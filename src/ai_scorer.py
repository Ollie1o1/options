"""AI-powered options candidate scorer.

Supports two providers configured in src/config_ai.py:
  - "openrouter"  — OpenAI-compatible API at openrouter.ai (free models available)
  - "anthropic"   — Anthropic Claude API

Features:
  - Same-day SQLite cache (ai_cache.py) — zero API calls on cache hits
  - Narrative context injection — raw numbers pre-translated to text
  - Two-pass scoring: Pass 1 = ticker context, Pass 2 = contract scoring
  - AI confidence field modulates final weight
  - Ticker-grouped batching
  - Fallback model chain on rate limit

API keys are loaded from a .env file in the project root.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config_ai import AI_CONFIG
from src.prompts import scoring_system_prompt, ticker_context_prompt, json_schema_instruction

logger = logging.getLogger(__name__)


# ── Load .env automatically ────────────────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass
    # Fallback: parse .env manually when python-dotenv is not installed
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # Strip inline comments (not inside quotes)
            if "#" in value and not value.startswith(("'", '"')):
                value = value[:value.index("#")]
            value = value.strip().strip("'\"")
            if key and not os.environ.get(key):
                os.environ[key] = value

_load_dotenv()


# ── JSON schema for contract scoring ──────────────────────────────────────────

# Tool schema for Anthropic provider
_SCORING_TOOL: dict[str, Any] = {
    "name": "score_options_batch",
    "description": "Score a batch of options candidates and return structured analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "ai_score": {"type": "number", "minimum": 0, "maximum": 100},
                        "reasoning": {"type": "string"},
                        "flags": {"type": "array", "items": {"type": "string"}},
                        "catalyst_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "iv_justified": {"type": "boolean"},
                        "ai_confidence": {"type": "number", "minimum": 0, "maximum": 10},
                    },
                    "required": ["id", "ai_score", "reasoning", "flags",
                                 "catalyst_risk", "iv_justified", "ai_confidence"],
                },
            }
        },
        "required": ["scores"],
    },
}


# ── Narrative context enrichment ───────────────────────────────────────────────

def _enrich_candidate_context(c: dict[str, Any], cfg: dict) -> str:
    """Compact key:value context — ~3x fewer tokens than prose version."""
    thr = cfg.get("narrative_thresholds", {})
    kv: list[str] = []

    # IV numerical precision (number+label with HV gap)
    iv_rank_val = c.get("iv_rank_30") or c.get("iv_rank")
    iv_abs = c.get("impliedVolatility")
    hv_val = c.get("hv_30d")
    if iv_rank_val is not None:
        pct = float(iv_rank_val) * 100
        label = "expensive" if pct > 70 else ("cheap" if pct < 30 else "neutral")
        iv_str = f"IV:{pct:.0f}%({label})"
        if iv_abs and hv_val and float(hv_val) > 0:
            gap = (float(iv_abs) - float(hv_val)) * 100
            iv_str += f" HV:{float(hv_val)*100:.1f}% gap:{gap:+.1f}pp"
        kv.append(iv_str)

    # Greek values (delta+gamma+theta as numbers)
    delta_v = c.get("delta") or c.get("abs_delta")
    gamma_v = c.get("gamma")
    theta_v = c.get("theta")
    greek_parts = []
    if delta_v is not None: greek_parts.append(f"δ:{float(delta_v):.2f}")
    if gamma_v is not None: greek_parts.append(f"γ:{float(gamma_v):.4f}")
    if theta_v is not None: greek_parts.append(f"θ:{float(theta_v):.3f}")
    if greek_parts:
        kv.append(" ".join(greek_parts))

    # Term structure as number
    ts = c.get("term_structure_spread")
    if ts is not None:
        ts_f = float(ts) * 100
        label = "backwd" if ts_f < 0 else "contango"
        kv.append(f"termstruct:{ts_f:+.1f}pp({label})")

    # Momentum confluence (new from Change 2)
    mom_conf = c.get("momentum_confluence")
    if mom_conf is not None:
        kv.append(f"mom-confluence:{float(mom_conf):.2f}")

    # Risk flag count
    rfc = c.get("risk_flag_count")
    if rfc is not None and int(rfc) >= 2:
        kv.append(f"RISK-FLAGS:{int(rfc)}")

    # Max pain
    max_pain = c.get("max_pain_strike")
    mp_dist = c.get("max_pain_dist_pct")
    if max_pain is not None and mp_dist is not None:
        mp_dist_f = float(mp_dist)
        if mp_dist_f < 2.0:
            kv.append(f"max-pain:${float(max_pain):.0f}(PINNING-{mp_dist_f:.1f}%)")
        elif mp_dist_f < 5.0:
            kv.append(f"max-pain:${float(max_pain):.0f}({mp_dist_f:.1f}%away)")

    iv_vs_hv = c.get("iv_vs_hv")
    if iv_vs_hv is not None:
        if iv_vs_hv >= thr.get("iv_vs_hv_rich", 0.05):
            kv.append(f"IVvHV:+{iv_vs_hv:.1%}(seller-edge)")
        elif iv_vs_hv <= thr.get("iv_vs_hv_cheap", -0.05):
            kv.append(f"IVvHV:{iv_vs_hv:.1%}(buyer-edge)")

    pop = c.get("pop_sim") or c.get("prob_profit")
    if pop is not None:
        label = "HIGH" if pop >= thr.get("pop_strong", 0.65) else ("LOW" if pop <= thr.get("pop_weak", 0.45) else "MED")
        kv.append(f"PoP:{pop:.0%}({label})")

    rr = c.get("rr_ratio")
    if rr is not None:
        label = "OK" if rr >= thr.get("rr_good", 1.5) else ("POOR" if rr <= thr.get("rr_poor", 0.75) else "fair")
        kv.append(f"RR:{rr:.1f}x({label})")

    theta = c.get("theta")
    premium = c.get("premium")
    if theta and premium and float(premium) > 0:
        bleed = abs(float(theta)) / float(premium)
        if bleed >= thr.get("theta_decay_high", 0.05):
            kv.append(f"theta:{bleed:.1%}/day(HIGH)")

    spread = c.get("spread_pct")
    if spread and float(spread) >= thr.get("spread_wide", 0.15):
        kv.append(f"spread:{spread:.0%}(WIDE)")

    rvol = c.get("rvol")
    if rvol and float(rvol) >= thr.get("rvol_unusual", 1.5):
        kv.append(f"rvol:{rvol:.1f}x(unusual)")

    opt_rvol = c.get("option_rvol")
    if opt_rvol is not None and float(opt_rvol) >= 3.0:
        kv.append(f"opt-rvol:{float(opt_rvol):.1f}x(CONTRACT-unusual)")

    if str(c.get("Earnings Play", "")).upper() == "YES":
        kv.append("EARNINGS(gap-risk)")

    be_dist = c.get("be_dist_pct")
    if be_dist is not None:
        kv.append(f"BE-dist:{float(be_dist):.1f}%")

    ann_ret = c.get("annualized_return")
    if ann_ret is not None and float(ann_ret) > 0:
        kv.append(f"ann-yield:{float(ann_ret):.0%}")

    warnings = []
    if c.get("macro_warning"):
        warnings.append("MACRO")
    if c.get("sr_warning"):
        warnings.append("SR")
    if c.get("decay_warning"):
        warnings.append("DECAY")
    if c.get("gamma_ramp"):
        warnings.append("GAMMA-RAMP")
    if warnings:
        kv.append("WARN:" + "+".join(warnings))

    headlines = c.get("_news_headlines", [])
    if headlines:
        kv.append("news:" + "|".join(f'"{h[:40]}"' for h in headlines[:2]))

    unusual_flow = c.get("_unusual_flow")
    if unusual_flow:
        try:
            n_contracts = len(unusual_flow) if isinstance(unusual_flow, list) else int(unusual_flow)
            kv.append(f"unusual-flow:{n_contracts}contracts")
        except Exception:
            pass

    iv_skew_rank = c.get("iv_skew_rank")
    if iv_skew_rank is not None:
        if float(iv_skew_rank) >= 0.80:
            kv.append(f"skew-rank:{float(iv_skew_rank):.0%}(ELEVATED-fear)")
        elif float(iv_skew_rank) <= 0.20:
            kv.append(f"skew-rank:{float(iv_skew_rank):.0%}(DEPRESSED-complacency)")

    vrp_regime = c.get("vrp_regime")
    vrp_mean = c.get("vrp_mean")
    if vrp_regime and vrp_regime != "UNKNOWN":
        vrp_str = f"{float(vrp_mean)*100:+.1f}%" if vrp_mean is not None else ""
        kv.append(f"VRP:{vrp_regime}{vrp_str}")

    predicted_crush = c.get("predicted_iv_crush")
    crush_conf = c.get("crush_confidence", "")
    if predicted_crush is not None and float(predicted_crush) > 0.05:
        kv.append(f"IV-crush:{float(predicted_crush)*100:.0f}%pts-predicted({crush_conf})")

    gamma_pin_dist = c.get("gamma_pin_dist_pct")
    max_gamma_strike = c.get("max_gamma_strike")
    if gamma_pin_dist is not None and max_gamma_strike is not None:
        dist_f = float(gamma_pin_dist)
        if dist_f < 3.0:
            kv.append(f"gamma-pin:${float(max_gamma_strike):.0f}(NEAR-{dist_f:.1f}%)")
        elif dist_f < 7.0:
            kv.append(f"gamma-pin:${float(max_gamma_strike):.0f}({dist_f:.1f}%away)")

    return " ".join(kv) if kv else "no-context"


# ── Public class ───────────────────────────────────────────────────────────────


class AIScorer:
    """Score options candidates using a configurable AI provider.

    Parameters
    ----------
    config:
        Dict of overrides applied on top of ``AI_CONFIG``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {**AI_CONFIG, **(config or {})}
        from src.config_validator import validate_ai_config
        validate_ai_config(self.config)
        self._clients: dict[str, Any] = {}   # keyed by api_key_env name
        self._cache = None
        self._api_call_count = 0
        self._api_token_estimate = 0
        self._api_retry_count = 0
        self._session_warnings_issued: set = set()
        if self.config.get("cache_enabled", True):
            try:
                from src.ai_cache import AIScoreCache
                self._cache = AIScoreCache()
                self._cache.clear_stale()
            except Exception as e:
                logger.warning("Cache init failed: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_candidates(
        self,
        df: pd.DataFrame,
        ticker_contexts: Optional[dict[str, dict]] = None,
    ) -> pd.DataFrame:
        """Score all rows in *df* and return a parallel DataFrame of AI results.

        Parameters
        ----------
        df:
            Screener picks DataFrame.
        ticker_contexts:
            Optional dict mapping symbol -> context dict from fetch_options_yfinance.
            Used for the two-pass ticker analysis.

        Returns
        -------
        DataFrame with columns: ai_score, ai_reasoning, ai_flags,
                                 catalyst_risk, iv_justified, ai_confidence
        """
        if df.empty:
            return _empty_results(df)

        # Run two-pass ticker analysis if enabled and contexts provided
        ticker_summaries: dict[str, dict] = {}
        if self.config.get("two_pass_enabled", True) and ticker_contexts:
            for symbol, ctx in ticker_contexts.items():
                try:
                    ticker_summaries[symbol] = self._score_ticker_context(symbol, ctx, df)
                except Exception as e:
                    logger.warning("Ticker context pass failed for %s: %s", symbol, e)

        candidates = self._extract_candidates(df, ticker_summaries)
        all_results: list[dict] = []
        cache_hits = 0

        # Check cache for each candidate
        uncached: list[dict] = []
        if self._cache:
            for c in candidates:
                cached = self._cache.get(c)
                if cached:
                    cached["id"] = c["_id"]
                    all_results.append(cached)
                    cache_hits += 1
                else:
                    uncached.append(c)
        else:
            uncached = candidates

        if cache_hits:
            print(f"  [ai_scorer] {cache_hits} candidate(s) loaded from cache.")

        if not uncached:
            return self._results_to_df(df, all_results)

        # Sort by ticker for grouped batching
        uncached.sort(key=lambda c: c.get("symbol", ""))

        batch_size = self.config["batch_size"]
        batches = [uncached[i: i + batch_size] for i in range(0, len(uncached), batch_size)]

        for i, batch in enumerate(batches):
            logger.info("Scoring batch %d/%d (%d candidates)...", i + 1, len(batches), len(batch))
            results = self._score_batch_with_retry(batch, batch_num=i + 1)
            # Write to cache — skip default/fallback results so they get re-scored next run
            if self._cache:
                cand_map = {c["_id"]: c for c in batch}
                for r in results:
                    if r.get("reasoning", "").startswith("Scoring unavailable"):
                        continue
                    raw = cand_map.get(r.get("id", ""))
                    if raw:
                        try:
                            self._cache.set(raw, r)
                        except Exception:
                            pass
            all_results.extend(results)
            if i < len(batches) - 1:
                time.sleep(0.5)

        if self._api_call_count > 0:
            logger.info("Session: %d API calls, ~%d tokens estimated", self._api_call_count, self._api_token_estimate)

        return self._results_to_df(df, all_results)

    # ── Ticker context pass (Pass 1) ───────────────────────────────────────────

    def _score_ticker_context(self, symbol: str, ctx: dict, df: pd.DataFrame) -> dict:
        """Run a lightweight ticker-level analysis and return a summary dict."""
        # Fast path: return cached result without any processing
        if self._cache:
            cached = self._cache.get_ticker_context(symbol)
            if cached:
                return cached

        underlying = None
        sym_rows = df[df["symbol"] == symbol] if "symbol" in df.columns else pd.DataFrame()
        if not sym_rows.empty and "underlying" in sym_rows.columns:
            underlying = float(sym_rows["underlying"].iloc[0])

        iv_rank = ctx.get("iv_rank")
        iv_pct = ctx.get("iv_percentile")
        hv = ctx.get("hv")
        term_spread = ctx.get("term_structure_spread")
        earnings = ctx.get("earnings_date")
        sentiment = ctx.get("sentiment_score")
        rvol = ctx.get("rvol")
        short_int = ctx.get("short_interest")
        headlines = ctx.get("news_headlines", [])
        earnings_move = ctx.get("earnings_move_data") or {}

        lines = [f"Ticker: {symbol}"]
        if underlying:
            lines.append(f"Current price: ${underlying:.2f}")
        if iv_rank is not None:
            regime_txt = "HIGH (expensive)" if iv_rank > 0.65 else ("LOW (cheap)" if iv_rank < 0.35 else "NORMAL")
            lines.append(f"IV Rank 30d: {iv_rank:.0%} -- {regime_txt}")
        if hv:
            lines.append(f"30d Realized HV: {hv:.1%}")
        if term_spread is not None:
            ts_label = "CONTANGO" if term_spread > 0.005 else ("BACKWARDATION" if term_spread < -0.005 else "FLAT")
            lines.append(f"Term structure: {term_spread:+.2%} ({ts_label})")
        if earnings:
            implied_move = earnings_move.get("implied_move_pct")
            hist_move = earnings_move.get("hist_avg_move")
            lines.append(f"Earnings: {earnings}")
            if implied_move:
                lines.append(f"  Implied earnings move: +/-{implied_move:.1%}" +
                              (f" vs hist avg +/-{hist_move:.1%}" if hist_move else ""))
        if sentiment is not None:
            sent_label = "Bullish" if sentiment > 0.05 else ("Bearish" if sentiment < -0.05 else "Neutral")
            lines.append(f"News sentiment: {sent_label} ({sentiment:+.3f})")
        if rvol:
            lines.append(f"Relative volume: {rvol:.2f}x")
        if short_int:
            lines.append(f"Short interest: {short_int:.1%}")
        # Prefer structured news_data when available (multi-source, with sentiment + analyst changes)
        news_data = ctx.get("news_data")
        if news_data is not None and hasattr(news_data, "top_headlines"):
            enriched_headlines = news_data.top_headlines[:3]
            if enriched_headlines:
                lines.append("Recent headlines: " + " | ".join(f'"{h}"' for h in enriched_headlines))
            if getattr(news_data, "has_negative_catalyst", False):
                lines.append("CATALYST RISK flagged in recent news")
            if getattr(news_data, "has_positive_catalyst", False):
                lines.append("POSITIVE CATALYST detected in recent news")
            if getattr(news_data, "unusual_news_volume", False):
                lines.append("UNUSUAL NEWS VOLUME in last 24h")
            analyst_changes = getattr(news_data, "analyst_changes", [])
            if analyst_changes:
                ac_strs = []
                for ac in analyst_changes[:3]:
                    grade_str = f"{ac.from_grade}→{ac.to_grade}" if ac.from_grade and ac.to_grade else (ac.to_grade or ac.action)
                    pt_str = f" PT:${ac.price_target:.0f}" if ac.price_target else ""
                    ac_strs.append(f"{ac.firm} {ac.action.upper()} {grade_str}{pt_str}")
                lines.append("Analyst changes (30d): " + " | ".join(ac_strs))
        elif headlines:
            lines.append("Recent headlines: " + " | ".join(f'"{h}"' for h in headlines[:3]))

        # Unusual options flow (from Polygon, when available)
        unusual_flow = ctx.get("unusual_options_flow")
        if unusual_flow:
            flow_parts = []
            for contract in unusual_flow[:3]:
                det = contract.get("details") or {}
                day = contract.get("day") or {}
                ctype = det.get("contract_type", "?").upper()
                strike = det.get("strike_price", "?")
                exp = det.get("expiration_date", "?")
                vol = day.get("volume", "?")
                dv = contract.get("_dollar_volume", 0)
                dv_k = f"${dv/1000:.0f}K" if dv else ""
                flow_parts.append(f"{ctype} {strike} {exp} vol={vol} {dv_k}")
            lines.append("Unusual options flow: " + " | ".join(flow_parts))

        # Company description (from Polygon, when available)
        company_desc = ctx.get("company_description", "")
        if company_desc:
            lines.append(f"Company: {company_desc[:200]}")

        market_cap = ctx.get("market_cap")
        if market_cap:
            mc_b = market_cap / 1e9
            lines.append(f"Market cap: ${mc_b:.1f}B")

        prompt = "\n".join(lines) + "\n\nProvide your ticker-level assessment."

        try:
            raw = self._chat_complete(
                system=ticker_context_prompt(),
                user=prompt,
                max_tokens=512,
            )
            parsed = _parse_json_single(raw)
            result = parsed if isinstance(parsed, dict) else {}
            if result and self._cache:
                self._cache.set_ticker_context(symbol, result)
            return result
        except Exception as e:
            logger.warning("Ticker context parse failed for %s: %s", symbol, e)
            return {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _key_env_for_model(self, model: str) -> str:
        """Return the env-var name that holds the API key for *model*."""
        from src.config_ai import resolve_api_key_env
        return resolve_api_key_env(model, self.config)

    def get_session_stats(self) -> dict:
        """Return a summary of this session's API and cache usage."""
        return {
            "api_calls":        self._api_call_count,
            "api_retries":      self._api_retry_count,
            "estimated_tokens": self._api_token_estimate,
            "cache_hits_today": self._cache.stats()["today"] if self._cache else 0,
        }

    def analyze_thematic_sentiment(self, headlines: list) -> dict:
        """Optional AI Pass 0: sector sentiment from headlines.

        Returns a dict mapping ETF ticker -> sentiment float in [-0.20, +0.20].
        Returns {} immediately when disabled or on any failure.
        """
        import hashlib

        if not self.config.get("thematic_analysis_enabled", False):
            return {}
        if not headlines:
            return {}

        # 30-min in-process cache keyed on MD5 of headlines
        if not hasattr(self, "_thematic_cache"):
            self._thematic_cache: dict = {}
        key = hashlib.md5("|".join(str(h) for h in headlines).encode()).hexdigest()
        cached = self._thematic_cache.get(key)
        if cached is not None:
            ts, val = cached
            if time.time() - ts < 1800:
                return val

        etf_to_sector = self.config.get("etf_to_sector", {
            "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
            "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
            "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
            "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
        })
        sector_list = ", ".join(f"{etf}={name}" for etf, name in etf_to_sector.items())

        system_prompt = (
            "You are a macro analyst. Given news headlines, score each SPDR sector ETF "
            "with a sentiment float in [-0.20, +0.20] (positive = tailwind, negative = headwind). "
            f"Sectors: {sector_list}. "
            "Return ONLY valid JSON like: {\"XLK\": 0.12, \"XLE\": -0.08}. No commentary."
        )
        user_prompt = "Headlines:\n" + "\n".join(f"- {h}" for h in headlines[:15])

        try:
            raw = self.safe_chat_complete(system=system_prompt, user=user_prompt, max_tokens=300)
            if not raw:
                return {}
            # Extract JSON object from response
            import re
            m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not m:
                return {}
            data = json.loads(m.group())
            result = {k: max(-0.20, min(0.20, float(v))) for k, v in data.items() if k in etf_to_sector}
            self._thematic_cache[key] = (time.time(), result)
            return result
        except Exception as exc:
            logger.debug("analyze_thematic_sentiment failed: %s", exc)
            return {}

    def safe_chat_complete(self, system: str, user: str, max_tokens: int = 400) -> Optional[str]:
        """Chat completion with fallback model chain. Returns None on all-retries-exhausted."""
        models = [
            self.config["model"],
            self.config.get("fallback_model"),
            self.config.get("second_fallback_model"),
            self.config.get("third_fallback_model"),
        ]
        models = [m for m in models if m]
        last_exc = None
        for model in models:
            for attempt in range(2):
                try:
                    return self._chat_complete(system=system, user=user, max_tokens=max_tokens, model=model)
                except Exception as e:
                    last_exc = e
                    is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                    time.sleep(5 * (2 ** attempt) if is_rate_limit else 1)
        logger.warning("safe_chat_complete failed after all models: %s", last_exc)
        return None

    def _get_client(self, key_env: str = None):
        """Return (and cache) an OpenAI/Anthropic client for the given key env var."""
        provider = self.config.get("provider", "openrouter")
        env_name = key_env or self.config["api_key_env"]
        if env_name in self._clients:
            return self._clients[env_name]
        api_key = os.environ.get(env_name)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {env_name!r} in your .env file.\n"
                "See .env.example for the format."
            )
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("Run:  pip install anthropic") from exc
            client = anthropic.Anthropic(api_key=api_key)
        else:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("Run:  pip install openai") from exc
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Ollie1o1/options",
                    "X-Title": "Options Screener AI Ranking",
                },
            )
        self._clients[env_name] = client
        return client

    def _chat_complete(self, system: str, user: str, max_tokens: int = None, model: str = None) -> str:
        """Generic chat completion (OpenAI-compatible path)."""
        provider = self.config.get("provider", "openrouter")
        use_model = model or self.config["model"]
        use_max_tokens = max_tokens or self.config["max_tokens"]
        client = self._get_client(self._key_env_for_model(use_model))

        timeout_secs = self.config.get("timeout", 30)
        if provider == "anthropic":
            response = client.messages.create(
                model=use_model,
                max_tokens=use_max_tokens,
                temperature=self.config["temperature"],
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=timeout_secs,
            )
            return response.content[0].text if (response.content and len(response.content) > 0) else ""
        else:
            response = client.chat.completions.create(
                model=use_model,
                max_tokens=use_max_tokens,
                temperature=self.config["temperature"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                timeout=timeout_secs,
            )
            return response.choices[0].message.content or ""

    def _score_batch_with_retry(self, batch: list[dict], batch_num: int = 1) -> list[dict]:
        """Score with exponential backoff; switches to fallback models on failure."""
        max_retries = 5
        delay = 5
        fallback        = self.config.get("fallback_model")
        second_fallback = self.config.get("second_fallback_model")
        third_fallback  = self.config.get("third_fallback_model")

        def _pick_model(attempt: int) -> str:
            if attempt <= 2:
                return self.config["model"]
            if attempt == 3:
                return fallback or self.config["model"]
            if attempt == 4:
                return second_fallback or fallback or self.config["model"]
            return third_fallback or second_fallback or fallback or self.config["model"]

        for attempt in range(1, max_retries + 1):
            use_model = _pick_model(attempt)
            try:
                result = self._score_batch(batch, model=use_model)
                self._api_call_count += 1
                estimated_tokens = len(batch) * 600  # rough estimate: ~600 tokens per candidate (prompt + response)
                self._api_token_estimate += estimated_tokens
                if self._api_call_count % 10 == 0:
                    print(f"  [ai_scorer] {self._api_call_count} API calls this session, ~{self._api_token_estimate:,} tokens estimated")
                return result
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if attempt < max_retries:
                    if is_rate_limit:
                        label = f"fallback ({use_model})" if attempt > 2 else "primary"
                        print(f"  [ai_scorer] batch {batch_num} rate-limited on {label} model, retrying in {delay}s...")
                    else:
                        print(f"  [ai_scorer] batch {batch_num} error (attempt {attempt}): {exc}")
                    self._api_retry_count += 1
                    time.sleep(delay if is_rate_limit else 1)
                    delay = min(delay * 2, 60)
                else:
                    logger.warning("Batch %d failed after %d attempts: %s", batch_num, attempt, exc)
                    return [_default_result(c["_id"]) for c in batch]
        return [_default_result(c["_id"]) for c in batch]

    def _score_batch(self, batch: list[dict], model: str = None) -> list[dict]:
        provider = self.config.get("provider", "openrouter")
        if provider == "anthropic":
            return self._score_batch_anthropic(batch, model=model)
        return self._score_batch_openai(batch, model=model)

    def _score_batch_openai(self, batch: list[dict], model: str = None) -> list[dict]:
        prompt = self._build_prompt(batch)
        raw = self._chat_complete(
            system=scoring_system_prompt(),
            user=prompt,
            model=model,
        )
        return _parse_json_response(raw)

    def _score_batch_anthropic(self, batch: list[dict], model: str = None) -> list[dict]:
        use_model = model or self.config["model"]
        client = self._get_client(self._key_env_for_model(use_model))
        prompt = self._build_prompt(batch, include_schema=False)
        response = client.messages.create(
            model=use_model,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            system=scoring_system_prompt(),
            tools=[_SCORING_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "score_options_batch":
                return block.input.get("scores", [])
        raise ValueError("Anthropic response missing tool_use block.")

    def _build_prompt(self, batch: list[dict], include_schema: bool = True) -> str:
        blocks = []
        for c in batch:
            narrative = _enrich_candidate_context(c, self.config)
            ticker_ctx = c.get("_ticker_context", {})

            # Header: contract identity on one line
            sym   = c.get("symbol", "?")
            ctype = str(c.get("type", "?")).upper()
            strike = c.get("strike", "?")
            exp    = str(c.get("expiration", "?"))[:10]
            udly   = c.get("underlying")
            prem   = c.get("premium")
            iv_r   = c.get("iv_rank")
            delta  = c.get("delta")
            pop    = c.get("prob_profit")
            ev     = c.get("ev_per_contract")
            rr     = c.get("rr_ratio")
            trend  = c.get("Trend_Aligned")
            earn   = c.get("Earnings Play", "NO")

            raw_parts = []
            if udly is not None:   raw_parts.append(f"udly:{float(udly):.1f}")
            if prem is not None:   raw_parts.append(f"prem:{float(prem):.2f}")
            if iv_r is not None:   raw_parts.append(f"iv_rank:{float(iv_r):.0%}")
            if delta is not None:  raw_parts.append(f"delta:{float(delta):.2f}")
            if pop is not None:    raw_parts.append(f"PoP:{float(pop):.0%}")
            if ev is not None:     raw_parts.append(f"EV:{float(ev):.0f}")
            if rr is not None:     raw_parts.append(f"RR:{float(rr):.1f}x")
            if trend is not None:  raw_parts.append(f"trend:{'Y' if trend else 'N'}")
            raw_parts.append(f"earn:{earn}")

            # Ticker context (one line)
            ctx_line = ""
            if ticker_ctx:
                # Cap ctx_line to avoid inflating prompt beyond max_tokens budget
                ctx_line = (f"\n  ctx: regime={ticker_ctx.get('regime','?')} "
                            f"cat={ticker_ctx.get('catalyst_risk','?')} "
                            f"bias={ticker_ctx.get('directional_bias','?')} "
                            f"summary=\"{str(ticker_ctx.get('summary',''))[:80]}\"")
                ctx_line = ctx_line[:200]

            # IV surface residual context
            iv_resid = c.get("iv_surface_residual")
            iv_surface_line = ""
            if iv_resid is not None and not (isinstance(iv_resid, float) and math.isnan(iv_resid)):
                if iv_resid < -0.02:
                    iv_surface_line = f"\n  IV Surface: CHEAP vs SVI surface (residual: {iv_resid:+.3f})"
                elif iv_resid > 0.02:
                    iv_surface_line = f"\n  IV Surface: RICH vs SVI surface (residual: {iv_resid:+.3f})"
                else:
                    iv_surface_line = f"\n  IV Surface: Fair vs SVI surface (residual: {iv_resid:+.3f})"

            block = (
                f"ID:{c['_id']} {sym} {ctype} ${strike} exp:{exp}\n"
                f"  {' '.join(raw_parts)}\n"
                f"  narrative: {narrative}"
                + ctx_line
                + iv_surface_line
            )
            blocks.append(block)

        body = "\n---\n".join(blocks)
        schema = f"\n{json_schema_instruction()}" if include_schema else \
            "\nCall the score_options_batch tool with your analysis."
        return f"Score {len(batch)} option(s):\n{body}{schema}"

    def _extract_candidates(
        self,
        df: pd.DataFrame,
        ticker_summaries: dict[str, dict],
    ) -> list[dict]:
        wanted = [f for f in self.config["fields_to_include"] if f in df.columns]
        candidates: list[dict] = []
        for idx, row in df.iterrows():
            c: dict[str, Any] = {"_id": str(idx)}
            for field in wanted:
                val = row[field]
                try:
                    is_na = pd.isna(val)
                except (TypeError, ValueError):
                    is_na = False
                c[field] = None if is_na else val

            # Attach ticker context
            sym = str(c.get("symbol", ""))
            if sym in ticker_summaries:
                c["_ticker_context"] = ticker_summaries[sym]

            candidates.append(c)
        return candidates

    def _results_to_df(self, original_df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
        score_map = {r["id"]: r for r in results if "id" in r}
        rows = []
        for idx in original_df.index:
            r = score_map.get(str(idx), _default_result(str(idx)))
            rows.append({
                "ai_score":      float(r.get("ai_score", 50)),
                "ai_reasoning":  str(r.get("reasoning", "")),
                "ai_flags":      ", ".join(r.get("flags") or []),
                "catalyst_risk": str(r.get("catalyst_risk", "medium")),
                "iv_justified":  bool(r.get("iv_justified", True)),
                "ai_confidence": float(r.get("ai_confidence", 5.0)),
            })
        return pd.DataFrame(rows, index=original_df.index)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_partial_scores(text: str) -> list[dict]:
    """Scan text at every brace depth and collect every complete, valid JSON
    object that contains both 'id' and 'ai_score' keys.

    Score objects are nested inside {"scores": [...]}, so they never return
    to depth-0; we must scan at all nesting levels.
    """
    scores: list[dict] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Try to find the matching closing brace for this opening brace
        depth = 0
        start = i
        end = -1
        for j in range(i, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            # This opening brace has no matching close (truncated envelope).
            # Skip it and keep scanning — inner objects may still be complete.
            i += 1
            continue
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            # Only keep score-leaf objects (have ai_score + id, not the "scores" envelope)
            if isinstance(obj, dict) and "ai_score" in obj and "id" in obj:
                scores.append(obj)
        except json.JSONDecodeError:
            pass
        i = end + 1
    return scores


def _parse_json_response(raw: str) -> list[dict]:
    """Parse AI scoring response, with partial-JSON recovery for truncated responses."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in response: {raw[:200]}")

    # Happy path: complete, valid JSON
    try:
        parsed = json.loads(text[start:end])
        scores = parsed.get("scores", [])
        if not isinstance(scores, list):
            raise ValueError(f"Expected 'scores' list, got: {type(scores)}")
        return scores
    except json.JSONDecodeError:
        pass

    # Recovery path: response was truncated mid-JSON.
    # Walk the text and extract every complete score object we can.
    recovered = _extract_partial_scores(text)
    if recovered:
        logger.warning(
            "AI response truncated — recovered %d/%s complete score object(s). "
            "Consider increasing max_tokens or reducing batch_size.",
            len(recovered), "?"
        )
        return recovered

    raise ValueError(f"No JSON object in response: {raw[:200]}")

def _parse_json_single(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    return json.loads(text[start:end])

def _default_result(id_: str) -> dict:
    return {
        "id": id_,
        "ai_score": 50.0,
        "reasoning": "Scoring unavailable - using neutral default.",
        "flags": [],
        "catalyst_risk": "medium",
        "iv_justified": True,
        "ai_confidence": 5.0,
    }

def _empty_results(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ai_score":      pd.Series(dtype=float),
            "ai_reasoning":  pd.Series(dtype=str),
            "ai_flags":      pd.Series(dtype=str),
            "catalyst_risk": pd.Series(dtype=str),
            "iv_justified":  pd.Series(dtype=bool),
            "ai_confidence": pd.Series(dtype=float),
        },
        index=df.index,
    )
