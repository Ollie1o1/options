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
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config_ai import AI_CONFIG

logger = logging.getLogger(__name__)


# ── Load .env automatically ────────────────────────────────────────────────────
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass

_load_dotenv()


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert options trader and quantitative analyst. You receive options candidates \
from a professional technical screener and evaluate each trade setup.

Scoring guidelines (0-100):
  80-100  Exceptional - strong edge, clear catalyst-free runway, favorable vol regime
  60-79   Good - solid thesis, manageable risk
  40-59   Average - mixed signals, some risks
  20-39   Weak - unfavorable conditions, poor risk/reward
  0-19    Avoid - multiple red flags

For each candidate evaluate:
1. Vol regime: is IV cheap or expensive vs realized? Does this favor buyers or sellers?
2. Catalyst risk: any earnings, macro events, or news that could gap the stock before expiry?
3. Trend/momentum: does the option direction align with price action?
4. Probability and risk/reward: is the PoP justified by the premium?
5. Screener warnings: treat macro_warning, sr_warning, decay_warning as serious flags.

The "narrative_context" field in each candidate already interprets the raw numbers -
use it as your primary reference, then verify against the raw metrics.

Provide ai_confidence (0-10): 10 = strong consistent evidence; 1 = limited data.
Keep reasoning concise (1-2 sentences). Use short uppercase flag strings."""


_TICKER_CONTEXT_PROMPT = """\
You are an expert options market analyst. Analyze this ticker's current market conditions \
and provide a structured assessment that will inform individual option contract scoring.

Respond in JSON: {"regime": "SELLER_EDGE|BUYER_EDGE|NEUTRAL", "catalyst_risk": "low|medium|high", \
"directional_bias": "bullish|bearish|neutral", "key_risks": ["risk1", "risk2"], \
"summary": "2-3 sentence market overview", "confidence": <1-10>}"""


# ── JSON schema for contract scoring ──────────────────────────────────────────

_JSON_SCHEMA = """\
Respond with ONLY a JSON object in exactly this format:
{
  "scores": [
    {
      "id": "<candidate id>",
      "ai_score": <0-100>,
      "reasoning": "<1-2 sentences>",
      "flags": ["FLAG1", "FLAG2"],
      "catalyst_risk": "<low|medium|high>",
      "iv_justified": <true|false>,
      "ai_confidence": <0-10>
    }
  ]
}"""

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
    """Translate raw metric values into a human-readable narrative string."""
    thr = cfg.get("narrative_thresholds", {})
    parts: list[str] = []

    # IV rank interpretation
    iv_rank = c.get("iv_rank")
    if iv_rank is not None:
        if iv_rank >= thr.get("iv_rank_high", 0.70):
            parts.append(f"IV is at {iv_rank:.0%} percentile -- EXPENSIVE vs history, favors premium sellers.")
        elif iv_rank <= thr.get("iv_rank_low", 0.30):
            parts.append(f"IV is at {iv_rank:.0%} percentile -- CHEAP vs history, favors option buyers.")
        else:
            parts.append(f"IV is at {iv_rank:.0%} percentile -- neutral regime.")

    # IV vs HV (seller/buyer edge)
    iv_vs_hv = c.get("iv_vs_hv")
    if iv_vs_hv is not None:
        if iv_vs_hv >= thr.get("iv_vs_hv_rich", 0.05):
            parts.append(f"IV exceeds realized HV by {iv_vs_hv:.1%} -- clear seller edge.")
        elif iv_vs_hv <= thr.get("iv_vs_hv_cheap", -0.05):
            parts.append(f"IV is {abs(iv_vs_hv):.1%} BELOW realized HV -- option is cheap vs realized vol, buyer edge.")
        else:
            parts.append(f"IV and HV are close ({iv_vs_hv:+.1%} spread) -- no strong vol edge.")

    # PoP interpretation
    pop = c.get("pop_sim") or c.get("prob_profit")
    if pop is not None:
        if pop >= thr.get("pop_strong", 0.65):
            parts.append(f"PoP {pop:.0%} -- high probability trade.")
        elif pop <= thr.get("pop_weak", 0.45):
            parts.append(f"PoP {pop:.0%} -- below even-odds, requires strong directional conviction.")
        else:
            parts.append(f"PoP {pop:.0%} -- moderate probability.")

    # Risk/Reward
    rr = c.get("rr_ratio")
    if rr is not None:
        if rr >= thr.get("rr_good", 1.5):
            parts.append(f"RR {rr:.1f}x -- favorable risk/reward.")
        elif rr <= thr.get("rr_poor", 0.75):
            parts.append(f"RR {rr:.1f}x -- unfavorable; premium is large relative to potential gain.")

    # Earnings risk
    earnings_play = c.get("Earnings Play") or c.get("event_flag", "")
    earnings_date = c.get("earnings_date")
    if str(earnings_play).upper() == "YES" or "EARNINGS" in str(earnings_play).upper():
        parts.append(f"EARNINGS NEARBY ({earnings_date}) -- elevated gap risk; IV may crush post-event.")

    # Volume / unusual flow
    rvol = c.get("rvol")
    if rvol and float(rvol) >= thr.get("rvol_unusual", 1.5):
        parts.append(f"RVOL {rvol:.1f}x -- unusual volume; potential institutional interest.")

    # Theta decay pressure
    theta = c.get("theta")
    premium = c.get("premium")
    if theta and premium and float(premium) > 0:
        daily_bleed_pct = abs(float(theta)) / float(premium)
        if daily_bleed_pct >= thr.get("theta_decay_high", 0.05):
            parts.append(f"High theta burn: {daily_bleed_pct:.1%}/day of premium -- time is working against buyer.")

    # Spread width
    spread = c.get("spread_pct")
    if spread and float(spread) >= thr.get("spread_wide", 0.15):
        parts.append(f"Spread is {spread:.1%} -- wide; slippage will be significant, use limit orders.")

    # Score drivers (what the technical model liked)
    drivers = c.get("score_drivers")
    if drivers:
        parts.append(f"Technical score driven by: {drivers}.")

    # Active warnings
    warnings = []
    if c.get("macro_warning"):
        warnings.append(str(c["macro_warning"]))
    if c.get("sr_warning"):
        warnings.append(str(c["sr_warning"]))
    if c.get("decay_warning"):
        warnings.append("High decay pressure")
    if warnings:
        parts.append("Screener warnings: " + "; ".join(warnings) + ".")

    # News headlines
    headlines = c.get("_news_headlines", [])
    if headlines:
        parts.append("Recent news: " + " | ".join(f'"{h}"' for h in headlines[:3]))

    return " ".join(parts) if parts else "Insufficient data for narrative context."


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
        self._client = None
        self._cache = None
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
            # Write to cache
            if self._cache:
                cand_map = {c["_id"]: c for c in batch}
                for r in results:
                    raw = cand_map.get(r.get("id", ""))
                    if raw:
                        try:
                            self._cache.set(raw, r)
                        except Exception:
                            pass
            all_results.extend(results)
            if i < len(batches) - 1:
                time.sleep(0.5)

        return self._results_to_df(df, all_results)

    # ── Ticker context pass (Pass 1) ───────────────────────────────────────────

    def _score_ticker_context(self, symbol: str, ctx: dict, df: pd.DataFrame) -> dict:
        """Run a lightweight ticker-level analysis and return a summary dict."""
        underlying = None
        sym_rows = df[df.get("symbol", pd.Series(dtype=str)) == symbol] if "symbol" in df.columns else pd.DataFrame()
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
        if headlines:
            lines.append("Recent headlines: " + " | ".join(f'"{h}"' for h in headlines[:3]))

        prompt = "\n".join(lines) + "\n\nProvide your ticker-level assessment."

        try:
            raw = self._chat_complete(
                system=_TICKER_CONTEXT_PROMPT,
                user=prompt,
                max_tokens=512,
            )
            parsed = _parse_json_single(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            logger.warning("Ticker context parse failed for %s: %s", symbol, e)
            return {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        provider = self.config.get("provider", "openrouter")
        api_key = os.environ.get(self.config["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {self.config['api_key_env']!r} in your .env file.\n"
                "See .env.example for the format."
            )
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("Run:  pip install anthropic") from exc
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("Run:  pip install openai") from exc
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Ollie1o1/options",
                    "X-Title": "Options Screener AI Ranking",
                },
            )
        return self._client

    def _chat_complete(self, system: str, user: str, max_tokens: int = None, model: str = None) -> str:
        """Generic chat completion (OpenAI-compatible path)."""
        client = self._get_client()
        provider = self.config.get("provider", "openrouter")
        use_model = model or self.config["model"]
        use_max_tokens = max_tokens or self.config["max_tokens"]

        if provider == "anthropic":
            response = client.messages.create(
                model=use_model,
                max_tokens=use_max_tokens,
                temperature=self.config["temperature"],
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text if response.content else ""
        else:
            response = client.chat.completions.create(
                model=use_model,
                max_tokens=use_max_tokens,
                temperature=self.config["temperature"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content or ""

    def _score_batch_with_retry(self, batch: list[dict], batch_num: int = 1) -> list[dict]:
        """Score with exponential backoff; switches to fallback models on failure."""
        max_retries = 4
        delay = 5
        fallback = self.config.get("fallback_model")
        second_fallback = self.config.get("second_fallback_model")

        def _pick_model(attempt: int) -> str:
            if attempt <= 2:
                return self.config["model"]
            if attempt == 3:
                return fallback or self.config["model"]
            return second_fallback or fallback or self.config["model"]

        for attempt in range(1, max_retries + 1):
            use_model = _pick_model(attempt)
            try:
                return self._score_batch(batch, model=use_model)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if attempt < max_retries:
                    if is_rate_limit:
                        label = f"fallback ({use_model})" if attempt > 2 else "primary"
                        print(f"  [ai_scorer] batch {batch_num} rate-limited on {label} model, retrying in {delay}s...")
                    else:
                        print(f"  [ai_scorer] batch {batch_num} error (attempt {attempt}): {exc}")
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
            system=_SYSTEM_PROMPT,
            user=prompt,
            model=model,
        )
        return _parse_json_response(raw)

    def _score_batch_anthropic(self, batch: list[dict], model: str = None) -> list[dict]:
        client = self._get_client()
        prompt = self._build_prompt(batch, include_schema=False)
        use_model = model or self.config["model"]
        response = client.messages.create(
            model=use_model,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            system=_SYSTEM_PROMPT,
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
            lines = [f"Candidate ID: {c['_id']}"]
            if ticker_ctx:
                lines.append(f"  [Ticker Analysis] Regime: {ticker_ctx.get('regime','N/A')} | "
                             f"Catalyst risk: {ticker_ctx.get('catalyst_risk','N/A')} | "
                             f"Bias: {ticker_ctx.get('directional_bias','N/A')}")
                summary = ticker_ctx.get("summary")
                if summary:
                    lines.append(f"  [Ticker Summary] {summary}")
            lines.append(f"  [Context] {narrative}")
            for k, v in c.items():
                if k.startswith("_"):
                    continue
                if v is None:
                    lines.append(f"  {k}: N/A")
                elif isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
            blocks.append("\n".join(lines))

        body = "\n\n---\n\n".join(blocks)
        schema = f"\n\n{_JSON_SCHEMA}" if include_schema else \
            "\n\nCall the score_options_batch tool with your analysis."
        return f"Score the following {len(batch)} option candidate(s).\n\n{body}{schema}"

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

def _parse_json_response(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in response: {raw[:200]}")
    parsed = json.loads(text[start:end])
    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError(f"Expected 'scores' list, got: {type(scores)}")
    return scores

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
