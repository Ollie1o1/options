#!/usr/bin/env python3
"""AI-enhanced options screener.

Runs the technical screener then enriches candidates with AI-powered analysis
(narrative context, two-pass ticker analysis, IV justification, catalyst awareness)
producing a final dynamically-weighted ranked list.

Usage:
    python ai_rank.py AAPL
    python ai_rank.py AAPL TSLA NVDA --detail
    python ai_rank.py --mode "Premium Selling" --dte-min 14 --dte-max 45 SPY QQQ
    python ai_rank.py AAPL --no-ai
    python ai_rank.py SPY --ai-weight 0.5 --output ranked.csv --detail
    python ai_rank.py AAPL MSFT --portfolio-check

Environment variables (set in .env):
    OPENROUTER_API_KEY   Your OpenRouter API key
    ANTHROPIC_API_KEY    Your Anthropic API key (if using anthropic provider)
"""

from __future__ import annotations

import argparse
import logging
import sys
import io
import time

import pandas as pd

# UTF-8 stdout for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _import_screener():
    from src.options_screener import run_scan, get_market_context
    return run_scan, get_market_context


def _import_ai():
    from src.ai_scorer import AIScorer
    from src.ranking import combine_scores, print_ranked_table, to_csv, to_json
    return AIScorer, combine_scores, print_ranked_table, to_csv, to_json


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai_rank",
        description="AI-enhanced options screener -- technical + AI scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("tickers", nargs="+", metavar="TICKER",
                   help="One or more ticker symbols")

    scan = p.add_argument_group("screener")
    scan.add_argument("--mode", default="Single-stock",
                      choices=["Single-stock", "Budget scan", "Discovery scan",
                               "Premium Selling", "Credit Spreads", "Iron Condor"])
    scan.add_argument("--dte-min",  type=int,   default=7,   metavar="DAYS")
    scan.add_argument("--dte-max",  type=int,   default=45,  metavar="DAYS")
    scan.add_argument("--budget",   type=float, default=None, metavar="$")
    scan.add_argument("--expiries", type=int,   default=8,   metavar="N",
                      help="Max expiration dates per ticker (default: 8)")
    scan.add_argument("--profile",  default="swing",
                      choices=["swing", "day", "position"])

    ai = p.add_argument_group("AI scoring")
    ai.add_argument("--no-ai",     action="store_true",
                    help="Skip AI scoring; rank by quality_score only")
    ai.add_argument("--ai-weight", type=float, default=None, metavar="W",
                    help="Base AI weight 0-1 (default: from config_ai.py)")
    ai.add_argument("--top",       type=int,   default=20,  metavar="N",
                    help="Candidates to display (default: 20)")
    ai.add_argument("--detail",    action="store_true",
                    help="Show full AI reasoning for each pick")
    ai.add_argument("--portfolio-check", action="store_true",
                    help="Run AI portfolio coherence analysis on top picks")

    out = p.add_argument_group("output")
    out.add_argument("--output",  metavar="FILE", help="Save ranked results to CSV")
    out.add_argument("--json",    metavar="FILE", help="Save ranked results to JSON")
    out.add_argument("--update-weights", action="store_true",
                     help="Run paper trade IC analysis, write ic_weights_cache.json, and exit")
    out.add_argument("--quiet",   action="store_true",
                     help="Suppress screener verbose output")
    out.add_argument("--verbose", action="store_true",
                     help="Enable debug logging for the AI scorer")

    return p


def score_and_rank(
    picks: pd.DataFrame,
    ticker_contexts: dict,
    vix_regime: str,
    ai_weight: float | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Core AI scoring pipeline — no argparse dependency."""
    AIScorer, combine_scores, _, _, _ = _import_ai()
    scorer = AIScorer(config={"ai_weight": ai_weight} if ai_weight is not None else None)
    ai_df = scorer.score_candidates(picks, ticker_contexts=ticker_contexts)
    kwargs: dict = {}
    if ai_weight is not None:
        kwargs["ai_weight"] = ai_weight
        kwargs["technical_weight"] = 1.0 - ai_weight
    ranked = combine_scores(picks, ai_df, vix_regime=vix_regime, **kwargs)
    if verbose:
        stats = scorer.get_session_stats()
        print(f"  [session] {stats['api_calls']} API calls · {stats['api_retries']} retries · ~{stats['estimated_tokens']:,} tokens · {stats['cache_hits_today']} cache hits")
    return ranked


def _get_ticker_contexts(picks: pd.DataFrame, max_expiries: int = 4) -> dict[str, dict]:
    """Fetch lightweight ticker context for two-pass AI analysis.

    Deprecated: ticker_contexts are now threaded through run_scan() return value.
    This function is kept as a fallback for cases where pre-fetched contexts are
    unavailable (e.g. direct invocation without run_scan).
    """
    from src.data_fetching import fetch_options_yfinance, _CHAIN_CACHE
    contexts: dict[str, dict] = {}
    if "symbol" not in picks.columns:
        return contexts
    for symbol in picks["symbol"].unique():
        try:
            if symbol in _CHAIN_CACHE:
                contexts[symbol] = _CHAIN_CACHE[symbol].get("context", {})
                continue
            result = fetch_options_yfinance(symbol, max_expiries=min(max_expiries, 2))
            contexts[symbol] = result.get("context", {})
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Could not fetch context for %s: %s", symbol, e)
    return contexts


def _portfolio_check(ranked: pd.DataFrame, scorer) -> None:
    """Run an AI portfolio coherence check on the top 5 picks."""
    top5 = ranked.head(5)
    if top5.empty:
        return

    lines = ["Portfolio coherence check -- top picks:"]
    for _, row in top5.iterrows():
        sym    = row.get("symbol", "")
        opt_t  = str(row.get("type", "")).upper()
        strike = row.get("strike", "")
        exp    = str(row.get("expiration", ""))[:10]
        final  = float(row.get("final_score", 0)) * 100
        cat    = row.get("catalyst_risk", "medium")
        flags  = row.get("ai_flags", "")
        lines.append(f"  #{int(row.get('rank',0))} {sym} {opt_t} ${strike} exp {exp} | Score:{final:.1f} | Cat:{cat} | {flags}")

    prompt = "\n".join(lines) + (
        "\n\nAs a portfolio risk analyst, assess: (1) directional concentration "
        "(too many calls or puts?), (2) correlation risk between tickers, "
        "(3) catalyst overlap (multiple earnings events?), "
        "(4) overall portfolio Greek balance. "
        "Provide 3-4 sentences of actionable portfolio-level commentary."
    )

    system = (
        "You are a portfolio risk manager specializing in options. "
        "Give concise, actionable portfolio-level risk commentary. "
        "Focus on concentration, correlation, and catalyst risks."
    )

    raw = scorer.safe_chat_complete(system=system, user=prompt, max_tokens=400)

    if raw is not None:
        print()
        print("=" * 70)
        print("  PORTFOLIO COHERENCE CHECK")
        print("=" * 70)
        print(raw.strip())
        print("=" * 70)
        print()
    else:
        print("\n[portfolio-check] Failed — all models exhausted.")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.update_weights:
        import json as _json
        from datetime import datetime as _dt
        try:
            from src.backtester import run_paper_trade_ic
            from src.options_screener import load_config, load_ic_adjusted_weights
        except ImportError as _e:
            print(f"Import error: {_e}")
            sys.exit(1)
        ic_data = run_paper_trade_ic()
        component_ic = ic_data.get("component_ic", {})
        try:
            with open("ic_weights_cache.json", "w") as _f:
                _json.dump({"component_ic": component_ic, "generated": str(_dt.now())}, _f, indent=2)
            print(f"ic_weights_cache.json written. component_ic: {component_ic}")
            from src.options_screener import _invalidate_ic_weights_cache
            _invalidate_ic_weights_cache()
        except Exception as _we:
            print(f"Warning: could not write ic_weights_cache.json: {_we}")
        config = load_config()
        adj = load_ic_adjusted_weights(config)
        print("Adjusted composite weights:", adj)
        sys.exit(0)

    run_scan, get_market_context = _import_screener()

    market_trend = "Unknown"
    volatility_regime = "Unknown"
    macro_risk = False
    tnx_change = 0.0
    try:
        market_trend, volatility_regime, macro_risk, tnx_change = get_market_context()
    except Exception:
        pass

    print(f"\nRunning screener for: {', '.join(args.tickers)}")

    results = run_scan(
        mode=args.mode,
        tickers=args.tickers,
        budget=args.budget,
        max_expiries=args.expiries,
        min_dte=args.dte_min,
        max_dte=args.dte_max,
        trader_profile=args.profile,
        logger=logging.getLogger("screener"),
        market_trend=market_trend,
        volatility_regime=volatility_regime,
        macro_risk_active=macro_risk,
        tnx_change_pct=tnx_change,
        verbose=not args.quiet,
    )

    picks: pd.DataFrame = results.picks

    if picks.empty:
        print("\nNo candidates found - nothing to rank.")
        sys.exit(0)

    ai_label = "Skipping AI scoring." if args.no_ai else "Starting AI scoring..."
    print(f"\nFound {len(picks)} candidate(s).  {ai_label}")

    AIScorer, combine_scores, print_ranked_table, to_csv, to_json = _import_ai()

    # Map volatility_regime string to low/normal/high key
    vix_regime_map = {"Low": "low", "Normal": "normal", "High": "high",
                      "low": "low", "normal": "normal", "high": "high"}
    vix_regime = vix_regime_map.get(str(volatility_regime), "normal")

    scorer = None

    if args.no_ai:
        ranked = _rank_without_ai(picks)
    else:
        ai_config_override: dict = {}
        if args.ai_weight is not None:
            ai_config_override["ai_weight"] = args.ai_weight

        try:
            scorer = AIScorer(config=ai_config_override or None)

            # Use pre-fetched ticker contexts from run_scan (avoids double fetch)
            ticker_contexts: dict = results.ticker_contexts
            from src.config_ai import AI_CONFIG
            if not ticker_contexts and AI_CONFIG.get("two_pass_enabled", True):
                print("  Fetching ticker context for two-pass analysis...")
                ticker_contexts = _get_ticker_contexts(picks, max_expiries=args.expiries)

            ai_df = scorer.score_candidates(picks, ticker_contexts=ticker_contexts)

            kwargs: dict = {}
            if args.ai_weight is not None:
                kwargs["ai_weight"] = args.ai_weight
                kwargs["technical_weight"] = 1.0 - args.ai_weight

            ranked = combine_scores(picks, ai_df, vix_regime=vix_regime, **kwargs)

            if args.verbose:
                _stats = scorer.get_session_stats()
                print(
                    f"  [session] {_stats['api_calls']} API calls · "
                    f"{_stats['api_retries']} retries · "
                    f"~{_stats['estimated_tokens']:,} tokens · "
                    f"{_stats['cache_hits_today']} cache hits"
                )

            # Show cache stats if cache is active
            if scorer._cache:
                stats = scorer._cache.stats()
                print(f"  Cache: {stats['today']} entries today / {stats['total']} total.")

        except EnvironmentError as exc:
            print(f"\nConfiguration error: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"\nAI scoring failed ({exc}). Falling back to technical ranking.")
            ranked = _rank_without_ai(picks)
            scorer = None

    print_ranked_table(ranked, top_n=args.top, verbose_reasoning=args.detail)

    if args.portfolio_check and not args.no_ai and scorer is not None:
        try:
            _portfolio_check(ranked, scorer)
        except Exception:
            pass

    if args.output:
        to_csv(ranked, args.output)
    if args.json:
        to_json(ranked, args.json)


def _rank_without_ai(picks: pd.DataFrame) -> pd.DataFrame:
    df = picks.copy()
    df["ai_score"]      = None
    df["ai_skipped"]    = True
    df["ai_reasoning"]  = ""
    df["ai_flags"]      = ""
    df["catalyst_risk"] = "medium"
    df["iv_justified"]  = True
    df["ai_confidence"] = 5.0
    df["score_divergence"]    = 0.0
    df["divergence_flag"]     = False
    df["divergence_direction"] = "---"
    df["divergence_adjusted"] = False
    df["final_score"]   = df["quality_score"].fillna(0)
    df["ai_weight_used"] = 0.0
    df["rank"] = df["final_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank")


if __name__ == "__main__":
    main()
