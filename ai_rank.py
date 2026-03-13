#!/usr/bin/env python3
"""AI-enhanced options screener.

Runs the technical screener then enriches the top candidates with
AI-powered analysis (IV justification, catalyst awareness, risk commentary)
using the Claude API, producing a final weighted ranked list.

Usage examples:
    python ai_rank.py TSLA AAPL NVDA
    python ai_rank.py --mode "Premium Selling" --dte-min 14 --dte-max 45 SPY QQQ
    python ai_rank.py AAPL --no-ai               # rank by quality_score only
    python ai_rank.py SPY --output ranked.csv    # save results to CSV
    python ai_rank.py SPY --json ranked.json     # save results to JSON
    python ai_rank.py --help

Environment variables:
    ANTHROPIC_API_KEY   Your Anthropic API key (required unless --no-ai is set)
"""

from __future__ import annotations

import argparse
import logging
import sys
import io

import pandas as pd

# Reconfigure stdout/stderr to UTF-8 on Windows so unicode symbols (✓, ★, Δ…)
# don't crash when the terminal is set to a legacy code page (e.g. CP1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Lazy import helpers (so --help works without heavy deps loaded) ─────────────


def _import_screener():
    from src.options_screener import run_scan, get_market_context  # noqa: PLC0415
    return run_scan, get_market_context


def _import_ai():
    from src.ai_scorer import AIScorer  # noqa: PLC0415
    from src.ranking import combine_scores, print_ranked_table, to_csv, to_json  # noqa: PLC0415
    return AIScorer, combine_scores, print_ranked_table, to_csv, to_json


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai_rank",
        description="AI-enhanced options screener — technical + AI scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "tickers",
        nargs="+",
        metavar="TICKER",
        help="One or more ticker symbols (e.g. AAPL TSLA SPY)",
    )

    # ── Screener options ───────────────────────────────────────────────────────
    scan = p.add_argument_group("screener")
    scan.add_argument(
        "--mode",
        default="Single-stock",
        choices=[
            "Single-stock",
            "Budget scan",
            "Discovery scan",
            "Premium Selling",
            "Credit Spreads",
            "Iron Condor",
        ],
        help="Screener mode (default: Single-stock)",
    )
    scan.add_argument(
        "--dte-min",
        type=int,
        default=7,
        metavar="DAYS",
        help="Minimum days to expiration (default: 7)",
    )
    scan.add_argument(
        "--dte-max",
        type=int,
        default=45,
        metavar="DAYS",
        help="Maximum days to expiration (default: 45)",
    )
    scan.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="$",
        help="Per-contract budget cap used in Budget scan mode",
    )
    scan.add_argument(
        "--expiries",
        type=int,
        default=8,
        metavar="N",
        help="Max expiration dates to fetch per ticker (default: 8)",
    )
    scan.add_argument(
        "--profile",
        default="swing",
        choices=["swing", "day", "position"],
        help="Trader profile that adjusts scoring weights (default: swing)",
    )

    # ── AI scoring options ─────────────────────────────────────────────────────
    ai = p.add_argument_group("AI scoring")
    ai.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI scoring and rank purely by the screener quality_score",
    )
    ai.add_argument(
        "--ai-weight",
        type=float,
        default=None,
        metavar="W",
        help=(
            "AI score contribution to final_score, 0–1 "
            "(default: from src/config_ai.py).  The technical weight becomes 1 - W."
        ),
    )
    ai.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of ranked candidates to display (default: 20)",
    )

    # ── Output options ─────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument(
        "--output",
        metavar="FILE",
        help="Save ranked results to a CSV file (e.g. output/ranked.csv)",
    )
    out.add_argument(
        "--json",
        metavar="FILE",
        help="Save ranked results to a JSON file (e.g. output/ranked.json)",
    )
    out.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the screener's verbose progress output",
    )
    out.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the AI scorer",
    )

    return p


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Fetch market context ───────────────────────────────────────────────────
    run_scan, get_market_context = _import_screener()

    market_trend      = "Unknown"
    volatility_regime = "Unknown"
    macro_risk        = False
    tnx_change        = 0.0
    try:
        market_trend, volatility_regime, macro_risk, tnx_change = get_market_context()
    except Exception:
        pass

    # ── Run the technical screener ─────────────────────────────────────────────
    tickers_display = ", ".join(args.tickers)
    print(f"\nRunning screener for: {tickers_display}")

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

    picks: pd.DataFrame = results.get("picks", pd.DataFrame())

    if picks.empty:
        print("\nNo candidates found — nothing to rank.")
        sys.exit(0)

    n = len(picks)
    ai_label = "Skipping AI scoring." if args.no_ai else "Starting AI scoring…"
    print(f"\nFound {n} candidate(s).  {ai_label}")

    # ── AI scoring layer ───────────────────────────────────────────────────────
    AIScorer, combine_scores, print_ranked_table, to_csv, to_json = _import_ai()

    if args.no_ai:
        ranked = _rank_without_ai(picks)
    else:
        ai_config_override: dict = {}
        if args.ai_weight is not None:
            ai_config_override["ai_weight"] = args.ai_weight

        try:
            scorer = AIScorer(config=ai_config_override or None)
            ai_df  = scorer.score_candidates(picks)

            kwargs: dict = {}
            if args.ai_weight is not None:
                kwargs["ai_weight"]        = args.ai_weight
                kwargs["technical_weight"] = 1.0 - args.ai_weight

            ranked = combine_scores(picks, ai_df, **kwargs)

        except EnvironmentError as exc:
            print(f"\nConfiguration error: {exc}")
            sys.exit(1)

        except Exception as exc:
            print(f"\nAI scoring failed ({exc}).  Falling back to technical ranking.")
            ranked = _rank_without_ai(picks)

    # ── Display & export ───────────────────────────────────────────────────────
    print_ranked_table(ranked, top_n=args.top)

    if args.output:
        to_csv(ranked, args.output)

    if args.json:
        to_json(ranked, args.json)


def _rank_without_ai(picks: pd.DataFrame) -> pd.DataFrame:
    """Return *picks* with stub AI columns, ranked by quality_score."""
    df = picks.copy()
    df["ai_score"]     = 50.0
    df["ai_reasoning"] = ""
    df["ai_flags"]     = ""
    df["catalyst_risk"] = "medium"
    df["iv_justified"] = True
    df["final_score"]  = df["quality_score"].fillna(0)
    df["rank"] = df["final_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank")


if __name__ == "__main__":
    main()
