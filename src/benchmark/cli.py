"""`python -m src.benchmark` — run the benchmark engine against real databases.

Read-only throughout: every database is opened `file:{path}?mode=ro`, and
this module never writes to `config.json` or any ledger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.benchmark.excess import (
    NULL_CASH, NULL_SPY, build_paired_frame, cumulative_dollar_comparison,
    daily_aggregate, daily_frame, load_closed_trades, paired_aggregate,
)
from src.benchmark.prices import DEFAULT_TOLERANCE_DAYS, stitch_spy_series
from src.benchmark.report import render_markdown, run_manifest


def build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Book vs. Cash and SPY buy-and-hold over identical "
                     "holding periods. Read-only; descriptive, not a "
                     "significance test.")
    ap.add_argument("--ledger-db", required=True,
                     help="Path to paper_trades.db")
    ap.add_argument("--ohlcv-db", required=True,
                     help="Path to data/equity_ohlcv.db")
    ap.add_argument("--archive-db", required=True,
                     help="Path to data/chain_archive.db")
    ap.add_argument("--out", required=True,
                     help="Path to write the markdown report to")
    ap.add_argument("--tolerance-days", type=int,
                     default=DEFAULT_TOLERANCE_DAYS,
                     help="SPY date-match tolerance in calendar days "
                          f"(default {DEFAULT_TOLERANCE_DAYS})")
    ap.add_argument("--n-boot", type=int, default=10000,
                     help="Bootstrap replicates for the per-trade cluster CI")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> str:
    """Runs the engine end-to-end and returns the rendered markdown."""
    spy_series, stitch = stitch_spy_series(args.ohlcv_db, args.archive_db)

    trades, load_report = load_closed_trades(args.ledger_db)
    print(f"[benchmark] loaded {load_report.loaded} closed trades "
          f"(dropped: {dict(load_report.dropped)})", file=sys.stderr)

    frames, match_stats = build_paired_frame(
        trades, spy_series, tolerance_days=args.tolerance_days)
    print(f"[benchmark] SPY match: {match_stats.exact} exact, "
          f"{match_stats.tolerant} tolerant, {match_stats.missing} missing",
          file=sys.stderr)

    per_trade = {
        name: paired_aggregate(df, n_boot=args.n_boot, seed=args.seed)
        for name, df in frames.items()
    }

    daily_df = daily_frame(trades, spy_series)
    daily_result = daily_aggregate(daily_df)
    print(f"[benchmark] daily series: {daily_result.n_days} aligned "
          "trading days", file=sys.stderr)

    cumulative = cumulative_dollar_comparison(
        trades, spy_series, tolerance_days=args.tolerance_days)

    manifest = run_manifest(args.ledger_db, load_report, match_stats, stitch,
                             seed=args.seed)

    md = render_markdown(manifest, cumulative, per_trade, daily_result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[benchmark] wrote {out_path}", file=sys.stderr)
    print(f"[benchmark] book P&L: ${cumulative.book_pnl_usd:,.2f} | "
          f"SPY-equiv (primary): ${cumulative.spy_equiv_primary_usd:,.2f} | "
          f"SPY-equiv (secondary, mean-concurrent): "
          f"${cumulative.spy_equiv_secondary_usd:,.2f}"
          if cumulative.spy_equiv_secondary_usd is not None else
          f"[benchmark] book P&L: ${cumulative.book_pnl_usd:,.2f}",
          file=sys.stderr)
    return md


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
