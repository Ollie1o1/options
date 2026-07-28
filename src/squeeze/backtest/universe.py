"""Study universe: which symbols can plausibly carry a squeeze grade.

Deliberately generous. The grader's own gate (short interest as a share of
float) does the real selection later; this only strips the long tail of names
too small or too thinly reported to produce a meaningful measurement, so the
price and EDGAR backfills aren't spent on noise.

Nothing here filters on outcomes, and nothing consults a present-day list of
tickers — selection uses only what FINRA reported on or before each settlement
date, so a name that later delisted is still in the universe for the dates it
was alive.
"""
from __future__ import annotations

import sqlite3
from typing import Set

from src.squeeze.backtest import DEFAULT_DB

MIN_SHARES_SHORT = 250_000   # below this, SI is noise / illiquid microcap
MIN_DATES = 3                # need history for a month-over-month trend


def study_symbols(db_path: str = DEFAULT_DB,
                  min_shares_short: int = MIN_SHARES_SHORT,
                  min_dates: int = MIN_DATES) -> Set[str]:
    """Symbols clearing the pre-filter on at least *min_dates* settlement dates."""
    conn = sqlite3.connect(db_path, timeout=60)
    rows = conn.execute(
        "SELECT symbol FROM si WHERE shares_short >= ? "
        "GROUP BY symbol HAVING COUNT(*) >= ?",
        (min_shares_short, min_dates)).fetchall()
    conn.close()
    return {r[0] for r in rows}


def coverage_report(db_path: str = DEFAULT_DB) -> dict:
    """How much of the raw FINRA panel the pre-filter keeps."""
    conn = sqlite3.connect(db_path, timeout=60)
    total_sym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM si").fetchone()[0]
    total_rows = conn.execute("SELECT COUNT(*) FROM si").fetchone()[0]
    kept_rows = conn.execute(
        "SELECT COUNT(*) FROM si WHERE shares_short >= ?", (MIN_SHARES_SHORT,)).fetchone()[0]
    conn.close()
    kept_sym = len(study_symbols(db_path))
    return {"symbols_total": total_sym, "symbols_kept": kept_sym,
            "rows_total": total_rows, "rows_kept": kept_rows,
            "row_pct": round(100.0 * kept_rows / total_rows, 1) if total_rows else 0.0}


if __name__ == "__main__":
    for k, v in coverage_report().items():
        print(f"  {k:16s} {v:,}" if isinstance(v, int) else f"  {k:16s} {v}")
