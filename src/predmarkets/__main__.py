"""Daily prediction-market archive job.

    python -m src.predmarkets            archive today's macro quotes
    python -m src.predmarkets --status   what has accumulated so far

Designed to be run by launchd once a day. It writes and reports; it never
scores anything, and nothing else in the screener reads this database yet.
That is deliberate — see src/predmarkets/__init__.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Optional, Sequence

from src.predmarkets import archive, kalshi


def run_archive(db: str, today: str,
                series: Sequence[str] = kalshi.MACRO_SERIES) -> int:
    """Fetch and store one day. Returns rows written."""
    quotes: List[kalshi.Quote] = []
    empty: List[str] = []
    for name in series:
        got = kalshi.fetch_series(name)
        if not got:
            empty.append(name)
        quotes.extend(got)

    conn = archive.connect(db)
    try:
        written = archive.record(conn, quotes, today)
    finally:
        conn.close()

    tight = sum(1 for q in quotes if q.mid() is not None)
    print(f"{today}: archived {written} quotes across "
          f"{len(series) - len(empty)}/{len(series)} series")
    print(f"  {tight} have a spread tight enough for a usable mid "
          f"({100.0 * tight / written:.0f}%)" if written else "  no quotes")
    if empty:
        print(f"  no open markets: {', '.join(empty)}")
    return written


def run_status(db: str) -> int:
    conn = archive.connect(db)
    try:
        n, days, first, last = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT archived_at), MIN(archived_at), "
            "MAX(archived_at) FROM pm_quotes").fetchone()
        print(f"{n:,} quotes over {days} day(s): {first} .. {last}")
        for series, c, d in conn.execute(
                "SELECT series, COUNT(*), COUNT(DISTINCT ticker) "
                "FROM pm_quotes GROUP BY series ORDER BY 2 DESC"):
            print(f"  {series:<20} {c:>6} rows  {d:>4} markets")
    finally:
        conn.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.predmarkets")
    parser.add_argument("--db", default=archive.DEFAULT_DB)
    parser.add_argument("--today", default=dt.date.today().isoformat())
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.status:
        return run_status(args.db)
    run_archive(args.db, args.today)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
