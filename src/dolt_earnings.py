"""Earnings dates + IV-crush study from the DoltHub `post-no-preference/earnings`
dataset, used to make the cohort backtest earnings-aware (long calls held through
earnings get crushed by the post-report IV collapse).

Cache-first into the same SQLite DB as the option chains.

CLI:
    python -m src.dolt_earnings --dates AAPL
    python -m src.dolt_earnings --iv-crush AAPL
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from statistics import mean, median
from typing import Any, Dict, List, Optional

import requests

EARNINGS_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master"
DEFAULT_CACHE = "data/dolt_options.db"

_DDL_EARN = """
CREATE TABLE IF NOT EXISTS earnings_cal (
    symbol TEXT, date TEXT, whn TEXT,
    PRIMARY KEY (symbol, date)
)
"""
_DDL_EARN_FETCHED = """
CREATE TABLE IF NOT EXISTS earnings_fetched (symbol TEXT PRIMARY KEY, fetched_at TEXT)
"""


def _ensure(db_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL_EARN)
        conn.execute(_DDL_EARN_FETCHED)


def _fetch_live(symbol: str) -> List[Dict[str, str]]:
    """Query earnings_calendar for one symbol (reuses dolt_options retry/backoff)."""
    from src import dolt_options as _do
    # Reuse the resilient _query but against the earnings endpoint.
    orig = _do.API_BASE
    try:
        _do.API_BASE = EARNINGS_API
        rows = _do._query(
            f"SELECT act_symbol, `date`, `when` FROM earnings_calendar "
            f"WHERE act_symbol='{symbol.upper()}'")
    finally:
        _do.API_BASE = orig
    return rows


def earnings_dates(symbol: str, db_path: str = DEFAULT_CACHE) -> List[str]:
    """Sorted list of earnings dates (ISO) for a symbol. Cache-first."""
    symbol = symbol.upper()
    _ensure(db_path)
    with sqlite3.connect(db_path) as conn:
        done = conn.execute("SELECT 1 FROM earnings_fetched WHERE symbol=?",
                            (symbol,)).fetchone()
        if done is None:
            rows = _fetch_live(symbol)
            for r in rows:
                conn.execute("INSERT OR REPLACE INTO earnings_cal (symbol,date,whn) VALUES (?,?,?)",
                             (symbol, r.get("date"), r.get("when")))
            conn.execute("INSERT OR REPLACE INTO earnings_fetched (symbol,fetched_at) VALUES (?,?)",
                         (symbol, _dt.datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        cur = conn.execute("SELECT date FROM earnings_cal WHERE symbol=? ORDER BY date", (symbol,))
        return [r[0] for r in cur.fetchall() if r[0]]


def earnings_in_window(symbol: str, start: str, end: str,
                       db_path: str = DEFAULT_CACHE) -> List[str]:
    """Earnings dates falling in [start, end] inclusive."""
    return [d for d in earnings_dates(symbol, db_path=db_path) if start <= d <= end]


def holds_through_earnings(symbol: str, entry_date: str, exit_date: str,
                           db_path: str = DEFAULT_CACHE) -> bool:
    """True if any earnings date falls strictly inside (entry_date, exit_date]."""
    return len([d for d in earnings_dates(symbol, db_path=db_path)
                if entry_date < d <= exit_date]) > 0


# ── IV crush study ──────────────────────────────────────────────────────────
def _atm_iv(chain, spot):
    cands = [c for c in chain if c.get("type") == "call" and c.get("strike")
             and c.get("iv")]
    if not cands:
        return None
    c = min(cands, key=lambda x: abs(x["strike"] - spot))
    return c["iv"]


def iv_crush(symbol: str, db_path: str = DEFAULT_CACHE,
             max_skip: int = 4) -> Dict[str, Any]:
    """For each earnings date with chains cached on both sides, measure ATM-IV
    before vs after. Returns mean/median absolute and relative crush."""
    from src import dolt_options as _do
    from src.dolt_validate import _spot_history
    spots = _spot_history(symbol)
    events = []
    for ed in earnings_dates(symbol, db_path=db_path):
        if ed < _do.COVERAGE_MIN or ed > _do.COVERAGE_MAX:
            continue
        # nearest spot for ATM reference
        spot = spots.get(ed) or next((spots[d] for d in sorted(spots) if d >= ed), None)
        if spot is None:
            continue
        # chain just BEFORE (search backward) and just AFTER (search forward)
        bd, before = _do.get_chain_near(symbol, ed, max_skip=max_skip, db_path=db_path, direction=-1)
        ad, after = _do.get_chain_near(symbol, ed, max_skip=max_skip, db_path=db_path, direction=1)
        if not before or not after or bd is None or ad is None or bd >= ad:
            continue
        iv_b, iv_a = _atm_iv(before, spot), _atm_iv(after, spot)
        if not iv_b or not iv_a:
            continue
        events.append({"date": ed, "iv_before": round(iv_b, 4), "iv_after": round(iv_a, 4),
                       "abs_crush": round(iv_b - iv_a, 4),
                       "rel_crush": round((iv_b - iv_a) / iv_b, 4) if iv_b else None})
    out: Dict[str, Any] = {"symbol": symbol, "n_events": len(events)}
    if events:
        out["mean_abs_crush"] = round(mean(e["abs_crush"] for e in events), 4)
        out["median_rel_crush"] = round(median(e["rel_crush"] for e in events
                                                if e["rel_crush"] is not None), 4)
        out["events"] = events
    return out


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Earnings dates + IV-crush from DoltHub")
    ap.add_argument("--dates", metavar="SYMBOL")
    ap.add_argument("--iv-crush", metavar="SYMBOL")
    ap.add_argument("--db", default=DEFAULT_CACHE)
    args = ap.parse_args()
    if args.dates:
        ds = earnings_dates(args.dates, db_path=args.db)
        print(f"{args.dates}: {len(ds)} earnings dates, {ds[:3]} ... {ds[-3:]}")
    if args.iv_crush:
        out = iv_crush(args.iv_crush, db_path=args.db)
        summary = {k: v for k, v in out.items() if k != "events"}
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    _cli()
