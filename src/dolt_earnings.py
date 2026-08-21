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
import logging
import sqlite3
import time as _time
from statistics import mean, median
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

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


def _is_stale(fetched_at: Optional[str], max_age_days: Optional[int]) -> bool:
    """True when a cached symbol is old enough to be worth re-querying.

    ``max_age_days=None`` means never — the original behaviour, which every
    existing caller relies on.
    """
    if max_age_days is None:
        return False
    if not fetched_at:
        return True
    try:
        when = _dt.datetime.fromisoformat(str(fetched_at))
    except ValueError:
        return True
    return (_dt.datetime.now() - when).days >= int(max_age_days)


def earnings_dates(symbol: str, db_path: str = DEFAULT_CACHE,
                   max_age_days: Optional[int] = None,
                   fetcher: Optional[Any] = None) -> List[str]:
    """Sorted list of earnings dates (ISO) for a symbol. Cache-first.

    ``max_age_days`` re-queries a symbol whose cache entry is at least that
    old. Without it a symbol is fetched exactly ONCE, ever — which is how the
    calendar came to hold every past quarter and almost no future one. Measured
    2026-08-20: 163 symbols cached, 18 with any date at or after that day, the
    oldest fetch marker 2026-06-15. Companies announce roughly three to four
    weeks ahead, so a weekly refresh is what keeps the earnings gate
    (src/earnings_gate.py) able to answer at all.

    Re-fetching only ever ADDS: a provider returning less than last time must
    not erase history, because `iv_crush` reads the same table. A failed fetch
    leaves the cache exactly as it was — an outage is not new information.
    """
    symbol = symbol.upper()
    _ensure(db_path)
    fetch = fetcher or _fetch_live
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT fetched_at FROM earnings_fetched WHERE symbol=?",
                           (symbol,)).fetchone()
        if row is None or _is_stale(row[0], max_age_days):
            try:
                rows = fetch(symbol)
            except Exception as exc:
                # Keep whatever is cached. A symbol that has never been fetched
                # stays unmarked so the next run tries again.
                logger.warning("earnings fetch failed for %s: %s", symbol, exc)
                rows = None
            if rows is not None:
                for r in rows:
                    conn.execute("INSERT OR REPLACE INTO earnings_cal (symbol,date,whn) VALUES (?,?,?)",
                                 (symbol, r.get("date"), r.get("when")))
                conn.execute("INSERT OR REPLACE INTO earnings_fetched (symbol,fetched_at) VALUES (?,?)",
                             (symbol, _dt.datetime.now().isoformat(timespec="seconds")))
                conn.commit()
        cur = conn.execute("SELECT date FROM earnings_cal WHERE symbol=? ORDER BY date", (symbol,))
        return [r[0] for r in cur.fetchall() if r[0]]


def refresh_symbols(symbols: List[str], db_path: str = DEFAULT_CACHE,
                    max_age_days: int = 7,
                    fetcher: Optional[Any] = None,
                    sleep_s: float = 1.5,
                    pause: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """Re-query a whole universe, reporting per symbol what it now holds.

    ``has_future`` is the number that decides whether the earnings gate can act
    on a symbol at all: a calendar of past quarters answers nothing about a
    holding period that starts today.

    One symbol's failure never stops the run — a universe refresh is 120-odd
    HTTP calls and losing the other 119 to one bad ticker would be absurd.

    **Paced.** Run flat out, 124 symbols tripped DoltHub's capacity wall on
    2026-08-20: 68 came back empty and the same symbols fetched cleanly one at
    a time a minute later. See [[project_dolthub_real_options]] — this source
    has always been capacity-walled. ``sleep_s`` is the gap between symbols.
    """
    today = _dt.date.today().isoformat()
    wait = pause if pause is not None else _time.sleep
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        wait(sleep_s)
        key = str(symbol).upper()
        try:
            before = _fetch_marker(key, db_path)
            dates = earnings_dates(key, db_path=db_path,
                                   max_age_days=max_age_days, fetcher=fetcher)
            after = _fetch_marker(key, db_path)
            entry: Dict[str, Any] = {
                "dates": len(dates),
                "latest": dates[-1] if dates else None,
                "has_future": bool(dates and dates[-1] >= today),
            }
            # `earnings_dates` deliberately swallows a fetch failure and hands
            # back the cache, because it runs on paths that must not break. A
            # REFRESH is a different job: a run that fetched nothing and
            # reported success would be the silent failure this repo keeps
            # finding. The marker only advances on a fetch that returned, so
            # comparing it is how the outcome is verified rather than assumed.
            if after == before:
                entry["error"] = "fetch did not complete; cache unchanged"
            out[key] = entry
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def scan_universe(config_path: str = "config.json",
                  ledger_path: str = "paper_trades.db",
                  candidates_path: str = "data/candidates.db") -> List[str]:
    """Every symbol the screener might have to judge.

    The configured watchlists are not enough on their own: the earnings gate
    has to answer for whatever actually reaches `log_trade`, which includes
    names the book has traded historically and names the scanner has recorded
    as candidates. Missing sources are skipped rather than raising — this is a
    maintenance helper, not a gate.
    """
    import json
    symbols: set = set()
    try:
        with open(config_path) as fh:
            for value in (json.load(fh).get("watchlists") or {}).values():
                if isinstance(value, list):
                    symbols.update(str(v).upper() for v in value)
    except Exception:
        pass
    for path, query in ((ledger_path, "SELECT DISTINCT ticker FROM trades"),
                        (candidates_path,
                         "SELECT DISTINCT symbol FROM candidates")):
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            symbols.update(str(r[0]).upper() for r in conn.execute(query)
                           if r[0])
            conn.close()
        except Exception:
            continue
    # Option symbols only: crypto rows and any junk key are not earnings names.
    return sorted(s for s in symbols
                  if s.isalpha() and 1 <= len(s) <= 5 and s not in ("BTC", "ETH"))


def _fetch_marker(symbol: str, db_path: str) -> Optional[str]:
    """`earnings_fetched.fetched_at` for a symbol, or None if never fetched."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT fetched_at FROM earnings_fetched WHERE symbol=?",
                (symbol.upper(),)).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


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
    ap.add_argument("--refresh", nargs="*", metavar="SYMBOL",
                    help="re-query symbols whose cache is older than "
                         "--max-age-days; no arguments means the whole scan "
                         "universe (watchlists + everything the book has "
                         "traded or scanned)")
    ap.add_argument("--max-age-days", type=int, default=7,
                    help="refresh a symbol cached at least this many days ago "
                         "(default 7; companies announce ~3-4 weeks ahead)")
    args = ap.parse_args()
    if args.refresh is not None:
        symbols = args.refresh or scan_universe()
        print(f"refreshing {len(symbols)} symbols, max age {args.max_age_days}d")
        report = refresh_symbols(symbols, db_path=args.db,
                                 max_age_days=args.max_age_days)
        future = [s for s, r in report.items() if r.get("has_future")]
        failed = [s for s, r in report.items() if r.get("error")]
        for sym in sorted(report):
            r = report[sym]
            note = r.get("error") or (
                f"{r['dates']} dates, latest {r['latest']}"
                f"{'  <- FUTURE' if r.get('has_future') else ''}")
            print(f"  {sym:6} {note}")
        print(f"\n{len(future)} of {len(report)} symbols carry a future "
              f"earnings date — that is what the earnings gate can act on.")
        if failed:
            print(f"{len(failed)} failed: {', '.join(sorted(failed))}")
    if args.dates:
        ds = earnings_dates(args.dates, db_path=args.db)
        print(f"{args.dates}: {len(ds)} earnings dates, {ds[:3]} ... {ds[-3:]}")
    if args.iv_crush:
        out = iv_crush(args.iv_crush, db_path=args.db)
        summary = {k: v for k, v in out.items() if k != "events"}
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    _cli()
