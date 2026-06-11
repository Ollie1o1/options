"""Alpha Vantage historical options — real per-contract history, budgeted.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.av_options --probe
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.av_options --backfill SPY \
        --start 2025-06-01 [--end 2025-12-31] [--budget 20]

Alpha Vantage's HISTORICAL_OPTIONS endpoint serves EOD option chains (with IV
and Greeks) going back 15+ years. The free tier allows ~25 requests/day; one
request = one symbol-day, so backfill is a *queue you drain daily*, not a bulk
download. Whether the free tier returns real data (vs a premium gate) varies —
run ``--probe`` with your key and it tells you which world you're in.

Rows land in the same ``data/chain_archive.db`` schema as the daily CBOE
snapshots, with ``source='alphavantage'`` — so backtests read one table
regardless of where the history came from. Resume state is kept per symbol in
``av_backfill_state``; rerunning the same command continues where it left off.

Key: ``ALPHAVANTAGE_API_KEY`` in the environment or .env (free at
https://www.alphavantage.co/support/#api-key).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.chain_archive import DEFAULT_DB, ensure_db, _COLUMNS

API_URL = ("https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
           "&symbol={symbol}&date={date}&apikey={key}")

_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS av_backfill_state (
    symbol TEXT PRIMARY KEY,
    last_done_date TEXT NOT NULL
);
"""


def _api_key() -> Optional[str]:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key:
        return key
    try:  # mirror ai_scorer's .env convention without requiring dotenv
        with open(".env") as f:
            for line in f:
                if line.strip().startswith("ALPHAVANTAGE_API_KEY="):
                    return line.strip().split("=", 1)[1] or None
    except OSError:
        pass
    return None


def classify_response(payload: Dict[str, Any]) -> str:
    """real_data | no_data | premium_required | rate_limited | bad_key | unknown."""
    if not isinstance(payload, dict):
        return "unknown"
    if isinstance(payload.get("data"), list):
        return "real_data" if payload["data"] else "no_data"
    info = str(payload.get("Information") or payload.get("Note") or "")
    if "premium" in info.lower():
        return "premium_required"
    if "rate limit" in info.lower() or "requests per day" in info.lower():
        return "rate_limited"
    if payload.get("Error Message") or "apikey" in info.lower():
        return "bad_key"
    return "unknown"


def parse_rows(payload: Dict[str, Any], symbol: str, date: str) -> List[Dict[str, Any]]:
    """Normalize AV contracts into the shared archive schema. Never raises."""
    rows: List[Dict[str, Any]] = []
    for o in (payload or {}).get("data") or []:
        try:
            rows.append({
                "symbol": symbol.upper(),
                "snap_date": date,
                "contract": o.get("contractID"),
                "type": str(o.get("type") or "").lower(),
                "strike": float(o["strike"]),
                "expiration": str(o.get("expiration"))[:10],
                "bid": _f(o.get("bid")), "ask": _f(o.get("ask")),
                "bid_size": _f(o.get("bid_size")), "ask_size": _f(o.get("ask_size")),
                "iv": _f(o.get("implied_volatility")),
                "delta": _f(o.get("delta")), "gamma": _f(o.get("gamma")),
                "theta": _f(o.get("theta")), "vega": _f(o.get("vega")),
                "rho": _f(o.get("rho")),
                "open_interest": _f(o.get("open_interest")),
                "volume": _f(o.get("volume")),
                "last_trade_time": None,
                "spot": None,                      # AV omits spot; join price history at read time
                "snapshot_ts": date,
                "source": "alphavantage",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _http_fetch(symbol: str, date: str, api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    import requests
    resp = requests.get(API_URL.format(symbol=symbol.upper(), date=date, key=api_key),
                        timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    status = classify_response(payload)
    return status, (parse_rows(payload, symbol, date) if status == "real_data" else [])


def _weekdays(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while d <= stop:
        if d.isoweekday() <= 5:
            yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def backfill(symbol: str, start: str, end: Optional[str] = None,
             db_path: str = DEFAULT_DB, budget: int = 20,
             fetch_fn: Optional[Callable] = None,
             api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch up to `budget` symbol-days (weekdays start..end), oldest first,
    resuming after the last completed date. Stops early on rate limit /
    premium gate / bad key and reports why."""
    fetch_fn = fetch_fn or _http_fetch
    api_key = api_key or _api_key()
    if not api_key:
        return {"fetched": 0, "rows": 0, "stopped": "no_api_key"}
    end = end or datetime.now().strftime("%Y-%m-%d")
    ensure_db(db_path)
    fetched = rows_written = 0
    stopped = "done"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_STATE_SCHEMA)
        row = conn.execute("SELECT last_done_date FROM av_backfill_state WHERE symbol=?",
                           (symbol.upper(),)).fetchone()
        resume_after = row[0] if row else None
        for date in _weekdays(start, end):
            if resume_after and date <= resume_after:
                continue
            if fetched >= budget:
                stopped = "budget"
                break
            status, rows = fetch_fn(symbol, date, api_key)
            if status in ("rate_limited", "premium_required", "bad_key", "unknown"):
                stopped = status
                break
            fetched += 1
            for r in rows:
                conn.execute(
                    f"INSERT OR IGNORE INTO chain_snapshots ({','.join(_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(_COLUMNS))})",
                    [r.get(c) for c in _COLUMNS])
                rows_written += 1
            conn.execute("INSERT INTO av_backfill_state (symbol, last_done_date) "
                         "VALUES (?, ?) ON CONFLICT(symbol) DO UPDATE SET last_done_date=?",
                         (symbol.upper(), date, date))
            conn.commit()
    return {"fetched": fetched, "rows": rows_written, "stopped": stopped}


def probe(api_key: Optional[str] = None) -> str:
    """One cheap request to find out what your key can do."""
    api_key = api_key or _api_key()
    if not api_key:
        return "no_api_key"
    status, _ = _http_fetch("IBM", "2024-01-04", api_key)
    return status


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Alpha Vantage historical options")
    ap.add_argument("--probe", action="store_true",
                    help="test the API key against one known symbol-day")
    ap.add_argument("--backfill", metavar="SYMBOL")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--budget", type=int, default=20,
                    help="max requests this run (free tier ~25/day)")
    args = ap.parse_args()

    if args.probe:
        status = probe()
        explain = {
            "real_data": "✅ your key returns REAL historical chains — backfill away",
            "no_data": "✅ key works (that sample day was empty) — backfill away",
            "premium_required": "🔒 free tier is gated for this endpoint — needs the "
                                "premium plan (~$50/mo) or skip AV entirely",
            "rate_limited": "⏳ daily limit already used — try again tomorrow",
            "bad_key": "❌ key invalid — check ALPHAVANTAGE_API_KEY",
            "no_api_key": "❌ no key found — set ALPHAVANTAGE_API_KEY in .env "
                          "(free: alphavantage.co/support/#api-key)",
            "unknown": "❓ unexpected response — inspect manually",
        }
        print(f"probe: {status}\n{explain.get(status, '')}")
        return

    if args.backfill:
        if not args.start:
            raise SystemExit("--backfill requires --start YYYY-MM-DD")
        r = backfill(args.backfill, start=args.start, end=args.end,
                     budget=args.budget)
        print(f"backfill {args.backfill.upper()}: fetched {r['fetched']} day(s), "
              f"{r['rows']} rows, stopped: {r['stopped']}")
        if r["stopped"] == "rate_limited":
            print("daily budget exhausted — rerun tomorrow, it resumes automatically")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
