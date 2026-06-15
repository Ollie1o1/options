"""Raw (unadjusted) daily close prices that MATCH the unadjusted option strikes.

The DoltHub `ohlcv` table is date-keyed and can't be scanned per-symbol within the
API deadline. Instead we take yfinance's split-adjusted close (fast) and UN-ADJUST
it with the tiny, selective DoltHub `split` table:

    raw(date) = adjusted(date) * Π ratio  for every split with ex_date > date

So a split name like NVDA (10:1 on 2024-06-10) reads ~$42 adjusted but ~$426 raw
in 2023 — matching the raw strikes, so it's no longer skipped by the split guard.

Cache-first into the same SQLite DB as the option chains.

CLI:
    python -m src.dolt_stocks --symbol NVDA
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from typing import Dict, List, Tuple

DEFAULT_CACHE = "data/dolt_options.db"
STOCKS_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master"

_DDL_CLOSE = """
CREATE TABLE IF NOT EXISTS stocks_close (
    symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date)
)
"""
_DDL_FETCHED = "CREATE TABLE IF NOT EXISTS stocks_fetched (symbol TEXT PRIMARY KEY, fetched_at TEXT)"


def _ensure(db_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL_CLOSE)
        conn.execute(_DDL_FETCHED)


def _fetch_splits(symbol: str) -> List[Tuple[str, float]]:
    """[(ex_date, ratio)] for a symbol from the DoltHub split table. ratio>1 = shares
    multiplied (price divided). Reuses dolt_options retry/backoff."""
    from src import dolt_options as _do
    orig = _do.API_BASE
    try:
        _do.API_BASE = STOCKS_API
        rows = _do._query(
            f"SELECT ex_date, to_factor, for_factor FROM split "
            f"WHERE act_symbol='{symbol.upper()}'")
    finally:
        _do.API_BASE = orig
    out = []
    for r in rows:
        try:
            ratio = float(r["to_factor"]) / float(r["for_factor"])
            out.append((r["ex_date"], ratio))
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            continue
    return out


def raw_from_adjusted(adjusted: Dict[str, float],
                      splits: List[Tuple[str, float]]) -> Dict[str, float]:
    """Pure un-adjust: raw(date) = adjusted(date) * Π ratio for ex_date > date."""
    out = {}
    for date, px in adjusted.items():
        factor = 1.0
        for ex_date, ratio in splits:
            if ex_date > date:        # split happens AFTER this date → un-adjust
                factor *= ratio
        out[date] = px * factor
    return out


def _yf_adjusted(symbol: str) -> Dict[str, float]:
    import warnings
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = yf.Ticker(symbol).history(period="6y")["Close"]   # split+div adjusted
    return {d.date().isoformat(): float(v) for d, v in h.items()}


def close_history(symbol: str, db_path: str = DEFAULT_CACHE) -> Dict[str, float]:
    """Raw close keyed by ISO date for a symbol. Cache-first."""
    symbol = symbol.upper()
    _ensure(db_path)
    with sqlite3.connect(db_path) as conn:
        done = conn.execute("SELECT 1 FROM stocks_fetched WHERE symbol=?", (symbol,)).fetchone()
        if done is None:
            adj = _yf_adjusted(symbol)
            splits = _fetch_splits(symbol)
            raw = raw_from_adjusted(adj, splits)
            for d, c in raw.items():
                conn.execute("INSERT OR REPLACE INTO stocks_close (symbol,date,close) VALUES (?,?,?)",
                             (symbol, d, c))
            conn.execute("INSERT OR REPLACE INTO stocks_fetched (symbol,fetched_at) VALUES (?,?)",
                         (symbol, _dt.datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        cur = conn.execute("SELECT date, close FROM stocks_close WHERE symbol=? ORDER BY date", (symbol,))
        return {d: c for d, c in cur.fetchall()}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Raw (split-unadjusted) daily close")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--db", default=DEFAULT_CACHE)
    args = ap.parse_args()
    h = close_history(args.symbol, db_path=args.db)
    ks = sorted(h)
    if ks:
        print(f"{args.symbol}: {len(h)} days, {ks[0]} ({h[ks[0]]:.2f}) .. {ks[-1]} ({h[ks[-1]]:.2f})")
    else:
        print(f"{args.symbol}: no data")


if __name__ == "__main__":
    _cli()
