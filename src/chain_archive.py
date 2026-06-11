"""Daily option-chain snapshots — the free historical dataset that compounds.

Real per-contract option history is the single biggest data gap for honest
backtesting (the lottery study had to model premiums with an IV proxy; paid
feeds run $29-90/mo). This module takes the other path: archive the *current*
chain once per trading day from the free CBOE feed, and in a few months the
backtests have real bid/ask/IV/Greeks history for the symbols we actually
trade — cost: $0 and patience.

Storage: ``data/chain_archive.db`` table ``chain_snapshots``, one row per
contract per day, filtered to tradeable territory (DTE ≤ max, moneyness within
a band) to keep growth sane (~10-15k rows/day for ~15 liquid symbols ≈
a few MB/month). Runs inside startup maintenance (parent only), throttled to
once per weekday afternoon — same self-healing model as the other jobs.

Config (config.json → data_archive): enabled, symbols, max_dte, moneyness_band.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

DEFAULT_DB = os.path.join("data", "chain_archive.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    symbol TEXT NOT NULL,
    snap_date TEXT NOT NULL,
    contract TEXT NOT NULL,
    type TEXT, strike REAL, expiration TEXT,
    bid REAL, ask REAL, bid_size REAL, ask_size REAL,
    iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
    open_interest REAL, volume REAL,
    last_trade_time TEXT, spot REAL, snapshot_ts TEXT, source TEXT,
    PRIMARY KEY (contract, snap_date)
);
CREATE INDEX IF NOT EXISTS idx_snap_symbol_date
    ON chain_snapshots (symbol, snap_date);
"""

_COLUMNS = ["symbol", "snap_date", "contract", "type", "strike", "expiration",
            "bid", "ask", "bid_size", "ask_size", "iv", "delta", "gamma",
            "theta", "vega", "rho", "open_interest", "volume",
            "last_trade_time", "spot", "snapshot_ts", "source"]


def ensure_db(db_path: str = DEFAULT_DB) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def filter_rows(rows: List[Dict[str, Any]], today: str,
                max_dte: int = 120, moneyness_band: float = 0.15,
                min_open_interest: float = 1) -> List[Dict[str, Any]]:
    """Keep contracts a swing trader could actually touch: 0 < DTE ≤ max_dte,
    strike within ±moneyness_band of spot, and some open interest (zero-OI far
    wings are junk and triple the storage). Pure.

    ±15% covers every realistic swing strike at these DTEs — even a 2.5σ
    lottery wing at 14d/20%vol sits ~12% OTM."""
    t0 = datetime.strptime(today, "%Y-%m-%d")
    kept = []
    for r in rows:
        try:
            dte = (datetime.strptime(str(r["expiration"])[:10], "%Y-%m-%d") - t0).days
            spot, strike = float(r.get("spot") or 0), float(r.get("strike") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if dte <= 0 or dte > max_dte:
            continue
        if spot <= 0 or strike <= 0 or abs(strike / spot - 1.0) > moneyness_band:
            continue
        if float(r.get("open_interest") or 0) < min_open_interest:
            continue
        kept.append(r)
    return kept


def archive_symbols(symbols: List[str], db_path: str = DEFAULT_DB,
                    today: Optional[str] = None,
                    fetcher: Optional[Callable] = None,
                    max_dte: int = 120, moneyness_band: float = 0.15,
                    min_open_interest: float = 1) -> int:
    """Snapshot each symbol once for `today`. Idempotent per (symbol, day);
    per-symbol failures are isolated. Returns rows written."""
    from src import cboe_client
    today = today or datetime.now().strftime("%Y-%m-%d")
    fetcher = fetcher or cboe_client.fetch_chain
    ensure_db(db_path)
    written = 0
    with sqlite3.connect(db_path) as conn:
        for sym in symbols:
            sym = sym.upper()
            try:
                done = conn.execute(
                    "SELECT 1 FROM chain_snapshots WHERE symbol=? AND snap_date=? LIMIT 1",
                    (sym, today)).fetchone()
                if done:
                    continue
                rows = filter_rows(fetcher(sym) or [], today,
                                   max_dte=max_dte, moneyness_band=moneyness_band,
                                   min_open_interest=min_open_interest)
                for r in rows:
                    values = [sym, today] + [r.get(c) for c in _COLUMNS[2:]]
                    conn.execute(
                        f"INSERT OR IGNORE INTO chain_snapshots ({','.join(_COLUMNS)}) "
                        f"VALUES ({','.join('?' * len(_COLUMNS))})", values)
                    written += 1
                conn.commit()
            except Exception:
                continue
    return written


def due_chain_archive(state: dict, today: str, weekday: int, hhmm: int) -> bool:
    """Once per weekday, in the afternoon (≥14:00 — close enough to EOD for a
    delayed feed). weekday: 1=Mon..7=Sun."""
    if weekday > 5 or hhmm < 1400:
        return False
    return (state or {}).get("last_chain_archive") != today


def archive_stats(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    """Quick inventory for display/CLI."""
    if not os.path.exists(db_path):
        return {"rows": 0, "symbols": 0, "days": 0, "first": None, "last": None}
    with sqlite3.connect(db_path) as conn:
        rows, symbols, days, first, last = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), COUNT(DISTINCT snap_date), "
            "MIN(snap_date), MAX(snap_date) FROM chain_snapshots").fetchone()
    return {"rows": rows, "symbols": symbols, "days": days,
            "first": first, "last": last}


if __name__ == "__main__":
    import json
    print(json.dumps(archive_stats(), indent=1))
