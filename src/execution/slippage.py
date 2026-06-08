"""Real-vs-paper fill tracking.

After placing a mirror-mode ticket, the human records what they actually filled at.
This logs intended-vs-actual so slippage (the gap between the model's limit and real
fills) can be measured before any automation is ever trusted.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional

CONTRACT_MULTIPLIER = 100


def _connect(db_path: str) -> sqlite3.Connection:
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fills ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  ts TEXT, ticket_id TEXT, ticker TEXT,"
        "  intended_price REAL, actual_price REAL, contracts INTEGER,"
        "  slippage_per_contract REAL, slippage_usd REAL)"
    )
    return conn


def record_fill(db_path: str, ticker: str, intended_price: float,
                actual_price: float, contracts: int,
                ticket_id: str = "", ts: Optional[str] = None) -> None:
    """Record an actual fill against the intended limit price.

    Slippage is signed: positive means you paid MORE than intended (worse)."""
    ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slip_pc = actual_price - intended_price
    slip_usd = slip_pc * contracts * CONTRACT_MULTIPLIER
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO fills (ts, ticket_id, ticker, intended_price, actual_price,"
            " contracts, slippage_per_contract, slippage_usd) VALUES (?,?,?,?,?,?,?,?)",
            (ts, ticket_id, ticker, intended_price, actual_price, contracts,
             slip_pc, slip_usd),
        )
        conn.commit()
    finally:
        conn.close()


def slippage_report(db_path: str) -> dict:
    """Aggregate slippage across all recorded fills."""
    if not os.path.exists(db_path):
        return {"n_fills": 0, "avg_slippage_per_contract": 0.0, "total_slippage_usd": 0.0}
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*), AVG(slippage_per_contract), "
            "COALESCE(SUM(slippage_usd), 0) FROM fills"
        ).fetchone()
    finally:
        conn.close()
    n = int(row[0] or 0)
    return {
        "n_fills": n,
        "avg_slippage_per_contract": round(float(row[1]), 6) if n else 0.0,
        "total_slippage_usd": round(float(row[2]), 2),
    }
