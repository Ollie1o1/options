"""Self-contained sqlite perps paper ledger. P&L via core.pnl. Safe by default:
nothing here ever sends an order; the (future) live path is gated elsewhere."""
from __future__ import annotations
import sqlite3
from typing import List
from .signals import Signal
from .sizing import Sizing
from src.core.pnl import realized_pnl

_SCHEMA = """
CREATE TABLE IF NOT EXISTS perp_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT, side TEXT, ts TEXT, entry REAL, stop REAL, target REAL,
  liq_price REAL, qty REAL, notional REAL, eff_leverage REAL,
  session TEXT, status TEXT DEFAULT 'open',
  exit_price REAL, exit_reason TEXT, pnl_pct REAL, pnl_usd REAL, closed_ts TEXT
);
"""


class PaperLedger:
    def __init__(self, db_path: str = "paper_trades_leverage.db"):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def open_position(self, sig: Signal, sizing: Sizing, liq_price: float) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO perp_trades (symbol, side, ts, entry, stop, target, "
                "liq_price, qty, notional, eff_leverage, session, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?, 'open')",
                (sig.symbol, sig.side, str(sig.ts), sig.entry, sig.stop, sig.target,
                 liq_price, sizing.qty, sizing.notional, sizing.eff_leverage,
                 sig.session))
            return cur.lastrowid

    def close_position(self, trade_id: int, exit_price: float,
                       reason: str) -> None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM perp_trades WHERE id=?",
                            (trade_id,)).fetchone()
            if row is None or row["status"] != "open":
                return
            pnl = realized_pnl(entry=row["entry"], exit_price=exit_price,
                               qty=row["qty"], side=row["side"], structure="debit")
            c.execute(
                "UPDATE perp_trades SET status='closed', exit_price=?, "
                "exit_reason=?, pnl_pct=?, pnl_usd=?, closed_ts=datetime('now') "
                "WHERE id=?",
                (exit_price, reason, pnl["pnl_pct"], pnl["pnl_usd"], trade_id))

    def open_positions(self) -> List[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM perp_trades WHERE status='open'")]

    def closed_positions(self) -> List[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM perp_trades WHERE status='closed'")]
