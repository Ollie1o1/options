"""Self-contained sqlite perps paper ledger. P&L via core.pnl. Safe by default:
nothing here ever sends an order; the (future) live path is gated elsewhere."""
from __future__ import annotations
import sqlite3
from typing import List
from typing import Optional
from .signals import Signal
from .sizing import Sizing
from src.core.pnl import realized_pnl
from src.paths import repo_path

# Module-level so a test can point the ledger somewhere else by patching THIS,
# rather than by chdir'ing and relying on a relative path to sandbox the write.
# That is how tests/leverage/test_cli.py used to isolate itself, and it is a
# sandbox that silently disappears the moment the path is anchored — which is
# exactly what happened here.
DEFAULT_LEDGER_DB = repo_path("paper_trades_leverage.db")

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
    def __init__(self, db_path: Optional[str] = None):
        # Resolved at call time, not import time, so patching DEFAULT_LEDGER_DB
        # takes effect. An explicit relative path still anchors; an absolute one
        # passes through.
        self.db_path = repo_path(db_path) if db_path else DEFAULT_LEDGER_DB
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

    def open_swing_position(self, symbol: str, side: str, ts, entry: float,
                            stop: float, qty, notional, eff_leverage) -> int:
        """Log a daily swing-breakout entry. The swing Signal has no fixed
        target/session, so target/liq are left null and session is tagged
        'swing' to distinguish it from the intraday ledger rows."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO perp_trades (symbol, side, ts, entry, stop, target, "
                "liq_price, qty, notional, eff_leverage, session, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?, 'swing', 'open')",
                (symbol, side, str(ts), entry, stop, None, None, qty, notional,
                 eff_leverage))
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
