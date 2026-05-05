"""Unit test for scripts/equity_stress_check.py.

Run directly:  venv/bin/python tests/test_equity_stress_check.py
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _create_minimal_paper_trades_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            type TEXT,
            entry_price REAL,
            status TEXT,
            quality_score REAL,
            strategy_name TEXT,
            entry_iv REAL,
            entry_delta REAL,
            entry_gamma REAL,
            entry_vega REAL,
            entry_theta REAL,
            weight_profile TEXT,
            long_strike REAL,
            spread_width REAL,
            net_credit REAL,
            max_loss_usd REAL,
            short_call_strike REAL,
            long_call_strike REAL,
            short_put_strike REAL,
            long_put_strike REAL
        )
        """
    )
    conn.commit()
    conn.close()


def _run_helper(db_path: Path, threshold_pct: float) -> str:
    env = os.environ.copy()
    env["EQUITY_STRESS_DB"] = str(db_path)
    env["EQUITY_STRESS_THRESHOLD_PCT"] = str(threshold_pct)
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/equity_stress_check.py")],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (proc.stdout or "") + (proc.stderr or "")


class TestEmptyBook(unittest.TestCase):
    def test_no_positions_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "p.db"
            _create_minimal_paper_trades_db(db)
            out = _run_helper(db, 100.0)
            self.assertIn("SAFE", out)
            self.assertIn("no open positions", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
