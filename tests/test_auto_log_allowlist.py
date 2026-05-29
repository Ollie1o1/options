"""Tests for the Phase 1 cohort quarantine allowlist and paper_only persistence.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_auto_log_allowlist -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest


class AllowlistDecisionTest(unittest.TestCase):
    """apply_auto_log_allowlist returns the right (action, flag) tuple."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.tmpdir, "config.json")
        cfg = {
            "auto_log": {
                "allowed_strategies": ["Long Call"],
                "paper_only_strategies": ["Bear Call", "Long Put", "Iron Condor", "Bull Put", "Short Put"],
            }
        }
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _call(self, strategy_name):
        from src.options_screener import apply_auto_log_allowlist
        return apply_auto_log_allowlist(
            {"strategy_name": strategy_name}, cfg_path=self.cfg_path
        )

    def test_allowed_strategy_returns_insert_0(self):
        """Long Call is in allowed_strategies → ("insert", 0)."""
        action, flag = self._call("Long Call")
        self.assertEqual(action, "insert")
        self.assertEqual(flag, 0)

    def test_paper_only_strategy_returns_insert_1(self):
        """Long Put is in paper_only_strategies → ("insert", 1)."""
        action, flag = self._call("Long Put")
        self.assertEqual(action, "insert")
        self.assertEqual(flag, 1)

    def test_unlisted_strategy_returns_drop(self):
        """A strategy not in either list → ("drop", None).

        "Bear Call" is in paper_only_strategies in the default config, so
        use a strategy that's genuinely unlisted: "Short Straddle".
        """
        action, flag = self._call("Short Straddle")
        self.assertEqual(action, "drop")
        self.assertIsNone(flag)

    def test_overlap_allowed_wins(self):
        """Strategy in BOTH lists → allowed wins → ("insert", 0)."""
        cfg = {
            "auto_log": {
                "allowed_strategies": ["Long Call", "Long Put"],
                "paper_only_strategies": ["Long Put", "Iron Condor"],
            }
        }
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f)
        action, flag = self._call("Long Put")
        self.assertEqual(action, "insert")
        self.assertEqual(flag, 0)


class PaperOnlyPersistenceTest(unittest.TestCase):
    """End-to-end: paper_only=1 written by log_trade is read back correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_trades.db")
        self.cfg_path = os.path.join(self.tmpdir, "config.json")
        # Minimal config so PaperManager doesn't choke on missing keys
        with open(self.cfg_path, "w") as f:
            json.dump({}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_paper_only_flag_persists(self):
        """log_trade with paper_only=1 stores 1 in the DB."""
        from src.paper_manager import PaperManager

        pm = PaperManager(db_path=self.db_path, config_path=self.cfg_path)

        trade = {
            "ticker": "AAPL",
            "expiration": "2026-07-18",
            "strike": 200.0,
            "type": "Put",
            "entry_price": 3.50,
            "quality_score": 0.72,
            "strategy_name": "Long Put",
            "paper_only": 1,
        }
        pm.log_trade(trade)

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT paper_only FROM trades WHERE ticker='AAPL' AND strategy_name='Long Put' LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, "Expected a row to be inserted")
        self.assertEqual(row[0], 1, f"Expected paper_only=1, got {row[0]}")

    def test_paper_only_defaults_to_zero_when_absent(self):
        """log_trade without paper_only key stores 0 (column default)."""
        from src.paper_manager import PaperManager

        pm = PaperManager(db_path=self.db_path, config_path=self.cfg_path)

        trade = {
            "ticker": "SPY",
            "expiration": "2026-07-18",
            "strike": 540.0,
            "type": "Call",
            "entry_price": 5.00,
            "quality_score": 0.80,
            "strategy_name": "Long Call",
        }
        pm.log_trade(trade)

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT paper_only FROM trades WHERE ticker='SPY' AND strategy_name='Long Call' LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, "Expected a row to be inserted")
        self.assertEqual(row[0], 0, f"Expected paper_only=0 (default), got {row[0]}")


if __name__ == "__main__":
    unittest.main()
