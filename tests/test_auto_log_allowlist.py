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


class CohortDteFloorTest(unittest.TestCase):
    """Long Calls below the cohort DTE floor log as data-only (paper_only=1).

    The floor protects the validation gate from short-horizon calls that would
    be force-closed by the time-exit before the swing thesis can play out.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.tmpdir, "config.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_cfg(self, cfg):
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f)

    def _call(self, trade):
        from src.options_screener import apply_auto_log_allowlist
        return apply_auto_log_allowlist(trade, cfg_path=self.cfg_path)

    def test_long_call_below_explicit_floor_is_paper_only(self):
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"],
                                       "cohort_min_dte": 30}})
        action, flag = self._call({"strategy_name": "Long Call", "dte": 14})
        self.assertEqual((action, flag), ("insert", 1))

    def test_long_call_at_or_above_floor_is_cohort_eligible(self):
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"],
                                       "cohort_min_dte": 30}})
        self.assertEqual(self._call({"strategy_name": "Long Call", "dte": 30}), ("insert", 0))
        self.assertEqual(self._call({"strategy_name": "Long Call", "dte": 45}), ("insert", 0))

    def test_unknown_dte_does_not_quarantine(self):
        # Backward-compatible: a trade with no DTE info stays cohort-eligible.
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"],
                                       "cohort_min_dte": 30}})
        self.assertEqual(self._call({"strategy_name": "Long Call"}), ("insert", 0))

    def test_floor_derived_from_time_exit_when_unset(self):
        # No explicit cohort_min_dte → derive time_exit_dte (21) + runway (9) = 30.
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"]},
                         "exit_rules": {"time_exit_dte": 21}})
        self.assertEqual(self._call({"strategy_name": "Long Call", "dte": 25}), ("insert", 1))
        self.assertEqual(self._call({"strategy_name": "Long Call", "dte": 31}), ("insert", 0))

    def test_floor_applies_to_expiration_date(self):
        from datetime import date, timedelta
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"],
                                       "cohort_min_dte": 30}})
        near = (date.today() + timedelta(days=12)).isoformat()
        far = (date.today() + timedelta(days=40)).isoformat()
        self.assertEqual(self._call({"strategy_name": "Long Call", "expiration": near}), ("insert", 1))
        self.assertEqual(self._call({"strategy_name": "Long Call", "expiration": far}), ("insert", 0))

    def test_paper_only_strategy_unaffected_by_floor(self):
        # A quarantined strategy stays paper_only regardless of DTE.
        self._write_cfg({"auto_log": {"allowed_strategies": ["Long Call"],
                                       "paper_only_strategies": ["Long Put"],
                                       "cohort_min_dte": 30}})
        self.assertEqual(self._call({"strategy_name": "Long Put", "dte": 45}), ("insert", 1))


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


class SpreadCondorPaperOnlyPersistenceTest(unittest.TestCase):
    """End-to-end: paper_only=1 written via log_spread / log_iron_condor is read back."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_spread_trades.db")
        self.cfg_path = os.path.join(self.tmpdir, "config.json")
        with open(self.cfg_path, "w") as f:
            json.dump({}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_spread_persists_paper_only_1(self):
        """log_spread with paper_only=1 stores 1 in the DB."""
        from src.paper_manager import PaperManager

        pm = PaperManager(db_path=self.db_path, config_path=self.cfg_path)

        spread = {
            "ticker": "SPY",
            "expiration": "2026-07-18",
            "short_strike": 540.0,
            "long_strike": 545.0,
            "type": "Bear Call",
            "net_credit": 0.85,
            "max_profit": 85.0,
            "max_loss": 415.0,
            "quality_score": 0.60,
            "paper_only": 1,
        }
        pm.log_spread(spread)

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT paper_only FROM trades WHERE ticker='SPY' AND strategy_name='Bear Call' LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, "Expected a row to be inserted by log_spread")
        self.assertEqual(row[0], 1, f"Expected paper_only=1, got {row[0]}")

    def test_log_iron_condor_persists_paper_only_1(self):
        """log_iron_condor with paper_only=1 stores 1 in the DB."""
        from src.paper_manager import PaperManager

        pm = PaperManager(db_path=self.db_path, config_path=self.cfg_path)

        condor = {
            "ticker": "QQQ",
            "expiration": "2026-07-18",
            "short_put_strike": 420.0,
            "long_put_strike": 415.0,
            "short_call_strike": 480.0,
            "long_call_strike": 485.0,
            "total_credit": 1.20,
            "max_profit": 120.0,
            "max_risk": 380.0,
            "quality_score": 0.55,
            "paper_only": 1,
        }
        pm.log_iron_condor(condor)

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT paper_only FROM trades WHERE ticker='QQQ' AND strategy_name='Iron Condor' LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, "Expected a row to be inserted by log_iron_condor")
        self.assertEqual(row[0], 1, f"Expected paper_only=1, got {row[0]}")


if __name__ == "__main__":
    unittest.main()
