"""Ledger integration for capital-at-risk: storage plus the auto-log budget gate.

unittest style on purpose — the options venv has no pytest, so these have to be
runnable locally as well as in CI.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager, _SCHEMA_VERSION


def _config(path, max_capital_at_risk=None):
    cfg = {
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25, "time_exit_dte": 21},
        "paper_trading": {
            "commission_per_contract": 0.65,
            "slippage_per_share": 0.05,
            "default_db_path": "paper_trades.db",
        },
        "auto_log": {"allowed_strategies": ["Long Call"]},
    }
    if max_capital_at_risk is not None:
        cfg["auto_log"]["max_capital_at_risk"] = max_capital_at_risk
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _long_call(**over):
    trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-20",
        "strike": 150.0,
        "type": "call",
        "entry_price": 3.50,
        "quality_score": 0.75,
        "strategy_name": "Long Call",
    }
    trade.update(over)
    return trade


def _short_put(**over):
    trade = {
        "ticker": "WFC",
        "expiration": "2026-06-20",
        "strike": 77.5,
        "type": "put",
        "entry_price": 1.52,
        "quality_score": 0.60,
        "strategy_name": "Short Put",
    }
    trade.update(over)
    return trade


class _LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")
        self.cfg = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute("SELECT * FROM trades").fetchall()


class TestSchema(_LedgerCase):
    def test_migration_adds_the_capital_at_risk_column(self):
        _config(self.cfg)
        PaperManager(db_path=self.db, config_path=self.cfg)
        with sqlite3.connect(self.db) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        self.assertIn("capital_at_risk", cols)

    def test_schema_version_is_bumped(self):
        _config(self.cfg)
        PaperManager(db_path=self.db, config_path=self.cfg)
        with sqlite3.connect(self.db) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, _SCHEMA_VERSION)
        self.assertGreaterEqual(_SCHEMA_VERSION, 16)


class TestStorage(_LedgerCase):
    def test_long_call_stores_the_debit_as_capital_at_risk(self):
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        pm.log_trade(_long_call())
        self.assertAlmostEqual(self.rows()[0]["capital_at_risk"], 350.0)

    def test_short_put_stores_collateral_not_the_credit(self):
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        pm.log_trade(_short_put())
        self.assertAlmostEqual(self.rows()[0]["capital_at_risk"], 7598.0)

    def test_credit_spread_stores_its_max_loss(self):
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        pm.log_spread({
            "ticker": "INTC",
            "expiration": "2026-06-20",
            "type": "Bull Put",
            "short_strike": 80.0,
            "long_strike": 79.0,
            "net_credit": 0.50,
            "max_loss": 50.0,
            "quality_score": 0.7,
        })
        self.assertAlmostEqual(self.rows()[0]["capital_at_risk"], 50.0)


class TestBudgetGate(_LedgerCase):
    def test_unaffordable_trade_is_rejected_when_a_cap_is_set(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        inserted = pm.log_trade(_short_put())
        self.assertFalse(inserted)
        self.assertEqual(len(self.rows()), 0)

    def test_affordable_trade_is_still_logged(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        inserted = pm.log_trade(_long_call())
        self.assertTrue(inserted)
        self.assertEqual(len(self.rows()), 1)

    def test_no_cap_configured_logs_everything(self):
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertTrue(pm.log_trade(_short_put()))
        self.assertEqual(len(self.rows()), 1)

    def test_explicit_override_bypasses_the_cap(self):
        # A deliberate manual entry is the user's call, not the feeder's.
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertTrue(pm.log_trade(_short_put(allow_unaffordable=True)))
        self.assertEqual(len(self.rows()), 1)

    def test_rejections_are_counted_not_silent(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        pm.log_trade(_short_put())
        pm.log_trade(_short_put(ticker="GE", strike=280.0, entry_price=7.0))
        self.assertEqual(pm.unaffordable_rejected, 2)

    def test_dedup_path_reports_rejection_as_not_inserted(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertFalse(pm.log_trade_if_new(_short_put()))
        self.assertEqual(len(self.rows()), 0)

    def test_spread_helper_reports_a_refused_row_as_not_inserted(self):
        # The wrappers returned True unconditionally, so a gated spread was
        # counted as logged by every caller that trusts the return value.
        _config(self.cfg, max_capital_at_risk=100)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        inserted = pm.log_spread_if_new({
            "ticker": "SPY", "expiration": "2026-06-20", "type": "Bull Put",
            "short_strike": 500.0, "long_strike": 495.0, "net_credit": 1.00,
            "max_loss": 400.0, "quality_score": 0.7,
        })
        self.assertIs(inserted, False)
        self.assertEqual(len(self.rows()), 0)

    def test_condor_helper_reports_a_refused_row_as_not_inserted(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        inserted = pm.log_iron_condor_if_new({
            "ticker": "AAPL", "expiration": "2026-06-20",
            "short_put_strike": 190.0, "long_put_strike": 185.0,
            "short_call_strike": 210.0, "long_call_strike": 215.0,
            "total_credit": 1.20, "max_risk": 3800.0, "quality_score": 0.6,
        })
        self.assertIs(inserted, False)
        self.assertEqual(len(self.rows()), 0)

    def test_affordable_spread_still_reports_inserted(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        inserted = pm.log_spread_if_new({
            "ticker": "INTC", "expiration": "2026-06-20", "type": "Bull Put",
            "short_strike": 80.0, "long_strike": 79.0, "net_credit": 0.50,
            "max_loss": 50.0, "quality_score": 0.7,
        })
        self.assertIs(inserted, True)
        self.assertEqual(len(self.rows()), 1)

    def test_unbounded_risk_is_rejected_under_a_cap(self):
        _config(self.cfg, max_capital_at_risk=750)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        naked_call = _long_call(strategy_name="Short Call", type="call")
        self.assertIs(pm.log_trade(naked_call), False)
        self.assertEqual(len(self.rows()), 0)


class TestStrategyBreakdown(_LedgerCase):
    def test_breakdown_reports_return_on_capital_at_risk(self):
        # Two closed long calls: $350 at risk each, +$70 and -$35 -> +5% on risk.
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        pm.log_trade(_long_call())
        pm.log_trade(_long_call(strike=155.0))
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "UPDATE trades SET status='CLOSED', pnl_pct=0.2, pnl_usd=70.0 "
                "WHERE strike=150.0"
            )
            conn.execute(
                "UPDATE trades SET status='CLOSED', pnl_pct=-0.1, pnl_usd=-35.0 "
                "WHERE strike=155.0"
            )
        breakdown = {r["strategy"]: r for r in pm.get_strategy_breakdown()}
        self.assertAlmostEqual(breakdown["Long Call"]["return_on_risk"], 0.05)
        self.assertAlmostEqual(breakdown["Long Call"]["capital_at_risk"], 700.0)


if __name__ == "__main__":
    unittest.main()
