"""The ledger charges currency conversion on closes.

A CAD account trading US-listed options pays Wealthsimple's conversion spread on
the premium in both directions. It is the largest single cost left in the model
now that commissions are $0, and it scales with position size rather than with
leg count — so leaving it out flatters big positions most.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager


def _config(path, fx_rate=None, commission=0.0):
    cfg = {
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25, "time_exit_dte": 21},
        "paper_trading": {
            "commission_per_contract": commission,
            "slippage_per_share": 0.05,
            "default_db_path": "paper_trades.db",
        },
    }
    if fx_rate is not None:
        cfg["paper_trading"]["fx_conversion_rate"] = fx_rate
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class TestFxConfiguration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.cfg = os.path.join(self.tmp.name, "c.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_the_conversion_rate_from_config(self):
        _config(self.cfg, fx_rate=0.015)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertAlmostEqual(pm._fx_conversion_rate, 0.015)

    def test_defaults_to_no_conversion_cost_when_unset(self):
        # An unconfigured ledger must not invent a cost.
        _config(self.cfg)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertEqual(pm._fx_conversion_rate, 0.0)

    def test_conversion_adds_to_friction_on_a_premium(self):
        # $1.00/share premium, 1.5% each way = $0.03/share on top of the spread.
        _config(self.cfg, fx_rate=0.015)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertAlmostEqual(pm._fx_per_share(1.00), 0.03)

    def test_no_conversion_cost_with_a_usd_account(self):
        _config(self.cfg, fx_rate=0.0)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertEqual(pm._fx_per_share(1.00), 0.0)

    def test_conversion_scales_with_the_premium(self):
        _config(self.cfg, fx_rate=0.015)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertAlmostEqual(pm._fx_per_share(4.00), 4 * pm._fx_per_share(1.00))

    def test_zero_commission_config_is_honoured(self):
        # Wealthsimple charges nothing per contract; the ledger must not add 0.65.
        _config(self.cfg, commission=0.0)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertEqual(pm._commission_per_contract, 0.0)
        self.assertAlmostEqual(pm._friction_per_share, 2 * 0.05)


if __name__ == "__main__":
    unittest.main()
