import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.pnl import realized_pnl

class TestRealizedPnl(unittest.TestCase):
    def test_long_debit_with_quantity(self):
        r = realized_pnl(entry=2561.03, exit_price=3115.05, qty=0.39, side="long", structure="debit")
        self.assertAlmostEqual(r["pnl_usd"], (3115.05 - 2561.03) * 0.39, places=4)
        self.assertAlmostEqual(r["pnl_pct"], (3115.05 - 2561.03) / 2561.03, places=6)

    def test_short_premium_expired_worthless(self):
        r = realized_pnl(entry=240.0, exit_price=0.0, qty=1.0, side="short", structure="debit")
        self.assertAlmostEqual(r["pnl_usd"], 240.0, places=6)
        self.assertAlmostEqual(r["pnl_pct"], 1.0, places=6)

    def test_credit_spread_close(self):
        r = realized_pnl(entry=250.0, exit_price=90.0, qty=2.0, side="short", structure="credit")
        self.assertAlmostEqual(r["pnl_usd"], (250.0 - 90.0) * 2.0, places=6)
        self.assertAlmostEqual(r["pnl_pct"], (250.0 - 90.0) / 250.0, places=6)

    def test_qty_defaults_to_one(self):
        r = realized_pnl(entry=100.0, exit_price=130.0, qty=None, side="long", structure="debit")
        self.assertAlmostEqual(r["pnl_usd"], 30.0, places=6)

    def test_long_debit_loss(self):
        r = realized_pnl(entry=100.0, exit_price=70.0, qty=2.0, side="long", structure="debit")
        self.assertAlmostEqual(r["pnl_usd"], -60.0, places=6)
        self.assertAlmostEqual(r["pnl_pct"], -0.30, places=6)

if __name__ == "__main__":
    unittest.main()
