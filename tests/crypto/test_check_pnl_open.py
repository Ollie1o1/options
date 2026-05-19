import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.crypto.check_pnl import _unrealized

class TestCheckPnlOpen(unittest.TestCase):
    def test_unrealized_scales_by_quantity(self):
        # long debit: entry 100, live 130, qty 0.5 → (130-100)*0.5 = 15
        usd, pct = _unrealized(entry=100.0, live=130.0, qty=0.5,
                               side="long", structure="debit")
        self.assertAlmostEqual(usd, 15.0, places=6)
        self.assertAlmostEqual(pct, 0.30, places=6)

    def test_unrealized_short_premium_decays(self):
        # short debit position: entry 50, live 10, qty 1.0 → +40 per unit
        usd, pct = _unrealized(entry=50.0, live=10.0, qty=1.0,
                               side="short", structure="debit")
        self.assertAlmostEqual(usd, 40.0, places=6)
        self.assertAlmostEqual(pct, 0.80, places=6)

if __name__ == "__main__":
    unittest.main()
