import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.crypto.screener import _quantity_for

class TestScreenerSizing(unittest.TestCase):
    def test_debit_quantity_from_premium(self):
        q = _quantity_for(structure="debit", entry_price=2561.03,
                          spread_width=None, net_credit=None)
        self.assertAlmostEqual(q, math.floor(999.0/2561.03*1e4)/1e4, places=6)

    def test_credit_quantity_from_max_loss(self):
        # width 1000, credit 250 → unit_risk 750 → 999/750
        q = _quantity_for(structure="credit", entry_price=250.0,
                          spread_width=1000.0, net_credit=250.0)
        self.assertAlmostEqual(q, math.floor(999.0/750.0*1e4)/1e4, places=6)

if __name__ == "__main__":
    unittest.main()
