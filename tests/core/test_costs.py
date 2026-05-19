import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.costs import round_turn_cost_pct_equity

class TestCosts(unittest.TestCase):
    def test_ten_x_reference_case(self):
        c = round_turn_cost_pct_equity(taker_fee=0.00055, slippage=0.0002,
                                       funding_share=0.0001, leverage=10.0)
        self.assertAlmostEqual(c, 0.014, places=6)

    def test_scales_linearly_with_leverage(self):
        a = round_turn_cost_pct_equity(0.00055, 0.0002, 0.0001, 2.0)
        b = round_turn_cost_pct_equity(0.00055, 0.0002, 0.0001, 4.0)
        self.assertAlmostEqual(b, 2 * a, places=9)

if __name__ == "__main__":
    unittest.main()
