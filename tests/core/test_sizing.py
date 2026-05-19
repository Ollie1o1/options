import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.sizing import capped_quantity

class TestCappedQuantity(unittest.TestCase):
    def test_debit_cap_targets_just_under_1000(self):
        q = capped_quantity(unit_risk=2561.03, cap_usd=1000.0)
        # floor to 4dp: assertAlmostEqual to 4 significant 4dp-floor places
        self.assertAlmostEqual(q, math.floor(999.0 / 2561.03 * 1e4) / 1e4, places=8)
        self.assertLess(q * 2561.03, 1000.0)

    def test_cheap_premium_still_capped_not_inflated(self):
        q = capped_quantity(unit_risk=77.72, cap_usd=1000.0)
        self.assertAlmostEqual(q, math.floor(999.0 / 77.72 * 1e4) / 1e4, places=8)

    def test_rounds_to_four_dp_never_over_cap(self):
        q = capped_quantity(unit_risk=333.33, cap_usd=1000.0)
        self.assertEqual(q, round(q, 4))
        self.assertLessEqual(q * 333.33, 1000.0)

    def test_zero_unit_risk_returns_zero(self):
        self.assertEqual(capped_quantity(unit_risk=0.0, cap_usd=1000.0), 0.0)

if __name__ == "__main__":
    unittest.main()
