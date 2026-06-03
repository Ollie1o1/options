import unittest
from src.leverage.risk import liquidation_price, passes_liquidation_safety


class TestLiquidation(unittest.TestCase):
    def test_long_liq_below_entry(self):
        liq = liquidation_price(entry=60000, side="long", leverage=5.0,
                                maint_margin=0.005)
        self.assertLess(liq, 60000)
        self.assertAlmostEqual(liq, 60000 * (1 - 1 / 5.0 + 0.005), places=2)

    def test_short_liq_above_entry(self):
        liq = liquidation_price(entry=60000, side="short", leverage=5.0,
                                maint_margin=0.005)
        self.assertGreater(liq, 60000)
        self.assertAlmostEqual(liq, 60000 * (1 + 1 / 5.0 - 0.005), places=2)

    def test_safety_rule_accepts_when_stop_3x_closer(self):
        self.assertTrue(passes_liquidation_safety(stop_dist=0.008, liq_dist=0.19))

    def test_safety_rule_rejects_when_stop_too_close_to_liq(self):
        self.assertFalse(passes_liquidation_safety(stop_dist=0.07, liq_dist=0.19))

    def test_safety_rule_boundary(self):
        self.assertTrue(passes_liquidation_safety(stop_dist=0.06, liq_dist=0.18))


if __name__ == "__main__":
    unittest.main()
