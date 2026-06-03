import unittest
from src.leverage.sizing import effective_leverage_size, Sizing


class TestEffectiveLeverageSize(unittest.TestCase):
    def test_quarter_kelly_with_2pct_cap(self):
        # p=0.42, b=2.0 -> Kelly f*=(0.42*2-0.58)/2=0.13; quarter=0.0325 -> capped 0.02
        s = effective_leverage_size(equity=1500, stop_distance_pct=0.008,
                                    kelly_p=0.42, kelly_b=2.0)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s.risk_frac, 0.02, places=4)
        self.assertAlmostEqual(s.risk_usd, 30.0, places=2)
        self.assertAlmostEqual(s.eff_leverage, 2.5, places=2)  # 0.02/0.008

    def test_tighter_stop_lands_in_band(self):
        s = effective_leverage_size(equity=1500, stop_distance_pct=0.005,
                                    kelly_p=0.42, kelly_b=2.0)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s.eff_leverage, 4.0, places=2)  # 0.02/0.005

    def test_hard_cap_5x(self):
        s = effective_leverage_size(equity=1500, stop_distance_pct=0.002,
                                    kelly_p=0.42, kelly_b=2.0)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s.eff_leverage, 5.0, places=6)

    def test_below_floor_returns_none(self):
        # 0.02/0.02 = 1.0x < 2x floor (un-leveraged buy-and-hold) -> skip
        s = effective_leverage_size(equity=1500, stop_distance_pct=0.02,
                                    kelly_p=0.42, kelly_b=2.0)
        self.assertIsNone(s)

    def test_notional_and_qty(self):
        s = effective_leverage_size(equity=1500, stop_distance_pct=0.005,
                                    kelly_p=0.42, kelly_b=2.0, price=60000.0)
        self.assertAlmostEqual(s.notional, 6000.0, places=2)  # 1500*4
        self.assertAlmostEqual(s.qty, 0.1, places=6)          # 6000/60000


if __name__ == "__main__":
    unittest.main()
