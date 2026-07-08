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


class TestMinNotionalGate(unittest.TestCase):
    # stop_distance_pct=0.005 -> eff_lev=0.02/0.005=4 (in the 2-5 band, so the
    # min_leverage early-return does NOT fire) -> notional = equity*4.
    def test_rejects_below_min_notional(self):
        # equity 20 -> notional 80 < $100 venue minimum -> None via the gate
        s = effective_leverage_size(equity=20.0, stop_distance_pct=0.005,
                                    price=60000.0, min_notional=100.0)
        self.assertIsNone(s)

    def test_allows_when_notional_meets_minimum(self):
        # equity 5000 -> notional 20000 >= minimum -> sized
        s = effective_leverage_size(equity=5000.0, stop_distance_pct=0.005,
                                    price=60000.0, min_notional=100.0)
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s.notional, 100.0)

    def test_default_min_notional_is_noop(self):
        # no min_notional passed -> unchanged behaviour
        s = effective_leverage_size(equity=20.0, stop_distance_pct=0.005,
                                    price=60000.0)
        self.assertIsNotNone(s)


if __name__ == "__main__":
    unittest.main()
