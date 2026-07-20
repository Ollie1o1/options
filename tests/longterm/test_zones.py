"""Tests for zone-state math (pure, no network)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import plan as P
from src.longterm import zones as Z


def mu():
    return P.PlanName("MU", [P.Tranche(750, 0.4), P.Tranche(650, 0.35), P.Tranche(550, 0.25)])


def snap(spot, sigma=0.041, high=1255.0, ma200=700.0):
    return Z.Snapshot("MU", spot=spot, high_52w=high, low_52w=103.0,
                      ma200=ma200, daily_sigma=sigma)


class TestAssess(unittest.TestCase):
    def test_watching_far_above(self):
        r = Z.assess(mu(), snap(848.95), set())
        self.assertEqual(r.state, Z.WATCHING)
        self.assertEqual(r.next_level, 750)
        self.assertAlmostEqual(r.distance_pct, (848.95 - 750) / 750 * 100, places=4)
        self.assertAlmostEqual(r.drawdown_pct, (848.95 / 1255.0 - 1) * 100, places=4)
        self.assertTrue(r.above_ma200)

    def test_near_within_one_sigma(self):
        # sigma 4.1% is wider than 2%: 750*1.04 = 780 is inside the NEAR band
        r = Z.assess(mu(), snap(780.0), set())
        self.assertEqual(r.state, Z.NEAR)

    def test_near_band_uses_wider_of_sigma_and_2pct(self):
        # sigma 0.5% → band is 2%; 764 is ~1.87% above 750 → NEAR
        self.assertEqual(Z.assess(mu(), snap(764.0, sigma=0.005), set()).state, Z.NEAR)
        # 766 is ~2.13% above → WATCHING
        self.assertEqual(Z.assess(mu(), snap(766.0, sigma=0.005), set()).state, Z.WATCHING)

    def test_in_zone_at_and_below_level(self):
        self.assertEqual(Z.assess(mu(), snap(750.0), set()).state, Z.IN_ZONE)
        self.assertEqual(Z.assess(mu(), snap(742.0), set()).state, Z.IN_ZONE)

    def test_filled_tranche_advances_ladder(self):
        # 750 filled → next open level is 650; spot 742 is WATCHING vs 650
        r = Z.assess(mu(), snap(742.0), {750.0})
        self.assertEqual(r.next_level, 650)
        self.assertEqual(r.state, Z.WATCHING)

    def test_fully_filled(self):
        r = Z.assess(mu(), snap(742.0), {750.0, 650.0, 550.0})
        self.assertEqual(r.state, Z.FILLED)
        self.assertIsNone(r.next_level)
        self.assertIsNone(r.distance_pct)

    def test_sigma_dist(self):
        r = Z.assess(mu(), snap(780.0), set())
        self.assertAlmostEqual(r.sigma_dist, (780.0 - 750) / (780.0 * 0.041), places=4)

    def test_zero_sigma_no_crash(self):
        r = Z.assess(mu(), snap(780.0, sigma=0.0), set())
        self.assertIsNone(r.sigma_dist)
        self.assertEqual(r.state, Z.WATCHING)

    def test_no_ma200(self):
        r = Z.assess(mu(), snap(780.0, ma200=None), set())
        self.assertIsNone(r.above_ma200)


if __name__ == "__main__":
    unittest.main()
