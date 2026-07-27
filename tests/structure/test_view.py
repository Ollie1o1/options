import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.structure import view as V
from src.structure.types import View


class TestBuildView(unittest.TestCase):
    def test_strong_positive_composite_is_bullish(self):
        v = V.build_view("AAPL", composite=0.8)
        self.assertEqual(v.direction, "BULLISH")
        self.assertGreater(v.confidence, 0.35)

    def test_strong_negative_composite_is_bearish_but_capped(self):
        v = V.build_view("NVDA", composite=-0.95)
        self.assertEqual(v.direction, "BEARISH")
        # docs/OUTLOOK_FINDINGS.md: bearish calls hit ~30%, so confidence
        # is hard-capped no matter how strong the signal looks.
        self.assertLessEqual(v.confidence, 0.4)

    def test_bullish_is_not_capped_at_bearish_level(self):
        v = V.build_view("AAPL", composite=0.95)
        self.assertGreater(v.confidence, 0.4)

    def test_weak_composite_collapses_to_neutral(self):
        v = V.build_view("SPY", composite=0.10)
        self.assertEqual(v.direction, "NEUTRAL")

    def test_drivers_are_carried_through(self):
        v = V.build_view("NVDA", composite=-0.9, drivers=["momentum -1.2z"])
        self.assertIn("momentum -1.2z", v.drivers)


class TestImpliedHit(unittest.TestCase):
    def test_bullish_confidence_raises_implied_hit(self):
        lo = V.implied_hit(View("A", "BULLISH", 0.2, []))
        hi = V.implied_hit(View("A", "BULLISH", 0.9, []))
        self.assertGreater(hi, lo)
        self.assertGreater(hi, 0.5)

    def test_bearish_confidence_LOWERS_implied_hit(self):
        # Intentional: the measured bearish base rate is 0.30, so more
        # bearish conviction means a WORSE expected hit rate.
        lo = V.implied_hit(View("A", "BEARISH", 0.1, []))
        hi = V.implied_hit(View("A", "BEARISH", 0.4, []))
        self.assertLess(hi, lo)
        self.assertLess(hi, 0.5)

    def test_neutral_is_a_coin_flip(self):
        self.assertAlmostEqual(V.implied_hit(View("A", "NEUTRAL", 0.0, [])), 0.5)
