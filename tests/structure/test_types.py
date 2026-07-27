import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.structure.types import (BEARISH_BASE_RATE, BEARISH_STRUCTURES,
                                 BULLISH_BASE_RATE, BULLISH_STRUCTURES,
                                 CREDIT_STRUCTURES, DEBIT_STRUCTURES,
                                 LEG_COUNT, Expression, Rejection,
                                 StructureMargin, View)


class TestTypes(unittest.TestCase):
    def test_leg_counts_match_real_structures(self):
        self.assertEqual(LEG_COUNT["Long Call"], 1)
        self.assertEqual(LEG_COUNT["Bull Put"], 2)
        self.assertEqual(LEG_COUNT["Iron Condor"], 4)

    def test_debit_and_credit_are_disjoint(self):
        self.assertFalse(DEBIT_STRUCTURES & CREDIT_STRUCTURES)

    def test_base_rates_cite_measured_asymmetry(self):
        # docs/OUTLOOK_FINDINGS.md: bullish 66-72%, bearish ~30%
        self.assertAlmostEqual(BULLISH_BASE_RATE, 0.68)
        self.assertAlmostEqual(BEARISH_BASE_RATE, 0.30)
        self.assertLess(BEARISH_BASE_RATE, 0.5)

    def test_directional_sets_cover_known_structures(self):
        self.assertIn("Bull Put", BULLISH_STRUCTURES)
        self.assertIn("Long Put", BEARISH_STRUCTURES)

    def test_dataclasses_construct(self):
        m = StructureMargin(strategy="Bull Put", n=68, wins=45, losses=23,
                            avg_win=116.0, avg_loss=70.0, breakeven_hit=0.375,
                            realized_hit=0.662, margin=0.287, state="ACTIVE",
                            ci_lo=0.10, ci_hi=0.45)
        self.assertEqual(m.state, "ACTIVE")
        v = View(symbol="NVDA", direction="BEARISH", confidence=0.31,
                 drivers=["momentum -1.2z"])
        self.assertEqual(v.direction, "BEARISH")
        e = Expression(strategy="Long Put", margin=0.122, breakeven_hit=0.229,
                       realized_hit=0.351, capital_required=340.0,
                       cost_drag_pct=3.3, legs=1)
        self.assertEqual(e.legs, 1)
        r = Rejection(strategy="Long Call", reason="BENCHED (margin -12.8)")
        self.assertIn("BENCHED", r.reason)
