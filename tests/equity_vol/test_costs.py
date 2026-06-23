"""Tests for the equity cost model + pricing re-export."""
from __future__ import annotations
import unittest
from src.equity_vol.costs import CostModel
from src.equity_vol import pricing


class CostTests(unittest.TestCase):
    def test_option_commissions_two_legs(self):
        self.assertAlmostEqual(CostModel().option_commissions(n_legs=2, contracts=1), 1.30, places=6)

    def test_hedge_cost_is_bps_of_abs_notional(self):
        # 1 bps of 10_000 = 1.0
        self.assertAlmostEqual(CostModel(hedge_slippage_bps=1.0).hedge_cost(-10_000), 1.0, places=6)

    def test_costs_never_negative(self):
        self.assertGreaterEqual(CostModel().hedge_cost(-5000), 0.0)


class PricingReexportTests(unittest.TestCase):
    def test_straddle_atm_positive_and_delta_small(self):
        self.assertGreater(pricing.straddle(100, 100, 0.08, 0.04, 0.3), 0.0)
        self.assertLess(abs(pricing.straddle_delta(100, 100, 0.08, 0.04, 0.3)), 0.2)


if __name__ == "__main__":
    unittest.main()
