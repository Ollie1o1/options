"""Fixed-tenor ATM call IV, the SI premium ratio, and the spread cost."""
import math
import unittest

from src.squeeze.sleeve import pricing


def _chain(dte_iv_pairs, spot=100.0, spread=0.02):
    rows = []
    for dte, iv in dte_iv_pairs:
        for strike in (spot * 0.9, spot, spot * 1.1):
            mid = max(0.5, spot * 0.1)
            rows.append({"dte": dte, "strike": strike, "iv": iv,
                         "bid": mid * (1 - spread), "ask": mid * (1 + spread),
                         "option_type": "call"})
    return rows


class AtmIvTest(unittest.TestCase):
    def test_an_exact_tenor_is_returned_as_is(self):
        got = pricing.atm_call_iv(_chain([(30, 0.80)]), 100.0, 30)
        self.assertAlmostEqual(got, 0.80, places=6)

    def test_interpolation_is_linear_in_total_variance(self):
        # 20d at 100% and 40d at 100% -> 30d must also be 100%
        got = pricing.atm_call_iv(_chain([(20, 1.0), (40, 1.0)]), 100.0, 30)
        self.assertAlmostEqual(got, 1.0, places=6)

    def test_interpolation_sits_between_the_bracketing_vols(self):
        got = pricing.atm_call_iv(_chain([(20, 0.60), (40, 1.00)]), 100.0, 30)
        self.assertGreater(got, 0.60)
        self.assertLess(got, 1.00)

    def test_a_lone_expiry_inside_the_tolerance_band_is_used(self):
        self.assertIsNotNone(pricing.atm_call_iv(_chain([(35, 0.80)]), 100.0, 30))

    def test_a_lone_expiry_outside_the_band_is_refused(self):
        self.assertIsNone(pricing.atm_call_iv(_chain([(120, 0.80)]), 100.0, 30))

    def test_puts_are_ignored(self):
        rows = _chain([(30, 0.80)])
        for r in rows:
            r["option_type"] = "put"
        self.assertIsNone(pricing.atm_call_iv(rows, 100.0, 30))

    def test_crossed_and_zero_bid_quotes_are_refused(self):
        rows = _chain([(30, 0.80)])
        for r in rows:
            r["bid"], r["ask"] = 5.0, 1.0
        self.assertIsNone(pricing.atm_call_iv(rows, 100.0, 30))

    def test_an_empty_chain_returns_none(self):
        self.assertIsNone(pricing.atm_call_iv([], 100.0, 30))


class SpreadAndPremiumTest(unittest.TestCase):
    def test_relative_spread_is_measured_against_the_mid(self):
        got = pricing.relative_spread(_chain([(30, 0.80)], spread=0.05), 100.0, 30)
        self.assertAlmostEqual(got, 0.10, places=6)

    def test_a_richer_implied_vol_costs_more_premium(self):
        got = pricing.premium_ratio(0.90, 0.80, 100.0, 30)
        self.assertGreater(got, 0.0)

    def test_equal_implied_vols_cost_the_same(self):
        got = pricing.premium_ratio(0.80, 0.80, 100.0, 30)
        self.assertAlmostEqual(got, 0.0, places=9)

    def test_a_cheaper_implied_vol_gives_a_negative_ratio(self):
        self.assertLess(pricing.premium_ratio(0.70, 0.80, 100.0, 30), 0.0)

    def test_non_positive_vols_return_none(self):
        self.assertIsNone(pricing.premium_ratio(0.0, 0.80, 100.0, 30))
