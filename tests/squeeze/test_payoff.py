"""Synthetic call return under the frozen ladder, both valuation variants."""
import unittest

from src.squeeze.sleeve import payoff


class PayoffTest(unittest.TestCase):
    SPOT = 100.0
    SIG_D = 0.05          # daily -> sigma_h = 0.05*sqrt(42) ~ 0.324
    IV = 0.80

    def test_a_flat_path_loses_but_not_the_whole_premium(self):
        # 90 DTE entry, sold at 42 trading days held: ~30 calendar days left,
        # so the central variant must recover real time value.
        got = payoff.synthetic_call_return(
            [self.SPOT] * 42, self.SPOT, self.SIG_D, self.IV, variant="central")
        self.assertLess(got, 0.0)
        self.assertGreater(got, -0.75)

    def test_the_conservative_variant_is_never_kinder_than_the_central_one(self):
        path = [self.SPOT * (1 + 0.01 * i) for i in range(42)]
        cen = payoff.synthetic_call_return(
            path, self.SPOT, self.SIG_D, self.IV, variant="central")
        con = payoff.synthetic_call_return(
            path, self.SPOT, self.SIG_D, self.IV, variant="conservative")
        self.assertLessEqual(con, cen + 1e-9)

    def test_a_flat_path_loses_everything_under_the_conservative_variant(self):
        got = payoff.synthetic_call_return(
            [self.SPOT] * 42, self.SPOT, self.SIG_D, self.IV,
            variant="conservative")
        self.assertAlmostEqual(got, -1.0, places=6)

    def test_a_large_spike_pays_multiples(self):
        path = [self.SPOT] * 5 + [self.SPOT * 2.0] * 37
        got = payoff.synthetic_call_return(
            path, self.SPOT, self.SIG_D, self.IV, variant="conservative")
        self.assertGreater(got, 1.0)

    def test_return_is_bounded_below_by_minus_one(self):
        path = [self.SPOT * 0.01] * 42
        for variant in ("central", "conservative"):
            got = payoff.synthetic_call_return(
                path, self.SPOT, self.SIG_D, self.IV, variant=variant)
            self.assertGreaterEqual(got, -1.0 - 1e-9)

    def test_non_finite_inputs_return_none(self):
        self.assertIsNone(payoff.synthetic_call_return(
            [self.SPOT] * 42, self.SPOT, 0.0, self.IV))
        self.assertIsNone(payoff.synthetic_call_return(
            [self.SPOT] * 42, self.SPOT, self.SIG_D, 0.0))

    def test_a_higher_strike_costs_less_and_pays_more_on_a_spike(self):
        path = [self.SPOT] * 5 + [self.SPOT * 2.0] * 37
        atm = payoff.synthetic_call_return(
            path, self.SPOT, self.SIG_D, self.IV, strike_mult=1.0,
            variant="conservative")
        otm = payoff.synthetic_call_return(
            path, self.SPOT, self.SIG_D, self.IV, strike_mult=1.15,
            variant="conservative")
        self.assertGreater(otm, atm)
