"""Tests for src/lab/pricing.py — what an option was worth, and how we know.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_lab_pricing -v
"""
import unittest

from src.lab import pricing as p


class SourceTest(unittest.TestCase):
    """Every price carries how it was obtained. Nothing pools the tiers."""

    def test_a_real_quote_is_marked_real(self):
        q = p.Quote(bid=2.00, ask=2.20, source="real_marks", iv=0.31)
        self.assertTrue(q.is_real)

    def test_a_modelled_price_is_marked_modelled(self):
        q = p.Quote(bid=1.98, ask=2.02, source="modeled", iv=0.30)
        self.assertFalse(q.is_real)

    def test_an_unknown_source_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            p.Quote(bid=1.0, ask=1.1, source="vibes", iv=0.3)


class BlackScholesQuoteTest(unittest.TestCase):
    """Tier 3: repricing from spot + a modelled vol."""

    def test_an_at_the_money_call_is_worth_roughly_point_four_sigma_root_t(self):
        # Standard approximation ATM call ~= 0.4 * S * sigma * sqrt(T) holds
        # only at a zero rate; at the default 4.5% the forward is higher and
        # the call is worth about $2 more.
        q = p.bs_quote("call", spot=100.0, strike=100.0, dte=365, iv=0.30, rate=0.0)
        self.assertAlmostEqual(q.mid, 0.4 * 100.0 * 0.30, delta=0.5)

    def test_carrying_a_rate_makes_a_call_worth_more_than_the_zero_rate_case(self):
        zero = p.bs_quote("call", 100.0, 100.0, dte=365, iv=0.30, rate=0.0)
        carry = p.bs_quote("call", 100.0, 100.0, dte=365, iv=0.30, rate=0.045)
        self.assertGreater(carry.mid, zero.mid)

    def test_it_is_marked_modelled_not_real(self):
        q = p.bs_quote("call", spot=100.0, strike=100.0, dte=90, iv=0.30)
        self.assertEqual(q.source, "modeled")

    def test_a_longer_dated_option_costs_more_than_a_shorter_one(self):
        short = p.bs_quote("call", spot=100.0, strike=100.0, dte=30, iv=0.30)
        long = p.bs_quote("call", spot=100.0, strike=100.0, dte=365, iv=0.30)
        self.assertGreater(long.mid, short.mid)

    def test_a_put_and_call_at_the_same_strike_obey_put_call_parity(self):
        c = p.bs_quote("call", spot=100.0, strike=100.0, dte=365, iv=0.30, rate=0.045)
        pu = p.bs_quote("put", spot=100.0, strike=100.0, dte=365, iv=0.30, rate=0.045)
        # C - P = S - K*exp(-rT)
        import math
        self.assertAlmostEqual(c.mid - pu.mid, 100.0 - 100.0 * math.exp(-0.045), places=4)

    def test_an_expired_option_is_worth_its_intrinsic(self):
        self.assertAlmostEqual(p.bs_quote("call", 110.0, 100.0, dte=0, iv=0.30).mid, 10.0)
        self.assertAlmostEqual(p.bs_quote("call", 90.0, 100.0, dte=0, iv=0.30).mid, 0.0)
        self.assertAlmostEqual(p.bs_quote("put", 90.0, 100.0, dte=0, iv=0.30).mid, 10.0)

    def test_a_modelled_quote_carries_a_spread_so_costs_are_never_free(self):
        """The mid-fill defect, prevented by construction: a synthetic quote
        must still have two sides, or a backtest on it pays no friction."""
        q = p.bs_quote("call", spot=100.0, strike=100.0, dte=90, iv=0.30)
        self.assertGreater(q.ask, q.bid)


class ModelledSpreadTest(unittest.TestCase):
    """How wide the synthetic market is. Calibrated from real archived quotes:
    a 0.30-0.60 delta single leg shows a ~1.7% half-spread at 5-30 DTE,
    ~1.1% at 31-60 and ~0.7% at 61-120 — narrower in relative terms as the
    premium grows."""

    def test_the_relative_half_spread_narrows_with_maturity(self):
        near = p.modeled_half_spread_frac(dte=20)
        mid = p.modeled_half_spread_frac(dte=45)
        far = p.modeled_half_spread_frac(dte=90)
        self.assertGreater(near, mid)
        self.assertGreater(mid, far)

    def test_it_matches_the_measured_archive_values(self):
        self.assertAlmostEqual(p.modeled_half_spread_frac(dte=20), 0.017, places=3)
        self.assertAlmostEqual(p.modeled_half_spread_frac(dte=45), 0.011, places=3)
        self.assertAlmostEqual(p.modeled_half_spread_frac(dte=90), 0.007, places=3)

    def test_beyond_the_measured_range_it_does_not_extrapolate_downward(self):
        """No data past 120 DTE. Holding the last measured value is a guess;
        assuming the trend continues is a guess that flatters the trade."""
        self.assertEqual(p.modeled_half_spread_frac(dte=400),
                         p.modeled_half_spread_frac(dte=120))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
