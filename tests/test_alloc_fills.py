"""Fills must cross the spread, never split it.

Mid-priced entries cost 27% of credit when actually crossed and INVERTED the
strategy ranking (docs/EXECUTION_TRUTH.md). A backtest that fills at mid will
reproduce that inversion and confidently recommend the wrong structure.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_fills -v
"""
from __future__ import annotations

import unittest

from src.alloc.fills import (Leg, SKIP_CROSSED, SKIP_MISSING, fill_price,
                             fill_with_reason, reverse)


def q(bid, ask):
    return (bid, ask)


class SingleLegFillTest(unittest.TestCase):
    def test_selling_fills_at_the_bid(self):
        self.assertAlmostEqual(
            fill_price([Leg(100.0, "put", "sell")],
                       {(100.0, "put"): q(1.00, 1.20)}), 1.00)

    def test_buying_fills_at_the_ask(self):
        self.assertAlmostEqual(
            fill_price([Leg(100.0, "put", "buy")],
                       {(100.0, "put"): q(1.00, 1.20)}), -1.20)

    def test_never_returns_the_mid(self):
        self.assertNotAlmostEqual(
            fill_price([Leg(100.0, "put", "sell")],
                       {(100.0, "put"): q(1.00, 1.20)}), 1.10)


class SpreadFillTest(unittest.TestCase):
    def test_bull_put_credit_is_short_bid_minus_long_ask(self):
        legs = [Leg(100.0, "put", "sell"), Leg(95.0, "put", "buy")]
        quotes = {(100.0, "put"): q(2.00, 2.30), (95.0, "put"): q(0.80, 1.00)}
        self.assertAlmostEqual(fill_price(legs, quotes), 1.00)

    def test_mid_pricing_would_have_flattered_it(self):
        legs = [Leg(100.0, "put", "sell"), Leg(95.0, "put", "buy")]
        quotes = {(100.0, "put"): q(2.00, 2.30), (95.0, "put"): q(0.80, 1.00)}
        self.assertLess(fill_price(legs, quotes), 2.15 - 0.90)

    def test_iron_condor_sums_all_four_legs(self):
        legs = [Leg(100.0, "put", "sell"), Leg(95.0, "put", "buy"),
                Leg(120.0, "call", "sell"), Leg(125.0, "call", "buy")]
        quotes = {(100.0, "put"): q(2.00, 2.30), (95.0, "put"): q(0.80, 1.00),
                  (120.0, "call"): q(1.50, 1.70), (125.0, "call"): q(0.50, 0.65)}
        self.assertAlmostEqual(fill_price(legs, quotes), 1.85)

    def test_debit_spread_is_negative(self):
        legs = [Leg(100.0, "call", "buy"), Leg(105.0, "call", "sell")]
        quotes = {(100.0, "call"): q(3.00, 3.30), (105.0, "call"): q(1.00, 1.20)}
        self.assertAlmostEqual(fill_price(legs, quotes), -2.30)

    def test_long_call_is_a_debit(self):
        self.assertAlmostEqual(
            fill_price([Leg(100.0, "call", "buy")],
                       {(100.0, "call"): q(3.00, 3.30)}), -3.30)


class SkipTest(unittest.TestCase):
    def test_missing_quote_returns_none(self):
        legs = [Leg(100.0, "put", "sell"), Leg(95.0, "put", "buy")]
        self.assertIsNone(fill_price(legs, {(100.0, "put"): q(2.00, 2.30)}))

    def test_missing_quote_reports_reason(self):
        self.assertEqual(fill_with_reason([Leg(100.0, "put", "sell")], {}),
                         (None, SKIP_MISSING))

    def test_crossed_quote_is_skipped(self):
        """bid > ask is a broken quote and must never be filled."""
        self.assertEqual(
            fill_with_reason([Leg(100.0, "put", "sell")],
                             {(100.0, "put"): q(2.50, 2.00)}),
            (None, SKIP_CROSSED))

    def test_none_bid_is_missing_not_free(self):
        self.assertEqual(
            fill_with_reason([Leg(100.0, "put", "sell")],
                             {(100.0, "put"): (None, 1.0)}),
            (None, SKIP_MISSING))

    def test_zero_bid_is_missing_not_free(self):
        self.assertEqual(
            fill_with_reason([Leg(100.0, "put", "sell")],
                             {(100.0, "put"): (0.0, 1.0)}),
            (None, SKIP_MISSING))

    def test_good_fill_reports_no_reason(self):
        price, reason = fill_with_reason([Leg(100.0, "put", "sell")],
                                         {(100.0, "put"): q(1.0, 1.2)})
        self.assertAlmostEqual(price, 1.0)
        self.assertIsNone(reason)

    def test_strike_lookup_tolerates_float_noise(self):
        """95.00000001 from arithmetic must still find the 95.0 quote."""
        legs = [Leg(95.0 + 1e-9, "put", "buy")]
        self.assertIsNotNone(fill_price(legs, {(95.0, "put"): q(0.8, 1.0)}))


class ReverseTest(unittest.TestCase):
    """Closing a position crosses the spread a SECOND time."""

    def test_reverse_flips_every_action(self):
        legs = [Leg(100.0, "put", "sell"), Leg(95.0, "put", "buy")]
        self.assertEqual([l.action for l in reverse(legs)], ["buy", "sell"])

    def test_round_trip_costs_the_full_spread_twice(self):
        legs = [Leg(100.0, "put", "sell")]
        quotes = {(100.0, "put"): q(1.00, 1.20)}
        opened = fill_price(legs, quotes)              # +1.00 received
        closed = fill_price(reverse(legs), quotes)     # -1.20 paid
        self.assertAlmostEqual(opened + closed, -0.20)  # the spread, twice

    def test_reverse_does_not_mutate_the_original(self):
        legs = [Leg(100.0, "put", "sell")]
        reverse(legs)
        self.assertEqual(legs[0].action, "sell")


if __name__ == "__main__":
    unittest.main()
