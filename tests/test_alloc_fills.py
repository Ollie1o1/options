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
            fill_price([Leg("2024-03-15", 100.0, "put", "sell")],
                       {("2024-03-15", 100.0, "put"): q(1.00, 1.20)}), 1.00)

    def test_buying_fills_at_the_ask(self):
        self.assertAlmostEqual(
            fill_price([Leg("2024-03-15", 100.0, "put", "buy")],
                       {("2024-03-15", 100.0, "put"): q(1.00, 1.20)}), -1.20)

    def test_never_returns_the_mid(self):
        self.assertNotAlmostEqual(
            fill_price([Leg("2024-03-15", 100.0, "put", "sell")],
                       {("2024-03-15", 100.0, "put"): q(1.00, 1.20)}), 1.10)


class SpreadFillTest(unittest.TestCase):
    def test_bull_put_credit_is_short_bid_minus_long_ask(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy")]
        quotes = {("2024-03-15", 100.0, "put"): q(2.00, 2.30), ("2024-03-15", 95.0, "put"): q(0.80, 1.00)}
        self.assertAlmostEqual(fill_price(legs, quotes), 1.00)

    def test_mid_pricing_would_have_flattered_it(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy")]
        quotes = {("2024-03-15", 100.0, "put"): q(2.00, 2.30), ("2024-03-15", 95.0, "put"): q(0.80, 1.00)}
        self.assertLess(fill_price(legs, quotes), 2.15 - 0.90)

    def test_iron_condor_sums_all_four_legs(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy"),
                Leg("2024-03-15", 120.0, "call", "sell"), Leg("2024-03-15", 125.0, "call", "buy")]
        quotes = {("2024-03-15", 100.0, "put"): q(2.00, 2.30), ("2024-03-15", 95.0, "put"): q(0.80, 1.00),
                  ("2024-03-15", 120.0, "call"): q(1.50, 1.70), ("2024-03-15", 125.0, "call"): q(0.50, 0.65)}
        self.assertAlmostEqual(fill_price(legs, quotes), 1.85)

    def test_debit_spread_is_negative(self):
        legs = [Leg("2024-03-15", 100.0, "call", "buy"), Leg("2024-03-15", 105.0, "call", "sell")]
        quotes = {("2024-03-15", 100.0, "call"): q(3.00, 3.30), ("2024-03-15", 105.0, "call"): q(1.00, 1.20)}
        self.assertAlmostEqual(fill_price(legs, quotes), -2.30)

    def test_long_call_is_a_debit(self):
        self.assertAlmostEqual(
            fill_price([Leg("2024-03-15", 100.0, "call", "buy")],
                       {("2024-03-15", 100.0, "call"): q(3.00, 3.30)}), -3.30)


class SkipTest(unittest.TestCase):
    def test_missing_quote_returns_none(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertIsNone(fill_price(legs, {("2024-03-15", 100.0, "put"): q(2.00, 2.30)}))

    def test_missing_quote_reports_reason(self):
        self.assertEqual(fill_with_reason([Leg("2024-03-15", 100.0, "put", "sell")], {}),
                         (None, SKIP_MISSING))

    def test_crossed_quote_is_skipped(self):
        """bid > ask is a broken quote and must never be filled."""
        self.assertEqual(
            fill_with_reason([Leg("2024-03-15", 100.0, "put", "sell")],
                             {("2024-03-15", 100.0, "put"): q(2.50, 2.00)}),
            (None, SKIP_CROSSED))

    def test_none_bid_is_missing_not_free(self):
        self.assertEqual(
            fill_with_reason([Leg("2024-03-15", 100.0, "put", "sell")],
                             {("2024-03-15", 100.0, "put"): (None, 1.0)}),
            (None, SKIP_MISSING))

    def test_zero_bid_is_missing_not_free(self):
        self.assertEqual(
            fill_with_reason([Leg("2024-03-15", 100.0, "put", "sell")],
                             {("2024-03-15", 100.0, "put"): (0.0, 1.0)}),
            (None, SKIP_MISSING))

    def test_good_fill_reports_no_reason(self):
        price, reason = fill_with_reason([Leg("2024-03-15", 100.0, "put", "sell")],
                                         {("2024-03-15", 100.0, "put"): q(1.0, 1.2)})
        self.assertAlmostEqual(price, 1.0)
        self.assertIsNone(reason)

    def test_strike_lookup_tolerates_float_noise(self):
        """95.00000001 from arithmetic must still find the 95.0 quote."""
        legs = [Leg("2024-03-15", 95.0 + 1e-9, "put", "buy")]
        self.assertIsNotNone(fill_price(legs, {("2024-03-15", 95.0, "put"): q(0.8, 1.0)}))


class ExpirationIsPartOfIdentityTest(unittest.TestCase):
    """A chain holds many expirations at once.

    Keying quotes on (strike, type) alone let a March 100-put collide with a
    June 100-put, so a leg got priced off an arbitrary expiry. That produced a
    17% win rate on a 25-delta put spread, which should win roughly 75%.
    """

    def test_same_strike_different_expiry_do_not_collide(self):
        from src.alloc.fills import quotes_from_chain
        chain = [
            {"expiration": "2024-03-15", "strike": 100.0, "type": "put",
             "bid": 1.00, "ask": 1.10},
            {"expiration": "2024-06-21", "strike": 100.0, "type": "put",
             "bid": 4.00, "ask": 4.20},
        ]
        quotes = quotes_from_chain(chain)
        self.assertEqual(len(quotes), 2)
        near = fill_price([Leg("2024-03-15", 100.0, "put", "sell")], quotes)
        far = fill_price([Leg("2024-06-21", 100.0, "put", "sell")], quotes)
        self.assertAlmostEqual(near, 1.00)
        self.assertAlmostEqual(far, 4.00)

    def test_a_leg_from_an_absent_expiry_does_not_fill(self):
        from src.alloc.fills import quotes_from_chain
        quotes = quotes_from_chain([
            {"expiration": "2024-03-15", "strike": 100.0, "type": "put",
             "bid": 1.00, "ask": 1.10}])
        self.assertIsNone(
            fill_price([Leg("2024-06-21", 100.0, "put", "sell")], quotes))


class ReverseTest(unittest.TestCase):
    """Closing a position crosses the spread a SECOND time."""

    def test_reverse_flips_every_action(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertEqual([l.action for l in reverse(legs)], ["buy", "sell"])

    def test_round_trip_costs_the_full_spread_twice(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell")]
        quotes = {("2024-03-15", 100.0, "put"): q(1.00, 1.20)}
        opened = fill_price(legs, quotes)              # +1.00 received
        closed = fill_price(reverse(legs), quotes)     # -1.20 paid
        self.assertAlmostEqual(opened + closed, -0.20)  # the spread, twice

    def test_reverse_does_not_mutate_the_original(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell")]
        reverse(legs)
        self.assertEqual(legs[0].action, "sell")


if __name__ == "__main__":
    unittest.main()


class TransactedSideTest(unittest.TestCase):
    """Only the side you actually transact on has to be a real price.

    Requiring both a bid and an ask on every leg rejected every far-OTM
    protective wing — those legitimately quote bid=0 and are BOUGHT at the ask —
    which silently excluded the mega-caps, the tightest-spread names available.
    """

    def test_buying_a_wing_with_no_bid_is_allowed(self):
        legs = [Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertAlmostEqual(
            fill_price(legs, {("2024-03-15", 95.0, "put"): (0.0, 0.10)}), -0.10)

    def test_selling_with_no_bid_is_still_refused(self):
        legs = [Leg("2024-03-15", 95.0, "put", "sell")]
        self.assertEqual(
            fill_with_reason(legs, {("2024-03-15", 95.0, "put"): (0.0, 0.10)}),
            (None, SKIP_MISSING))

    def test_buying_with_no_ask_is_refused(self):
        """No ask means nothing to buy. (bid must be 0 too, or it is crossed.)"""
        legs = [Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertEqual(
            fill_with_reason(legs, {("2024-03-15", 95.0, "put"): (0.0, 0.0)}),
            (None, SKIP_MISSING))

    def test_bid_above_ask_is_crossed_not_missing(self):
        """A crossed quote is diagnosed as crossed, whichever side we transact."""
        legs = [Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertEqual(
            fill_with_reason(legs, {("2024-03-15", 95.0, "put"): (0.05, 0.0)}),
            (None, SKIP_CROSSED))

    def test_spread_with_a_zero_bid_wing_fills(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"),
                Leg("2024-03-15", 95.0, "put", "buy")]
        quotes = {("2024-03-15", 100.0, "put"): (2.00, 2.30),
                  ("2024-03-15", 95.0, "put"): (0.0, 0.10)}
        self.assertAlmostEqual(fill_price(legs, quotes), 1.90)
