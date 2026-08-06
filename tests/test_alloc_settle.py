"""Expiry settles at intrinsic value, not at whatever the quotes say.

The bug this prevents: an illiquid long leg with a zero bid being "sold" for
nothing while the short leg is bought back at a full ask, producing a loss
larger than a defined-risk spread can physically sustain.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_settle -v
"""
from __future__ import annotations

import unittest

from src.alloc.fills import Leg
from src.alloc.settle import implied_spot, intrinsic, settle

EXP = "2024-03-15"


def _row(strike, typ, bid, ask, expiration=EXP):
    return {"expiration": expiration, "strike": float(strike), "type": typ,
            "bid": bid, "ask": ask}


class ImpliedSpotTest(unittest.TestCase):
    def test_parity_recovers_the_underlying(self):
        # spot 100: at K=100 call and put are equal; at K=95 call is 5 richer
        chain = [_row(100, "call", 2.9, 3.1), _row(100, "put", 2.9, 3.1),
                 _row(95, "call", 6.9, 7.1), _row(95, "put", 1.9, 2.1)]
        self.assertAlmostEqual(implied_spot(chain, EXP), 100.0, places=2)

    def test_median_ignores_one_bad_strike(self):
        chain = [_row(100, "call", 2.9, 3.1), _row(100, "put", 2.9, 3.1),
                 _row(95, "call", 6.9, 7.1), _row(95, "put", 1.9, 2.1),
                 _row(90, "call", 50.0, 60.0), _row(90, "put", 0.0, 0.1)]
        self.assertAlmostEqual(implied_spot(chain, EXP), 100.0, places=1)

    def test_other_expiries_are_ignored(self):
        chain = [_row(100, "call", 2.9, 3.1), _row(100, "put", 2.9, 3.1),
                 _row(100, "call", 9.9, 10.1, "2024-06-21"),
                 _row(100, "put", 0.9, 1.1, "2024-06-21")]
        self.assertAlmostEqual(implied_spot(chain, EXP), 100.0, places=2)

    def test_no_dual_quoted_strike_returns_none(self):
        self.assertIsNone(implied_spot([_row(100, "call", 1.0, 1.1)], EXP))

    def test_empty_chain_returns_none(self):
        self.assertIsNone(implied_spot([], EXP))

    def test_crossed_quotes_are_excluded(self):
        chain = [_row(100, "call", 5.0, 1.0), _row(100, "put", 5.0, 1.0)]
        self.assertIsNone(implied_spot(chain, EXP))


class IntrinsicTest(unittest.TestCase):
    def test_otm_put_is_worthless(self):
        self.assertEqual(intrinsic(Leg(EXP, 95.0, "put", "sell"), 100.0), 0.0)

    def test_itm_put_is_strike_minus_spot(self):
        self.assertEqual(intrinsic(Leg(EXP, 105.0, "put", "sell"), 100.0), 5.0)

    def test_otm_call_is_worthless(self):
        self.assertEqual(intrinsic(Leg(EXP, 105.0, "call", "sell"), 100.0), 0.0)

    def test_itm_call_is_spot_minus_strike(self):
        self.assertEqual(intrinsic(Leg(EXP, 95.0, "call", "sell"), 100.0), 5.0)

    def test_intrinsic_is_never_negative(self):
        self.assertGreaterEqual(
            intrinsic(Leg(EXP, 1.0, "put", "sell"), 100.0), 0.0)


class SettleTest(unittest.TestCase):
    def _bull_put(self):
        return [Leg(EXP, 100.0, "put", "sell"), Leg(EXP, 95.0, "put", "buy")]

    def test_fully_otm_spread_settles_at_zero(self):
        """Both legs expire worthless: you keep the whole credit."""
        self.assertEqual(settle(self._bull_put(), spot=110.0), 0.0)

    def test_fully_itm_spread_settles_at_minus_the_width(self):
        """Max loss, and never worse than the width."""
        self.assertEqual(settle(self._bull_put(), spot=80.0), -5.0)

    def test_partially_itm_spread_is_between(self):
        self.assertEqual(settle(self._bull_put(), spot=97.0), -3.0)

    def test_loss_can_never_exceed_the_width(self):
        """The bug that motivated this module: -6.20 on a $5-wide spread."""
        for spot in (0.5, 10.0, 50.0, 94.9, 99.9, 200.0):
            self.assertGreaterEqual(settle(self._bull_put(), spot), -5.0,
                                    f"spread lost more than its width at {spot}")

    def test_bear_call_mirrors_the_bull_put(self):
        legs = [Leg(EXP, 100.0, "call", "sell"), Leg(EXP, 105.0, "call", "buy")]
        self.assertEqual(settle(legs, spot=90.0), 0.0)
        self.assertEqual(settle(legs, spot=120.0), -5.0)

    def test_long_call_settles_positive_when_itm(self):
        self.assertEqual(settle([Leg(EXP, 100.0, "call", "buy")], 110.0), 10.0)

    def test_long_call_settles_at_zero_when_otm(self):
        self.assertEqual(settle([Leg(EXP, 100.0, "call", "buy")], 90.0), 0.0)

    def test_naked_short_put_loss_is_unbounded_below(self):
        """Unlike a spread, this genuinely can lose more than any width."""
        self.assertEqual(settle([Leg(EXP, 100.0, "put", "sell")], 10.0), -90.0)

    def test_iron_condor_settles_on_the_breached_side_only(self):
        legs = [Leg(EXP, 90.0, "put", "sell"), Leg(EXP, 85.0, "put", "buy"),
                Leg(EXP, 110.0, "call", "sell"), Leg(EXP, 115.0, "call", "buy")]
        self.assertEqual(settle(legs, spot=100.0), 0.0)      # inside the wings
        self.assertEqual(settle(legs, spot=80.0), -5.0)      # put side breached
        self.assertEqual(settle(legs, spot=120.0), -5.0)     # call side breached


if __name__ == "__main__":
    unittest.main()
