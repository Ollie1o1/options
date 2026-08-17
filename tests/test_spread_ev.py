"""A defined-risk structure's EV is the structure's, not its short leg's.

`enrich_credit_spreads` carried `_SHORT_LEG_SCORE_COLS` over from "the
risk-bearing leg". That list holds the 0-1 component RANKS — reasonable to
proxy off the short leg — but it also held the dollar EV LEVELS, and a naked
short option's EV is simply a different instrument's number. Observed live
2026-08-17:

    QQQ 744/745  width 1  max_profit $42.00  ev_per_contract 126.53
    QQQ 744/746  width 2  max_profit $81.50  ev_per_contract 126.53

Identical, because both share short strike 744. `EV/$risk` then printed +1.065
against a `Rwd/$risk` of 0.587 — an expected value nearly double the best
possible outcome, which cannot happen.

The replacement nets the legs rather than inventing a probability model. With
`g(leg) = fair_value - market_price` (the edge to a BUYER, which is exactly
what `ev_gross_per_contract` already holds per leg), a credit vertical is short
the near leg and long the far one, so

    gross(spread) = g(long) - g(short)
    cost(spread)  = cost(long) + cost(short)     # two legs, each crossed twice
    net(spread)   = gross - cost

That is model-consistent with the single-leg path — same Black-Scholes on
realized vol, same round-trip cost model — and needs nothing new.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.spread_scoring import enrich_credit_spreads


def _leg(strike, opt_type, premium, gross, cost, **kw):
    row = {"symbol": "QQQ", "expiration": "2026-09-18", "strike": float(strike),
           "type": opt_type, "premium": premium,
           "ev_gross_per_contract": gross, "ev_cost_per_contract": cost,
           "ev_per_contract": gross - cost, "ev_noise": 10.0,
           "pop_score": 0.5, "liquidity_score": 0.5, "theta_score": 0.5,
           "spread_score": 0.5, "momentum_score": 0.5, "iv_rank_score": 0.5,
           "catalyst_score": 0.5, "vega_dollar": 20.0}
    row.update(kw)
    return row


def _scored(*legs):
    return pd.DataFrame(list(legs))


def _spread(short_k, long_k, net_credit, max_profit, max_loss, typ="Bear Call"):
    return pd.DataFrame([{
        "symbol": "QQQ", "expiration": "2026-09-18", "type": typ,
        "short_strike": float(short_k), "long_strike": float(long_k),
        "net_credit": net_credit, "max_profit": max_profit,
        "max_loss": max_loss, "quality_score": 0.5,
    }])


CFG: dict = {}


class TestTheSpreadPricesBothLegs(unittest.TestCase):

    def test_ev_is_the_netted_legs_not_the_short_leg(self):
        """short gross +90, long gross +30, costs 4 and 3.

        gross = g(long) - g(short) = 30 - 90 = -60
        cost  = 4 + 3 = 7
        net   = -67
        """
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0),
                         _leg(746, "call", 0.40, 30.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertAlmostEqual(out["ev_gross_per_contract"].iloc[0], -60.0)
        self.assertAlmostEqual(out["ev_cost_per_contract"].iloc[0], 7.0)
        self.assertAlmostEqual(out["ev_per_contract"].iloc[0], -67.0)

    def test_two_widths_off_one_short_strike_no_longer_agree(self):
        """The exact reproduction: same short leg, different long legs."""
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0),
                         _leg(745, "call", 0.70, 55.0, 3.5),
                         _leg(746, "call", 0.40, 30.0, 3.0))
        narrow = enrich_credit_spreads(_spread(744, 745, 0.50, 50.0, 50.0),
                                       scored, CFG)["ev_per_contract"].iloc[0]
        wide = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                     scored, CFG)["ev_per_contract"].iloc[0]
        self.assertNotAlmostEqual(narrow, wide,
                                  msg="EV still ignores the long leg")

    def test_the_short_legs_own_ev_is_not_what_survives(self):
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0),
                         _leg(746, "call", 0.40, 30.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        short_leg_ev = 90.0 - 4.0
        self.assertNotAlmostEqual(out["ev_per_contract"].iloc[0], short_leg_ev)

    def test_cost_counts_both_legs(self):
        """A vertical crosses two spreads to open and two to close."""
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0),
                         _leg(746, "call", 0.40, 30.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertGreater(out["ev_cost_per_contract"].iloc[0], 4.0)

    def test_a_put_spread_nets_the_same_way(self):
        scored = _scored(_leg(750, "put", 1.10, 70.0, 4.0),
                         _leg(748, "put", 0.50, 25.0, 3.0))
        out = enrich_credit_spreads(
            _spread(750, 748, 0.60, 60.0, 140.0, typ="Bull Put"), scored, CFG)
        self.assertAlmostEqual(out["ev_gross_per_contract"].iloc[0], -45.0)


class TestAbsentIsNotZero(unittest.TestCase):
    """The same rule the single-leg path applies to a missing HV basis."""

    def _is_absent(self, value):
        return value is None or (isinstance(value, float) and np.isnan(value))

    def test_no_long_leg_means_no_structure_ev(self):
        """Without the long leg the structure cannot be priced at all, and the
        short leg's number is exactly the wrong answer to fall back on."""
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertTrue(self._is_absent(out["ev_per_contract"].iloc[0]))

    def test_a_nan_leg_poisons_the_structure_not_silently_zeroes_it(self):
        scored = _scored(_leg(744, "call", 1.20, float("nan"), 4.0),
                         _leg(746, "call", 0.40, 30.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertTrue(self._is_absent(out["ev_per_contract"].iloc[0]))

    def test_a_missing_short_leg_still_returns_the_row(self):
        """Pre-existing behaviour: the row survives un-enriched."""
        scored = _scored(_leg(746, "call", 0.40, 30.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertEqual(len(out), 1)


class TestTheImpossibilityIsGone(unittest.TestCase):
    """EV must not exceed what the structure can possibly pay."""

    def test_ev_does_not_exceed_max_profit_on_a_fairly_priced_spread(self):
        """Two legs priced with the SAME edge per dollar of premium net to
        almost nothing — which is the honest answer for a fair spread, and
        the opposite of the +1.065-of-risk the old copy produced."""
        scored = _scored(_leg(744, "call", 1.20, 12.0, 4.0),
                         _leg(746, "call", 0.40, 4.0, 3.0))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        ev = out["ev_per_contract"].iloc[0]
        self.assertLessEqual(ev, 80.0, "EV exceeds the structure's max profit")

    def test_the_score_columns_still_come_from_the_short_leg(self):
        """Only the dollar LEVELS changed. The 0-1 ranks are still proxied off
        the risk-bearing leg, which was always the intent."""
        scored = _scored(_leg(744, "call", 1.20, 90.0, 4.0, pop_score=0.77),
                         _leg(746, "call", 0.40, 30.0, 3.0, pop_score=0.11))
        out = enrich_credit_spreads(_spread(744, 746, 0.80, 80.0, 120.0),
                                    scored, CFG)
        self.assertAlmostEqual(out["pop_score"].iloc[0], 0.77)


if __name__ == "__main__":
    unittest.main()
