"""Tests for the evidence-based lottery-ticket selector.

The selector scores far-OTM option candidates on factors the academic
literature actually rewards (vol cheapness, momentum, convexity sweet-spot,
liquidity, catalyst-gated-on-cheap-IV) and applies hard disqualifiers for the
Boyer-Vorkink "lottery trap". Pure functions, offline.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.lottery.test_selector -v
"""
from __future__ import annotations

import copy
import unittest

from src.lottery.selector import (
    DEFAULT_LOTTERY_CONFIG,
    score_candidate,
    select_best,
)


def _base_candidate(**overrides):
    """A neutral, eligible candidate; tests override one feature at a time."""
    c = {
        "ticker": "TEST",
        "direction": "call",
        "iv": 0.40,            # implied vol used to price
        "realized_vol": 0.40,  # trailing realized vol
        "iv_rank": 0.50,       # 0..1 percentile of IV
        "momentum": 0.0,       # signed, aligned with direction
        "strike_sigma": 2.0,   # std-devs OTM over option life
        "spread_pct": 0.10,    # bid-ask as fraction of premium
        "open_interest": 1000,
        "has_catalyst": False,
    }
    c.update(overrides)
    return c


class ScoreCandidateTests(unittest.TestCase):

    def test_cheap_vol_scores_higher_than_rich_vol(self):
        """Goyal-Saretto: realized > implied (cheap option) should score higher."""
        cheap = _base_candidate(realized_vol=0.55, iv=0.40, iv_rank=0.25)
        rich = _base_candidate(realized_vol=0.30, iv=0.55, iv_rank=0.80)
        s_cheap = score_candidate(cheap, DEFAULT_LOTTERY_CONFIG)["score"]
        s_rich = score_candidate(rich, DEFAULT_LOTTERY_CONFIG)["score"]
        self.assertGreater(s_cheap, s_rich)

    def test_trend_aligned_scores_higher_than_countertrend(self):
        """Positive aligned momentum should beat negative."""
        up = _base_candidate(momentum=0.08)
        down = _base_candidate(momentum=-0.08)
        self.assertGreater(
            score_candidate(up, DEFAULT_LOTTERY_CONFIG)["score"],
            score_candidate(down, DEFAULT_LOTTERY_CONFIG)["score"],
        )

    def test_convexity_peaks_in_sweet_spot_not_at_extreme(self):
        """A strike near the sweet spot (~2 sigma) beats a 2.9-sigma moonshot."""
        sweet = _base_candidate(strike_sigma=2.0)
        extreme = _base_candidate(strike_sigma=2.9)
        self.assertGreater(
            score_candidate(sweet, DEFAULT_LOTTERY_CONFIG)["score"],
            score_candidate(extreme, DEFAULT_LOTTERY_CONFIG)["score"],
        )

    def test_catalyst_helps_only_when_iv_is_cheap(self):
        """Catalyst with cheap IV should beat the same catalyst with pumped IV."""
        cheap_cat = _base_candidate(has_catalyst=True, realized_vol=0.55, iv=0.40, iv_rank=0.3)
        rich_cat = _base_candidate(has_catalyst=True, realized_vol=0.30, iv=0.60, iv_rank=0.9)
        # rich_cat is disqualified by iv_rank, so compare the catalyst component directly
        comp_cheap = score_candidate(cheap_cat, DEFAULT_LOTTERY_CONFIG)["components"]["catalyst"]
        rich_cat_ok = _base_candidate(has_catalyst=True, realized_vol=0.30, iv=0.60, iv_rank=0.7)
        comp_rich = score_candidate(rich_cat_ok, DEFAULT_LOTTERY_CONFIG)["components"]["catalyst"]
        self.assertGreater(comp_cheap, comp_rich)

    def test_components_are_reported(self):
        res = score_candidate(_base_candidate(), DEFAULT_LOTTERY_CONFIG)
        for key in ("vol_cheapness", "momentum", "convexity", "liquidity", "catalyst"):
            self.assertIn(key, res["components"])


class DisqualifierTests(unittest.TestCase):

    def test_wide_spread_disqualifies(self):
        c = _base_candidate(spread_pct=0.60)
        self.assertTrue(score_candidate(c, DEFAULT_LOTTERY_CONFIG)["disqualified"])

    def test_too_far_otm_disqualifies(self):
        c = _base_candidate(strike_sigma=3.5)
        self.assertTrue(score_candidate(c, DEFAULT_LOTTERY_CONFIG)["disqualified"])

    def test_iv_rank_too_high_disqualifies(self):
        c = _base_candidate(iv_rank=0.95)
        self.assertTrue(score_candidate(c, DEFAULT_LOTTERY_CONFIG)["disqualified"])

    def test_thin_open_interest_disqualifies(self):
        c = _base_candidate(open_interest=5)
        self.assertTrue(score_candidate(c, DEFAULT_LOTTERY_CONFIG)["disqualified"])


class SelectBestTests(unittest.TestCase):

    def test_picks_highest_scoring_eligible(self):
        good = _base_candidate(ticker="GOOD", realized_vol=0.55, iv=0.40, iv_rank=0.25, momentum=0.08)
        meh = _base_candidate(ticker="MEH", realized_vol=0.30, iv=0.50, iv_rank=0.70, momentum=-0.05)
        pick = select_best([meh, good], DEFAULT_LOTTERY_CONFIG)
        self.assertEqual(pick["ticker"], "GOOD")

    def test_returns_none_when_all_disqualified(self):
        bad = _base_candidate(spread_pct=0.9)
        self.assertIsNone(select_best([bad], DEFAULT_LOTTERY_CONFIG))

    def test_weights_are_adaptable(self):
        """Changing weights must change the ranking — proves the system is tunable."""
        a = _base_candidate(ticker="A", realized_vol=0.60, iv=0.40, iv_rank=0.2, momentum=-0.05)  # cheap, weak mom
        b = _base_candidate(ticker="B", realized_vol=0.40, iv=0.40, iv_rank=0.5, momentum=0.10)    # fair vol, strong mom
        cfg_vol = copy.deepcopy(DEFAULT_LOTTERY_CONFIG)
        cfg_vol["weights"] = {"vol_cheapness": 0.90, "momentum": 0.025, "convexity": 0.025,
                              "liquidity": 0.025, "catalyst": 0.025}
        cfg_mom = copy.deepcopy(DEFAULT_LOTTERY_CONFIG)
        cfg_mom["weights"] = {"vol_cheapness": 0.025, "momentum": 0.90, "convexity": 0.025,
                              "liquidity": 0.025, "catalyst": 0.025}
        self.assertEqual(select_best([a, b], cfg_vol)["ticker"], "A")
        self.assertEqual(select_best([a, b], cfg_mom)["ticker"], "B")


if __name__ == "__main__":
    unittest.main()
