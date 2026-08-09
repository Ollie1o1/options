"""Three components measured negative, and deleted rather than re-fitted.

The reason this file exists at all: a weight can drift back. The IV-rank result
was withdrawn as a scoring input in `docs/ATTRIBUTION_20260808.md` §4f on
2026-08-08 and was still weighted 0.15 and 0.12 the next morning, because a
document said so and nothing enforced it. A note in a markdown file is not a
guard.

What is pinned here is the DIRECTION and its evidence, not a fitted number.
Each of these is a component whose measured contribution was negative, set to
zero; none of them is a value chosen to make a backtest look better. The
distinction matters because `docs/SCORE_AUDIT_20260807.md` §1 established that
re-fitting weights on this ledger is fitting to noise — the composite does not
rank on it, and its top score quintile is the worst cell in it. Deleting a
measured-negative term is a different act from fitting one.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_weight_deletions_20260809 -v
"""
from __future__ import annotations

import json
import os
import unittest

from src.spread_scoring import (DEFAULT_IRON_WEIGHTS, DEFAULT_SPREAD_WEIGHTS,
                                _weighted_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config() -> dict:
    with open(os.path.join(ROOT, "config.json")) as fh:
        return json.load(fh)


class ShippedWeightsTest(unittest.TestCase):
    """config.json and the in-code fallbacks must agree, or a fresh install
    silently scores differently from this one."""

    def test_iv_rank_carries_no_weight_on_credit_spreads(self):
        # Component rank IC -0.1187 on the ledger's verticals, and the same
        # quantity in the allocation backtest inverts out of sample:
        # q_spread +7.88% in-sample -> -12.50% on the 2020-21 holdout.
        self.assertEqual(_config()["credit_spread_weights"]["iv_rank"], 0.0)
        self.assertEqual(DEFAULT_SPREAD_WEIGHTS["iv_rank"], 0.0)

    def test_iv_rank_carries_no_weight_on_condors(self):
        self.assertEqual(_config()["iron_condor_weights"]["iv_rank"], 0.0)
        self.assertEqual(DEFAULT_IRON_WEIGHTS["iv_rank"], 0.0)

    def test_pop_carries_no_weight_on_condors(self):
        # Rank IC -0.3115 (p 0.001) while carrying the LARGEST weight, with
        # the mechanism identified: spearman(pop_score, net_credit) = -0.7197.
        self.assertEqual(_config()["iron_condor_weights"]["pop"], 0.0)
        self.assertEqual(DEFAULT_IRON_WEIGHTS["pop"], 0.0)

    def test_pop_is_KEPT_on_verticals(self):
        """The evidence is structure-specific and the change must be too.

        The same component measures -0.0594 at p = 0.354 on the verticals.
        Deleting on that would be exactly the small-sample tuning this repo
        keeps getting burned by, so it stays.
        """
        self.assertEqual(_config()["credit_spread_weights"]["pop"], 0.25)
        self.assertEqual(DEFAULT_SPREAD_WEIGHTS["pop"], 0.25)

    def test_the_components_that_measured_POSITIVE_are_untouched(self):
        for table, key, want in (("credit_spread_weights", "credit_to_width", 0.2),
                                 ("credit_spread_weights", "return_on_risk", 0.1),
                                 ("credit_spread_weights", "theta", 0.08),
                                 ("iron_condor_weights", "theta", 0.08),
                                 ("iron_condor_weights", "delta_neutral", 0.15)):
            with self.subTest(table=table, key=key):
                self.assertEqual(_config()[table][key], want)


class ZeroWeightIsADeletionTest(unittest.TestCase):
    """A zeroed component must be REMOVED from the composite, not merely
    multiplied by zero while still sitting in the denominator.

    `_weighted_score` renormalises over the weights it summed, so a 0.0 weight
    adds nothing to either side and the survivors share the full scale. If it
    instead divided by a fixed total, zeroing three components would cap every
    condor's score at 0.58 and compress the whole board toward the bottom —
    which would look like the change had made every candidate worse.
    """

    def _row(self, **scores):
        return {f"{k}_score": v for k, v in scores.items()}

    def test_a_zero_weight_component_cannot_move_the_score(self):
        weights = {"a": 0.5, "b": 0.5, "dead": 0.0}
        cols = {"a": "a_score", "b": "b_score", "dead": "dead_score"}
        lo = _weighted_score(self._row(a=0.8, b=0.8, dead=0.0), cols, weights)
        hi = _weighted_score(self._row(a=0.8, b=0.8, dead=1.0), cols, weights)
        self.assertEqual(lo, hi)

    def test_the_survivors_still_reach_a_full_score(self):
        weights = {"a": 0.5, "b": 0.5, "dead": 0.0}
        cols = {"a": "a_score", "b": "b_score", "dead": "dead_score"}
        s = _weighted_score(self._row(a=1.0, b=1.0, dead=0.0), cols, weights)
        self.assertAlmostEqual(s, 1.0, places=9)

    def test_the_shipped_condor_table_still_spans_the_full_range(self):
        cols = {"pop": "pop_score", "credit_to_width": "credit_to_width_score",
                "delta_neutral": "delta_neutral_score",
                "iv_rank": "iv_rank_score", "liquidity": "liquidity_score",
                "theta": "theta_score", "spread": "spread_score"}
        best = {c: 1.0 for c in cols.values()}
        worst = {c: 0.0 for c in cols.values()}
        self.assertAlmostEqual(
            _weighted_score(best, cols, DEFAULT_IRON_WEIGHTS), 1.0, places=9)
        self.assertAlmostEqual(
            _weighted_score(worst, cols, DEFAULT_IRON_WEIGHTS), 0.0, places=9)


class TheOrderingActuallyChangesTest(unittest.TestCase):
    """A weight change nobody can observe is not a change.

    The point of dropping condor `pop` is that the high-PoP / tiny-credit
    structures stop being ranked first. This builds exactly that pair and
    asserts the ordering inverts.
    """

    def test_the_tiny_credit_high_pop_condor_no_longer_outranks(self):
        cols = {"pop": "pop_score", "credit_to_width": "credit_to_width_score",
                "delta_neutral": "delta_neutral_score",
                "iv_rank": "iv_rank_score", "liquidity": "liquidity_score",
                "theta": "theta_score", "spread": "spread_score"}
        # Both carry credit_to_width_score = 1.0, and that is not a
        # simplification — it is what the real data does. CONDOR_COMPOSITE §3
        # reports median c/w of 0.374 / 0.365 / 0.360 / 0.337 across the four
        # pop quartiles, and the condor normalisation is
        # clip((c2w - 0.10) / 0.20), which SATURATES at c/w >= 0.30. Every
        # observed quartile is above that, so the component is pinned at 1.0
        # across the whole cohort and discriminates nothing despite its 0.20
        # weight. That is why pop was free to dominate: the term that should
        # have priced the credit was constant.
        #
        # Q4 of §3: highest pop, $1.25 credit against a $4 width, -37.8%
        # median return on a 26.7% win rate. High pop is also high friction
        # (spearman(pop, round_trip) = +0.5991, §7), hence the weak theta and
        # spread scores.
        thin = {"pop_score": 0.95, "credit_to_width_score": 1.0,
                "delta_neutral_score": 0.5, "iv_rank_score": 0.5,
                "liquidity_score": 0.5, "theta_score": 0.2,
                "spread_score": 0.2}
        # Q1: lowest pop, $11.34 credit against a $31 width, +15.5% median on
        # 74.2% wins.
        fat = {"pop_score": 0.30, "credit_to_width_score": 1.0,
               "delta_neutral_score": 0.5, "iv_rank_score": 0.5,
               "liquidity_score": 0.5, "theta_score": 0.8,
               "spread_score": 0.8}

        old = {"pop": 0.30, "credit_to_width": 0.20, "delta_neutral": 0.15,
               "iv_rank": 0.12, "liquidity": 0.10, "theta": 0.08,
               "spread": 0.05}
        self.assertGreater(_weighted_score(thin, cols, old),
                           _weighted_score(fat, cols, old),
                           "the OLD weights are supposed to rank it backwards")
        self.assertLess(_weighted_score(thin, cols, DEFAULT_IRON_WEIGHTS),
                        _weighted_score(fat, cols, DEFAULT_IRON_WEIGHTS))


if __name__ == "__main__":
    unittest.main()
