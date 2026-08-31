"""Cross-validation that does not leak, and statistics that account for search.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_validate -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.alloc.validate import (cpcv_splits, deflated_sharpe, effective_n,
                                expected_max_sharpe, pbo_from_pairs,
                                purge_embargo, sharpe)


class CpcvSplitTest(unittest.TestCase):
    def test_number_of_paths_is_n_choose_k(self):
        self.assertEqual(len(cpcv_splits(80, n_blocks=8, k=2)), math.comb(8, 2))

    def test_train_and_test_never_overlap(self):
        for train, test in cpcv_splits(80, n_blocks=8, k=2):
            self.assertEqual(set(train) & set(test), set())

    def test_every_index_is_used(self):
        seen = set()
        for train, test in cpcv_splits(80, n_blocks=8, k=2):
            seen |= set(train) | set(test)
        self.assertEqual(seen, set(range(80)))

    def test_default_arguments_work(self):
        self.assertTrue(cpcv_splits(80))

    def test_degenerate_inputs_return_nothing(self):
        self.assertEqual(cpcv_splits(0), [])
        self.assertEqual(cpcv_splits(80, n_blocks=2, k=2), [])


class PurgeEmbargoTest(unittest.TestCase):
    def test_samples_overlapping_the_test_label_are_purged(self):
        kept = purge_embargo(list(range(50)), list(range(50, 60)),
                             holding_days=5, embargo_days=0)
        self.assertNotIn(49, kept)
        self.assertNotIn(45, kept)
        self.assertIn(44, kept)

    def test_embargo_removes_samples_after_the_test_block(self):
        kept = purge_embargo(list(range(100)), list(range(50, 60)),
                             holding_days=0, embargo_days=5)
        for i in range(60, 65):
            self.assertNotIn(i, kept)
        self.assertIn(66, kept)

    def test_longer_holding_purges_more(self):
        few = purge_embargo(list(range(50)), list(range(50, 60)), 2, 0)
        many = purge_embargo(list(range(50)), list(range(50, 60)), 20, 0)
        self.assertGreater(len(few), len(many))

    def test_default_embargo_applies(self):
        kept = purge_embargo(list(range(100)), list(range(50, 60)),
                             holding_days=0)
        self.assertNotIn(61, kept)

    def test_empty_test_set_keeps_everything(self):
        self.assertEqual(purge_embargo([1, 2, 3], [], 5), [1, 2, 3])


class ExpectedMaxSharpeTest(unittest.TestCase):
    """The bar a result must clear rises with the size of the search."""

    def test_one_trial_sets_no_bar(self):
        self.assertEqual(expected_max_sharpe(1), 0.0)

    def test_more_trials_raise_the_bar(self):
        self.assertLess(expected_max_sharpe(10), expected_max_sharpe(1000))

    def test_the_bar_is_positive_for_a_real_search(self):
        self.assertGreater(expected_max_sharpe(50), 0.0)


class DeflatedSharpeTest(unittest.TestCase):
    def _returns(self, mean, n=250, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(mean, 0.01, n)

    def test_more_trials_lowers_the_deflated_sharpe(self):
        r = self._returns(0.001)
        self.assertGreater(deflated_sharpe(r, 1, len(r)),
                           deflated_sharpe(r, 1000, len(r)))

    def test_a_strong_strategy_survives_a_small_search(self):
        r = self._returns(0.004)
        self.assertGreater(deflated_sharpe(r, 5, len(r)), 0.95)

    def test_a_marginal_strategy_dies_under_a_large_search(self):
        r = self._returns(0.0002)
        self.assertLess(deflated_sharpe(r, 5000, len(r)), 0.5)

    def test_flat_returns_do_not_raise(self):
        self.assertIsInstance(deflated_sharpe(np.zeros(100), 10, 100), float)

    def test_too_few_returns_is_zero_not_a_crash(self):
        self.assertEqual(deflated_sharpe([0.01, 0.02], 10, 2), 0.0)

    def test_negative_skew_is_penalised(self):
        """Short premium's shape must not be flattered."""
        rng = np.random.default_rng(3)
        base = rng.normal(0.002, 0.01, 300)
        skewed = base.copy()
        skewed[:5] = -0.15                      # rare large losses
        self.assertLess(deflated_sharpe(skewed, 10, len(skewed)),
                        deflated_sharpe(base, 10, len(base)))

    def test_default_trial_variance_branch(self):
        r = self._returns(0.002)
        self.assertIsInstance(deflated_sharpe(r, 10, len(r)), float)


class DeflatedSharpeSampleSizeTest(unittest.TestCase):
    def _returns(self, mean, n=200, seed=1):
        rng = np.random.default_rng(seed)
        return rng.normal(mean, 0.05, n)

    def test_n_eff_is_required(self):
        """A default here is how the wrong number ships unnoticed."""
        with self.assertRaises(TypeError):
            deflated_sharpe(self._returns(0.01), 10)   # type: ignore[call-arg]

    def test_fewer_effective_observations_lower_the_dsr(self):
        r = self._returns(0.012)
        self.assertGreater(deflated_sharpe(r, 200, 200),
                           deflated_sharpe(r, 200, 46))

    def test_more_trials_still_lower_the_dsr(self):
        r = self._returns(0.012)
        self.assertGreater(deflated_sharpe(r, 1, 200),
                           deflated_sharpe(r, 1000, 200))

    def test_row_count_can_promote_where_clusters_reject(self):
        """The measured defect: same returns, two verdicts, gate bar 0.5."""
        rng = np.random.default_rng(3)
        r = rng.normal(0.0135, 0.05, 253)
        self.assertGreaterEqual(deflated_sharpe(r, 200, 253), 0.5)
        self.assertLess(deflated_sharpe(r, 200, 58), 0.5)

    def test_too_few_effective_observations_returns_zero(self):
        self.assertEqual(deflated_sharpe(self._returns(0.01), 10, 2), 0.0)


class EffectiveNTest(unittest.TestCase):
    """How many INDEPENDENT observations a set of overlapping trades carries.

    Row count is not it. Trades whose holding periods overlap are scored on the
    same price path, and counting them as independent is what inflates a
    deflated Sharpe into a promotion.
    """

    def test_disjoint_intervals_each_count(self):
        n = effective_n(["2024-01-01", "2024-02-01", "2024-03-01"],
                        ["2024-01-05", "2024-02-05", "2024-03-05"])
        self.assertEqual(n, 3)

    def test_identical_intervals_count_once(self):
        n = effective_n(["2024-01-01"] * 10, ["2024-01-05"] * 10)
        self.assertEqual(n, 1)

    def test_same_entry_day_collapses(self):
        """Ten trades opened the same morning share that day's move."""
        n = effective_n(["2024-01-01"] * 10, ["2024-01-03"] * 10)
        self.assertEqual(n, 1)

    def test_overlapping_holds_collapse_even_on_distinct_entry_days(self):
        """The case entry-day clustering misses entirely."""
        starts = ["2024-01-01", "2024-01-02", "2024-01-03"]
        ends = ["2024-01-30", "2024-01-31", "2024-02-01"]
        self.assertEqual(effective_n(starts, ends), 1)

    def test_touching_intervals_are_treated_as_overlapping(self):
        """A trade closing the day another opens shared that day's path."""
        self.assertEqual(
            effective_n(["2024-01-01", "2024-01-05"],
                        ["2024-01-05", "2024-01-09"]), 1)

    def test_integer_days_work_as_well_as_dates(self):
        self.assertEqual(effective_n([0, 40], [30, 70]), 2)

    def test_empty_is_zero(self):
        self.assertEqual(effective_n([], []), 0)

    def test_mismatched_lengths_use_the_shorter(self):
        self.assertEqual(effective_n([0, 40, 80], [30, 70]), 2)

    def test_never_exceeds_distinct_starts_or_row_count(self):
        """The invariant the whole change rests on."""
        starts = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-03-01"]
        ends = ["2024-01-10", "2024-01-10", "2024-01-11", "2024-03-10"]
        n = effective_n(starts, ends)
        self.assertLessEqual(n, len(set(starts)))
        self.assertLessEqual(len(set(starts)), len(starts))
        self.assertEqual(n, 2)


class SharpeTest(unittest.TestCase):
    def test_positive_mean_gives_positive_sharpe(self):
        self.assertGreater(sharpe([0.01, 0.02, 0.015, 0.011]), 0)

    def test_constant_series_is_zero_not_infinite(self):
        self.assertEqual(sharpe([0.01] * 10), 0.0)

    def test_single_observation_is_zero(self):
        self.assertEqual(sharpe([0.01]), 0.0)


class PboTest(unittest.TestCase):
    def test_consistent_winner_gives_low_pbo(self):
        pairs = [([3.0, 1.0, 2.0], [3.0, 1.0, 2.0]) for _ in range(20)]
        self.assertLess(pbo_from_pairs(pairs), 0.1)

    def test_inverted_ranking_gives_high_pbo(self):
        pairs = [([3.0, 1.0, 2.0], [1.0, 3.0, 2.0]) for _ in range(20)]
        self.assertGreater(pbo_from_pairs(pairs), 0.9)

    def test_no_paths_is_zero(self):
        self.assertEqual(pbo_from_pairs([]), 0.0)

    def test_mismatched_lengths_are_skipped(self):
        self.assertEqual(pbo_from_pairs([([1.0, 2.0], [1.0])]), 0.0)


if __name__ == "__main__":
    unittest.main()
