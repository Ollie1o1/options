"""The threshold sweep is a search, and a search needs deflating.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_backtester_dsr -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.alloc.validate import deflated_sharpe, effective_n
from src.backtester import THRESHOLD_SWEEP, THRESHOLD_SWEEP_TRIALS


class ThresholdSweepTrialCountTest(unittest.TestCase):
    def test_the_published_count_is_derived_from_the_sequence(self):
        """The published trial count must match the loop that ran.

        It did not: the loop was `np.arange(0.3, 0.9, 0.05)`, whose bounds read
        as twelve steps but which yields thirteen — the last lands on
        0.8999999999999999 and slips under the exclusive stop. A hardcoded
        count can drift from its loop; a derived one cannot.
        """
        self.assertEqual(THRESHOLD_SWEEP_TRIALS, len(THRESHOLD_SWEEP))

    def test_the_sweep_still_covers_what_arange_covered(self):
        """Replacing arange must not silently drop or add a threshold."""
        self.assertEqual(THRESHOLD_SWEEP_TRIALS,
                         len(np.arange(0.3, 0.9, 0.05)))
        np.testing.assert_allclose(np.array(THRESHOLD_SWEEP),
                                   np.arange(0.3, 0.9, 0.05), atol=1e-12)

    def test_the_thresholds_are_exact_not_float_drifted(self):
        self.assertEqual(THRESHOLD_SWEEP[-1], 0.90)
        self.assertEqual(THRESHOLD_SWEEP[0], 0.30)


class BacktestEffectiveNTest(unittest.TestCase):
    def test_overlapping_daily_entries_collapse(self):
        """One trade per day held 30 days is not 60 observations."""
        days = list(range(60))
        exits = [d + 30 for d in days]
        self.assertEqual(effective_n(days, exits), 2)

    def test_deflating_the_sweep_lowers_the_reported_confidence(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.01, 0.05, 60)
        self.assertGreaterEqual(deflated_sharpe(r, 1, 60),
                                deflated_sharpe(r, THRESHOLD_SWEEP_TRIALS, 60))


if __name__ == "__main__":
    unittest.main()
