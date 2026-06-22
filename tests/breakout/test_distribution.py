"""Tests for the forward-return Distribution + baseline climatology."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.distribution import Distribution, baseline_distribution


class DistributionTests(unittest.TestCase):
    def test_cdf_is_monotone(self):
        d = Distribution(np.linspace(-0.5, 0.5, 1001))
        self.assertLessEqual(d.prob_ge(0.2), d.prob_ge(0.0))
        self.assertGreaterEqual(d.prob_le(0.2), d.prob_le(0.0))

    def test_prob_ge_and_le_complement(self):
        d = Distribution(np.linspace(-1, 1, 1001))
        self.assertAlmostEqual(d.prob_ge(0.0) + d.prob_le(0.0), 1.0, places=2)

    def test_band_is_ordered_and_brackets_median(self):
        d = Distribution(np.linspace(-1, 1, 1001))
        lo, hi = d.band(0.1, 0.9)
        self.assertLess(lo, d.point())
        self.assertLess(d.point(), hi)

    def test_baseline_prob_matches_empirical_frequency(self):
        rng = np.random.default_rng(1)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
        d = baseline_distribution(close, t=len(close) - 1, horizon=21, n=6000, seed=2)
        # symmetric-ish driftless series: P(>=0) near 0.5
        self.assertTrue(0.3 < d.prob_ge(0.0) < 0.7)


if __name__ == "__main__":
    unittest.main()
