"""Tests for the forward-return Distribution + baseline climatology."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.distribution import Distribution, baseline_distribution, parametric_distribution, DEFAULT_PARAMS


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


class ParametricTests(unittest.TestCase):
    def _feat(self, trend, vol=0.015, mom=0.0, dhigh=-0.2, dlow=0.2):
        return {"trend_200": trend, "trend_20": trend, "vol_20": vol,
                "mom_1m": mom, "dist_52w_high": dhigh, "dist_52w_low": dlow}

    def test_scale_grows_with_vol(self):
        lo = parametric_distribution(self._feat(0.0, vol=0.01), 21, seed=3)
        hi = parametric_distribution(self._feat(0.0, vol=0.04), 21, seed=3)
        self.assertGreater(hi.quantile(0.9) - hi.quantile(0.1),
                           lo.quantile(0.9) - lo.quantile(0.1))

    def test_breakout_proximity_fattens_upper_tail(self):
        near_high = parametric_distribution(
            self._feat(0.10, dhigh=-0.01, dlow=0.40), 21, seed=4)
        near_low = parametric_distribution(
            self._feat(-0.10, dhigh=-0.40, dlow=0.01), 21, seed=4)
        self.assertGreater(near_high.prob_ge(0.10), near_low.prob_ge(0.10))

    def test_missing_features_falls_back_gracefully(self):
        d = parametric_distribution({}, 21, seed=5)
        self.assertIsInstance(d.point(), float)

    def test_returns_never_below_total_loss(self):
        # extreme vol must not produce impossible (< -100%) simple returns
        d = parametric_distribution(self._feat(0.0, vol=0.50), 63, n=20000, seed=9)
        self.assertGreaterEqual(d.samples.min(), -1.0)
        lo, hi = d.band(0.1, 0.9)
        self.assertGreaterEqual(lo, -1.0)


if __name__ == "__main__":
    unittest.main()
