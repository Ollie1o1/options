"""Tests for breakout point-in-time features — pure, no look-ahead."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout import features as F


def _ramp(n, start=100.0, step=1.0):
    c = np.array([start + i * step for i in range(n)], dtype=float)
    return c, c + 1.0, c - 1.0, np.full(n, 1000.0)


class FeatureMathTests(unittest.TestCase):
    def test_trend_above_ma_is_positive(self):
        c, h, l, v = _ramp(260)  # rising series -> price above its MAs
        self.assertGreater(F.trend(c, len(c) - 1, 200), 0)

    def test_dist_52w_high_zero_at_new_high(self):
        c, h, l, v = _ramp(300)  # last point is the 52w high
        self.assertAlmostEqual(F.dist_52w_high(c, len(c) - 1), 0.0, places=6)

    def test_dist_52w_low_positive_above_low(self):
        c, h, l, v = _ramp(300)
        self.assertGreater(F.dist_52w_low(c, len(c) - 1), 0.0)

    def test_insufficient_history_returns_none(self):
        c, h, l, v = _ramp(10)
        self.assertIsNone(F.trend(c, 5, 200))

    def test_vector_has_all_keys(self):
        c, h, l, v = _ramp(300)
        fv = F.feature_vector(c, h, l, v, len(c) - 1)
        for k in ("trend_20", "mom_12_1", "vol_20", "dist_52w_high",
                  "rsi_14", "bollinger_b", "atr_14", "vol_surge", "gap_freq"):
            self.assertIn(k, fv)


class NoLookAheadTests(unittest.TestCase):
    def test_feature_at_t_unchanged_by_future(self):
        rng = np.random.default_rng(0)
        c = 100 + np.cumsum(rng.normal(0, 1, 400))
        h, l, v = c + 1, c - 1, np.full(400, 1000.0)
        t = 300
        before = F.feature_vector(c[: t + 1], h[: t + 1], l[: t + 1], v[: t + 1], t)
        after = F.feature_vector(c, h, l, v, t)  # extra future data appended
        for k in before:
            self.assertEqual(before[k], after[k], f"{k} leaked future data")


if __name__ == "__main__":
    unittest.main()
