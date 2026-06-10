"""Tests for the outlook backtest evaluation math — pure, offline."""
from __future__ import annotations

import unittest

from src.outlook.backtest import evaluate_calls, spearman_ic


class SpearmanTests(unittest.TestCase):
    def test_monotonic_increasing_is_one(self):
        self.assertAlmostEqual(spearman_ic([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=6)

    def test_monotonic_decreasing_is_minus_one(self):
        self.assertAlmostEqual(spearman_ic([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=6)

    def test_too_few_points_returns_none(self):
        self.assertIsNone(spearman_ic([1], [2]))


class EvaluateCallsTests(unittest.TestCase):
    def _records(self):
        return [
            {"direction": "BULLISH", "fwd": 0.05, "bench_fwd": 0.02},   # abs+ rel+
            {"direction": "BULLISH", "fwd": -0.03, "bench_fwd": 0.01},  # abs- rel-
            {"direction": "BEARISH", "fwd": -0.04, "bench_fwd": 0.01},  # abs+ rel+
            {"direction": "NEUTRAL", "fwd": 0.00, "bench_fwd": 0.0},    # skipped
        ]

    def test_overall_directional_hit_rate(self):
        r = evaluate_calls(self._records())
        self.assertEqual(r["n_calls"], 3)
        self.assertAlmostEqual(r["hit_rate"], 2 / 3, places=6)

    def test_split_by_direction(self):
        r = evaluate_calls(self._records())
        self.assertAlmostEqual(r["bullish_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(r["bearish_hit_rate"], 1.0, places=6)
        self.assertEqual(r["n_bullish"], 2)
        self.assertEqual(r["n_bearish"], 1)

    def test_relative_hit_rate_beats_market(self):
        # rec1 (0.05>0.02) yes, rec2 (-0.03<0.01) no, rec3 bearish (-0.04<0.01) yes
        r = evaluate_calls(self._records())
        self.assertAlmostEqual(r["relative_hit_rate"], 2 / 3, places=6)

    def test_empty_safe(self):
        r = evaluate_calls([])
        self.assertEqual(r["n_calls"], 0)
        self.assertEqual(r["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
