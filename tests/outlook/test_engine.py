"""Tests for the outlook scoring engine — cross-sectional ranking, offline."""
from __future__ import annotations

import unittest

from src.outlook.engine import (
    DEFAULT_OUTLOOK_CONFIG, zscore_map, rank_universe, classify,
)


class ZScoreTests(unittest.TestCase):
    def test_mean_zero_and_order_preserved(self):
        z = zscore_map({"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertAlmostEqual(sum(z.values()), 0.0, places=6)
        self.assertLess(z["a"], z["c"])

    def test_constant_values_map_to_zero(self):
        z = zscore_map({"a": 5.0, "b": 5.0})
        self.assertEqual(z["a"], 0.0)
        self.assertEqual(z["b"], 0.0)

    def test_none_values_ignored(self):
        z = zscore_map({"a": 1.0, "b": None, "c": 3.0})
        self.assertNotIn("b", z)


class ClassifyTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(classify(1.0, DEFAULT_OUTLOOK_CONFIG), "BULLISH")
        self.assertEqual(classify(-1.0, DEFAULT_OUTLOOK_CONFIG), "BEARISH")
        self.assertEqual(classify(0.0, DEFAULT_OUTLOOK_CONFIG), "NEUTRAL")


class RankUniverseTests(unittest.TestCase):
    def _universe(self):
        # strong: high on every factor; weak: low on every factor; mid: middling
        return {
            "STRONG": {"mom_12_1": 0.40, "trend_score": 0.20, "relative_strength": 0.15, "reversal_1m": 0.02},
            "MID":    {"mom_12_1": 0.10, "trend_score": 0.05, "relative_strength": 0.00, "reversal_1m": 0.00},
            "WEAK":   {"mom_12_1": -0.30, "trend_score": -0.15, "relative_strength": -0.12, "reversal_1m": -0.02},
        }

    def test_strong_ranks_bullish_first_weak_bearish_last(self):
        ranked = rank_universe(self._universe(), DEFAULT_OUTLOOK_CONFIG)
        self.assertEqual(ranked[0]["ticker"], "STRONG")
        self.assertEqual(ranked[-1]["ticker"], "WEAK")
        self.assertEqual(ranked[0]["direction"], "BULLISH")
        self.assertEqual(ranked[-1]["direction"], "BEARISH")

    def test_drivers_are_reported(self):
        ranked = rank_universe(self._universe(), DEFAULT_OUTLOOK_CONFIG)
        top = ranked[0]
        self.assertTrue(top["drivers"])  # non-empty
        self.assertIn("conviction", top)
        self.assertGreaterEqual(top["conviction"], 50)

    def test_weights_are_adaptable(self):
        """Zeroing momentum and leaning on reversal flips the ranking."""
        uni = {
            "A": {"mom_12_1": 0.50, "trend_score": 0.0, "relative_strength": 0.0, "reversal_1m": -0.10},
            "B": {"mom_12_1": -0.50, "trend_score": 0.0, "relative_strength": 0.0, "reversal_1m": 0.10},
        }
        import copy
        cfg = copy.deepcopy(DEFAULT_OUTLOOK_CONFIG)
        cfg["weights"] = {"mom_12_1": 1.0, "trend_score": 0, "relative_strength": 0, "reversal_1m": 0}
        self.assertEqual(rank_universe(uni, cfg)[0]["ticker"], "A")
        cfg["weights"] = {"mom_12_1": 0, "trend_score": 0, "relative_strength": 0, "reversal_1m": 1.0}
        self.assertEqual(rank_universe(uni, cfg)[0]["ticker"], "B")


if __name__ == "__main__":
    unittest.main()
