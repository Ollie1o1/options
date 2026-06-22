"""Tests for breakout backtest metrics — pure math."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout import metrics as M


class MetricTests(unittest.TestCase):
    def test_brier_perfect_is_zero(self):
        self.assertAlmostEqual(M.brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 0.0)

    def test_brier_worst_is_one(self):
        self.assertAlmostEqual(M.brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0])), 1.0)

    def test_auc_perfect_ranking_is_one(self):
        self.assertAlmostEqual(M.auc(np.array([0.1, 0.2, 0.6, 0.8]),
                                     np.array([0, 0, 1, 1])), 1.0, places=6)

    def test_auc_single_class_is_none(self):
        self.assertIsNone(M.auc(np.array([0.2, 0.3]), np.array([1, 1])))

    def test_band_coverage_counts_inside(self):
        los = np.array([-0.1, -0.1, -0.1])
        his = np.array([0.1, 0.1, 0.1])
        realized = np.array([0.0, 0.2, -0.05])  # 2 of 3 inside
        self.assertAlmostEqual(M.band_coverage(los, his, realized), 2 / 3, places=6)

    def test_skill_score_positive_when_better(self):
        self.assertAlmostEqual(M.skill_score(0.1, 0.2), 0.5, places=6)

    def test_ece_zero_for_calibrated(self):
        rng = np.random.default_rng(0)
        p = rng.random(20000)
        y = (rng.random(20000) < p).astype(float)
        self.assertLess(M.expected_calibration_error(p, y, 10), 0.03)

    def test_pinball_loss_median_is_half_mae(self):
        pred = np.array([0.0, 0.0, 0.0])
        realized = np.array([1.0, -1.0, 0.5])
        # at q=0.5 pinball loss = 0.5 * mean(|realized - pred|)
        self.assertAlmostEqual(M.pinball_loss(pred, realized, 0.5),
                               0.5 * np.mean(np.abs(realized - pred)), places=6)


if __name__ == "__main__":
    unittest.main()
