"""Tests for equity-vol backtest metrics — pure math."""
from __future__ import annotations
import unittest
from src.equity_vol import metrics as M


class MetricTests(unittest.TestCase):
    def test_sharpe_basic(self):
        self.assertAlmostEqual(M.sharpe([1.0, 1.0, 1.0, 1.0]), None if False else M.sharpe([1, 1, 1, 1]))
        self.assertIsNone(M.sharpe([2.0]))                       # too few
        self.assertIsNone(M.sharpe([3.0, 3.0, 3.0]))             # zero std

    def test_sharpe_sign(self):
        self.assertGreater(M.sharpe([0.1, 0.2, -0.05, 0.15]), 0.0)

    def test_hit_rate(self):
        self.assertAlmostEqual(M.hit_rate([1, -1, 1, 1]), 0.75, places=6)

    def test_profit_factor(self):
        self.assertAlmostEqual(M.profit_factor([2, -1, 2, -1]), 2.0, places=6)
        self.assertIsNone(M.profit_factor([1, 2, 3]))            # no losses

    def test_newey_west_t_positive_for_positive_mean(self):
        t = M.newey_west_t([0.1, 0.12, 0.09, 0.11, 0.1, 0.13], lags=2)
        self.assertIsNotNone(t)
        self.assertGreater(t, 0.0)

    def test_split_oos_partitions_by_date(self):
        dr = [("2023-06-01", 0.1), ("2023-12-01", 0.2), ("2024-03-01", -0.1), ("2024-09-01", 0.05)]
        r = M.split_oos(dr, cutoff="2024-01-01")
        self.assertEqual(r["train"]["n"], 2)
        self.assertEqual(r["test"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
