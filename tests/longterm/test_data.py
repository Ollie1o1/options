"""Tests for the pure snapshot builder (no network)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import data as D


class TestSnapshotFromCloses(unittest.TestCase):
    def test_basic_fields(self):
        closes = [100.0 + i for i in range(260)]  # 100 … 359, rising
        s = D.snapshot_from_closes("MU", closes)
        self.assertEqual(s.ticker, "MU")
        self.assertEqual(s.spot, 359.0)
        self.assertEqual(s.high_52w, 359.0)
        self.assertEqual(s.low_52w, 100.0)
        self.assertAlmostEqual(s.ma200, sum(closes[-200:]) / 200.0)
        self.assertEqual(s.closes, closes)

    def test_sigma_from_last_63_log_returns(self):
        closes = [100.0] * 200 + [100.0 * (1.01 ** i) for i in range(64)]
        s = D.snapshot_from_closes("X", closes)
        # constant 1% log-return tail → sigma ≈ 0 variance? last 63 returns all ln(1.01)
        self.assertAlmostEqual(s.daily_sigma, 0.0, places=6)

    def test_short_history_no_ma200(self):
        closes = [50.0 + (i % 7) for i in range(120)]
        s = D.snapshot_from_closes("X", closes)
        self.assertIsNone(s.ma200)
        self.assertGreater(s.daily_sigma, 0.0)

    def test_too_short_returns_none(self):
        self.assertIsNone(D.snapshot_from_closes("X", [1.0] * 29))

    def test_nonfinite_closes_dropped(self):
        closes = [100.0] * 100 + [float("nan")] + [101.0] * 100
        s = D.snapshot_from_closes("X", closes)
        self.assertTrue(math.isfinite(s.spot))
        self.assertEqual(len(s.closes), 200)


if __name__ == "__main__":
    unittest.main()
