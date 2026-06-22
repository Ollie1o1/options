# tests/breakout/test_engine.py
"""Tests for the breakout engine orchestrator (offline, injected series)."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.data import Series
from src.breakout.engine import live_forecasts


def _series(close):
    close = np.asarray(close, float)
    return Series([f"d{i:05d}" for i in range(len(close))],
                  close, close + 1, close - 1, np.full(len(close), 1000.0))


class EngineTests(unittest.TestCase):
    def test_live_forecasts_shape(self):
        rng = np.random.default_rng(0)
        s = {"AAA": _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))}
        rows = live_forecasts(s, model="parametric")
        self.assertEqual(len(rows), 3)  # 3 horizons
        r = rows[0]
        for k in ("ticker", "horizon", "point", "band", "up_prob", "down_prob"):
            self.assertIn(k, r)
        self.assertTrue(0.0 <= r["up_prob"] <= 1.0)

    def test_short_history_skipped_not_crashed(self):
        s = {"AAA": _series(np.full(50, 100.0))}  # too short for 52w features
        rows = live_forecasts(s, model="parametric")
        self.assertIsInstance(rows, list)  # no exception


if __name__ == "__main__":
    unittest.main()
