"""Tests for the walk-forward breakout backtest: planted signal must score
skill, pure noise must not (overfit tripwire)."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.data import Series
from src.breakout.backtest import collect_samples, run_backtest


def _series_from_close(close):
    close = np.asarray(close, dtype=float)
    dates = [f"2010-{1 + i // 21:02d}".replace("2010-13", "2011-01") for i in range(len(close))]
    dates = [f"d{i:05d}" for i in range(len(close))]  # opaque but ordered
    return Series(dates, close, close + 1, close - 1, np.full(len(close), 1000.0))


class BacktestTests(unittest.TestCase):
    def test_collect_returns_aligned_arrays(self):
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
        s = {"AAA": _series_from_close(close)}
        out = collect_samples(s, horizon=21, model="baseline", step=21)
        self.assertEqual(len(out["up_probs"]), len(out["up_outcomes"]))
        self.assertGreater(len(out["up_probs"]), 0)

    def test_noise_has_near_zero_skill(self):
        rng = np.random.default_rng(7)
        s = {f"T{i}": _series_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 900))))
             for i in range(6)}
        r = run_backtest(s, model="parametric", step=21, seed=1)
        self.assertLess(abs(r["EOM"]["skill_vs_baseline"]), 0.25)

    def test_calibration_keeps_probs_in_range(self):
        rng = np.random.default_rng(3)
        s = {f"T{i}": _series_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 900))))
             for i in range(4)}
        r = run_backtest(s, model="baseline", step=21, seed=1)
        self.assertGreaterEqual(r["EOM"]["brier"], 0.0)
        self.assertLessEqual(r["EOM"]["brier"], 1.0)


if __name__ == "__main__":
    unittest.main()
