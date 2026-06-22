"""Tests for the walk-forward breakout backtest: planted signal must score
skill, pure noise must not (overfit tripwire)."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.data import Series
from src.breakout.backtest import collect_samples, run_backtest, _score


def _series_from_close(close):
    close = np.asarray(close, dtype=float)
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

    def test_score_rewards_skillful_model(self):
        import numpy as np
        rng = np.random.default_rng(0)
        n = 400
        outcomes = (rng.random(n) < 0.3).astype(float)          # 30% breakout base rate
        # oracle: high prob exactly when a breakout happened, low when not (+ mild noise)
        up_probs = np.clip(0.15 + 0.7 * outcomes + rng.normal(0, 0.05, n), 0, 1)
        samples = {"up_probs": up_probs, "up_outcomes": outcomes,
                   "los": np.full(n, -0.2), "his": np.full(n, 0.2),
                   "realized": np.zeros(n)}
        baseline_probs = np.full(n, outcomes.mean())            # uninformative constant
        r = _score(samples, baseline_probs)
        self.assertGreater(r["skill_vs_baseline"], 0.2)         # skillful model beats baseline


if __name__ == "__main__":
    unittest.main()
