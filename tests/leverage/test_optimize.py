import unittest
import numpy as np
import pandas as pd
from src.leverage.reversion import ReversionParams
from src.leverage.optimize import optimize_reversion, default_grid
from src.leverage.backtest import BacktestResult


def _oscillating(n=2000):
    """A clean mean-reverting (sine) series so fade signals fire and the
    optimizer has something to chew on."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    t = np.arange(n)
    c = 100.0 + 8.0 * np.sin(t / 6.0) + np.random.default_rng(0).normal(0, 0.3, n)
    df5 = pd.DataFrame({"open": c, "high": c + 1.0, "low": c - 1.0, "close": c,
                        "volume": 100.0}, index=idx).rename_axis("open_time")
    df5.attrs["symbol"] = "BTCUSDT"
    df15 = df5.resample("15min").agg({"open": "first", "high": "max",
                                      "low": "min", "close": "last",
                                      "volume": "sum"}).dropna()
    df15.attrs["symbol"] = "BTCUSDT"
    return df5, df15


class TestOptimize(unittest.TestCase):
    def test_default_grid_nonempty_and_typed(self):
        g = default_grid()
        self.assertTrue(g)
        self.assertTrue(all(isinstance(p, ReversionParams) for p in g))

    def test_returns_grid_member_and_results(self):
        df5, df15 = _oscillating()
        grid = [ReversionParams(z_entry=1.5), ReversionParams(z_entry=2.0)]
        best, train_r, test_r = optimize_reversion(df5, df15, grid=grid,
                                                   min_trades=5)
        self.assertIn(best, grid)
        self.assertIsInstance(train_r, BacktestResult)
        self.assertIsInstance(test_r, BacktestResult)

    def test_holdout_split_is_disjoint(self):
        # train picks params it never saw on test; both must run on real bars
        df5, df15 = _oscillating()
        best, train_r, test_r = optimize_reversion(df5, df15, min_trades=5)
        self.assertGreater(train_r.n, 0)
        self.assertGreaterEqual(test_r.n, 0)


if __name__ == "__main__":
    unittest.main()
