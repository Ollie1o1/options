"""Drift neutralizer: detrend OHLC so cumulative drift -> 0 while the bar-to-bar
oscillation (and thus ATR / bands / z-scores) is preserved. This is the diagnostic
behind the 2026-06-03 finding that reversion's 'edge' was pure down-drift capture."""
import unittest
import numpy as np
import pandas as pd

from src.leverage.backtest import neutralize_drift


def _ohlc(mu_per_bar: float, n: int = 3000, vol: float = 0.002,
          seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = mu_per_bar + rng.normal(0.0, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close, "volume": 1.0},
                      index=idx)
    df.attrs["symbol"] = "BTCUSDT"
    return df


def _mean_logret(df: pd.DataFrame) -> float:
    return float(np.log(df["close"]).diff().dropna().mean())


def _std_logret(df: pd.DataFrame) -> float:
    return float(np.log(df["close"]).diff().dropna().std())


class TestNeutralizeDrift(unittest.TestCase):
    def test_removes_drift_from_5m(self):
        df5 = _ohlc(mu_per_bar=-5e-4)  # strong downtrend
        self.assertLess(_mean_logret(df5), -1e-4)
        d5, _ = neutralize_drift(df5, df5)
        self.assertAlmostEqual(_mean_logret(d5), 0.0, places=7)

    def test_preserves_oscillation(self):
        df5 = _ohlc(mu_per_bar=5e-4)  # uptrend
        d5, _ = neutralize_drift(df5, df5)
        # subtracting a constant from each log-return leaves the std unchanged
        self.assertAlmostEqual(_std_logret(df5), _std_logret(d5), places=9)

    def test_detrends_15m_consistently(self):
        df5 = _ohlc(mu_per_bar=-3e-4)
        df15 = df5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
        df15.attrs["symbol"] = "BTCUSDT"
        _, d15 = neutralize_drift(df5, df15)
        # 15m bars carry ~3x the per-bar drift; after neutralizing it -> ~0
        self.assertLess(abs(_mean_logret(d15)), 1e-4)

    def test_preserves_attrs_and_shape(self):
        df5 = _ohlc(mu_per_bar=1e-4)
        d5, d15 = neutralize_drift(df5, df5)
        self.assertEqual(d5.attrs.get("symbol"), "BTCUSDT")
        self.assertEqual(len(d5), len(df5))
        self.assertEqual(list(d5.columns), list(df5.columns))

    def test_idempotent(self):
        # after one pass the realized drift is ~0, so a second pass barely moves
        # anything — neutralizing is idempotent on an already-trendless series.
        df5 = _ohlc(mu_per_bar=-5e-4)
        d5, _ = neutralize_drift(df5, df5)
        d5b, _ = neutralize_drift(d5, d5)
        self.assertTrue(np.allclose(d5["close"].to_numpy(),
                                    d5b["close"].to_numpy(), rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
