import unittest
import numpy as np
import pandas as pd
from src.leverage.indicators import (ema, atr, donchian_high, donchian_low,
                                      rvol_pctile)


class TestIndicators(unittest.TestCase):
    def test_ema_matches_pandas(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = ema(s, span=3)
        exp = s.ewm(span=3, adjust=False).mean()
        np.testing.assert_allclose(out.values, exp.values)

    def test_atr_constant_range(self):
        n = 50
        df = pd.DataFrame({
            "high": np.full(n, 110.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 105.0),
        })
        a = atr(df, period=14)
        self.assertAlmostEqual(a.iloc[-1], 10.0, places=6)

    def test_donchian(self):
        df = pd.DataFrame({"high": [1, 2, 3, 4, 5.0], "low": [1, 1, 1, 0, 2.0]})
        self.assertEqual(donchian_high(df, n=3).iloc[4], 4.0)  # max(2,3,4)
        self.assertEqual(donchian_low(df, n=3).iloc[4], 0.0)   # min(1,1,0)

    def test_rvol_pctile_bounds(self):
        rng = np.random.default_rng(0)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
        p = rvol_pctile(close, window=30, lookback=200)
        last = p.dropna().iloc[-1]
        self.assertGreaterEqual(last, 0.0)
        self.assertLessEqual(last, 100.0)


if __name__ == "__main__":
    unittest.main()
