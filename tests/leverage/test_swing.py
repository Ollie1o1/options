import unittest

import numpy as np
import pandas as pd

from src.leverage import swing


def _ohlc(closes, highs=None, lows=None):
    closes = list(closes)
    highs = highs or [c * 1.001 for c in closes]
    lows = lows or [c * 0.999 for c in closes]
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": [1.0] * len(closes)}, index=idx)


def _features(n, entry_i, atr_val=5.0, ma_offset=20.0, closes=None, highs=None, lows=None):
    """Hand-built features frame for engine tests: a single long breakout at
    entry_i, MA held below price (no regime exit), constant ATR."""
    closes = closes if closes is not None else [100.0] * n
    highs = highs if highs is not None else [c + 0.5 for c in closes]
    lows = lows if lows is not None else [c - 0.5 for c in closes]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    f = pd.DataFrame({"close": closes, "high": highs, "low": lows,
                      "atr": [atr_val] * n,
                      "ma": [c - ma_offset for c in closes],
                      "long_brk": [i == entry_i for i in range(n)],
                      "short_brk": [False] * n}, index=idx)
    return f


class TestIndicators(unittest.TestCase):
    def test_atr_constant_range(self):
        # high-low = 4 every bar, no gaps -> ATR converges to 4
        closes = [100.0] * 30
        df = _ohlc(closes, highs=[102.0] * 30, lows=[98.0] * 30)
        a = swing.atr(df, period=14).iloc[-1]
        self.assertAlmostEqual(a, 4.0, places=6)

    def test_compute_features_flags_breakout(self):
        closes = [100.0] * 10 + [110.0]      # jump above prior highs and MA
        f = swing.compute_features(_ohlc(closes), lookback=3, ma=5, atr_period=3)
        self.assertTrue(bool(f["long_brk"].iloc[-1]))
        self.assertFalse(bool(f["short_brk"].iloc[-1]))


class TestCalibration(unittest.TestCase):
    def test_fallback_when_too_few_entries(self):
        f = _features(30, entry_i=22)
        self.assertEqual(swing.calibrate_stop_k(f, "long", horizon=2),
                         swing.K_FALLBACK)

    def test_percentile_of_winner_mae(self):
        # 10 identical winning breakouts, each with MAE = 0.5 ATR -> k = 0.5
        block = []
        c, h, l, brk = [], [], [], []
        for _ in range(10):
            # entry bar: close 100, then dip (low 95 = 0.5 ATR adverse), then win 105
            c += [100.0, 98.0, 105.0, 105.0]
            h += [100.5, 99.0, 106.0, 105.5]
            l += [99.5, 95.0, 104.0, 104.5]
            brk += [True, False, False, False]
        n = len(c)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        f = pd.DataFrame({"close": c, "high": h, "low": l, "atr": [10.0] * n,
                          "ma": [x - 20 for x in c], "long_brk": brk,
                          "short_brk": [False] * n}, index=idx)
        k = swing.calibrate_stop_k(f, "long", horizon=2, pct=80)
        self.assertAlmostEqual(k, 0.5, places=3)


class TestBacktest(unittest.TestCase):
    def test_uptrend_long_is_profitable(self):
        closes = [100.0] * 23 + [100 + i for i in range(1, 7)]   # rises after entry
        f = _features(len(closes), entry_i=22, atr_val=5.0, closes=closes)
        trades = swing.backtest(f, k_long=1.0, k_short=1.0, cost=0.0)
        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0].r_multiple, 0)
        self.assertEqual(trades[0].side, "long")

    def test_immediate_reversal_stops_near_minus_1R(self):
        closes = [100.0] * 23 + [96.0] + [96.0] * 5
        lows = [c - 0.5 for c in closes]
        lows[23] = 94.0                       # pierces stop at 100 - 1*5 = 95
        f = _features(len(closes), entry_i=22, atr_val=5.0, closes=closes, lows=lows)
        trades = swing.backtest(f, k_long=1.0, k_short=1.0, cost=0.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].reason, "stop")
        self.assertAlmostEqual(trades[0].r_multiple, -1.0, places=2)


class TestSignalAndWalkForward(unittest.TestCase):
    def test_no_signal_when_flat(self):
        df = _ohlc([100.0] * 150)
        self.assertIsNone(swing.latest_signal(df, stop_k=2.0))

    def test_signal_on_breakout(self):
        closes = [100.0] * 130 + [100 + i * 0.1 for i in range(1, 20)] + [115.0]
        sig = swing.latest_signal(_ohlc(closes), stop_k=2.0)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "long")
        self.assertLess(sig.stop, sig.price)        # long stop below price
        self.assertGreater(sig.risk_pct, 0)

    def test_walk_forward_runs_and_calibrates(self):
        rng = np.random.default_rng(0)
        steps = rng.normal(0.001, 0.02, 600)
        closes = 100 * np.cumprod(1 + steps)
        df = _ohlc(closes.tolist())
        wf = swing.walk_forward(df, train_frac=0.6)
        self.assertGreater(wf["k_long"], 0)
        self.assertIn("out_sample", wf)
        self.assertIn("n", wf["out_sample"])


if __name__ == "__main__":
    unittest.main()
