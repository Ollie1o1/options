import unittest
import numpy as np
import pandas as pd
from src.leverage.reversion import ReversionParams, generate_reversion_signals


def _frame(closes):
    n = len(closes)
    idx = pd.date_range("2026-05-01 00:00", periods=n, freq="5min", tz="UTC")
    c = np.array(closes, float)
    df = pd.DataFrame({"open": c, "high": c + 5, "low": c - 5, "close": c,
                       "volume": 100.0}, index=idx).rename_axis("open_time")
    df.attrs["symbol"] = "BTCUSDT"
    return df


class TestReversion(unittest.TestCase):
    def setUp(self):
        # flat at 100 for the lookback, then a sharp spike up far beyond 2 sigma
        base = [100.0 + (0.5 if i % 2 else -0.5) for i in range(40)]  # tiny wiggle
        self.up = _frame(base + [130.0])     # last bar spikes way above mean
        self.down = _frame(base + [70.0])    # last bar craters below mean

    def test_spike_up_emits_short_toward_mean(self):
        sigs = generate_reversion_signals(self.up, self.up, ReversionParams())
        last = [s for s in sigs if s.ts == self.up.index[-1]]
        self.assertTrue(last, "expected a fade signal on the >2sigma spike")
        s = last[0]
        self.assertEqual(s.side, "short")     # fade the up-move
        self.assertLess(s.target, s.entry)    # target is the mean, below entry
        self.assertGreater(s.stop, s.entry)   # stop is further up
        self.assertEqual(s.session, "reversion")

    def test_drop_down_emits_long_toward_mean(self):
        sigs = generate_reversion_signals(self.down, self.down, ReversionParams())
        last = [s for s in sigs if s.ts == self.down.index[-1]]
        self.assertTrue(last)
        s = last[0]
        self.assertEqual(s.side, "long")
        self.assertGreater(s.target, s.entry)  # mean above entry
        self.assertLess(s.stop, s.entry)

    def test_no_signal_when_not_stretched(self):
        flat = _frame([100.0 + (0.5 if i % 2 else -0.5) for i in range(60)])
        sigs = generate_reversion_signals(flat, flat, ReversionParams())
        self.assertEqual(sigs, [])

    def test_higher_threshold_emits_fewer(self):
        loose = generate_reversion_signals(self.up, self.up, ReversionParams(z_entry=1.5))
        strict = generate_reversion_signals(self.up, self.up, ReversionParams(z_entry=3.0))
        self.assertGreaterEqual(len(loose), len(strict))


if __name__ == "__main__":
    unittest.main()
