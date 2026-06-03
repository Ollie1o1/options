import unittest
import numpy as np
import pandas as pd
from src.leverage.signals import Params, Signal, generate_signals, _in_session

# Disable the rvol band for the geometry/trigger/session tests so they isolate
# the breakout + trend + volume + session logic. The vol-band filter gets its
# own dedicated, deterministic test below.
_NOBAND = Params(vol_pctile_lo=0.0, vol_pctile_hi=100.0)


def _frames(n=600):
    """Aligned 5m + 15m frames, monotone uptrend (so every bar makes a new
    Donchian high and the 15m EMA slope is up). Volume is flat at 100 except a
    1000-surge on every IN-SESSION bar past the warmup, so >=1 long signal must
    fire. n=600 (~50h) clears the rvol warmup (~230 bars) and spans several
    session windows. The per-bar close advance (30) exceeds the intrabar
    high cushion (20) so each close genuinely breaks the prior bar's Donchian
    high (which is built from `high`, not `close`). Returns (df5, df15)."""
    start = pd.Timestamp("2026-05-01 00:00", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="5min")
    close = 60000.0 + np.arange(n) * 30.0
    vol = np.full(n, 100.0)
    for i in range(n):
        if i > 260 and _in_session(idx[i]) is not None:
            vol[i] = 1000.0
    df5 = pd.DataFrame({"open": close, "high": close + 20, "low": close - 20,
                        "close": close, "volume": vol}, index=idx
                       ).rename_axis("open_time")
    df5.attrs["symbol"] = "BTCUSDT"
    n15 = n // 3
    idx15 = pd.date_range(start, periods=n15, freq="15min")
    c15 = 60000.0 + np.arange(n15) * 6.0
    df15 = pd.DataFrame({"open": c15, "high": c15 + 30, "low": c15 - 30,
                         "close": c15, "volume": 300.0}, index=idx15
                        ).rename_axis("open_time")
    df15.attrs["symbol"] = "BTCUSDT"
    return df5, df15


class TestSignals(unittest.TestCase):
    def test_breakout_emits_long_in_uptrend(self):
        df5, df15 = _frames()
        sigs = generate_signals(df5, df15, _NOBAND)
        longs = [s for s in sigs if s.side == "long"]
        self.assertTrue(longs)
        s = longs[0]
        self.assertGreater(s.target, s.entry)   # long target above entry
        self.assertLess(s.stop, s.entry)        # long stop below entry
        self.assertEqual(s.symbol, "BTCUSDT")
        self.assertIsNotNone(_in_session(s.ts))  # only fires in-session

    def test_no_volume_surge_no_signal(self):
        df5, df15 = _frames()
        df5["volume"] = 100.0  # remove every surge
        sigs = generate_signals(df5, df15, _NOBAND)
        self.assertEqual(sigs, [])

    def test_outside_session_no_signal(self):
        df5, df15 = _frames()
        df5 = df5.copy()
        df5["volume"] = 100.0
        out_bars = [i for i in range(260, len(df5))
                    if _in_session(df5.index[i]) is None]
        for i in out_bars:
            df5.iloc[i, df5.columns.get_loc("volume")] = 1000.0
        sigs = generate_signals(df5, df15, _NOBAND)
        self.assertEqual(sigs, [])

    def test_exit_levels_use_atr_mults(self):
        df5, df15 = _frames()
        s = next(x for x in generate_signals(df5, df15, _NOBAND)
                 if x.side == "long")
        self.assertAlmostEqual(s.stop, s.entry - Params().atr_stop_mult * s.atr,
                               places=2)
        self.assertAlmostEqual(s.target, s.entry + Params().atr_target_mult * s.atr,
                               places=2)

    def test_vol_band_filter_suppresses_when_rvol_too_high(self):
        # Ramp the per-bar move amplitude so realized vol is monotonically rising;
        # the final bars' rvol percentile is then ~100 -> outside the default
        # [30,85] band -> otherwise-valid breakouts are suppressed.
        df5, df15 = _frames()
        amp = np.linspace(1.0, 400.0, len(df5))
        wob = np.where(np.arange(len(df5)) % 2 == 0, amp, -amp)
        df5 = df5.copy()
        df5["close"] = 60000.0 + np.arange(len(df5)) * 30.0 + wob
        df5["high"] = df5["close"] + 20
        df5["low"] = df5["close"] - 20
        with_band = generate_signals(df5, df15, Params())          # default band
        without_band = generate_signals(df5, df15, _NOBAND)        # band off
        self.assertLess(len(with_band), len(without_band))


if __name__ == "__main__":
    unittest.main()
