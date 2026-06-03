"""The 6-parameter intraday-momentum rule. PURE: no I/O, no project state, so
the exact same code runs in backtest, paper, and (future) live.

Long rule (short = mirror):
  1. Regime: EMA(ema_len,15m) slope > 0 AND rvol percentile in [lo, hi].
  2. Trigger: close breaks prior donchian_n-bar high (5m).
  3. Confirm: breakout-bar volume > 1.5x rolling-median volume.
  4. Session: within 2h after 00:00 UTC or 13:30 UTC.
Exits (precomputed): stop atr_stop_mult*ATR, target atr_target_mult*ATR,
trail trail_mult*ATR after +1.5 ATR, time stop 4h / session end.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from .indicators import ema, atr, donchian_high, donchian_low, rvol_pctile

_VOL_CONFIRM = 1.5
_VOL_MED_WIN = 20
_SESSIONS_UTC = (0, 13.5)   # 00:00 and 13:30
_SESSION_HOURS = 2.0
_TRAIL_ARM_ATR = 1.5


@dataclass(frozen=True)
class Params:
    donchian_n: int = 20
    ema_len: int = 50
    vol_pctile_lo: float = 30.0
    vol_pctile_hi: float = 85.0
    atr_stop_mult: float = 1.2
    atr_target_mult: float = 2.2
    trail_mult: float = 1.0


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: str            # 'long' | 'short'
    ts: pd.Timestamp
    entry: float
    atr: float
    stop: float
    target: float
    trail_trigger: float  # price at which trailing arms (+1.5 ATR favorable)
    session: str
    confidence: float


def _in_session(ts: pd.Timestamp) -> Optional[str]:
    hour = ts.hour + ts.minute / 60.0
    for start in _SESSIONS_UTC:
        if start <= hour < start + _SESSION_HOURS:
            return "asia-open" if start == 0 else "us-open"
    return None


def generate_signals(df5: pd.DataFrame, df15: pd.DataFrame,
                     params: Params) -> List[Signal]:
    """Emit a Signal for every bar whose conditions fire. Backtest consumes the
    whole list; the `signal` CLI uses only the final bar."""
    symbol = df5.attrs.get("symbol", "BTCUSDT")
    a = atr(df5, period=14)
    dhi = donchian_high(df5, params.donchian_n)
    dlo = donchian_low(df5, params.donchian_n)
    vol_med = df5["volume"].rolling(_VOL_MED_WIN).median()
    rvp = rvol_pctile(df5["close"], window=30, lookback=200)
    out: List[Signal] = []
    e15 = ema(df15["close"], span=params.ema_len)

    for i in range(len(df5)):
        ts = df5.index[i]
        sess = _in_session(ts)
        if sess is None:
            continue
        atr_i = a.iloc[i]
        if not (atr_i > 0):
            continue
        rv = rvp.iloc[i]
        if pd.isna(rv) or not (params.vol_pctile_lo <= rv <= params.vol_pctile_hi):
            continue
        vmed = vol_med.iloc[i]
        if pd.isna(vmed) or df5["volume"].iloc[i] <= _VOL_CONFIRM * vmed:
            continue
        close = df5["close"].iloc[i]
        # regime via 15m EMA slope as of this 5m bar
        sub15 = e15[e15.index <= ts]
        if len(sub15) < 2:
            continue
        up = sub15.iloc[-1] > sub15.iloc[-2]
        down = sub15.iloc[-1] < sub15.iloc[-2]
        conf = min(1.0, (df5["volume"].iloc[i] / vmed) / 4.0)
        if up and not pd.isna(dhi.iloc[i]) and close > dhi.iloc[i]:
            out.append(Signal(symbol, "long", ts, close, atr_i,
                              close - params.atr_stop_mult * atr_i,
                              close + params.atr_target_mult * atr_i,
                              close + _TRAIL_ARM_ATR * atr_i, sess, conf))
        elif down and not pd.isna(dlo.iloc[i]) and close < dlo.iloc[i]:
            out.append(Signal(symbol, "short", ts, close, atr_i,
                              close + params.atr_stop_mult * atr_i,
                              close - params.atr_target_mult * atr_i,
                              close - _TRAIL_ARM_ATR * atr_i, sess, conf))
    return out
