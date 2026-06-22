"""Point-in-time feature library for the breakout engine. Every function reads
only data up to index t (inclusive) and returns None on insufficient history —
the same no-look-ahead discipline as src/outlook/factors.py."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

TRADING_MONTH = 21
TRADING_YEAR = 252


def trend(close: np.ndarray, t: int, window: int) -> Optional[float]:
    if t < window:
        return None
    ma = float(np.mean(close[t - window + 1: t + 1]))
    return None if ma <= 0 else close[t] / ma - 1.0


def _ret(close: np.ndarray, t: int, lag: int) -> Optional[float]:
    if t < lag or close[t - lag] <= 0:
        return None
    return float(close[t] / close[t - lag] - 1.0)


def mom_12_1(close, t):
    if t < TRADING_YEAR or close[t - TRADING_YEAR] <= 0:
        return None
    return float(close[t - TRADING_MONTH] / close[t - TRADING_YEAR] - 1.0)


def realized_vol(close: np.ndarray, t: int, window: int) -> Optional[float]:
    if t < window:
        return None
    seg = close[t - window: t + 1]
    rets = np.diff(seg) / seg[:-1]
    return float(np.std(rets)) if len(rets) else None


def dist_52w_high(close: np.ndarray, t: int) -> Optional[float]:
    if t < TRADING_YEAR:
        return None
    hi = float(np.max(close[t - TRADING_YEAR + 1: t + 1]))
    return None if hi <= 0 else close[t] / hi - 1.0   # <= 0, zero at a new high


def dist_52w_low(close: np.ndarray, t: int) -> Optional[float]:
    if t < TRADING_YEAR:
        return None
    lo = float(np.min(close[t - TRADING_YEAR + 1: t + 1]))
    return None if lo <= 0 else close[t] / lo - 1.0   # >= 0, zero at a new low


def rsi_14(close: np.ndarray, t: int, window: int = 14) -> Optional[float]:
    if t < window:
        return None
    d = np.diff(close[t - window: t + 1])
    gain = float(np.mean(np.clip(d, 0, None)))
    loss = float(np.mean(np.clip(-d, 0, None)))
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger_b(close: np.ndarray, t: int, window: int = 20) -> Optional[float]:
    if t < window:
        return None
    seg = close[t - window + 1: t + 1]
    mu, sd = float(np.mean(seg)), float(np.std(seg))
    if sd == 0:
        return 0.5
    lower, upper = mu - 2 * sd, mu + 2 * sd
    return (close[t] - lower) / (upper - lower)


def atr_14(high, low, close, t, window: int = 14) -> Optional[float]:
    if t < window:
        return None
    tr = []
    for i in range(t - window + 1, t + 1):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]),
                      abs(low[i] - close[i - 1])))
    return float(np.mean(tr)) / close[t] if close[t] > 0 else None


def vol_surge(volume: np.ndarray, t: int, window: int = 20) -> Optional[float]:
    if t < window:
        return None
    avg = float(np.mean(volume[t - window: t]))
    return None if avg <= 0 else float(volume[t] / avg - 1.0)


def gap_freq(close: np.ndarray, t: int, window: int = 20, thresh: float = 0.02) -> Optional[float]:
    if t < window:
        return None
    seg = close[t - window: t + 1]
    rets = np.abs(np.diff(seg) / seg[:-1])
    return float(np.mean(rets > thresh))


def feature_vector(close, high, low, volume, t: int) -> Dict[str, Optional[float]]:
    return {
        "trend_20": trend(close, t, 20),
        "trend_50": trend(close, t, 50),
        "trend_200": trend(close, t, 200),
        "mom_12_1": mom_12_1(close, t),
        "mom_3m": _ret(close, t, 63),
        "mom_1m": _ret(close, t, 21),
        "vol_20": realized_vol(close, t, 20),
        "vol_60": realized_vol(close, t, 60),
        "dist_52w_high": dist_52w_high(close, t),
        "dist_52w_low": dist_52w_low(close, t),
        "rsi_14": rsi_14(close, t),
        "bollinger_b": bollinger_b(close, t),
        "atr_14": atr_14(high, low, close, t),
        "vol_surge": vol_surge(volume, t),
        "gap_freq": gap_freq(close, t),
    }
