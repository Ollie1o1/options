"""Pure technical-analysis primitives over pandas frames/series. No I/O."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder via EMA). df needs high, low, close."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def donchian_high(df: pd.DataFrame, n: int) -> pd.Series:
    """Highest high of the prior n bars (excludes the current bar)."""
    return df["high"].shift(1).rolling(n).max()


def donchian_low(df: pd.DataFrame, n: int) -> pd.Series:
    return df["low"].shift(1).rolling(n).min()


def rvol_pctile(close: pd.Series, window: int = 30,
                lookback: int = 200) -> pd.Series:
    """Percentile (0-100) of current rolling realized vol within its own
    trailing `lookback` distribution. Realized vol = std of log returns over
    `window`, annualization-free (relative comparison only)."""
    logret = np.log(close / close.shift(1))
    rv = logret.rolling(window).std()
    return rv.rolling(lookback).apply(
        lambda x: (x[-1] >= x).mean() * 100.0, raw=True)
