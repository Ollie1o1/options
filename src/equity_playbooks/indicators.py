"""Pure, causal (zero-lookahead) vectorized equity indicators.

Shared by both Phase 1 preregistered backtests and Phase 2 live Radar scanners
so backtested rules and live rules never drift.
"""
from __future__ import annotations

import numpy as np


def sma(closes: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average using strictly closes[t-window+1 : t+1]."""
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if n < window or window <= 0:
        return out
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1:] = (csum[window:] - csum[:-window]) / float(window)
    return out


def wilder_rsi(closes: np.ndarray, period: int = 2) -> np.ndarray:
    """Wilder's smoothed RSI computed strictly causally up to index t."""
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if n <= period or period <= 0:
        return out
    deltas = np.diff(arr)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    if avg_loss == 0.0:
        out[period] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    inv_p = 1.0 / float(period)
    keep = (period - 1.0) * inv_p
    for i in range(period, n - 1):
        avg_gain = avg_gain * keep + float(gains[i]) * inv_p
        avg_loss = avg_loss * keep + float(losses[i]) * inv_p
        if avg_loss == 0.0:
            out[i + 1] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return out


def mom_12_1_series(closes: np.ndarray, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """12-month minus 1-month return: close[t - skip] / close[t - lookback] - 1."""
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if n <= lookback:
        return out
    past = arr[: n - lookback]
    recent = arr[lookback - skip : n - skip]
    valid = (past > 0.0) & np.isfinite(past) & np.isfinite(recent)
    res = np.full(past.size, np.nan, dtype=float)
    res[valid] = recent[valid] / past[valid] - 1.0
    out[lookback:] = res
    return out


def trailing_median_dollar_vol(closes: np.ndarray, volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Trailing `window`-bar median of close * volume (strictly causal)."""
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = c.size
    out = np.full(n, np.nan, dtype=float)
    if n < window or window <= 0:
        return out
    dv = c * np.where(np.isfinite(v) & (v > 0.0), v, 0.0)
    # Strided sliding window view is zero-copy and fast in numpy.
    windows = np.lib.stride_tricks.sliding_window_view(dv, window)
    out[window - 1:] = np.median(windows, axis=1)
    return out


def trailing_rv(closes: np.ndarray, window: int = 20, annualize: float = 252.0) -> np.ndarray:
    """Trailing `window`-bar annualized realized volatility of daily log returns."""
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if n <= window or window < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        lret = np.diff(np.log(arr))
    windows = np.lib.stride_tricks.sliding_window_view(lret, window)
    stds = np.std(windows, axis=1, ddof=1)
    out[window:] = stds * np.sqrt(annualize)
    return out


def bad_jump_mask(closes: np.ndarray, max_abs_daily_ret: float = 0.45) -> np.ndarray:
    """Boolean mask True on bars with |close[t]/close[t-1] - 1| > threshold."""
    arr = np.asarray(closes, dtype=float)
    n = arr.size
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask
    prev = arr[:-1]
    curr = arr[1:]
    valid = (prev > 0.0) & np.isfinite(prev) & np.isfinite(curr)
    rets = np.zeros(n - 1, dtype=float)
    rets[valid] = np.abs(curr[valid] / prev[valid] - 1.0)
    mask[1:] = (~valid) | (rets > max_abs_daily_ret)
    return mask
