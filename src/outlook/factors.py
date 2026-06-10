"""Forward-looking factor library for the sector/asset outlook engine.

Each factor turns a daily close series into a directional signal (positive =
bullish, negative = bearish). The factor set is deliberately small and chosen
for documented predictive power at the 1-3 month horizon:

  mom_12_1          12-month return excluding the most recent month
                    (Jegadeesh-Titman / Asness cross-sectional momentum)
  trend_score       price vs its 200-day MA (time-series momentum / trend)
  reversal_1m       contrarian: negative of the last-month return
                    (short-horizon mean reversion)
  relative_strength instrument momentum minus a benchmark's (rotation signal)

All functions return None when there is insufficient history, and never look
ahead — they read only the series up to the given index (default: the end).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

TRADING_MONTH = 21
TRADING_YEAR = 252


def _at(closes: Sequence[float], t: Optional[int]) -> int:
    return (len(closes) - 1) if t is None else t


def mom_12_1(closes: Sequence[float], t: Optional[int] = None) -> Optional[float]:
    """12-month return, skipping the most recent month."""
    t = _at(closes, t)
    if t < TRADING_YEAR:
        return None
    past = closes[t - TRADING_YEAR]
    recent = closes[t - TRADING_MONTH]
    if past <= 0:
        return None
    return recent / past - 1.0


def trend_score(closes: Sequence[float], t: Optional[int] = None) -> Optional[float]:
    """Price relative to its 200-day moving average (fractional gap)."""
    t = _at(closes, t)
    if t < 200:
        return None
    window = closes[t - 199: t + 1]
    ma200 = sum(window) / len(window)
    if ma200 <= 0:
        return None
    return closes[t] / ma200 - 1.0


def reversal_1m(closes: Sequence[float], t: Optional[int] = None) -> Optional[float]:
    """Contrarian short-term signal: negative of the last-month return."""
    t = _at(closes, t)
    if t < TRADING_MONTH:
        return None
    prev = closes[t - TRADING_MONTH]
    if prev <= 0:
        return None
    return -(closes[t] / prev - 1.0)


def relative_strength(
    closes: Sequence[float], bench: Sequence[float],
    t: Optional[int] = None, lookback: int = 63,
) -> Optional[float]:
    """Instrument's return minus the benchmark's over `lookback` days."""
    t = _at(closes, t)
    if t < lookback or len(bench) <= t:
        return None
    inst = closes[t] / closes[t - lookback] - 1.0 if closes[t - lookback] > 0 else None
    b = bench[t] / bench[t - lookback] - 1.0 if bench[t - lookback] > 0 else None
    if inst is None or b is None:
        return None
    return inst - b


__all__ = ["mom_12_1", "trend_score", "reversal_1m", "relative_strength"]
