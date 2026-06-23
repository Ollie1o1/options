"""Pure backtest metrics for the equity-vol study: per-trade Sharpe, a
Newey-West-adjusted t-stat of the mean, hit rate, profit factor, and an
out-of-sample train/test split by date."""
from __future__ import annotations
from typing import List, Optional, Tuple
import math


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sharpe(returns) -> Optional[float]:
    xs = [float(x) for x in returns]
    if len(xs) < 2:
        return None
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    return None if sd == 0 else m / sd


def newey_west_t(returns, lags: int = 5) -> Optional[float]:
    xs = [float(x) for x in returns]
    n = len(xs)
    if n < 3:
        return None
    m = _mean(xs)
    e = [x - m for x in xs]
    gamma0 = sum(v * v for v in e) / n
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        gammaL = sum(e[t] * e[t - L] for t in range(L, n)) / n
        var += 2.0 * w * gammaL
    if var <= 0:
        return None
    se = math.sqrt(var / n)
    return None if se == 0 else m / se


def hit_rate(returns) -> float:
    xs = [float(x) for x in returns]
    return sum(1 for x in xs if x > 0) / len(xs) if xs else 0.0


def profit_factor(returns) -> Optional[float]:
    gains = sum(x for x in returns if x > 0)
    losses = sum(-x for x in returns if x < 0)
    return None if losses == 0 else gains / losses


def _agg(xs: List[float]) -> dict:
    return {"n": len(xs), "mean": (_mean(xs) if xs else None), "sharpe": sharpe(xs)}


def split_oos(dated_returns: List[Tuple[str, float]], cutoff: str = "2024-01-01") -> dict:
    train = [r for d, r in dated_returns if d < cutoff]
    test = [r for d, r in dated_returns if d >= cutoff]
    return {"train": _agg(train), "test": _agg(test)}
