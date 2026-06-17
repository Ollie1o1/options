"""Per-trade + portfolio risk metrics with overlap-aware significance. Pure."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def max_drawdown(equity: Sequence[float]) -> float:
    peak = -math.inf
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return float(mdd)


def cvar(pnl: Sequence[float], q: float = 0.05) -> float:
    a = sorted(float(x) for x in pnl)
    if not a:
        return float("nan")
    k = max(1, int(round(len(a) * q)))
    return float(sum(a[:k]) / k)


def summarize(pnl: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([float(x) for x in pnl], dtype=float)
    if a.size == 0:
        return {"n": 0}
    gains = a[a > 0].sum()
    losses = -a[a < 0].sum()
    eq = np.cumsum(a)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "std": sd, "hit_rate": float((a > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total": float(a.sum()), "max_drawdown": max_drawdown(eq),
        "cvar5": cvar(a, 0.05), "worst": float(a.min()), "best": float(a.max()),
    }


def newey_west_tstat(pnl: Sequence[float], lag: int) -> float:
    """t-stat of the mean with a Newey-West HAC variance (Bartlett weights),
    accounting for the autocorrelation that overlapping trades induce."""
    a = np.asarray([float(x) for x in pnl])
    n = a.size
    if n < 3:
        return 0.0
    x = a - a.mean()
    var = float((x * x).mean())
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1)
        cov = float((x[L:] * x[:-L]).mean())
        var += 2 * w * cov
    se = math.sqrt(max(var, 1e-18) / n)
    return float(a.mean() / se) if se > 0 else 0.0


def block_bootstrap_ci(pnl: Sequence[float], block: int = 5, iters: int = 1000,
                       seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Moving-block bootstrap CI for the mean — preserves the serial dependence
    that overlapping trades create (an iid bootstrap would understate the CI)."""
    a = np.asarray([float(x) for x in pnl])
    n = a.size
    if n == 0:
        return (0.0, 0.0)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    nb = max(1, math.ceil(n / block))
    means = []
    for _ in range(iters):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        means.append(a[idx].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)
