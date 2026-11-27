"""Moving-block bootstrap and Holm correction for daily return series."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def block_bootstrap_means(x: Sequence[float], block: int = 10, draws: int = 10_000,
                          seed: int = 20261127, chunk: int = 1_000, circular: bool = False) -> np.ndarray:
    """Means of `draws` moving-block resamples of the series `x` (in order).

    `circular=True` wraps blocks around the end (every observation equally likely)."""
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([])
    block = max(1, min(block, n))
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    out = np.empty(draws)
    done = 0
    while done < draws:
        k = min(chunk, draws - done)
        if circular:
            starts = rng.integers(0, n, size=(k, n_blocks))
            idx = ((starts[:, :, None] + offsets) % n).reshape(k, -1)[:, :n]
        else:
            starts = rng.integers(0, n - block + 1, size=(k, n_blocks))
            idx = (starts[:, :, None] + offsets).reshape(k, -1)[:, :n]
        out[done:done + k] = arr[idx].mean(axis=1)
        done += k
    return out


def summarise(x: Sequence[float], block: int = 10, draws: int = 10_000,
              seed: int = 20261127) -> Dict[str, float]:
    """Mean, 95% percentile CI and a two-sided bootstrap p-value (centred)."""
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": float(arr.mean()) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "hit": float("nan")}
    means = block_bootstrap_means(arr, block, draws, seed)
    m = float(arr.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(np.mean(np.abs(means - m) >= abs(m)))
    return {"n": n, "mean": m, "lo": float(lo), "hi": float(hi),
            "p": max(p, 1.0 / draws), "hit": float(np.mean(arr > 0))}


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Holm step-down: adjusted p and reject flag per key."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, float]] = {}
    running = 0.0
    still_rejecting = True
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        reject = still_rejecting and p <= alpha / (m - i)
        still_rejecting = reject
        out[k] = {"p": p, "p_holm": running, "reject": bool(reject)}
    return out
