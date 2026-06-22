"""In-house isotonic regression (pool-adjacent-violators) for probability
calibration. No sklearn. Maps raw model probabilities to calibrated ones using a
monotone non-decreasing fit on (prob, outcome) holdout pairs."""
from __future__ import annotations
from typing import Tuple
import numpy as np


def fit_isotonic(probs: np.ndarray, outcomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    order = np.argsort(probs)
    x = probs[order]
    y = outcomes[order].copy()
    w = np.ones_like(y)
    # pool-adjacent-violators
    blocks = [[y[k], w[k], x[k]] for k in range(len(y))]
    merged = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            v2, w2, x2 = merged.pop()
            v1, w1, x1 = merged.pop()
            tot = w1 + w2
            merged.append([(v1 * w1 + v2 * w2) / tot, tot, x1])
    xk, yk = [], []
    idx = 0
    for v, wt, _ in merged:
        xk.append(x[idx])
        yk.append(v)
        idx += int(wt)
    return np.array(xk), np.clip(np.array(yk), 0.0, 1.0)


def apply_isotonic(cal: Tuple[np.ndarray, np.ndarray], p: float) -> float:
    xk, yk = cal
    if len(xk) == 0:
        return float(np.clip(p, 0.0, 1.0))
    return float(np.clip(np.interp(p, xk, yk, left=yk[0], right=yk[-1]), 0.0, 1.0))
