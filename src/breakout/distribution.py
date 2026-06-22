"""Forward-return distribution objects for the breakout engine.

A Distribution is represented by a sorted sample array, so every model — the
unconditional baseline and the featureful parametric model — produces the same
object and is scored the same way. Quantiles and tail probabilities read off the
empirical sample."""
from __future__ import annotations
import numpy as np


class Distribution:
    def __init__(self, samples: np.ndarray):
        self.samples = np.sort(np.asarray(samples, dtype=float))

    def point(self) -> float:
        return float(np.median(self.samples))

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    def band(self, lo: float, hi: float):
        return self.quantile(lo), self.quantile(hi)

    def prob_ge(self, x: float) -> float:
        return float(np.mean(self.samples >= x))

    def prob_le(self, x: float) -> float:
        return float(np.mean(self.samples <= x))


def baseline_distribution(close: np.ndarray, t: int, horizon: int,
                          n: int = 4000, seed: int = 0) -> Distribution:
    """Unconditional climatology: bootstrap from historical overlapping
    horizon-day returns observed up to t. No features — the control model."""
    if t < horizon + 1:
        return Distribution(np.zeros(n))
    hist = close[: t + 1]
    fwd = hist[horizon:] / hist[:-horizon] - 1.0   # overlapping h-day returns
    fwd = fwd[np.isfinite(fwd)]
    if fwd.size == 0:
        return Distribution(np.zeros(n))
    rng = np.random.default_rng(seed)
    return Distribution(rng.choice(fwd, size=n, replace=True))
