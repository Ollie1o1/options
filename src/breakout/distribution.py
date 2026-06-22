"""Forward-return distribution objects for the breakout engine.

A Distribution is represented by a sorted sample array, so every model — the
unconditional baseline and the featureful parametric model — produces the same
object and is scored the same way. Quantiles and tail probabilities read off the
empirical sample."""
from __future__ import annotations
import math
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


DEFAULT_PARAMS = {"df": 4.0, "drift_gain": 0.5, "skew_gain": 3.0, "vol_floor": 0.005}


def _f(features: dict, key: str, default: float = 0.0) -> float:
    v = features.get(key)
    return float(v) if v is not None else default


def parametric_distribution(features: dict, horizon: int, params: dict | None = None,
                            n: int = 4000, seed: int = 0) -> Distribution:
    """Skew Student-t forward-return distribution conditioned on features.

    scale = daily vol * sqrt(horizon); location = drift_gain * 1-month momentum;
    right-skew (upper tail fattened) rises with trend and proximity to the 52w
    high, left-skew with proximity to the 52w low. Hand-set params (v1) keep it
    interpretable and overfit-resistant."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)
    vol = max(_f(features, "vol_20", 0.015), p["vol_floor"])
    scale = vol * math.sqrt(horizon)
    loc = p["drift_gain"] * _f(features, "mom_1m", 0.0)
    # skew driver: positive when bullish/near-high, negative when bearish/near-low
    drive = (_f(features, "trend_200", 0.0)
             + _f(features, "dist_52w_high", -0.2)   # ~0 near high (bullish)
             + _f(features, "dist_52w_low", 0.2))    # large near high
    gamma = math.exp(p["skew_gain"] * 0.1 * math.tanh(drive))
    z = rng.standard_t(p["df"], size=n)
    z = np.where(z >= 0, z * gamma, z / gamma)       # gamma>1 fattens upper tail
    samples = loc + scale * z
    samples = np.maximum(samples, -1.0)   # a simple return cannot be below -100%
    return Distribution(samples)


def make_distribution(series, t: int, horizon: int, model: str, seed: int = 0) -> Distribution:
    """Build the forward-return distribution for a Series at index t under the
    chosen model ('baseline' or 'parametric')."""
    from src.breakout import features as _F
    if model == "baseline":
        return baseline_distribution(series.close, t, horizon, seed=seed)
    fv = _F.feature_vector(series.close, series.high, series.low, series.volume, t)
    return parametric_distribution(fv, horizon, seed=seed)
