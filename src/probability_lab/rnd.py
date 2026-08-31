"""Risk-neutral density (RND) extraction via Breeden-Litzenberger.

q(K) = e^{rT} d^2C/dK^2, where C(K) is reconstructed from the SVI smile so the
second derivative is taken on a smooth, arbitrage-controlled call curve rather
than on noisy raw market prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.iv_surface import fit_svi_slice
from src.utils import bs_call

_GRID_N = 500
_LO, _HI = 0.40, 2.50  # grid spans [0.40 S, 2.50 S]


@dataclass
class Density:
    K: np.ndarray
    pdf: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.pdf = np.asarray(self.pdf, dtype=float)
        # Cumulative distribution via trapezoid, anchored at 0 on the left edge.
        self._cdf = np.concatenate([[0.0],
                                    np.cumsum(0.5 * (self.pdf[1:] + self.pdf[:-1])
                                              * np.diff(self.K))])

    def prob_above(self, x: float) -> float:
        return float(np.clip(1.0 - np.interp(x, self.K, self._cdf), 0.0, 1.0))

    def prob_below(self, x: float) -> float:
        return float(np.clip(np.interp(x, self.K, self._cdf), 0.0, 1.0))

    def prob_between(self, a: float, b: float) -> float:
        return max(0.0, self.prob_below(b) - self.prob_below(a))

    def mean(self) -> float:
        return float(np.trapezoid(self.K * self.pdf, self.K))

    def quantile(self, p: float) -> float:
        return float(np.interp(p, self._cdf, self.K))

    def expected_payoff(self, fn: Callable[[np.ndarray], np.ndarray]) -> float:
        return float(np.trapezoid(fn(self.K) * self.pdf, self.K))

    def prob_payoff_exceeds(self, fn: Callable[[np.ndarray], np.ndarray],
                            thr: float) -> float:
        mask = (fn(self.K) > thr).astype(float)
        return float(np.clip(np.trapezoid(mask * self.pdf, self.K), 0.0, 1.0))

    def moments(self, basis: str) -> dict:
        """Risk-neutral mean, variance, skewness and kurtosis.

        `basis` is REQUIRED and has no default, because the two conventions
        give different numbers from the same density and either could
        reasonably be called "the risk-neutral skew":

          "price"  — moments of S_T itself. A flat 30% smile at T=0.25 gives
                     skewness +0.456, positive purely from the lognormal shape.
          "logret" — moments of ln(S_T / S_0). The same flat smile gives 0.0,
                     so a non-zero reading is the volatility smile talking
                     rather than the baseline.

        Quoting one as the other misstates both the sign and the magnitude, so
        the caller names it. Kurtosis is RAW, not excess: a Gaussian reads 3.0.

        Skewness and kurtosis are undefined for a degenerate density; variance
        at or below zero yields NaN for both rather than a divide-by-zero.
        """
        if basis not in ("price", "logret"):
            raise ValueError(
                f"basis must be 'price' or 'logret', got {basis!r} — "
                "the two give different numbers and cannot be guessed")

        K, pdf = self.K, self.pdf
        if basis == "logret":
            # ln(K) needs K > 0, and the grid starts at 0.40*S so this only
            # trims a numerically empty left edge.
            keep = K > 1e-12
            K, pdf = K[keep], pdf[keep]
            x = np.log(K)
        else:
            x = K

        area = float(np.trapezoid(pdf, K))
        if not np.isfinite(area) or area <= 0:
            return {"mean": float("nan"), "variance": float("nan"),
                    "skewness": float("nan"), "kurtosis": float("nan")}
        pdf = pdf / area

        mean = float(np.trapezoid(x * pdf, K))
        var = float(np.trapezoid((x - mean) ** 2 * pdf, K))
        if not np.isfinite(var) or var <= 0:
            return {"mean": mean, "variance": var,
                    "skewness": float("nan"), "kurtosis": float("nan")}
        m3 = float(np.trapezoid((x - mean) ** 3 * pdf, K))
        m4 = float(np.trapezoid((x - mean) ** 4 * pdf, K))
        return {
            "mean": mean,
            "variance": var,
            "skewness": m3 / var ** 1.5,
            "kurtosis": m4 / var ** 2,
        }


def _normalize(K: np.ndarray, raw: np.ndarray) -> np.ndarray:
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    area = np.trapezoid(raw, K)
    if area <= 0:
        raise ValueError("degenerate RND (non-positive area)")
    return raw / area


def _lognormal_pdf(K: np.ndarray, S: float, sigma: float, T: float,
                   r: float) -> np.ndarray:
    T = max(T, 1e-9)
    sigma = max(sigma, 1e-6)
    mu = np.log(S) + (r - 0.5 * sigma ** 2) * T
    s = sigma * np.sqrt(T)
    with np.errstate(divide="ignore"):
        pdf = np.exp(-((np.log(K) - mu) ** 2) / (2 * s ** 2)) / (
            K * s * np.sqrt(2 * np.pi))
    return pdf


def rnd_from_smile(strikes, ivs, T: float, S: float, r: float,
                   confidence_out: Optional[dict] = None) -> Density:
    """Build the RND from a smile. Falls back to an ATM-lognormal (flagged)
    when the SVI fit fails on a thin/short slice. Never raises on bad slices."""
    K = np.linspace(_LO * S, _HI * S, _GRID_N)
    params = fit_svi_slice(strikes, ivs, T, S)
    if params is not None:
        k = np.log(K / S)
        iv = np.clip(params.iv(k), 1e-3, None)
        C = np.asarray(bs_call(S, K, T, r, iv), dtype=float)
        d2 = np.gradient(np.gradient(C, K), K)
        raw = np.exp(r * T) * d2
        try:
            pdf = _normalize(K, raw)
            if confidence_out is not None:
                confidence_out["source"] = "svi"
                confidence_out["fit_quality"] = params.fit_quality
            return Density(K, pdf)
        except ValueError:
            pass  # fall through to lognormal
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    atm = float(ivs[np.argmin(np.abs(strikes - S))]) if len(ivs) else 0.3
    pdf = _normalize(K, _lognormal_pdf(K, S, atm, T, r))
    if confidence_out is not None:
        confidence_out["source"] = "lognormal_fallback"
        confidence_out["fit_quality"] = 0.0
    return Density(K, pdf)


__all__ = ["Density", "rnd_from_smile"]
