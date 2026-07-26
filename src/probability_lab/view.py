"""Tilt the market RND into a user view.

Shift the log-mean by ln(1+drift) and scale dispersion around that mean by
vol_mult, working in log-price space so the market density's *shape* (skew,
kurtosis) is preserved. The user tilts the market's own belief rather than
replacing it with a Gaussian. drift=0, vol_mult=1 is the identity.
"""
from __future__ import annotations

import numpy as np

from src.probability_lab.rnd import Density


def apply_view(market: Density, drift_pct: float, vol_mult: float) -> Density:
    K = market.K
    R = np.log(K)                              # log-price grid
    # Market density over R: g(R) = K * pdf_K(K)  (change of variables dR = dK/K).
    g = K * market.pdf
    area = np.trapezoid(g, R)
    if area <= 0:
        return market
    g = g / area
    c = float(np.trapezoid(R * g, R))          # market mean log-price
    shift = np.log(1.0 + drift_pct)
    vm = max(vol_mult, 1e-6)
    # Target g_v(R) = g( c + (R - c - shift)/vm ) / vm  (shift + scale about c).
    src = c + (R - c - shift) / vm
    g_v = np.interp(src, R, g, left=0.0, right=0.0) / vm
    pdf_v = g_v / K                            # back to K-space
    pdf_v = np.clip(np.where(np.isfinite(pdf_v), pdf_v, 0.0), 0.0, None)
    a = np.trapezoid(pdf_v, K)
    if a <= 0:
        return market
    return Density(K, pdf_v / a)


__all__ = ["apply_view"]
