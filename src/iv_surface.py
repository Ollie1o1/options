"""
SVI (Stochastic Volatility Inspired) parameterization for IV surface fitting.

Fits w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)) per expiration,
where w = total variance (IV^2 * T), k = log-moneyness ln(K/S).

Computes iv_surface_residual: (market_IV - fitted_IV) / fitted_IV.
Positive = expensive vs fair surface, negative = cheap.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m: float, sigma: float) -> np.ndarray:
    """SVI parameterization: total variance w(k)."""
    dk = k - m
    return a + b * (rho * dk + np.sqrt(dk ** 2 + sigma ** 2))


def _svi_iv(k: np.ndarray, T: float, a: float, b: float, rho: float,
            m: float, sigma: float) -> np.ndarray:
    """Convert SVI total variance to implied volatility."""
    w = _svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)


def _svi_objective(params: np.ndarray, k: np.ndarray,
                   market_var: np.ndarray) -> float:
    """Sum of squared errors between market and fitted total variance."""
    a, b, rho, sigma, m = params
    fitted = _svi_total_variance(k, a, b, rho, m, sigma)
    return np.sum((market_var - fitted) ** 2)


def _enforce_constraints(params: np.ndarray) -> np.ndarray:
    """Project parameters onto the feasible set for no-arbitrage."""
    a, b, rho, sigma, m = params
    b = max(b, 1e-6)
    rho = np.clip(rho, -0.999, 0.999)
    sigma = max(sigma, 0.001)
    # No-arbitrage: a + b*sigma*sqrt(1 - rho^2) >= 0
    floor = -b * sigma * np.sqrt(1.0 - rho ** 2)
    a = max(a, floor)
    return np.array([a, b, rho, sigma, m])


def _fit_single_expiry(k: np.ndarray, market_iv: np.ndarray,
                       T: float) -> Tuple[Optional[np.ndarray], float]:
    """
    Fit SVI params for one expiration slice.

    Returns (params, fit_quality) where params is (a, b, rho, sigma, m)
    or None if fitting fails, and fit_quality is in [0, 1].
    """
    market_var = market_iv ** 2 * T
    mean_var = np.mean(market_var)

    x0 = np.array([mean_var, 0.1, -0.3, 0.3, 0.0])

    # Nelder-Mead with a penalty wrapper to respect constraints
    def penalised(params):
        a, b, rho, sigma, m = params
        penalty = 0.0
        if b < 0:
            penalty += 1e6 * b ** 2
        if abs(rho) >= 1.0:
            penalty += 1e6 * (abs(rho) - 0.999) ** 2
        if sigma < 0.001:
            penalty += 1e6 * (0.001 - sigma) ** 2
        arb_floor = -b * max(sigma, 0.001) * np.sqrt(1.0 - min(rho ** 2, 0.998))
        if a < arb_floor:
            penalty += 1e6 * (arb_floor - a) ** 2
        return _svi_objective(params, k, market_var) + penalty

    try:
        res = minimize(penalised, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8,
                                "fatol": 1e-10, "adaptive": True})
        if not res.success and res.fun > mean_var * len(k):
            return None, 0.0
        params = _enforce_constraints(res.x)
        fit_quality = max(0.0, 1.0 - res.fun / max(mean_var * len(k), 1e-10))
        return params, fit_quality
    except Exception:
        return None, 0.0


def fit_svi_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SVI surface across all expirations and compute residuals.

    Parameters
    ----------
    df : DataFrame with columns:
        strike, underlying, impliedVolatility, T_years, expiration

    Returns
    -------
    DataFrame with added columns ``iv_surface_residual`` and ``iv_surface_confidence``.
        residual = (market_IV - fitted_IV) / fitted_IV
        Positive means contract is expensive vs the fitted vol surface.
        Set to 0.0 where fitting is not possible.
        confidence is the per-expiry fit quality in [0, 1].
    """
    df = df.copy()
    df["iv_surface_residual"] = 0.0
    df["iv_surface_confidence"] = 0.0
    df["iv_surface_fitted"] = False

    required = {"strike", "underlying", "impliedVolatility", "T_years", "expiration"}
    if not required.issubset(df.columns):
        return df

    for exp, grp in df.groupby("expiration"):
        idx = grp.index
        if len(grp) < 5:
            continue

        S = grp["underlying"].iloc[0]
        if S <= 0:
            continue

        strikes = grp["strike"].values.astype(float)
        market_iv = grp["impliedVolatility"].values.astype(float)
        T = grp["T_years"].iloc[0]

        if T <= 0:
            continue

        # Filter out zero/nan IVs
        valid = (market_iv > 0) & np.isfinite(market_iv) & (strikes > 0)
        if valid.sum() < 5:
            continue

        k = np.log(strikes[valid] / S)
        iv_valid = market_iv[valid]

        params, fit_quality = _fit_single_expiry(k, iv_valid, T)
        if params is None:
            import logging
            logging.getLogger(__name__).info(
                "SVI fit failed for expiry %s (%d valid points, T=%.3f) — residuals set to NaN",
                exp, valid.sum(), T,
            )
            df.loc[idx, "iv_surface_residual"] = np.nan
            continue

        a, b, rho, sigma, m = params

        # Compute fitted IV for ALL rows in this expiry group
        k_all = np.log(strikes / S)
        fitted_iv = _svi_iv(k_all, T, a, b, rho, m, sigma)

        # Residual where we have valid market IV and fitted IV
        safe = (fitted_iv > 1e-6) & (market_iv > 0) & np.isfinite(market_iv)
        residuals = np.where(safe, (market_iv - fitted_iv) / fitted_iv, 0.0)

        df.loc[idx, "iv_surface_residual"] = residuals
        df.loc[idx, "iv_surface_confidence"] = fit_quality
        df.loc[idx, "iv_surface_fitted"] = True

    return df


__all__ = ["fit_svi_surface"]
