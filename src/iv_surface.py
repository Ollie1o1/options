"""
SVI (Stochastic Volatility Inspired) parameterization for IV surface fitting.

Fits w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)) per expiration,
where w = total variance (IV^2 * T), k = log-moneyness ln(K/S).

Computes iv_surface_residual: (market_IV - fitted_IV) / fitted_IV.
Positive = expensive vs fair surface, negative = cheap.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParams:
    """Fitted SVI parameters for a single expiry slice."""
    a: float
    b: float
    rho: float
    sigma: float
    m: float
    T: float
    fit_quality: float

    def iv(self, k: np.ndarray) -> np.ndarray:
        """Implied vol at log-moneyness k = ln(K/S)."""
        return _svi_iv(np.asarray(k, dtype=float), self.T,
                       self.a, self.b, self.rho, self.m, self.sigma)


def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m: float, sigma: float) -> np.ndarray:
    """SVI parameterization: total variance w(k)."""
    dk = k - m
    return a + b * (rho * dk + np.sqrt(dk ** 2 + sigma ** 2))


def _svi_iv(k: np.ndarray, T: float, a: float, b: float, rho: float,
            m: float, sigma: float) -> np.ndarray:
    """Convert SVI total variance to implied volatility."""
    if T <= 0:
        T = 1e-10
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
    # float(), not the bare numpy scalar: `budget` below divides a float by
    # this, and an untyped reduction makes that expression Any/Any — which is
    # how the loose typing here went unnoticed until `sse` was made explicit.
    mean_var = float(np.mean(market_var))

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
        if not res.success:
            return None, 0.0

        # Both the accept/reject decision and the reported quality are measured
        # on the parameters this function RETURNS, not on `res.x`.
        #
        # They are different points. `_enforce_constraints` projects `res.x`
        # onto the feasible set, and it moves it in ~60% of fitted slices.
        # `res.fun` is also the PENALISED objective, not the fit error. Judging
        # on `res.fun` while returning the projected params let a degenerate
        # corner through: on a realistic skew the search escaped to b -> 0,
        # rho -> -1, sigma -> 5.4e6, which flattens w(k) to a constant. That
        # was reported as fit_quality 0.9905 while the returned parameters
        # scored 0.0000 against the same data — and it flowed on as a confident
        # `iv_surface_residual` computed from a surface that described nothing.
        params = _enforce_constraints(res.x)
        budget = max(mean_var * len(k), 1e-10)
        sse = float(_svi_objective(params, k, market_var))

        # Reject the fit if the returned parameters miss the data by more than
        # the whole variance budget. (The convergence test was once `and`, which
        # let a converged-but-garbage fit slip through; it is now a separate
        # early return above.)
        if not np.isfinite(sse) or sse > budget:
            return None, 0.0
        fit_quality = max(0.0, 1.0 - sse / budget)
        return params, fit_quality
    except Exception:
        return None, 0.0


def fit_svi_slice(strikes, market_iv, T: float, S: float) -> Optional[SVIParams]:
    """Fit SVI to a single expiry slice.

    Parameters
    ----------
    strikes, market_iv : 1-D arrays (raw strikes, not log-moneyness).
    T : years to expiry.  S : underlying spot.

    Returns SVIParams or None when the slice is too thin / the fit fails.
    Reuses the same ``_fit_single_expiry`` optimizer that drives the surface
    residual signal, so both paths stay consistent.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_iv = np.asarray(market_iv, dtype=float)
    if S <= 0 or T <= 0:
        return None
    valid = (market_iv > 0) & np.isfinite(market_iv) & (strikes > 0)
    if valid.sum() < 5:
        return None
    k = np.log(strikes[valid] / S)
    params, quality = _fit_single_expiry(k, market_iv[valid], T)
    if params is None:
        return None
    a, b, rho, sigma, m = params
    return SVIParams(a=float(a), b=float(b), rho=float(rho), sigma=float(sigma),
                     m=float(m), T=float(T), fit_quality=float(quality))


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
            # Routine on thin / short-DTE expiries and already handled below by
            # setting residuals to NaN. DEBUG, not INFO: the root logger prints
            # bare messages at INFO, so this would interleave with the report.
            import logging
            logging.getLogger(__name__).debug(
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
        residuals = np.where(safe, (market_iv - fitted_iv) / fitted_iv, np.nan)

        df.loc[idx, "iv_surface_residual"] = residuals
        df.loc[idx, "iv_surface_confidence"] = fit_quality
        df.loc[idx, "iv_surface_fitted"] = True

    return df


__all__ = ["fit_svi_surface", "fit_svi_slice", "SVIParams"]
