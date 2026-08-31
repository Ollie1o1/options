"""
SVI (Stochastic Volatility Inspired) parameterization for IV surface fitting.

Fits w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)) per expiration,
where w = total variance (IV^2 * T), k = log-moneyness ln(K/S).

Computes iv_surface_residual: (market_IV - fitted_IV) / fitted_IV.
Positive = expensive vs fair surface, negative = cheap.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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


# Asymptotic-slope ceiling on total variance. w(k)/|k| -> b(1 +/- rho) as
# k -> +/-inf, so b(1+|rho|) bounds how steeply the wings may rise. Past this
# the wings are steeper than any arbitrage-free surface permits and the implied
# density goes negative out there.
#
# NECESSARY, NOT SUFFICIENT. This bounds the WINGS. It does not certify the
# whole slice free of butterfly arbitrage — that requires Gatheral's g(k) >= 0
# across the strip, which is not computed here. Naming it "no butterfly
# arbitrage" would claim more than it checks.
MAX_WING_SLOPE = 4.0


def _enforce_constraints(params: np.ndarray) -> np.ndarray:
    """Project parameters onto the feasible set for no-arbitrage."""
    a, b, rho, sigma, m = params
    b = max(b, 1e-6)
    rho = np.clip(rho, -0.999, 0.999)
    sigma = max(sigma, 0.001)
    # Wing slope: b(1+|rho|) < MAX_WING_SLOPE. Capping b rather than rho keeps
    # the smile's asymmetry, which carries the skew signal, and gives up only
    # the wing steepness that was unattainable anyway. Strictly below the
    # ceiling, not at it, so the returned params satisfy a strict inequality.
    wing = b * (1.0 + abs(rho))
    if wing >= MAX_WING_SLOPE:
        b = (MAX_WING_SLOPE / (1.0 + abs(rho))) * (1.0 - 1e-9)
    # No-arbitrage: a + b*sigma*sqrt(1 - rho^2) >= 0. Computed AFTER the wing
    # cap, since that changes b and therefore the floor.
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
        # Steer the search away from arbitrageable wings, so the projection in
        # `_enforce_constraints` is a rounding rather than a rescue.
        wing = b * (1.0 + abs(rho))
        if wing >= MAX_WING_SLOPE:
            penalty += 1e6 * (wing - MAX_WING_SLOPE) ** 2
        return _svi_objective(params, k, market_var) + penalty

    try:
        res = minimize(penalised, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8,
                                "fatol": 1e-10, "adaptive": True})
        # `res.success` is deliberately NOT a gate. Nelder-Mead reports success
        # only on meeting xatol=1e-8 AND fatol=1e-10 over five badly-scaled
        # parameters, which it usually cannot; it exhausts maxiter and reports
        # failure while sitting on an excellent fit. Over 120 realistic slices
        # 62% reported failure, and EVERY ONE of those scored above 0.95
        # against its own data (median 0.9999) — so the flag was discarding
        # nearly two thirds of good fits and leaving `iv_surface_residual` NaN
        # on most expiries.
        #
        # It is uninformative in the other direction too: the degenerate corner
        # described below converged with success=True. Fit adequacy is measured
        # by the SSE budget check, which is the thing that actually looks at
        # the data.

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


def calendar_arbitrage(slices: Sequence[SVIParams],
                       k_grid: Optional[np.ndarray] = None,
                       tol: float = 1e-12) -> dict:
    """Check that total variance never decreases with maturity.

    w(k, T2) >= w(k, T1) for every T2 > T1 and every k. A dip means a calendar
    spread is priced below zero: the longer-dated option would be cheaper in
    total variance than the shorter one covering the same strike.

    Slices are sorted by T here, so an unordered input is not mistaken for a
    violation — the caller's list order carries no meaning.

    Returns a dict with `arbitrage_free`, `n_violations`, `worst_drop` (the
    largest w decrease found, 0.0 when clean) and `violations`, each naming the
    maturity pair and log-moneyness where the surface dipped.

    This is a REPORT, not a filter. It does not modify or reject any slice:
    calendar arbitrage is a property of a surface, while fitting happens one
    expiry at a time, so the only honest place to act on it is a caller that
    holds every slice at once.
    """
    ordered = sorted((s for s in slices if s.T > 0), key=lambda s: s.T)
    if len(ordered) < 2:
        return {"arbitrage_free": True, "n_violations": 0,
                "worst_drop": 0.0, "violations": []}

    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 81)
    k_grid = np.asarray(k_grid, dtype=float)

    violations = []
    worst = 0.0
    for lo, hi in zip(ordered, ordered[1:]):
        w_lo = _svi_total_variance(k_grid, lo.a, lo.b, lo.rho, lo.m, lo.sigma)
        w_hi = _svi_total_variance(k_grid, hi.a, hi.b, hi.rho, hi.m, hi.sigma)
        drop = w_lo - w_hi                      # positive => w fell with T
        bad = drop > tol
        if not bad.any():
            continue
        j = int(np.argmax(drop))
        worst = max(worst, float(drop[j]))
        violations.append({
            "T_lo": float(lo.T),
            "T_hi": float(hi.T),
            "k": float(k_grid[j]),
            "w_lo": float(w_lo[j]),
            "w_hi": float(w_hi[j]),
            "drop": float(drop[j]),
            "n_points": int(bad.sum()),
        })

    return {
        "arbitrage_free": not violations,
        "n_violations": len(violations),
        "worst_drop": worst,
        "violations": violations,
    }


__all__ = ["fit_svi_surface", "fit_svi_slice", "SVIParams",
           "calendar_arbitrage", "MAX_WING_SLOPE"]
