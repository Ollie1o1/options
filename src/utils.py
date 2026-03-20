#!/usr/bin/env python3
"""Utility helpers for the options screener.

This module provides high-performance, vectorized implementations of
Black-Scholes Greeks and pricing functions using NumPy and SciPy.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Union, Any

# --- Basic Utilities ---

def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float, returning default on failure."""
    try:
        if val is None:
            return default
        fval = float(val)
        return fval if np.isfinite(fval) else default
    except (ValueError, TypeError):
        return default

def norm_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal cumulative distribution function."""
    return norm.cdf(x)

def norm_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal probability density function."""
    return norm.pdf(x)

# --- Black-Scholes Core ---

def _d1d2(S, K, T, r, sigma, q=0.0):
    """Compute d1 and d2 for Black-Scholes with optional continuous dividend yield q."""
    S = np.asanyarray(S)
    K = np.asanyarray(K)
    T = np.asanyarray(T)
    r = np.asanyarray(r)
    sigma = np.asanyarray(sigma)
    q = np.asanyarray(q)

    # Merton (1973): replace S with S*exp(-q*T) in d1/d2
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    # Handle scalar results for backward compatibility with 'is None' checks
    if d1.ndim == 0:
        if np.isnan(d1):
            return None, None
        return float(d1), float(d2)

    return d1, d2

def bs_call(S, K, T, r, sigma, q=0.0):
    """Black-Scholes call option price with optional continuous dividend yield q."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def bs_put(S, K, T, r, sigma, q=0.0):
    """Black-Scholes put option price with optional continuous dividend yield q."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def bs_price(option_type, S, K, T, r, sigma, q=0.0):
    """Black-Scholes theoretical option price with optional continuous dividend yield q."""
    S_arr = np.asanyarray(S)
    if isinstance(option_type, str):
        if option_type.lower() == "call":
            return bs_call(S, K, T, r, sigma, q)
        else:
            return bs_put(S, K, T, r, sigma, q)
    else:
        option_type = np.asanyarray(option_type)
        price = np.empty_like(S_arr, dtype=float)
        is_call = np.char.lower(option_type.astype(str)) == "call"

        if np.any(is_call):
            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v
            price[is_call] = bs_call(_get(S, is_call), _get(K, is_call), _get(T, is_call), _get(r, is_call), _get(sigma, is_call), q)

        if np.any(~is_call):
            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v
            price[~is_call] = bs_put(_get(S, ~is_call), _get(K, ~is_call), _get(T, ~is_call), _get(r, ~is_call), _get(sigma, ~is_call), q)

        return price

def bs_delta(option_type, S, K, T, r, sigma, q=0.0):
    """Black-Scholes delta with optional continuous dividend yield q."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    if isinstance(option_type, str):
        if option_type.lower() == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    else:
        option_type = np.asanyarray(option_type)
        delta = np.empty_like(d1, dtype=float)
        is_call = np.char.lower(option_type.astype(str)) == "call"
        disc_q = np.exp(-q * np.asanyarray(T))
        delta[is_call] = disc_q * norm.cdf(d1[is_call]) if np.ndim(disc_q) == 0 else disc_q[is_call] * norm.cdf(d1[is_call])
        delta[~is_call] = (disc_q * (norm.cdf(d1[~is_call]) - 1.0)) if np.ndim(disc_q) == 0 else disc_q[~is_call] * (norm.cdf(d1[~is_call]) - 1.0)
        return delta

def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Black-Scholes gamma with optional continuous dividend yield q."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def bs_vega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega (per 1% change in IV) with optional continuous dividend yield q."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0
    return vega

def bs_theta(option_type, S, K, T, r, sigma, q=0.0):
    """Black-Scholes theta (daily) with optional continuous dividend yield q."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    q = np.asanyarray(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if isinstance(option_type, str):
            if option_type.lower() == "call":
                p2 = r * K * np.exp(-r * T) * norm.cdf(d2)
                p3 = q * S * np.exp(-q * T) * norm.cdf(d1)
                return (p1 - p2 + p3) / 365.0
            else:
                p2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
                p3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
                return (p1 + p2 - p3) / 365.0
        else:
            option_type = np.asanyarray(option_type)
            theta = np.empty_like(d1, dtype=float)
            is_call = np.char.lower(option_type.astype(str)) == "call"

            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v

            if np.any(is_call):
                p2_c = _get(r, is_call) * _get(K, is_call) * np.exp(-_get(r, is_call) * _get(T, is_call)) * norm.cdf(d2[is_call])
                p3_c = _get(q, is_call) * _get(S, is_call) * np.exp(-_get(q, is_call) * _get(T, is_call)) * norm.cdf(d1[is_call]) if np.ndim(q) > 0 else float(q) * _get(S, is_call) * np.exp(-float(q) * _get(T, is_call)) * norm.cdf(d1[is_call])
                theta[is_call] = (p1[is_call] - p2_c + p3_c) / 365.0

            if np.any(~is_call):
                p2_p = _get(r, ~is_call) * _get(K, ~is_call) * np.exp(-_get(r, ~is_call) * _get(T, ~is_call)) * norm.cdf(-d2[~is_call])
                p3_p = _get(q, ~is_call) * _get(S, ~is_call) * np.exp(-_get(q, ~is_call) * _get(T, ~is_call)) * norm.cdf(-d1[~is_call]) if np.ndim(q) > 0 else float(q) * _get(S, ~is_call) * np.exp(-float(q) * _get(T, ~is_call)) * norm.cdf(-d1[~is_call])
                theta[~is_call] = (p1[~is_call] + p2_p - p3_p) / 365.0

            return theta

def bs_rho(option_type, S, K, T, r, sigma, q=0.0):
    """Black-Scholes rho (per 1% change in rates) with optional continuous dividend yield q."""
    _, d2 = _d1d2(S, K, T, r, sigma, q)
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(option_type, str):
            if option_type.lower() == "call":
                return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
            else:
                return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
        else:
            option_type = np.asanyarray(option_type)
            rho = np.empty_like(d2, dtype=float)
            is_call = np.char.lower(option_type.astype(str)) == "call"

            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v

            if np.any(is_call):
                rho[is_call] = _get(K, is_call) * _get(T, is_call) * np.exp(-_get(r, is_call) * _get(T, is_call)) * norm.cdf(d2[is_call]) / 100.0

            if np.any(~is_call):
                rho[~is_call] = -_get(K, ~is_call) * _get(T, ~is_call) * np.exp(-_get(r, ~is_call) * _get(T, ~is_call)) * norm.cdf(-d2[~is_call]) / 100.0

            return rho


def bs_charm(option_type, S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes charm (dDelta/dTime) — daily delta decay.
    Negative for calls (delta decays toward 0), positive for puts.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if d1 is None:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        charm_raw = -norm.pdf(d1) * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if isinstance(option_type, str):
        if option_type.lower() == "call":
            return charm_raw / 365.0
        else:
            return (-charm_raw) / 365.0
    else:
        option_type = np.asanyarray(option_type)
        is_call = np.char.lower(option_type.astype(str)) == "call"
        charm = np.where(is_call, charm_raw, -charm_raw)
        return charm / 365.0


def bs_vanna(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes vanna (dDelta/dVol = dVega/dSpot).
    Vanna = -n(d1) * d2 / sigma
    Positive vanna: delta increases as IV rises.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if d1 is None:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        vanna = -norm.pdf(d1) * d2 / np.maximum(sigma, 1e-9)
    return vanna


# --- American Option Pricing (Barone-Adesi-Whaley 1987) ---

def baw_american_put(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, max_iter: int = 50, tol: float = 1e-6,
) -> float:
    """Barone-Adesi-Whaley American put price (scalar).

    Adds the early exercise premium on top of the European BS put price.
    For OTM puts (screener focus, delta 0.15-0.45), the premium is small but
    material for ITM puts and near-expiry positions.

    Parameters
    ----------
    S, K, T, r, sigma : float  — spot, strike, time-to-expiry (years), rate, IV
    q : float                  — continuous dividend yield (default 0 = no dividend)
    """
    import math
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)

    intrinsic = max(K - S, 0.0)
    if T <= 1e-6 or sigma <= 1e-9:
        return intrinsic

    p_euro = float(bs_put(S, K, T, r, sigma))

    rT = r * T
    exp_rT = math.exp(-rT)
    if abs(1.0 - exp_rT) < 1e-9:
        return p_euro

    M = 2.0 * r / (sigma ** 2 * (1.0 - exp_rT))
    N = 2.0 * (r - q) / sigma ** 2

    disc = (N - 1.0) ** 2 + 4.0 * M
    if disc < 0:
        return p_euro

    q2 = 0.5 * (-(N - 1.0) - math.sqrt(disc))
    if q2 >= 0:
        return p_euro   # no finite critical price for this parameter set

    exp_qT = math.exp(-q * T)

    # Find S* via bisection in (epsilon, K).
    # f(x) = (K - x) - [p(x) + (1 - exp(-qT)*N(-d1(x))) * x / q2]
    # Early exercise is always optimal when no root exists in (0, K) — i.e. f(K) > 0.

    def _f(x: float) -> float:
        d1v, _ = _d1d2(x, K, T, r, sigma)
        if d1v is None:
            return 0.0
        Nd1x = float(norm.cdf(-float(d1v)))
        ps = float(bs_put(x, K, T, r, sigma))
        return (K - x) - (ps + (1.0 - exp_qT * Nd1x) * x / q2)

    # Check boundary at K: if f(K) >= 0, early exercise is optimal for all S < K
    try:
        f_at_K = _f(K * (1.0 - 1e-6))
    except Exception:
        f_at_K = 1.0  # conservative: assume early exercise always optimal

    if f_at_K >= 0:
        # No valid S_star < K — for any S < K, exercise early
        if S < K:
            return max(intrinsic, p_euro)
        return p_euro

    # Bisection search for root in [K*epsilon, K)
    lo, hi = K * 1e-4, K * (1.0 - 1e-6)
    if _f(lo) <= 0:
        return p_euro  # f negative throughout: no early exercise

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = _f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            lo = mid
            break
        if fm > 0:
            lo = mid
        else:
            hi = mid

    S_star = lo

    if S <= S_star:
        return max(intrinsic, p_euro)

    d1_star_val, _ = _d1d2(S_star, K, T, r, sigma)
    if d1_star_val is None:
        return p_euro
    Nd1_star = float(norm.cdf(-float(d1_star_val)))

    A2 = -(S_star / q2) * (1.0 - exp_qT * Nd1_star)
    american_price = p_euro + A2 * (S / S_star) ** q2
    return max(american_price, intrinsic)


def baw_american_call(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, max_iter: int = 50, tol: float = 1e-6,
) -> float:
    """Barone-Adesi-Whaley American call price (scalar).

    For non-dividend-paying stocks (q=0), American call = European call.
    Early exercise becomes relevant only when q > 0.
    """
    import math
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)

    c_euro = float(bs_call(S, K, T, r, sigma))
    intrinsic = max(S - K, 0.0)

    if q <= 0:
        return c_euro   # no early exercise for non-dividend-paying calls
    if T <= 1e-6 or sigma <= 1e-9:
        return intrinsic

    rT = r * T
    exp_rT = math.exp(-rT)
    if abs(1.0 - exp_rT) < 1e-9:
        return c_euro

    M = 2.0 * r / (sigma ** 2 * (1.0 - exp_rT))
    N = 2.0 * (r - q) / sigma ** 2

    disc = (N - 1.0) ** 2 + 4.0 * M
    if disc < 0:
        return c_euro

    q1 = 0.5 * (-(N - 1.0) + math.sqrt(disc))
    if q1 <= 0:
        return c_euro

    exp_qT = math.exp(-q * T)
    S_star = K / max(1.0 - 1.0 / q1, 1e-6)

    for _ in range(max_iter):
        d1_s_val, _ = _d1d2(S_star, K, T, r, sigma)
        if d1_s_val is None:
            return c_euro
        d1_s = float(d1_s_val)

        Nd1 = float(norm.cdf(d1_s))
        c_s = float(bs_call(S_star, K, T, r, sigma))

        rhs = c_s + (1.0 - exp_qT * Nd1) * S_star / q1
        f = (S_star - K) - rhs

        call_delta = exp_qT * Nd1
        gamma_s = float(norm.pdf(d1_s)) / max(S_star * sigma * math.sqrt(T), 1e-12)
        d_rhs = call_delta + (1.0 - exp_qT * Nd1) / q1 + exp_qT * S_star * gamma_s / q1
        df = 1.0 - d_rhs
        if abs(df) < 1e-12:
            break

        S_star_new = max(S_star - f / df, 1e-3)
        if abs(S_star_new - S_star) < tol:
            S_star = S_star_new
            break
        S_star = S_star_new

    if S >= S_star:
        return max(intrinsic, c_euro)

    d1_star_val, _ = _d1d2(S_star, K, T, r, sigma)
    if d1_star_val is None:
        return c_euro
    Nd1_star = float(norm.cdf(float(d1_star_val)))

    A1 = (S_star / q1) * (1.0 - exp_qT * Nd1_star)
    american_price = c_euro + A1 * (S / S_star) ** q1
    return max(american_price, intrinsic)


def american_price(
    option_type: str, S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """Dispatch to BAW American pricer (put or call)."""
    if option_type.lower() == "call":
        return baw_american_call(S, K, T, r, sigma, q)
    return baw_american_put(S, K, T, r, sigma, q)


def early_exercise_premium(
    option_type: str, S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """Return the early exercise premium: American price - European price.

    Useful for flagging options where European pricing materially understates value.
    """
    if option_type.lower() == "call":
        a = baw_american_call(S, K, T, r, sigma, q)
        e = float(bs_call(S, K, T, r, sigma))
    else:
        a = baw_american_put(S, K, T, r, sigma, q)
        e = float(bs_put(S, K, T, r, sigma))
    return max(0.0, a - e)


# --- Formatting Helpers ---

def format_pct(x: Any) -> str:
    """Format a number as a percentage string."""
    try:
        if x is None or (isinstance(x, (float, np.float64)) and not np.isfinite(x)):
            return "-"
        return f"{100.0 * float(x):.1f}%"
    except (ValueError, TypeError):
        return "-"

def format_money(x: Any) -> str:
    """Format a number as a currency string."""
    try:
        if x is None or (isinstance(x, (float, np.float64)) and not np.isfinite(x)):
            return "-"
        return f"${float(x):.2f}"
    except (ValueError, TypeError):
        return "-"

def determine_moneyness(row: Any) -> str:
    """Determine if option is ITM, ATM, or OTM based on strike vs underlying."""
    try:
        strike = float(row["strike"])
        underlying = float(row["underlying"])
        opt_type = str(row["type"]).lower()

        if underlying > 0 and abs(strike - underlying) / underlying <= 0.015:
            return "ATM"
        if opt_type == "call":
            return "ITM" if strike < underlying else "OTM"
        else:  # put
            return "ITM" if strike > underlying else "OTM"
    except (ValueError, TypeError, KeyError):
        return "---"

__all__ = [
    "safe_float",
    "norm_cdf",
    "norm_pdf",
    "bs_call",
    "bs_put",
    "bs_delta",
    "bs_price",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_charm",
    "bs_vanna",
    "_d1d2",
    "baw_american_put",
    "baw_american_call",
    "american_price",
    "early_exercise_premium",
    "format_pct",
    "format_money",
    "determine_moneyness",
]