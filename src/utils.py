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

def _d1d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    S = np.asanyarray(S)
    K = np.asanyarray(K)
    T = np.asanyarray(T)
    r = np.asanyarray(r)
    sigma = np.asanyarray(sigma)
    
    # Avoid division by zero and log of zero/negative
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    # Handle scalar results for backward compatibility with 'is None' checks
    if d1.ndim == 0:
        if np.isnan(d1):
            return None, None
        return float(d1), float(d2)

    return d1, d2

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def bs_put(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def bs_price(option_type, S, K, T, r, sigma):
    """Black-Scholes theoretical option price."""
    S_arr = np.asanyarray(S)
    if isinstance(option_type, str):
        if option_type.lower() == "call":
            return bs_call(S, K, T, r, sigma)
        else:
            return bs_put(S, K, T, r, sigma)
    else:
        option_type = np.asanyarray(option_type)
        price = np.empty_like(S_arr, dtype=float)
        is_call = np.char.lower(option_type.astype(str)) == "call"
        
        if np.any(is_call):
            # Extract scalar or array values for the call legs
            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v
            price[is_call] = bs_call(_get(S, is_call), _get(K, is_call), _get(T, is_call), _get(r, is_call), _get(sigma, is_call))
        
        if np.any(~is_call):
            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v
            price[~is_call] = bs_put(_get(S, ~is_call), _get(K, ~is_call), _get(T, ~is_call), _get(r, ~is_call), _get(sigma, ~is_call))
            
        return price

def bs_delta(option_type, S, K, T, r, sigma):
    """Black-Scholes delta."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    if isinstance(option_type, str):
        if option_type.lower() == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0
    else:
        option_type = np.asanyarray(option_type)
        delta = np.empty_like(d1, dtype=float)
        is_call = np.char.lower(option_type.astype(str)) == "call"
        delta[is_call] = norm.cdf(d1[is_call])
        delta[~is_call] = norm.cdf(d1[~is_call]) - 1.0
        return delta

def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1% change in IV)."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    return vega

def bs_theta(option_type, S, K, T, r, sigma):
    """Black-Scholes theta (daily)."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if isinstance(option_type, str):
            if option_type.lower() == "call":
                p2 = r * K * np.exp(-r * T) * norm.cdf(d2)
                return (p1 - p2) / 365.0
            else:
                p2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
                return (p1 + p2) / 365.0
        else:
            option_type = np.asanyarray(option_type)
            theta = np.empty_like(d1, dtype=float)
            is_call = np.char.lower(option_type.astype(str)) == "call"
            
            def _get(v, mask): return v[mask] if isinstance(v, np.ndarray) else v
            
            if np.any(is_call):
                p2_c = _get(r, is_call) * _get(K, is_call) * np.exp(-_get(r, is_call) * _get(T, is_call)) * norm.cdf(d2[is_call])
                theta[is_call] = (p1[is_call] - p2_c) / 365.0
            
            if np.any(~is_call):
                p2_p = _get(r, ~is_call) * _get(K, ~is_call) * np.exp(-_get(r, ~is_call) * _get(T, ~is_call)) * norm.cdf(-d2[~is_call])
                theta[~is_call] = (p1[~is_call] + p2_p) / 365.0
                
            return theta

def bs_rho(option_type, S, K, T, r, sigma):
    """Black-Scholes rho (per 1% change in rates)."""
    _, d2 = _d1d2(S, K, T, r, sigma)
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
    """Determine if option is ITM or OTM based on strike vs underlying."""
    try:
        strike = float(row["strike"])
        underlying = float(row["underlying"])
        opt_type = str(row["type"]).lower()
        
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
    "_d1d2",
    "format_pct",
    "format_money",
    "determine_moneyness",
]