#!/usr/bin/env python3
"""
Monte Carlo simulation for options probability calculations using Geometric Brownian Motion.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.stats import norm as _norm


def monte_carlo_pop(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    premium: float,
    option_type: str = "call",
    n_simulations: int = 10000,
    random_seed: Optional[int] = None,
    jump_intensity: float = 2.0,
    jump_mean: float = -0.02,
    jump_vol: float = 0.04,
    is_short: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Probability of Profit (PoP) and Probability of Touch (PoT) using
    Merton Jump Diffusion Monte Carlo.

    Jump diffusion captures the fat tails and negative skew observed in real equity
    returns, producing more accurate PoP estimates than plain GBM — particularly for
    OTM options where tail risk dominates.

    When is_short=True, the position is a credit/premium-selling trade:
    the seller profits when the option's intrinsic value at expiry is less than
    the premium collected.

    Default jump parameters (calibrated to US equity average):
        jump_intensity = 2.0   ~2 significant jumps per year
        jump_mean      = -0.02  average -2% jump size (negative skew)
        jump_vol       = 0.04   4% jump size standard deviation
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or premium <= 0:
            return None, None

        rng = np.random.default_rng(random_seed)

        # Time step (daily granularity for touch detection)
        n_steps = max(int(T * 252), 1)
        dt = T / n_steps

        # Risk-neutral drift correction for jump component (keeps E[S_T] = S * e^(rT))
        jump_corr = jump_intensity * (np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0)
        drift = (r - jump_corr - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Diffusion shocks: (n_simulations, n_steps)
        z = rng.standard_normal((n_simulations, n_steps))

        # Jump component: Bernoulli approximation (prob = lambda*dt per step)
        jump_occurs = rng.random((n_simulations, n_steps)) < (jump_intensity * dt)
        jump_sizes = rng.normal(jump_mean, jump_vol, (n_simulations, n_steps))

        # Log-return per step = diffusion + occasional jump
        log_steps = drift + diffusion * z + np.where(jump_occurs, jump_sizes, 0.0)

        # Cumulative price paths
        cum_log = np.cumsum(log_steps, axis=1)
        prices = S * np.exp(
            np.concatenate([np.zeros((n_simulations, 1)), cum_log], axis=1)
        )  # (n_simulations, n_steps+1)

        final_prices = prices[:, -1]

        if is_short:
            # Short/credit position: seller profits when intrinsic < premium
            if option_type.lower() == "call":
                payoff = np.maximum(final_prices - K, 0)
                touched = np.any(prices >= K, axis=1)
            else:
                payoff = np.maximum(K - final_prices, 0)
                touched = np.any(prices <= K, axis=1)
            profitable = payoff < premium
        else:
            # Long/debit position: buyer profits when past breakeven
            if option_type.lower() == "call":
                breakeven = K + premium
                profitable = final_prices > breakeven
                touched = np.any(prices >= K, axis=1)
            else:
                breakeven = K - premium
                profitable = final_prices < breakeven
                touched = np.any(prices <= K, axis=1)

        return float(np.mean(profitable)), float(np.mean(touched))

    except Exception:
        return None, None


def monte_carlo_expected_value(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    premium: float,
    option_type: str = "call",
    n_simulations: int = 10000,
    random_seed: Optional[int] = None
) -> Optional[float]:
    """
    Calculate expected value of an option position using Monte Carlo simulation.
    
    Returns:
        Expected profit/loss per share (multiply by 100 for per-contract value)
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        
        rng = np.random.default_rng(random_seed)

        # Simulate final stock prices
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T)
        random_shocks = rng.standard_normal(n_simulations)
        
        final_prices = S * np.exp(drift + diffusion * random_shocks)
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount to present value and subtract premium
        expected_payoff = np.mean(payoffs) * np.exp(-r * T)
        expected_value = expected_payoff - premium
        
        return float(expected_value)
    
    except Exception:
        return None


def batch_monte_carlo_pop(
    S_arr,
    K_arr,
    T_arr,
    sigma_arr,
    r: float,
    premium_arr,
    option_types,
    n_simulations: int = 5000,
    is_short: bool = False,
    random_seed=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch Monte Carlo PoP + analytical PoT across all contracts at once.

    PoP: single-step Merton jump-diffusion simulation — no per-step path loop needed,
         just simulates the final distribution of S_T for all N contracts simultaneously.
    PoT: exact GBM reflection-principle formula — zero simulation cost.

    Replaces the serial df.apply(_calc_mc_pop) pattern, giving ~20-50x speedup.
    Returns (pop_arr, pot_arr) each of shape (N,), NaN where inputs are invalid.
    """
    S_arr = np.asarray(S_arr, dtype=float)
    K_arr = np.asarray(K_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)
    sigma_arr = np.asarray(sigma_arr, dtype=float)
    premium_arr = np.asarray(premium_arr, dtype=float)
    types_arr = np.array([str(t).lower() for t in option_types])

    N = len(S_arr)
    pop_out = np.full(N, np.nan)
    pot_out = np.full(N, np.nan)

    valid = (S_arr > 0) & (K_arr > 0) & (T_arr > 0) & (sigma_arr > 0) & (premium_arr > 0)
    if not valid.any():
        return pop_out, pot_out

    S = S_arr[valid]
    K = K_arr[valid]
    T = T_arr[valid]
    sigma = sigma_arr[valid]
    prem = premium_arr[valid]
    types = types_arr[valid]
    Nv = S.shape[0]

    rng = np.random.default_rng(random_seed)

    # --- PoP: single-step jump-diffusion (same params as monte_carlo_pop) ---
    jump_intensity = 2.0
    jump_mean = -0.02
    jump_vol = 0.04
    jump_corr = jump_intensity * (np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0)
    mu_adj = r - jump_corr - 0.5 * sigma ** 2  # (Nv,)

    Z = rng.standard_normal((Nv, n_simulations))
    n_jumps = rng.poisson(jump_intensity * T[:, None], size=(Nv, n_simulations))
    jump_log = rng.normal(jump_mean, jump_vol, (Nv, n_simulations)) * n_jumps
    log_ret = mu_adj[:, None] * T[:, None] + sigma[:, None] * np.sqrt(T[:, None]) * Z + jump_log
    final_prices = S[:, None] * np.exp(log_ret)  # (Nv, n_sims)

    is_call = types == "call"  # (Nv,)

    if is_short:
        payoff = np.where(
            is_call[:, None],
            np.maximum(final_prices - K[:, None], 0),
            np.maximum(K[:, None] - final_prices, 0),
        )
        profitable = payoff < prem[:, None]
    else:
        breakeven = np.where(is_call, K + prem, K - prem)  # (Nv,)
        profitable = np.where(
            is_call[:, None],
            final_prices > breakeven[:, None],
            final_prices < breakeven[:, None],
        )

    pop_valid = np.mean(profitable.astype(float), axis=1)  # (Nv,)

    # --- PoT: analytical GBM barrier-touch probability (reflection principle) ---
    # For down-barrier (put, K typically < S):
    #   P(min S_t <= K) = Φ((h - μT)/(σ√T)) + exp(2μh/σ²) × Φ((h + μT)/(σ√T))
    # For up-barrier (call, K typically > S):
    #   P(max S_t >= K) = Φ((-h - μT)/(σ√T)) + exp(-2μh/σ²) × Φ((-h + μT)/(σ√T))
    # where h = ln(K/S), μ = r - σ²/2
    mu_gbm = r - 0.5 * sigma ** 2
    h = np.log(K / S)
    sqrt_T = np.sqrt(T)
    denom = sigma * sqrt_T

    pot_down = np.clip(
        _norm.cdf((h - mu_gbm * T) / denom)
        + np.exp(np.clip(2 * mu_gbm * h / sigma ** 2, -500, 500))
        * _norm.cdf((h + mu_gbm * T) / denom),
        0.0, 1.0,
    )
    pot_up = np.clip(
        _norm.cdf((-h - mu_gbm * T) / denom)
        + np.exp(np.clip(-2 * mu_gbm * h / sigma ** 2, -500, 500))
        * _norm.cdf((-h + mu_gbm * T) / denom),
        0.0, 1.0,
    )
    pot_valid = np.where(is_call, pot_up, pot_down)

    pop_out[valid] = pop_valid
    pot_out[valid] = pot_valid
    return pop_out, pot_out
