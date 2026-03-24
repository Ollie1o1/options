#!/usr/bin/env python3
"""
Monte Carlo simulation for options probability calculations using Geometric Brownian Motion.
"""

import numpy as np
from typing import Optional, Tuple


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
