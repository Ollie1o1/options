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
    random_seed: Optional[int] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Probability of Profit (PoP) and Probability of Touch (PoT) using Monte Carlo simulation.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        sigma: Annualized volatility (IV)
        r: Risk-free rate
        premium: Option premium paid
        option_type: "call" or "put"
        n_simulations: Number of Monte Carlo paths to simulate
        random_seed: Optional seed for reproducibility
    
    Returns:
        Tuple of (probability_of_profit, probability_of_touch)
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or premium <= 0:
            return None, None
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Time step (daily granularity for touch detection)
        n_steps = max(int(T * 252), 1)  # Trading days
        dt = T / n_steps
        
        # Pre-compute constants
        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random normal samples: shape (n_simulations, n_steps)
        random_shocks = np.random.standard_normal((n_simulations, n_steps))
        
        # Initialize price paths
        prices = np.zeros((n_simulations, n_steps + 1))
        prices[:, 0] = S
        
        # Simulate GBM paths
        for t in range(1, n_steps + 1):
            prices[:, t] = prices[:, t - 1] * np.exp(drift + diffusion * random_shocks[:, t - 1])
        
        # Final prices at expiration
        final_prices = prices[:, -1]
        
        # Calculate break-even price
        if option_type.lower() == "call":
            breakeven = K + premium
            # Profit if final price > breakeven
            profitable = final_prices > breakeven
            # Touch if any price in path >= strike
            touched = np.any(prices >= K, axis=1)
        else:  # put
            breakeven = K - premium
            # Profit if final price < breakeven
            profitable = final_prices < breakeven
            # Touch if any price in path <= strike
            touched = np.any(prices <= K, axis=1)
        
        # Calculate probabilities
        pop = np.mean(profitable)
        pot = np.mean(touched)
        
        return float(pop), float(pot)
    
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
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Simulate final stock prices
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T)
        random_shocks = np.random.standard_normal(n_simulations)
        
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
