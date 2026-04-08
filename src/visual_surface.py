"""Analytical calculation logic for option P&L and Greeks grids.
This module returns structured data (numpy arrays) and contains no display or print logic.
"""

import numpy as np
from .utils import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta


def compute_pnl_grid(option_type, S, K, T, r, sigma, entry_price,
                     q=0.0, n_price=40, n_iv=20,
                     price_range=(-0.25, 0.25), iv_range=(-0.50, 0.50)):
    """Compute P&L grid over price shocks x IV shocks using full BS repricing."""
    price_shocks = np.linspace(price_range[0], price_range[1], n_price)
    iv_shocks = np.linspace(iv_range[0], iv_range[1], n_iv)

    price_mesh, iv_mesh = np.meshgrid(price_shocks, iv_shocks, indexing='ij')
    S_grid = S * (1.0 + price_mesh)
    sigma_grid = np.maximum(sigma * (1.0 + iv_mesh), 0.01)

    new_prices = bs_price(option_type, S_grid, K, T, r, sigma_grid, q)
    pnl = new_prices - entry_price

    return price_shocks, iv_shocks, pnl


def compute_greek_grid(greek_name, option_type, S, K, T, r, sigma,
                       q=0.0, n_price=40, n_iv=20,
                       price_range=(-0.25, 0.25), iv_range=(-0.50, 0.50)):
    """Compute a Greek value grid over price shocks x IV shocks.

    greek_name: one of 'delta', 'gamma', 'vega', 'theta'
    """
    price_shocks = np.linspace(price_range[0], price_range[1], n_price)
    iv_shocks = np.linspace(iv_range[0], iv_range[1], n_iv)

    price_mesh, iv_mesh = np.meshgrid(price_shocks, iv_shocks, indexing='ij')
    S_grid = S * (1.0 + price_mesh)
    sigma_grid = np.maximum(sigma * (1.0 + iv_mesh), 0.01)

    # bs_gamma and bs_vega don't take option_type as first arg
    if greek_name == 'delta':
        greek_grid = bs_delta(option_type, S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'gamma':
        greek_grid = bs_gamma(S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'vega':
        greek_grid = bs_vega(S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'theta':
        greek_grid = bs_theta(option_type, S_grid, K, T, r, sigma_grid, q)
    else:
        raise ValueError(f"Unknown greek: {greek_name}")

    return price_shocks, iv_shocks, greek_grid
