"""Straddle price + greeks. Thin wrappers over src.utils Black-Scholes greeks.

Per 1 unit of underlying (1 BTC/ETH); prices in USD when S is in USD. Pure.
"""
from __future__ import annotations

from src.utils import bs_call, bs_put, bs_delta, bs_gamma, bs_vega, bs_theta


def straddle(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return abs(S - K)
    return bs_call(S, K, T, r, sigma, q) + bs_put(S, K, T, r, sigma, q)


def straddle_delta(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return 1.0 if S > K else (-1.0 if S < K else 0.0)
    return bs_delta("call", S, K, T, r, sigma, q) + bs_delta("put", S, K, T, r, sigma, q)


def straddle_gamma(S, K, T, r, sigma, q=0.0):
    return 0.0 if T <= 0 else 2.0 * bs_gamma(S, K, T, r, sigma, q)


def straddle_vega(S, K, T, r, sigma, q=0.0):
    return 0.0 if T <= 0 else 2.0 * bs_vega(S, K, T, r, sigma, q)


def straddle_theta(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return 0.0
    return bs_theta("call", S, K, T, r, sigma, q) + bs_theta("put", S, K, T, r, sigma, q)
