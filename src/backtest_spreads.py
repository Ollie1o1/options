"""Vertical-spread synthetic simulation for the hypothesis sweep.

Bull Put / Bear Call verticals, built from the same BS primitives as
backtest_optimizer.py's naked short-put engine (imported, not copied),
priced with per-leg friction from the fitted spread surface instead of a
flat round-trip slippage. See
docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from src.backtest_optimizer import bs_call_price, bs_put_price

PROFIT_TARGET = 0.50
STOP_LOSS_MULT = 2.0
EXIT_DTE_MIN = 21


def simulate_vertical_pnl(
    S0: float,
    future_closes: np.ndarray,
    K_short: float,
    K_long: float,
    sigma: float,
    r: float,
    entry_dte: int,
    option_type: str,
    rel_short: float,
    rel_long: float,
    profit_target: float = PROFIT_TARGET,
    stop_mult: float = STOP_LOSS_MULT,
    min_dte: int = EXIT_DTE_MIN,
) -> Tuple[float, int]:
    """Simulate a credit vertical (short K_short / long K_long) and return
    (pnl_pct, exit_offset).

    pnl_pct is a fraction of the entry mid credit (short leg premium minus
    long leg premium, before friction) — matching backtest_optimizer's
    simulate_pnl convention of scoring against the raw BS entry price, not
    the friction-adjusted amount actually received.

    `option_type` is "put" (Bull Put: K_short > K_long) or "call" (Bear
    Call: K_short < K_long). The function does not validate strike
    ordering — a malformed vertical simply prices to a non-positive entry
    credit, refused below.
    """
    price_fn = bs_put_price if option_type == "put" else bs_call_price
    T0 = entry_dte / 365.0
    short_prem = price_fn(S0, K_short, T0, r, sigma)
    long_prem = price_fn(S0, K_long, T0, r, sigma)
    entry_mid_credit = short_prem - long_prem
    if entry_mid_credit <= 0:
        return 0.0, 0

    received = short_prem * (1 - rel_short) - long_prem * (1 + rel_long)
    stop_val = entry_mid_credit * stop_mult
    take_val = entry_mid_credit * profit_target

    for i, S in enumerate(future_closes):
        dte = entry_dte - i - 1
        T = max(dte / 365.0, 1 / 365.0)
        short_val = price_fn(float(S), K_short, T, r, sigma)
        long_val = price_fn(float(S), K_long, T, r, sigma)
        mid_net = short_val - long_val
        exit_now = (mid_net <= take_val) or (mid_net >= stop_val) or (dte <= min_dte)
        if exit_now or i == len(future_closes) - 1:
            exit_cost = short_val * (1 + rel_short) - long_val * (1 - rel_long)
            pnl_pct = (received - exit_cost) / max(entry_mid_credit, 1e-8)
            return float(pnl_pct), i
    return 0.0, 0
