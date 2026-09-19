"""Vertical-spread synthetic simulation for the hypothesis sweep.

Bull Put / Bear Call verticals, built from the same BS primitives as
backtest_optimizer.py's naked short-put engine (imported, not copied),
priced with per-leg friction from the fitted spread surface instead of a
flat round-trip slippage. See
docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest_optimizer import (
    ENTRY_DTE, ROLL_STEP_DAYS, RISK_FREE_RATE, SLIPPAGE_PCT, TARGET_DELTA,
    WEIGHT_KEYS, _get_yf, _hv_30, bs_call_price, bs_put_price,
    compute_component_scores, strike_for_call_delta, strike_for_delta,
)
from src.spread_surface import DEFAULT_SURFACE_PATH, SpreadSurface, load_surface

PROFIT_TARGET = 0.50
STOP_LOSS_MULT = 2.0
EXIT_DTE_MIN = 21
WING_DELTA = 0.10

_SPREAD_IDX = WEIGHT_KEYS.index("spread")


@dataclass
class SpreadTrade:
    symbol: str
    entry_date: Any
    exit_date: Any
    pnl_pct: float
    components: np.ndarray
    credit_to_width: float


def load_default_surface(path: str = DEFAULT_SURFACE_PATH) -> Tuple[Optional[SpreadSurface], str]:
    """Load the fitted spread surface, or report the flat-friction fallback.

    Returns (None, "fallback_flat") both when the file is absent and when it
    exists but carries no cells — either way there is no real measurement to
    use, and a caller must not blend a guess with real data unlabeled.
    """
    if not os.path.exists(path):
        return None, "fallback_flat"
    surface = load_surface(path)
    if not surface.cells:
        return None, "fallback_flat"
    return surface, "surface"


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


def backtest_ticker_vertical(
    symbol: str,
    period: str = "5y",
    r: float = RISK_FREE_RATE,
    entry_dte: int = ENTRY_DTE,
    roll_step: int = ROLL_STEP_DAYS,
    target_delta: float = TARGET_DELTA,
    wing_delta: float = WING_DELTA,
    option_type: str = "put",
    stop_mult: float = STOP_LOSS_MULT,
    profit_target: float = PROFIT_TARGET,
    surface: Optional[SpreadSurface] = None,
) -> Optional[List[SpreadTrade]]:
    """Roll a credit vertical forward over one ticker's price history.

    Mirrors backtest_optimizer.backtest_ticker's loop structure (same
    warmup, same roll step, same component-score inputs) but prices two
    legs instead of one and tracks real calendar dates natively.
    """
    try:
        raw = _get_yf().download(symbol, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw["Close"].dropna().squeeze()
        if not isinstance(closes, pd.Series) or len(closes) < 300:
            return None
        vols = raw.get("Volume", pd.Series(0, index=raw.index))
        if isinstance(vols, pd.DataFrame):
            vols = vols.iloc[:, 0]
        vols = vols.reindex(closes.index).fillna(0)

        warmup = 260
        step_indices = range(warmup, len(closes) - entry_dte - 5, roll_step)
        strike_fn = strike_for_delta if option_type == "put" else strike_for_call_delta
        price_fn = bs_put_price if option_type == "put" else bs_call_price
        short_target = -target_delta if option_type == "put" else target_delta
        wing_target = -wing_delta if option_type == "put" else wing_delta

        trades: List[SpreadTrade] = []
        for idx in step_indices:
            S0 = float(closes.iloc[idx])
            if S0 <= 0:
                continue
            sigma = _hv_30(closes, idx)
            if sigma is None:
                continue

            T = entry_dte / 365.0
            try:
                K_short = strike_fn(S0, T, r, sigma, target_delta=short_target)
                K_long = strike_fn(S0, T, r, sigma, target_delta=wing_target)
            except Exception:
                continue
            if option_type == "put":
                if not (0 < K_long < K_short < S0):
                    continue
            else:
                if not (S0 < K_short < K_long):
                    continue

            if surface is not None:
                rel_short, _ = surface.oi_collapsed_relative(
                    abs_delta=target_delta, dte=float(entry_dte))
                rel_long, _ = surface.oi_collapsed_relative(
                    abs_delta=wing_delta, dte=float(entry_dte))
            else:
                rel_short = SLIPPAGE_PCT / 2.0
                rel_long = SLIPPAGE_PCT / 2.0

            future = closes.iloc[idx + 1: idx + entry_dte + 1].values
            if len(future) < entry_dte // 2:
                continue

            pnl, exit_offset = simulate_vertical_pnl(
                S0, future, K_short, K_long, sigma, r, entry_dte, option_type,
                rel_short, rel_long, profit_target=profit_target,
                stop_mult=stop_mult)
            if not np.isfinite(pnl):
                continue

            hv_history = [_hv_30(closes, i)
                         for i in range(max(idx - 252, 30), idx, 5)]
            hv_history = [v for v in hv_history if v is not None]
            hv_pct_rank = (float(np.mean(np.array(hv_history) < sigma))
                          if hv_history else 0.5)
            hv_6m = [_hv_30(closes, i) for i in range(max(idx - 130, 30), idx, 5)]
            hv_6m = [v for v in hv_6m if v is not None]
            hv_ratio = sigma / max(float(np.mean(hv_6m)), 0.01) if hv_6m else 1.0
            rsi_window = closes.iloc[max(0, idx - 20):idx].diff().dropna()
            gains = rsi_window.clip(lower=0).rolling(14).mean()
            losses = (-rsi_window).clip(lower=0).rolling(14).mean()
            raw_g = float(gains.iloc[-1]) if len(gains) > 0 else 0
            raw_l = float(losses.iloc[-1]) if len(losses) > 0 else 1e-8
            avg_g = raw_g if math.isfinite(raw_g) else 0
            avg_l = raw_l if math.isfinite(raw_l) else 1e-8
            rsi = 100 - 100 / (1 + avg_g / max(avg_l, 1e-8))
            rsi_score = float(np.clip(rsi / 100.0, 0, 1))
            vol_window = vols.iloc[max(0, idx - 252):idx]
            vol_rank = (float(np.mean(vol_window < vols.iloc[idx]))
                       if len(vol_window) else 0.5)

            comp = compute_component_scores(
                S0, K_short, T, r, sigma,
                hv_pct_rank=hv_pct_rank,
                hv_ratio=float(np.clip(hv_ratio, 0.1, 5.0)),
                rsi_score=rsi_score, vol_rank=vol_rank,
                mode="short_put" if option_type == "put" else "long_call",
            ).copy()
            avg_rel = (rel_short + rel_long) / 2.0
            comp[_SPREAD_IDX] = float(np.clip(1.0 - avg_rel / 0.10, 0.0, 1.0))

            entry_prem = price_fn(S0, K_short, T, r, sigma) - price_fn(S0, K_long, T, r, sigma)
            received = (price_fn(S0, K_short, T, r, sigma) * (1 - rel_short)
                       - price_fn(S0, K_long, T, r, sigma) * (1 + rel_long))
            width = abs(K_short - K_long)
            credit_to_width = received / width if width > 0 else 0.0

            entry_date = closes.index[idx]
            exit_date = closes.index[idx + 1 + exit_offset]

            trades.append(SpreadTrade(
                symbol=symbol, entry_date=entry_date, exit_date=exit_date,
                pnl_pct=pnl, components=comp, credit_to_width=credit_to_width,
            ))

        return trades if len(trades) >= 5 else None
    except Exception:
        return None
