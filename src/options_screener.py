#!/usr/bin/env python3
"""
Options Screener (Top 5 low / 5 medium / 5 high by premium)

Features:
- Fetches options chains via yfinance (Yahoo Finance data; check terms).
- Scores contracts by liquidity (volume/OI), spread tightness, delta quality, and IV balance.
- Categorizes by premium into low/medium/high and picks top 5 in each.
- User-friendly prompts, input validation, and formatted console output.

Note:
- Not financial advice. For personal/informational use only.
- Data availability and timeliness depend on the data provider.
"""

import sys
import math
import os
import csv
import json
import logging
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
import functools
import random


# Dependency checks
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import yfinance as yf
except Exception:
    missing.append("yfinance")
try:
    import numpy as np
except Exception:
    missing.append("numpy")
if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

from .data_fetching import (
    get_underlying_price,
    get_risk_free_rate,
    get_vix_level,
    determine_vix_regime,
    get_market_context,
    fetch_options_yfinance,
    retry_with_backoff,
    get_dynamic_tickers
)
from .utils import (
    safe_float,
    norm_cdf,
    norm_pdf,
    bs_call,
    bs_put,
    bs_delta,
    bs_price,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    _d1d2,
    format_pct,
    format_money,
    determine_moneyness,
)
from .filters import (
    filter_options,
    categorize_by_premium,
    pick_top_per_bucket
)
from .paper_manager import PaperManager

# Enhanced CLI modules
try:
    from . import formatting as fmt
    from .trade_analysis import (
        generate_trade_thesis,
        calculate_entry_exit_levels,
        calculate_confidence_score,
        categorize_by_strategy,
        assess_risk_factors,
        format_trade_plan
    )
    from tqdm import tqdm
    HAS_ENHANCED_CLI = True
except ImportError as e:
    HAS_ENHANCED_CLI = False
    print(f"Enhanced CLI features unavailable: {e}")
    print("Install with: pip install colorama tqdm")

# Optional imports (relative to this package)
try:
    from .simulation import monte_carlo_pop
    HAS_SIMULATION = True
except ImportError:
    HAS_SIMULATION = False

try:
    from .visualize_results import create_visualizations
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False





def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file with fallback defaults."""
    default_config = {
        "weights": {
            "ev_score": 0.30,
            "liquidity": 0.20,
            "iv_advantage": 0.15,
            "pop": 0.15,
            "spread": 0.10,
            "delta_quality": 0.10
        },
        # New composite quality score weights (can be overridden in config.json)
        "composite_weights": {
            "pop": 0.18,
            "em_realism": 0.12,
            "rr": 0.15,
            "momentum": 0.10,
            "iv_rank": 0.10,
            "liquidity": 0.15,
            "catalyst": 0.05,
            "theta": 0.10,
            "ev": 0.05,
            "trader_pref": 0.10
        },
        "moneyness_band": 0.15,
        "target_delta": 0.40,
        "earnings_buffer_days": 5,
        "monte_carlo_simulations": 10000,
        "exit_rules": {
            "take_profit": 0.50,
            "stop_loss": -0.25
        }
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    except Exception:
        return default_config






def calculate_probability_of_profit(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray], premium: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate probability of profit at expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        premium = np.asanyarray(premium)
        
        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"
            
        # Break-even point
        breakeven = np.where(is_call, K + premium, K - premium)
        
        # Probability that stock will be beyond break-even
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (np.log(S / breakeven) - (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        
        pop = np.where(is_call, norm_cdf(d), 1.0 - norm_cdf(d))
        
        if np.isscalar(option_type) and np.isscalar(S):
            return float(pop)
        return pop
    except Exception:
        return None


def calculate_expected_move(S: Union[float, np.ndarray], sigma: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate expected move (1 standard deviation) until expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        sigma = np.asanyarray(sigma)
        T = np.asanyarray(T)
        move = S * sigma * np.sqrt(T)
        if move.ndim == 0:
            return float(move)
        return move
    except Exception:
        return None


def calculate_probability_of_touch(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate probability that option will touch the strike price before expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        
        scalar_input = isinstance(option_type, str) and S.ndim == 0

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        # Probability of touching is approximately 2 * delta for ATM options
        # More precise: P(touch) ≈ 2 * N(d2)
        with np.errstate(divide='ignore', invalid='ignore'):
            d2 = (np.log(S / K) - (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

        # Scalar fast-path: avoid boolean indexing on 0-d arrays
        if scalar_input:
            is_otm = (K > S) if is_call else (K < S)
            if is_otm:
                pot_val = 2 * norm_cdf(float(d2)) if is_call else 2 * (1.0 - norm_cdf(float(d2)))
                return float(np.clip(pot_val, 0.0, 1.0))
            return 1.0

        pot = np.ones_like(S, dtype=float)
        call_otm = is_call & (K > S)
        put_otm = (~is_call) & (K < S)
        pot[call_otm] = 2 * norm_cdf(d2[call_otm])
        pot[put_otm] = 2 * (1.0 - norm_cdf(d2[put_otm]))
        return np.clip(pot, 0.0, 1.0)
    except Exception:
        return None


def calculate_risk_reward(
    option_type: Union[str, np.ndarray],
    premium: Union[float, np.ndarray],
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    expected_move: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
    """Calculate max loss, break-even, and risk/reward ratio (Vectorized).

    Uses the prompt's definition:
      - target_price = stock_price ± 0.75 * EM
      - RR = max_gain_if_target_hit / premium
    where gains and premium are measured per share.
    """
    try:
        premium = np.asanyarray(premium)
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        
        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        max_loss = premium * 100  # Per contract

        # Break-even price
        breakeven = np.where(is_call, K + premium, K - premium)

        # Compute max gain at target using expected move when available
        if expected_move is not None:
            expected_move = np.asanyarray(expected_move)
            target_price = np.where(is_call, S + 0.75 * expected_move, S - 0.75 * expected_move)
            payoff_per_share = np.where(is_call, np.maximum(0.0, target_price - K), np.maximum(0.0, K - target_price))
        else:
            # Fallback: simple heuristic target if EM is unavailable
            target_price = np.where(is_call, S * 1.5, S * 0.5)
            payoff_per_share = np.where(is_call, np.maximum(0.0, target_price - K), np.maximum(0.0, K - target_price))

        max_gain_per_share = np.maximum(0.0, payoff_per_share - premium)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            risk_reward_ratio = np.where(premium > 0, max_gain_per_share / premium, 0.0)

        if premium.ndim == 0:
            return float(max_loss), float(breakeven), float(risk_reward_ratio)
            
        return max_loss, breakeven, risk_reward_ratio
    except Exception:
        return None, None, None


def calculate_metrics(
    df: pd.DataFrame,
    risk_free_rate: float,
    earnings_date: Optional[datetime],
    config: Dict,
    iv_rank: Optional[float],
    iv_percentile: Optional[float],
    sentiment_score: Optional[float],
    macro_risk_active: bool,
    sector_perf: Dict,
    tnx_change_pct: float
) -> pd.DataFrame:
    """Calculates all objective mathematical metrics and merges external data."""
    
    # --- Institutional Flow & Sentiment ---
    df["Vol_OI_Ratio"] = df["volume"] / df["openInterest"].replace(0, np.nan)
    df["Unusual_Whale"] = (df["Vol_OI_Ratio"] > 1.5) & (df["volume"] > 500)
    df["high_premium_turnover"] = (df["premium"] * df["volume"] * 100) > 25000

    def _sentiment_tag(score):
        if score is None or pd.isna(score):
            return "Neutral"
        if score > 0.05:
            return "Bullish"
        elif score < -0.05:
            return "Bearish"
        else:
            return "Neutral"

    df["sentiment_tag"] = df["sentiment_score"].apply(_sentiment_tag)

    # --- Earnings Volatility Logic ---
    df["Earnings Play"] = "NO"
    if earnings_date:
        df.loc[(df["exp_dt"] > earnings_date), "Earnings Play"] = "YES"

    df["is_underpriced"] = False
    earnings_mask = df["Earnings Play"] == "YES"
    if earnings_mask.any():
        df.loc[earnings_mask, "is_underpriced"] = df.loc[earnings_mask, "impliedVolatility"] < df.loc[earnings_mask, "hv_30d"]

    # --- Trend Alignment Filter ---
    # Require price above BOTH SMA-20 and SMA-50 for calls (and below both for puts)
    # to confirm a genuine medium-term trend rather than just short-term noise.
    df["Trend_Aligned"] = False
    has_sma50 = "sma_50" in df.columns and df["sma_50"].notna().any()
    if has_sma50:
        df.loc[
            (df["type"] == "call") & (df["underlying"] > df["sma_20"]) & (df["underlying"] > df["sma_50"]),
            "Trend_Aligned"
        ] = True
        df.loc[
            (df["type"] == "put") & (df["underlying"] < df["sma_20"]) & (df["underlying"] < df["sma_50"]),
            "Trend_Aligned"
        ] = True
    else:
        df.loc[(df["type"] == "call") & (df["underlying"] > df["sma_20"]), "Trend_Aligned"] = True
        df.loc[(df["type"] == "put") & (df["underlying"] < df["sma_20"]), "Trend_Aligned"] = True

    # --- VECTORIZED GREEKS ---
    S_vals = df["underlying"].values
    K_vals = df["strike"].values
    T_vals = df["T_years"].values
    IV_vals = np.maximum(1e-9, df["impliedVolatility"].values)
    types_vals = df["type"].values

    df["delta"] = bs_delta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    df["abs_delta"] = np.abs(df["delta"].values)
    df["gamma"] = bs_gamma(S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    df["vega"] = bs_vega(S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    df["theta"] = bs_theta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    df["rho"] = bs_rho(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals)

    # --- ADVANCED METRICS ---
    df["expected_move"] = calculate_expected_move(S_vals, IV_vals, T_vals)
    is_call = np.char.lower(types_vals.astype(str)) == "call"

    # Probability of Profit: breakeven-based formula P(S_T > K+prem) for calls,
    # P(S_T < K-prem) for puts — correctly accounts for premium cost unlike 1-delta.
    prem_vals = df["premium"].values
    pop_arr = calculate_probability_of_profit(types_vals, S_vals, K_vals, T_vals, IV_vals, prem_vals)
    if pop_arr is None:
        pop_arr = 1.0 - df["abs_delta"].values
    df["prob_profit"] = np.clip(pop_arr, 0.0, 1.0)

    df["prob_touch"] = calculate_probability_of_touch(types_vals, S_vals, K_vals, T_vals, IV_vals)
    max_loss, breakeven, rr_ratio = calculate_risk_reward(types_vals, prem_vals, S_vals, K_vals, df["expected_move"].values)
    df["max_loss"] = max_loss
    df["breakeven"] = breakeven
    df["rr_ratio"] = rr_ratio

    # Break-even realism
    be_vals = np.where(is_call, K_vals + prem_vals, K_vals - prem_vals)
    req_move = np.where(is_call, np.maximum(0.0, be_vals - S_vals), np.maximum(0.0, S_vals - be_vals))
    em = df["expected_move"].values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(em > 0, req_move / em, np.nan)
    em_realism = np.full_like(ratio, 0.5)
    em_realism[ratio <= 0.5] = 1.0
    em_realism[(ratio > 0.5) & (ratio <= 1.0)] = 0.7
    em_realism[ratio > 1.0] = np.maximum(0.1, em[ratio > 1.0] / (req_move[ratio > 1.0] + 1e-9))
    df["required_move"] = req_move
    df["em_realism_score"] = em_realism

    # Theta Decay Pressure
    dte_vals = np.maximum(df["T_years"].values * 365.0, 1.0)
    tdp_raw = (df["premium"].values * 100.0) / dte_vals
    df["theta_decay_pressure"] = tdp_raw / np.maximum(df["abs_delta"].values, 0.1)

    # IV vs HV comparison
    if "hv_30d" in df.columns and df["hv_30d"].notna().any():
        df["iv_vs_hv"] = df["impliedVolatility"] - df["hv_30d"]
        df["iv_hv_ratio"] = df["impliedVolatility"] / df["hv_30d"].replace(0, float('nan'))
    else:
        df["iv_vs_hv"] = 0.0
        df["iv_hv_ratio"] = 1.0
    
    # --- OI Wall Detection (Optimized) ---
    df["oi_wall_warning"] = ""

    # Find max OI strikes per expiration and type using vectorized groupby
    max_oi_idx = df.groupby(["expiration", "type"])["openInterest"].idxmax()

    for (expiry, opt_type), idx in max_oi_idx.items():
        if pd.isna(idx):
            continue

        wall_strike = df.loc[idx, "strike"]

        # Get all strikes for this expiration and type, sorted
        mask = (df["expiration"] == expiry) & (df["type"] == opt_type)
        strikes_sorted = df.loc[mask, "strike"].sort_values().unique()

        # Find adjacent strike to the wall
        wall_pos = np.searchsorted(strikes_sorted, wall_strike)

        if opt_type == "call":
            # For calls, warn at wall and one strike below
            adjacent_strike = strikes_sorted[wall_pos - 1] if wall_pos > 0 else None
            warning_strikes = [wall_strike] + ([adjacent_strike] if adjacent_strike is not None else [])
            warning_mask = mask & df["strike"].isin(warning_strikes)
            df.loc[warning_mask, "oi_wall_warning"] = "LIMITED UPSIDE"
        else:  # put
            # For puts, warn at wall and one strike above
            adjacent_strike = strikes_sorted[wall_pos + 1] if wall_pos < len(strikes_sorted) - 1 else None
            warning_strikes = [wall_strike] + ([adjacent_strike] if adjacent_strike is not None else [])
            warning_mask = mask & df["strike"].isin(warning_strikes)
            df.loc[warning_mask, "oi_wall_warning"] = "LIMITED DOWNSIDE"

    # IV Skew
    df["iv_skew"] = np.nan
    for (exp, strike), group in df.groupby(["expiration", "strike"]):
        if len(group) == 2:
            call_iv = group[group["type"] == "call"]["impliedVolatility"].values
            put_iv = group[group["type"] == "put"]["impliedVolatility"].values
            if len(call_iv) > 0 and len(put_iv) > 0:
                df.loc[group.index, "iv_skew"] = put_iv[0] - call_iv[0]
    df["iv_skew"] = df["iv_skew"].ffill().bfill().fillna(0.0)

    # IV Skew Directional Alignment
    # Positive skew (put IV > call IV) = market hedging downside → favour puts
    # Negative/flat skew = normal regime → favours calls
    skew_vals = df["iv_skew"].values
    df["skew_alignment_score"] = np.where(
        df["type"] == "call",
        np.clip(0.5 - skew_vals * 4.0, 0.0, 1.0),   # calls: better when skew is low/negative
        np.clip(0.5 + skew_vals * 4.0, 0.0, 1.0),   # puts:  better when skew is positive
    )

    # Gamma/Theta Efficiency: explosive payoff potential per unit of daily time decay
    # Higher ratio = more leverage per dollar of daily premium bleed
    df["gamma_theta_ratio"] = np.abs(df["gamma"].values) / np.maximum(np.abs(df["theta"].values), 1e-9)

    # Flags
    df["liquidity_flag"] = "GOOD"
    df.loc[(df["volume"] < 10) & (df["openInterest"] < 100), "liquidity_flag"] = "POOR"
    df.loc[(df["volume"] >= 10) & (df["volume"] < 50) & (df["openInterest"] >= 100) & (df["openInterest"] < 500), "liquidity_flag"] = "FAIR"
    df["spread_flag"] = "OK"
    df.loc[df["spread_pct"] > 0.10, "spread_flag"] = "WIDE"
    df.loc[df["spread_pct"] > 0.20, "spread_flag"] = "VERY_WIDE"
    
    # External data
    df["iv_rank"] = iv_rank if iv_rank is not None else pd.NA
    df["iv_percentile"] = iv_percentile if iv_percentile is not None else pd.NA
    df["event_flag"] = "OK"
    if earnings_date is not None:
        eb_days = config.get("earnings_buffer_days", 5)
        for idx, row in df.iterrows():
            if pd.notna(row["exp_dt"]):
                days_to_e = abs((row["exp_dt"].replace(tzinfo=None) - earnings_date.replace(tzinfo=None)).days)
                if days_to_e <= eb_days: df.at[idx, "event_flag"] = "EARNINGS_NEARBY"
    
    # Monte Carlo
    if HAS_SIMULATION:
        n_sims = config.get("monte_carlo_simulations", 10000)
        def _calc_mc_pop(row):
            pop_sim, pot_sim = monte_carlo_pop(S=safe_float(row["underlying"]), K=safe_float(row["strike"]), T=safe_float(row["T_years"]), sigma=safe_float(row["impliedVolatility"]), r=risk_free_rate, premium=safe_float(row["premium"]), option_type=row["type"], n_simulations=n_sims)
            return pd.Series({"pop_sim": pop_sim, "pot_sim": pot_sim})
        mc_res = df.apply(_calc_mc_pop, axis=1)
        df["pop_sim"], df["pot_sim"] = mc_res["pop_sim"], mc_res["pot_sim"]
    else:
        df["pop_sim"], df["pot_sim"] = pd.NA, pd.NA

    # Blend MC PoP (60%) with analytical PoP (40%) when simulation data is available.
    # MC captures path-dependency and jump risk; analytical gives a stable floor.
    if HAS_SIMULATION:
        mc_valid = df["pop_sim"].notna()
        if mc_valid.any():
            df.loc[mc_valid, "prob_profit"] = (
                0.6 * df.loc[mc_valid, "pop_sim"].astype(float)
                + 0.4 * df.loc[mc_valid, "prob_profit"]
            ).clip(0.0, 1.0)

    # Theoretical value and P(ITM) using market IV (for display/reference)
    d1, d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    p_itm = np.where(is_call, norm_cdf(d2), norm_cdf(-d2))
    with np.errstate(divide='ignore', invalid='ignore'):
        theo_payoff = np.where(is_call, S_vals * np.exp(risk_free_rate * T_vals) * norm_cdf(d1) - K_vals * norm_cdf(d2), K_vals * norm_cdf(-d2) - S_vals * np.exp(risk_free_rate * T_vals) * norm_cdf(-d1))
    df["p_itm"], df["theo_value"] = p_itm, theo_payoff

    # HV-adjusted EV: BS(realized_vol) - market_price
    # Positive = options cheap vs realized vol (edge for buyers)
    # Negative = options expensive vs realized vol (edge for sellers)
    hv_arr = np.maximum(df["hv_30d"].fillna(df["impliedVolatility"]).values, 1e-9)
    hv_d1, hv_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, hv_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        hv_payoff = np.where(is_call,
            S_vals * np.exp(risk_free_rate * T_vals) * norm_cdf(hv_d1) - K_vals * norm_cdf(hv_d2),
            K_vals * norm_cdf(-hv_d2) - S_vals * np.exp(risk_free_rate * T_vals) * norm_cdf(-hv_d1))
    df["ev_per_contract"] = 100.0 * (hv_payoff - prem_vals) - (100.0 * prem_vals * df["spread_pct"].fillna(0.0).values)

    # Warnings
    df["Theta_Burn_Rate"] = np.where(df["premium"] > 0, np.abs(df["theta"].values) / df["premium"].values, 0.0)
    df["decay_warning"] = df["Theta_Burn_Rate"] > 0.06
    df["sr_warning"] = ""
    df.loc[(df["type"] == "call") & (df["underlying"] > df["high_20"] * 0.98), "sr_warning"] = "NEAR RESISTANCE"
    df.loc[(df["type"] == "put") & (df["underlying"] < df["low_20"] * 1.02), "sr_warning"] = "NEAR SUPPORT"

    # Professional Filters
    df["macro_warning"] = "⛔ MACRO RISK" if macro_risk_active else ""
    df["max_pain_warning"] = ""
    if sector_perf:
        stock_ret, sector_ret = sector_perf.get("ticker_return", 0.0), sector_perf.get("sector_return", 0.0)
        if "max_pain" in df.columns:
            mp, und, dte = pd.to_numeric(df["max_pain"], errors='coerce'), pd.to_numeric(df["underlying"], errors='coerce'), pd.to_numeric(df["T_years"], errors='coerce') * 365.0
            mask_mp = mp.notna() & und.notna() & (dte < 3)
            df.loc[mask_mp & ((und - mp).abs() / mp > 0.05), "max_pain_warning"] = "⚠️ FIGHTING MAX PAIN"
        if stock_ret > 0 and sector_ret < -0.015:
            df["macro_warning"] = np.where(df["macro_warning"] != "", df["macro_warning"] + " | FAKE-OUT DIVERGENCE", "FAKE-OUT DIVERGENCE")
    RATE_SENSITIVE = {"QQQ", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX"}
    if tnx_change_pct > 0.025:
        df["yield_warning"] = np.where(df["symbol"].isin(RATE_SENSITIVE), "📉 RATES UP", "")
    else:
        df["yield_warning"] = ""
    
    return df


def calculate_scores(
    df: pd.DataFrame,
    config: Dict,
    vix_regime_weights: Dict,
    trader_profile: str,
    mode: str,
    min_dte: int,
    max_dte: int
) -> pd.DataFrame:
    """Calculates subjective quality scores using normalization and weights."""
    
    def rank_norm(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1: return pd.Series([0.5] * n, index=s.index)
        r = s.rank(method="average", na_option="keep")
        return (r - 1.0) / (n - 1.0)

    # Base features
    vol_n, oi_n = rank_norm(df["volume"].fillna(0)), rank_norm(df["openInterest"].fillna(0))
    sp_cap = config.get("spread_score_cap", 0.25)
    sp = df["spread_pct"].replace([pd.NA, pd.NaT], float("inf")).clip(lower=0, upper=sp_cap)
    spread_score = 1.0 - (sp / sp_cap)
    d_target = config.get("target_delta", 0.40)
    delta_quality = (1.0 - (df["abs_delta"] - d_target).abs() / max(d_target, 1e-6)).clip(0, 1)
    iv_n = rank_norm(df["impliedVolatility"].fillna(df["impliedVolatility"].median()))
    iv_quality = 1.0 - (2.0 * (iv_n - 0.5).abs())
    liquidity = 0.5 * (vol_n + oi_n)
    pop_score = df["prob_profit"].fillna(0.5).clip(0, 1)
    rr_raw = pd.to_numeric(df["rr_ratio"], errors='coerce').fillna(0.0)
    # Smooth linear mapping [0.5 → 0, 4.0 → 1] instead of hard step thresholds
    rr_score = np.clip((rr_raw - 0.5) / 3.5, 0.0, 1.0)
    ev_score = rank_norm(df["ev_per_contract"].fillna(df["ev_per_contract"].median()))
    em_realism_score = pd.to_numeric(df["em_realism_score"], errors='coerce').fillna(0.5).clip(0, 1)
    theta_raw = df["theta_decay_pressure"].replace([pd.NA, pd.NaT], np.nan)
    theta_score = (1.0 - rank_norm(theta_raw.fillna(theta_raw.median()))).clip(0, 1)
    theta_score = theta_score.where((df["T_years"] * 365.0) > 7, theta_score * 0.7)
    
    ret_score = rank_norm(pd.to_numeric(df.get("ret_5d", pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0))
    rsi_vals = pd.to_numeric(df.get("rsi_14", pd.Series(np.nan, index=df.index)), errors="coerce")
    rsi_score = (1.0 - (abs((rsi_vals - 50.0) / 50.0))).clip(0, 1)
    atr_score = rank_norm(pd.to_numeric(df.get("atr_trend", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0))
    momentum_score = 0.4 * ret_score.fillna(0.5) + 0.3 * rsi_score.fillna(0.5) + 0.3 * atr_score.fillna(0.5)
    
    iv_pct_series = pd.to_numeric(df.get("iv_percentile_30", df.get("iv_percentile", pd.Series(np.nan, index=df.index))), errors="coerce")
    iv_rank_score = iv_pct_series.clip(0, 1).fillna(0.5) if mode == "Premium Selling" else (1.0 - iv_pct_series.clip(0, 1)).fillna(0.5)
    catalyst_score = pd.Series(0.3, index=df.index).mask(df["event_flag"] == "EARNINGS_NEARBY", 0.8)
    dte_norm = ((df["T_years"] * 365.0 - min_dte) / max(1, (max_dte - min_dte))).clip(0, 1)
    trader_pref_score = (0.6 * liquidity + 0.4 * spread_score) if trader_profile.lower().startswith("day") else (0.5 * delta_quality + 0.5 * dte_norm)

    # IV Edge Score: rewards options where market IV is below realized HV (cheap options for buyers).
    # For premium sellers the logic is flipped — high IV vs HV is the edge.
    iv_vs_hv = df.get("iv_vs_hv", pd.Series(0.0, index=df.index)).fillna(0.0)
    if mode == "Premium Selling":
        iv_edge_score = ((iv_vs_hv.clip(-0.2, 0.2) + 0.2) / 0.4).clip(0, 1)
    else:
        iv_edge_score = ((-iv_vs_hv.clip(-0.2, 0.2) + 0.2) / 0.4).clip(0, 1)

    # IV Skew Directional Alignment (computed in calculate_metrics)
    skew_align_score = pd.to_numeric(
        df.get("skew_alignment_score", pd.Series(0.5, index=df.index)), errors='coerce'
    ).fillna(0.5).clip(0, 1)

    # Gamma/Theta Efficiency (rank-normalised, capped at 95th pct to handle outliers)
    gt_raw = pd.to_numeric(
        df.get("gamma_theta_ratio", pd.Series(0.0, index=df.index)), errors='coerce'
    ).fillna(0.0)
    gt_cap = gt_raw.quantile(0.95) if len(gt_raw) > 10 else gt_raw.max()
    gamma_theta_score = rank_norm(gt_raw.clip(upper=max(gt_cap, 1e-9))).fillna(0.5)

    # Weight Application
    if mode == "Premium Selling":
        weights = config.get("premium_selling_weights", {})
        ror_score = rank_norm(df["return_on_risk"].fillna(df["return_on_risk"].median()))
        w = {k: weights.get(k, 0.0) for k in ["pop", "return_on_risk", "iv_rank", "liquidity", "theta", "ev", "trader_pref"]}
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (w["pop"]*pop_score + w["return_on_risk"]*ror_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity + w["theta"]*theta_score + w["ev"]*ev_score + w["trader_pref"]*trader_pref_score) / w_sum
    else:
        dw = {
            "pop": 0.20, "em_realism": 0.10, "rr": 0.12, "momentum": 0.08,
            "iv_rank": 0.07, "liquidity": 0.12, "catalyst": 0.04, "theta": 0.08,
            "ev": 0.10, "trader_pref": 0.07, "iv_edge": 0.08, "skew_align": 0.05, "gamma_theta": 0.04
        }
        cw = config.get("composite_weights", {}) or {}
        w = {k: cw.get(k, dw[k]) for k in dw}
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (
            w["pop"]*pop_score + w["em_realism"]*em_realism_score + w["rr"]*rr_score
            + w["momentum"]*momentum_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity
            + w["catalyst"]*catalyst_score + w["theta"]*theta_score + w["ev"]*ev_score
            + w["trader_pref"]*trader_pref_score + w["iv_edge"]*iv_edge_score
            + w["skew_align"]*skew_align_score + w["gamma_theta"]*gamma_theta_score
        ) / w_sum

    # Adjustments
    df.loc[df["event_flag"] == "EARNINGS_NEARBY", "quality_score"] -= 0.05
    # Reward earnings plays where IV is actually underpriced vs realized vol
    if "Earnings Play" in df.columns and "is_underpriced" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["is_underpriced"] == True), "quality_score"] += 0.08
    df.loc[df["Trend_Aligned"] == True, "quality_score"] += 0.15
    df.loc[df["decay_warning"] == True, "quality_score"] -= 0.20
    df.loc[df["sr_warning"] != "", "quality_score"] -= 0.10
    if "seasonal_win_rate" in df.columns:
        df.loc[df["seasonal_win_rate"] >= 0.8, "quality_score"] += 0.10
        df.loc[df["seasonal_win_rate"] <= 0.2, "quality_score"] -= 0.10
    df.loc[df["oi_wall_warning"] != "", "quality_score"] -= 0.10
    df["squeeze_play"] = (df["is_squeezing"] == True) & (df["Unusual_Whale"] == True)
    df.loc[df["squeeze_play"], "quality_score"] += 0.25
    if df["macro_warning"].str.contains("MACRO RISK").any(): df["quality_score"] -= 0.10
    
    # Save components
    df["quality_score"] = df["quality_score"].clip(0, 1)
    df["ev_score"] = ev_score
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"], df["delta_quality"], df["iv_quality"] = liquidity, delta_quality, iv_quality
    df["spread_score"], df["theta_score"], df["momentum_score"] = spread_score, theta_score, momentum_score
    df["iv_rank_score"], df["catalyst_score"] = iv_rank_score, catalyst_score
    df["iv_advantage_score"] = iv_edge_score  # mode-aware: buyers rewarded for IV < HV
    
    return df


def enrich_and_score(
    df: pd.DataFrame,
    min_dte: int,
    max_dte: int,
    risk_free_rate: float,
    config: Dict,
    vix_regime_weights: Dict,
    trader_profile: str = "swing",
    mode: str = "Single-stock",
    iv_rank: Optional[float] = None,
    iv_percentile: Optional[float] = None,
    earnings_date: Optional[datetime] = None,
    sentiment_score: Optional[float] = None,
    seasonal_win_rate: Optional[float] = None,
    term_structure_spread: Optional[float] = None,
    macro_risk_active: bool = False,
    sector_perf: Dict = {},
    tnx_change_pct: float = 0.0,
) -> pd.DataFrame:
    # Prepare
    now = datetime.now(timezone.utc)
    df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
    df = df[df["exp_dt"].notna()].copy()
    df["T_years"] = (df["exp_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    df = df[(df["T_years"] > min_dte / 365.0) & (df["T_years"] < max_dte / 365.0)].copy()

    for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "underlying"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    mb = config.get("moneyness_band", 0.15)
    if "underlying" in df.columns and "strike" in df.columns:
        df = df[(df["strike"] >= df["underlying"] * (1 - mb)) & (df["strike"] <= df["underlying"] * (1 + mb))].copy()

    # Only use valid bid/ask prices (> 0), otherwise fall back to lastPrice
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")

    # Calculate mid only when both bid and ask are valid (> 0)
    valid_bid = (df["bid"].notna()) & (df["bid"] > 0)
    valid_ask = (df["ask"].notna()) & (df["ask"] > 0)
    valid_quotes = valid_bid & valid_ask

    df["mid"] = np.where(valid_quotes, (df["bid"] + df["ask"]) / 2.0, np.nan)
    df["premium"] = df["mid"].where(df["mid"].notna() & (df["mid"] > 0.0), df["lastPrice"])

    # For spread calculation, set bid/ask to NaN if invalid (filled later)
    df.loc[~valid_bid, "bid"] = np.nan
    df.loc[~valid_ask, "ask"] = np.nan

    if mode == "Premium Selling":
        df = df[df['type'] == 'put'].copy()
        if df.empty: return df
        df['return_on_risk'] = df['premium'] / df['strike']

    df = df[(df["premium"].notna()) & (df["premium"] > 0)].copy()
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    valid_spread = pd.to_numeric(df["spread_pct"], errors='coerce').notna() & np.isfinite(df["spread_pct"].astype(float))
    df.loc[~valid_spread, "spread_pct"] = float("inf")

    fc = config.get("filters", {})
    df = df[df["spread_pct"] <= fc.get("max_bid_ask_spread_pct", 0.40)].copy()
    df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors='coerce').fillna(0)
    df = df[(df["volume"] >= fc.get("min_volume", 50)) | (df["openInterest"] >= fc.get("min_open_interest", 10))].copy()

    if df.empty: return df

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors='coerce')
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    ov_iv_m = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(ov_iv_m if pd.notna(ov_iv_m) else 0.25)

    # 1. Call Helper: Metrics
    df = calculate_metrics(df, risk_free_rate, earnings_date, config, iv_rank, iv_percentile, sentiment_score, macro_risk_active, sector_perf, tnx_change_pct)

    # 2. Call Helper: Scores
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte)

    # Final Filters
    d_min, d_max = (0.20, 0.40) if mode == "Premium Selling" else (fc.get("delta_min", 0.15), fc.get("delta_max", 0.35))
    df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    if mode != "Premium Selling": df = df[df["rr_ratio"] >= 0.25].copy()

    if df.empty: return df
    
    # Sorting
    df = df.sort_values(["Unusual_Whale", "quality_score", "volume", "openInterest"], ascending=[False, False, False, False]).reset_index(drop=True)
    return df



def find_vertical_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies vertical spreads from a DataFrame of single options.
    """
    spreads = []

    # Identify "Buy" candidates
    buy_candidates = df[df["quality_score"] > 0.7].copy()

    for _, buy_leg in buy_candidates.iterrows():
        # Find potential "Sell" candidates in the same expiry
        if buy_leg["type"] == "call":
            sell_candidates = df[
                (df["expiration"] == buy_leg["expiration"]) &
                (df["type"] == buy_leg["type"]) &
                (df["symbol"] == buy_leg["symbol"]) &
                (df["strike"] > buy_leg["strike"]) & # OTM
                (df["strike"] <= buy_leg["strike"] + 2) # 1 or 2 strikes away
            ]
        else: # Put
            sell_candidates = df[
                (df["expiration"] == buy_leg["expiration"]) &
                (df["type"] == buy_leg["type"]) &
                (df["symbol"] == buy_leg["symbol"]) &
                (df["strike"] < buy_leg["strike"]) & # OTM
                (df["strike"] >= buy_leg["strike"] - 2) # 1 or 2 strikes away
            ]

        for _, sell_leg in sell_candidates.iterrows():
            if sell_leg["openInterest"] > 0 and sell_leg["volume"] > 0:
                spread_cost = buy_leg["premium"] - sell_leg["premium"]
                strike_width = abs(sell_leg["strike"] - buy_leg["strike"])
                max_profit = strike_width - spread_cost
                risk = spread_cost

                if risk > 0 and max_profit > 1.5 * risk:
                    spreads.append({
                    "symbol": buy_leg["symbol"],
                    "type": f"{buy_leg['type'].upper()} Spread",
                    "long_strike": buy_leg["strike"],
                    "short_strike": sell_leg["strike"],
                    "expiration": buy_leg["expiration"],
                    "spread_cost": spread_cost,
                    "max_profit": max_profit,
                    "risk": risk,
                    "underlying": buy_leg["underlying"]
                })

    return pd.DataFrame(spreads) if spreads else pd.DataFrame()


def find_credit_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies high-probability Bull Put and Bear Call credit spreads.
    """
    spreads = []

    # --- Bull Put Spreads (Sell a Put, Buy a lower Put) ---
    # Short leg candidates: Delta between -0.15 and -0.40 (Relaxed)
    put_df = df[df['type'] == 'put'].copy()
    short_put_candidates = put_df[
        (put_df['delta'] >= -0.40) & (put_df['delta'] <= -0.15)
    ].copy()

    for _, short_leg in short_put_candidates.iterrows():
        # Find potential long legs (protection) 1 or 2 strikes lower
        # Find potential long legs (protection) 1 or 2 strikes lower
        strikes = sorted(put_df[
            (put_df['expiration'] == short_leg['expiration']) &
            (put_df['symbol'] == short_leg['symbol'])
        ]['strike'].unique(), reverse=True)

        try:
            current_strike_index = strikes.index(short_leg['strike'])
        except ValueError:
            continue

        potential_long_strikes = []
        if current_strike_index + 1 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 1])
        if current_strike_index + 2 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 2])

        long_leg_candidates = put_df[
            (put_df['expiration'] == short_leg['expiration']) &
            (put_df['symbol'] == short_leg['symbol']) &
            (put_df['strike'].isin(potential_long_strikes))
        ]

        for _, long_leg in long_leg_candidates.iterrows():
            strike_width = short_leg['strike'] - long_leg['strike']
            net_credit = short_leg['premium'] - long_leg['premium']

            # Profitability Filter: Net Credit > 0.20 * Strike Width (Relaxed)
            if net_credit > (0.20 * strike_width):
                spreads.append({
                    "symbol": short_leg['symbol'],
                    "type": "Bull Put",
                    "short_strike": short_leg['strike'],
                    "long_strike": long_leg['strike'],
                    "expiration": short_leg['expiration'],
                    "net_credit": net_credit,
                    "max_profit": net_credit * 100,
                    "max_loss": (strike_width - net_credit) * 100,
                    "quality_score": (short_leg['quality_score'] + long_leg['quality_score']) / 2
                })

    # --- Bear Call Spreads (Sell a Call, Buy a higher Call) ---
    call_df = df[df['type'] == 'call'].copy()
    # Short leg candidates: Delta between 0.15 and 0.40 (Relaxed)
    short_call_candidates = call_df[
        (call_df['delta'] >= 0.15) & (call_df['delta'] <= 0.40)
    ].copy()

    for _, short_leg in short_call_candidates.iterrows():
        # Find potential long legs (protection) 1 or 2 strikes higher
        strikes = sorted(call_df[
            (call_df['expiration'] == short_leg['expiration']) &
            (call_df['symbol'] == short_leg['symbol'])
        ]['strike'].unique())

        try:
            current_strike_index = strikes.index(short_leg['strike'])
        except ValueError:
            continue

        potential_long_strikes = []
        if current_strike_index + 1 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 1])
        if current_strike_index + 2 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 2])

        long_leg_candidates = call_df[
            (call_df['expiration'] == short_leg['expiration']) &
            (call_df['symbol'] == short_leg['symbol']) &
            (call_df['strike'].isin(potential_long_strikes))
        ]

        for _, long_leg in long_leg_candidates.iterrows():
            strike_width = long_leg['strike'] - short_leg['strike']
            net_credit = short_leg['premium'] - long_leg['premium']

            # Profitability Filter: Net Credit > 0.20 * Strike Width (Relaxed)
            if net_credit > (0.20 * strike_width):
                spreads.append({
                    "symbol": short_leg['symbol'],
                    "type": "Bear Call",
                    "short_strike": short_leg['strike'],
                    "long_strike": long_leg['strike'],
                    "expiration": short_leg['expiration'],
                    "net_credit": net_credit,
                    "max_profit": net_credit * 100,
                    "max_loss": (strike_width - net_credit) * 100,
                    "quality_score": (short_leg['quality_score'] + long_leg['quality_score']) / 2
                })

    return pd.DataFrame(spreads).sort_values(by="quality_score", ascending=False) if spreads else pd.DataFrame()


def find_iron_condors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies Iron Condor opportunities (Bull Put Spread + Bear Call Spread).
    
    An Iron Condor sells premium on both sides expecting the stock to stay range-bound.
    Includes strict liquidity requirements and delta neutrality checks.
    """
    condors = []
    
    # Separate puts and calls
    put_df = df[df['type'] == 'put'].copy()
    call_df = df[df['type'] == 'call'].copy()
    
    # Strict liquidity filter for all legs (volume > 50, OI > 500)
    put_df = put_df[(put_df['volume'] > 50) & (put_df['openInterest'] > 500)].copy()
    call_df = call_df[(call_df['volume'] > 50) & (call_df['openInterest'] > 500)].copy()
    
    if put_df.empty or call_df.empty:
        return pd.DataFrame()
    
    # Group by symbol and expiration
    for (symbol, exp), group_data in df.groupby(['symbol', 'expiration']):
        put_group = put_df[(put_df['symbol'] == symbol) & (put_df['expiration'] == exp)]
        call_group = call_df[(call_df['symbol'] == symbol) & (call_df['expiration'] == exp)]
        
        if put_group.empty or call_group.empty:
            continue
        
        # --- PUT WING (Bull Put Spread) ---
        # Short Put: Delta between -0.30 and -0.20
        short_put_candidates = put_group[
            (put_group['delta'] >= -0.30) & (put_group['delta'] <= -0.20)
        ].copy()
        
        best_put_spread = None
        best_put_credit = 0
        
        for _, short_put in short_put_candidates.iterrows():
            # Long Put: abs(delta) < 0.15 (closer to 0, further OTM) AND lower strike
            # This ensures the long put is a protective wing, not ITM
            long_put_candidates = put_group[
                (put_group['delta'].abs() < 0.15) &
                (put_group['strike'] < short_put['strike'])
            ]
            
            for _, long_put in long_put_candidates.iterrows():
                put_width = short_put['strike'] - long_put['strike']
                put_credit = short_put['premium'] - long_put['premium']
                
                if put_credit > best_put_credit and put_credit > 0:
                    best_put_credit = put_credit
                    best_put_spread = {
                        'short_put': short_put,
                        'long_put': long_put,
                        'put_width': put_width,
                        'put_credit': put_credit
                    }
        
        if not best_put_spread:
            continue
        
        # --- CALL WING (Bear Call Spread) ---
        # Short Call: Delta between 0.20 and 0.30
        short_call_candidates = call_group[
            (call_group['delta'] >= 0.20) & (call_group['delta'] <= 0.30)
        ].copy()
        
        best_call_spread = None
        best_call_credit = 0
        
        for _, short_call in short_call_candidates.iterrows():
            # Long Call: Delta < 0.15 (further OTM) AND higher strike
            long_call_candidates = call_group[
                (call_group['delta'] < 0.15) &
                (call_group['strike'] > short_call['strike'])
            ]
            
            for _, long_call in long_call_candidates.iterrows():
                call_width = long_call['strike'] - short_call['strike']
                call_credit = short_call['premium'] - long_call['premium']
                
                if call_credit > best_call_credit and call_credit > 0:
                    best_call_credit = call_credit
                    best_call_spread = {
                        'short_call': short_call,
                        'long_call': long_call,
                        'call_width': call_width,
                        'call_credit': call_credit
                    }
        
        if not best_call_spread:
            continue
        
        # --- COMBINE AND FILTER ---
        total_credit = best_put_spread['put_credit'] + best_call_spread['call_credit']
        max_width = max(best_put_spread['put_width'], best_call_spread['call_width'])
        max_risk = max_width - total_credit
        
        # Filter: Must collect at least 1/5 of the width as credit (relaxed from 1/3)
        min_credit = 0.20 * max_width
        if total_credit <= min_credit or max_risk <= 0:
            continue
        
        # Delta Neutrality Check: abs(short_put_delta + short_call_delta) < 0.10
        short_put_delta = best_put_spread['short_put']['delta']
        short_call_delta = best_call_spread['short_call']['delta']
        net_delta = short_put_delta + short_call_delta
        
        if abs(net_delta) >= 0.10:
            continue  # Too directional
        
        # Calculate metrics
        return_on_risk = total_credit / max_risk if max_risk > 0 else 0
        avg_quality = (
            best_put_spread['short_put']['quality_score'] +
            best_put_spread['long_put']['quality_score'] +
            best_call_spread['short_call']['quality_score'] +
            best_call_spread['long_call']['quality_score']
        ) / 4
        
        condors.append({
            'symbol': symbol,
            'expiration': exp,
            'short_put_strike': best_put_spread['short_put']['strike'],
            'long_put_strike': best_put_spread['long_put']['strike'],
            'short_call_strike': best_call_spread['short_call']['strike'],
            'long_call_strike': best_call_spread['long_call']['strike'],
            'put_credit': best_put_spread['put_credit'],
            'call_credit': best_call_spread['call_credit'],
            'total_credit': total_credit,
            'max_width': max_width,
            'max_risk': max_risk * 100,  # Per contract
            'return_on_risk': return_on_risk,
            'net_delta': net_delta,
            'quality_score': avg_quality
        })
    
    return pd.DataFrame(condors).sort_values(by="return_on_risk", ascending=False) if condors else pd.DataFrame()



def format_analysis_row(row: pd.Series, chain_iv_median: float, mode: str) -> str:
    """Formats the second detail row for the screener report (analysis)."""
    parts = []

    # IV context
    iv = row.get("impliedVolatility", pd.NA)
    if pd.notna(iv) and math.isfinite(iv):
        rel = "≈" if abs(float(iv) - chain_iv_median) <= 0.02 else ("above" if iv > chain_iv_median else "below")
        parts.append(f"IV: {format_pct(iv)} ({rel} median)")

    # Probability of Profit
    pop = row.get("prob_profit", pd.NA)
    if pd.notna(pop) and math.isfinite(pop):
        parts.append(f"PoP: {format_pct(pop)}")

    # Risk/Reward or Return on Risk
    if mode == "Premium Selling":
        ror = row.get("return_on_risk", pd.NA)
        if pd.notna(ror):
            parts.append(f"RoR: {format_pct(ror)}")
    else:
        rr = row.get("rr_ratio", pd.NA)
        if pd.notna(rr):
            parts.append(f"RR: {float(rr):.1f}x")

    # Momentum
    rsi = row.get("rsi_14", pd.NA)
    ret5 = row.get("ret_5d", pd.NA)
    if pd.notna(rsi) and pd.notna(ret5):
        parts.append(f"Momentum: RSI {float(rsi):.0f}, 5d {format_pct(ret5)}")

    # Sentiment
    sentiment = row.get("sentiment_tag", "Neutral")
    parts.append(f"Sentiment: {sentiment}")

    # Quality Score
    quality = row.get('quality_score', 0.0)
    parts.append(f"Quality: {quality:.2f}")

    # Earnings Play
    if row.get("Earnings Play") == "YES":
        underpriced_status = "Underpriced" if row.get("is_underpriced") else "Overpriced"
        parts.append(f"Earnings: YES ({underpriced_status})")

    # Stock Price and DTE
    stock_price = row.get('underlying', 0.0)
    dte = int(row.get('T_years', 0) * 365)
    parts.append(f"Stock: {format_money(stock_price)} | DTE: {dte}d")

    # --- Seasonality ---
    if pd.notna(row.get("seasonal_win_rate")):
        win_rate = row["seasonal_win_rate"]
        current_month_name = datetime.now().strftime("%b")
        parts.append(f"{current_month_name} Hist: {win_rate:.0%}")

    # --- Warnings & Squeeze---
    if row.get("decay_warning"):
        parts.append("HIGH DECAY RISK")
    if row.get("sr_warning"):
        parts.append(row["sr_warning"])
    if row.get("oi_wall_warning"):
        parts.append(row["oi_wall_warning"])
    if row.get("squeeze_play"):
        parts.append("🔥 SQUEEZE PLAY")

    # Invisible Filters Tags
    if row.get("macro_warning"):
        parts.append(row["macro_warning"])
    if row.get("max_pain_warning"):
        parts.append(row["max_pain_warning"])
    if row.get("yield_warning"):
        parts.append(row["yield_warning"])
    if row.get("high_premium_turnover"):
        parts.append("🐋 WHALE FLOW")

    return " | ".join(parts)


def format_mechanics_row(row: pd.Series) -> str:
    """Formats the first detail row for the screener report (market mechanics)."""
    parts = []

    # Liquidity
    vol = int(row.get('volume', 0))
    oi = int(row.get('openInterest', 0))
    parts.append(f"Vol: {vol} OI: {oi}")

    # Spread
    sp = row.get("spread_pct", pd.NA)
    parts.append(f"Spread: {format_pct(sp)}")

    # Delta
    d = row.get("delta", pd.NA)
    if pd.notna(d) and math.isfinite(d):
        parts.append(f"Delta: {d:+.2f}")

    # Greeks
    gamma = row.get("gamma", pd.NA)
    vega = row.get("vega", pd.NA)
    theta = row.get("theta", pd.NA)
    if pd.notna(gamma) and pd.notna(vega) and pd.notna(theta):
        parts.append(f"Greeks: Γ {gamma:.3f}, V {vega:.2f}, Θ {theta:.2f}")

    # Cost
    cost = row.get('premium', 0.0) * 100
    parts.append(f"Cost: {format_money(cost)}")

    return " | ".join(parts)


def print_executive_summary(df_picks: pd.DataFrame, config: Dict, mode: str = "Discovery",
                            market_trend: str = "Unknown", volatility_regime: str = "Unknown",
                            macro_risk: bool = False, num_tickers: int = 0):
    """
    Print an executive summary with top picks and key warnings.

    Args:
        df_picks: DataFrame with all options
        config: Configuration dictionary
        mode: Scan mode
        market_trend: Market trend (Bullish/Bearish/Sideways)
        volatility_regime: VIX regime (Low/Normal/High)
        macro_risk: Whether macro risk is active
        num_tickers: Number of tickers scanned
    """
    if df_picks.empty:
        return

    if not HAS_ENHANCED_CLI:
        # Fallback to simple summary
        print(f"\n{'='*80}")
        print(f"  SUMMARY: Found {len(df_picks)} opportunities")
        print(f"{'='*80}\n")
        return

    # Use new formatting
    width = config.get('display', {}).get('terminal_width', 100)

    print("\n" + fmt.draw_box("⚡ EXECUTIVE SUMMARY", width, double=True))

    # Market Context
    vix = get_vix_level()
    vix_str = f"{vix:.1f}" if vix else "N/A"

    context_parts = []
    if mode in ["Discovery", "Budget"]:
        context_parts.append(f"{mode} Scan ({num_tickers} tickers)")
    else:
        context_parts.append(mode)

    print(f"\n{fmt.format_header('📊 MARKET CONTEXT', '')}")
    trend_color = fmt.Colors.GREEN if market_trend == "Bullish" else (
        fmt.Colors.RED if market_trend == "Bearish" else fmt.Colors.YELLOW
    )
    vol_color = fmt.Colors.GREEN if volatility_regime == "Low" else (
        fmt.Colors.RED if volatility_regime == "High" else fmt.Colors.YELLOW
    )

    print(f"   Trend: {fmt.colorize(market_trend, trend_color, bold=True)} | "
          f"VIX: {fmt.colorize(vix_str, vol_color)} ({volatility_regime}) | "
          f"Risk: {fmt.colorize('HIGH', fmt.Colors.RED, bold=True) if macro_risk else fmt.colorize('LOW', fmt.Colors.GREEN)}")

    # Top 3 Opportunities
    print(f"\n{fmt.format_header('🏆 TOP 3 OPPORTUNITIES', '')}")

    top3 = df_picks.nlargest(3, 'quality_score')

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        symbol = row.get('symbol', 'N/A')
        strike = row.get('strike', 0)
        opt_type = row.get('type', 'call').upper()
        premium = row.get('premium', 0)
        pop = row.get('prob_profit', 0)
        rr = row.get('rr_ratio', 0)
        ev = row.get('ev_per_contract', 0)
        quality = row.get('quality_score', 0)

        stars, _ = fmt.format_quality_score(quality)

        # Box for each pick
        print(fmt.draw_separator(width - 4, fmt.BoxChars.HORIZONTAL))
        print(f"{fmt.BoxChars.VERTICAL} {i}. {symbol} ${strike} {opt_type} @ ${premium:.2f} • "
              f"{fmt.format_pop(pop)} PoP • {fmt.format_rr(rr)} RR • {fmt.format_ev(ev)} EV • {stars}")

        # Thesis
        thesis = generate_trade_thesis(row) if HAS_ENHANCED_CLI else "Standard setup"
        print(f"{fmt.BoxChars.VERTICAL}    💡 {thesis}")

        # Entry/Exit
        if config.get('display', {}).get('show_entry_exit_levels', True):
            levels = calculate_entry_exit_levels(row, config)
            print(f"{fmt.BoxChars.VERTICAL}    📍 Entry: ≤${levels['entry_price']:.2f} | "
                  f"Target: ${levels['profit_target']:.2f} (+50%) | "
                  f"Stop: ${levels['stop_loss']:.2f} (-25%)")

    print(fmt.draw_separator(width - 4, fmt.BoxChars.HORIZONTAL))

    # Warnings
    print(f"\n{fmt.format_header('⚠️  WATCH OUT', '')}")

    high_spread = df_picks[df_picks['spread_pct'] > 0.20]
    if not high_spread.empty:
        print(fmt.format_warning(f"{len(high_spread)} options with spreads >20% - use limit orders!"))

    neg_ev = df_picks[df_picks['ev_per_contract'] < 0]
    if not neg_ev.empty:
        print(fmt.format_warning(f"{len(neg_ev)} trades have negative expected value"))

    earnings = df_picks[df_picks.get('Earnings Play', 'NO') == 'YES']
    if not earnings.empty:
        print(fmt.format_warning(f"{len(earnings)} earnings plays - IV crush risk post-announcement"))

    low_liquid = df_picks[(df_picks['volume'] < 100) | (df_picks['openInterest'] < 100)]
    if not low_liquid.empty:
        print(fmt.format_warning(f"{len(low_liquid)} low-liquidity options - execution risk"))

    print("\n" + fmt.draw_separator(width, fmt.BoxChars.D_HORIZONTAL))
    print()


def print_report(df_picks: pd.DataFrame, underlying_price: float, rfr: float, num_expiries: int, min_dte: int, max_dte: int, mode: str = "Single-stock", budget: Optional[float] = None, market_trend: str = "Unknown", volatility_regime: str = "Unknown"):
    """Enhanced report with context, formatting, top pick, and summary."""
    if df_picks.empty:
        print("No picks available after filtering.")
        return
    
    chain_iv_median = df_picks["impliedVolatility"].median(skipna=True)
    
    # Header with context
    print("\n" + "="*80)
    if mode == "Budget scan":
        print(f"  OPTIONS SCREENER REPORT - MULTI-TICKER (Budget: ${budget:.2f})")
    elif mode == "Discovery scan":
        unique_tickers = df_picks["symbol"].nunique() if "symbol" in df_picks.columns else 1
        print(f"  OPTIONS SCREENER REPORT - DISCOVERY MODE ({unique_tickers} Tickers)")
    elif mode == "Premium Selling":
        unique_tickers = df_picks["symbol"].nunique() if "symbol" in df_picks.columns else 1
        print(f"  OPTIONS SCREENER REPORT - PREMIUM SELLING ({unique_tickers} Tickers)")
    else:
        symbol_name = df_picks.iloc[0]['symbol'] if "symbol" in df_picks.columns and not df_picks.empty else "UNKNOWN"
        print(f"  OPTIONS SCREENER REPORT - {symbol_name}")
    print("="*80)
    
    if mode == "Single-stock":
        print(f"  Stock Price: ${underlying_price:.2f}")
    elif mode == "Budget scan":
        print(f"  Budget Constraint: ${budget:.2f} per contract (premium × 100)")
        print(f"  Categories: LOW ($0-${budget*0.33:.2f}) | MEDIUM (${budget*0.33:.2f}-${budget*0.66:.2f}) | HIGH (${budget*0.66:.2f}-${budget:.2f})")
    elif mode in ["Discovery scan", "Premium Selling"]:
        print(f"  Scan Type: Top opportunities across all price ranges (no budget limit)")
        print(f"  Categories: LOW (bottom 33%) | MEDIUM (middle 33%) | HIGH (top 33%) by premium")
    print(f"  Market Status: Trend is {market_trend} | Volatility is {volatility_regime}")
    print(f"  Risk-Free Rate: {rfr*100:.2f}% (13-week Treasury)")
    print(f"  Expirations Scanned: {num_expiries}")
    print(f"  DTE Range: {min_dte} - {max_dte} days")
    print(f"  Chain Median IV: {format_pct(chain_iv_median)}")
    print(f"  Mode: {mode}")
    print("="*80)

    def header(txt: str):
        print("\n" + "─" * 80)
        print(f"  {txt}")
        print("─" * 80)

    # Print each bucket with summary stats
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df_picks[df_picks["price_bucket"] == bucket]
        if sub.empty:
            continue
        
        # Bucket header
        header(f"{bucket} PREMIUM (Top {len(sub)} Picks)")
        
        # Category summary stats
        avg_iv = sub["impliedVolatility"].mean()
        avg_spread = sub["spread_pct"].mean(skipna=True)
        median_delta = sub["abs_delta"].median()
        print(f"  Summary: Avg IV {format_pct(avg_iv)} | Avg Spread {format_pct(avg_spread)} | Median |Δ| {median_delta:.2f}\n")
        
        # Column headers (add Ticker for multi-stock modes)
        if mode in ["Budget scan", "Discovery scan", "Premium Selling"]:
            print(f"  {'Tkr':<5} {'Whale':<3} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<7} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*81)
        else:
            print(f"  {'Whale':<3} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<8} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*79)
        
        for _, r in sub.iterrows():
            exp = pd.to_datetime(r["expiration"]).date()
            moneyness = determine_moneyness(r)
            dte = int(r["T_years"] * 365)
            whale_emoji = "🐋" if r.get("high_premium_turnover", False) else ""
            
            # Main line with aligned columns
            if mode in ["Budget scan", "Discovery scan", "Premium Selling"]:
                print(
                    f"  {r['symbol']:<5} "
                    f"{whale_emoji:<3} "
                    f"{r['type'].upper():<5} "
                    f"{r['strike']:>7.2f} "
                    f"{exp} "
                    f"{format_money(r['premium']):<8} "
                    f"{format_pct(r['impliedVolatility']):<7} "
                    f"{int(r['openInterest']):>6} "
                    f"{int(r['volume']):>6} "
                    f"{r['delta']:>+6.2f} "
                    f"{moneyness:<4}"
                )
            else:
                print(
                    f"  {whale_emoji:<3} "
                    f"{r['type'].upper():<5} "
                    f"{r['strike']:>7.2f} "
                    f"{exp} "
                    f"{format_money(r['premium']):<8} "
                    f"{format_pct(r['impliedVolatility']):<7} "
                    f"{int(r['openInterest']):>7} "
                    f"{int(r['volume']):>7} "
                    f"{r['delta']:>+6.2f} "
                    f"{moneyness:<4}"
                )
            # Rationale
            mechanics_line = format_mechanics_row(r)
            analysis_line = format_analysis_row(r, chain_iv_median, mode)

            print(f"    ↳ Mechanics: {mechanics_line}")
            print(f"    ↳ Analysis:  {analysis_line}")
            
            # --- NEW: Institutional Metrics ---
            if "short_interest" in r and pd.notna(r["short_interest"]):
                si_val = r["short_interest"] * 100
                print(f"      • Short Interest: {si_val:.2f}%")
            
            if "rvol" in r and pd.notna(r["rvol"]):
                print(f"      • RVOL: {r['rvol']:.2f}x")
            
            if "gex_flip_price" in r and pd.notna(r["gex_flip_price"]):
                print(f"      • GEX Flip: ${r['gex_flip_price']:.2f}")
            
            if "vwap" in r and pd.notna(r["vwap"]):
                 print(f"      • VWAP: ${r['vwap']:.2f}")
            
            print("") # Newline


def print_spreads_report(df_spreads: pd.DataFrame):
    """Prints a report of the vertical spreads found."""
    if df_spreads.empty:
        return

    print("\n" + "="*80)
    print("  VERTICAL SPREADS REPORT")
    print("="*80)

    print(f"  {'Symbol':<7} {'Type':<12} {'Long Strike':<12} {'Short Strike':<13} {'Expiration':<12} {'Cost':<8} {'Max Profit':<12} {'Risk':<8}")
    print("  " + "-"*78)

    for _, row in df_spreads.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        print(
            f"  {row['symbol']:<7} "
            f"{row['type']:<12} "
            f"{row['long_strike']:>11.2f} "
            f"{row['short_strike']:>12.2f} "
            f"{exp} "
            f"{format_money(row['spread_cost']):<8} "
            f"{format_money(row['max_profit']):<12} "
            f"{format_money(row['risk']):<8}"
        )


def print_credit_spreads_report(df_spreads: pd.DataFrame):
    """Prints a dedicated report for credit spreads."""
    if df_spreads.empty:
        print("\nNo credit spreads meeting the criteria were found.")
        return

    print("\n" + "="*80)
    print("  CREDIT SPREADS REPORT (INCOME ENGINE)")
    print("="*80)

    header = f"  {'Symbol':<7} {'Type':<10} {'Short Strike':<13} {'Long Strike':<12} {'Expiration':<12} {'Credit':<8} {'Max Profit':<12} {'Max Loss':<10} {'Score':<5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, row in df_spreads.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        print(
            f"  {row['symbol']:<7} "
            f"{row['type']:<10} "
            f"{row['short_strike']:>12.2f} "
            f"{row['long_strike']:>11.2f} "
            f"{exp} "
            f"{format_money(row['net_credit']):<8} "
            f"{format_money(row['max_profit']):<12} "
            f"{format_money(row['max_loss']):<10} "
            f"{row['quality_score']:.2f}"
        )


def print_iron_condor_report(df_condors: pd.DataFrame):
    """Prints a dedicated report for iron condors."""
    if df_condors.empty:
        print("\nNo iron condors meeting the criteria were found.")
        return

    print("\n" + "="*120)
    print("  IRON CONDOR REPORT (RANGE-BOUND STRATEGIES)")
    print("="*120)

    header = (
        f"  {'Symbol':<7} {'Exp':<12} "
        f"{'Put Wing':<20} {'Call Wing':<20} "
        f"{'Credit':<8} {'Max Risk':<10} {'RoR':<8} {'Net Δ':<8} {'Score':<5}"
    )
    print(header)
    print("  " + "-" * 118)

    for _, row in df_condors.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        
        # Format wings
        put_wing = f"{row['long_put_strike']:.0f}/{row['short_put_strike']:.0f}"
        call_wing = f"{row['short_call_strike']:.0f}/{row['long_call_strike']:.0f}"
        
        # Format delta with sign
        delta_sign = "+" if row['net_delta'] >= 0 else ""
        
        print(
            f"  {row['symbol']:<7} {exp} "
            f"{put_wing:<20} {call_wing:<20} "
            f"{format_money(row['total_credit']):<8} "
            f"{format_money(row['max_risk']):<10} "
            f"{row['return_on_risk']:.2f}x    "
            f"{delta_sign}{row['net_delta']:>+.3f}   "
            f"{row['quality_score']:.2f}"
        )
    
    print("\n  💡 Iron Condors profit from range-bound movement with defined risk on both sides.")
    print("  📊 Net Delta shows directional bias (closer to 0 = more neutral)")


def export_to_csv(df_picks: pd.DataFrame, mode: str, budget: Optional[float] = None) -> str:
    """Export picks to CSV with timestamp."""
    try:
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/options_picks_{mode.replace(' ', '_')}_{timestamp}.csv"
        
        # Select relevant columns for export
        export_cols = [
            "symbol", "type", "strike", "expiration", "premium", "underlying",
            "delta", "gamma", "vega", "theta", "rho", "impliedVolatility", "hv_30d", "iv_vs_hv", "iv_rank", "iv_percentile",
            "iv_rank_30", "iv_percentile_30", "iv_rank_90", "iv_percentile_90",
            "volume", "openInterest", "spread_pct", "Vol_OI_Ratio", "Unusual_Whale",
            "sentiment_score", "sentiment_tag",
            "Earnings Play", "is_underpriced",
            "prob_profit", "pop_sim", "expected_move", "required_move", "em_realism_score",
            "theta_decay_pressure", "theta_score",
            "prob_touch", "pot_sim", "p_itm",
            "max_loss", "breakeven", "rr_ratio", "return_on_risk",
            "theo_value", "ev_per_contract", "ev_score",
            "liquidity_score", "momentum_score", "iv_rank_score", "catalyst_score",
            "ret_5d", "rsi_14", "atr_trend",
            "quality_score", "liquidity_flag", "spread_flag", "event_flag", "price_bucket",
            "short_interest", "rvol", "gex_flip_price", "vwap", "high_premium_turnover"
        ]
        
        # Filter to existing columns
        export_cols = [c for c in export_cols if c in df_picks.columns]
        
        df_picks[export_cols].to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")
        return None


def log_trade_entry(df_picks: pd.DataFrame, mode: str) -> None:
    """Log trade entries for future P/L tracking.

    Adds a unique entry_id so rows can be reliably joined/updated later.
    """
    try:
        # Create trades_log directory if it doesn't exist
        os.makedirs("trades_log", exist_ok=True)
        
        log_file = "trades_log/entries.csv"
        file_exists = os.path.exists(log_file)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', newline='') as f:
            fieldnames = [
                'entry_id', 'timestamp', 'mode', 'symbol', 'type', 'strike', 'expiration',
                'entry_premium', 'entry_underlying', 'delta', 'iv', 'hv', 'iv_rank',
                'prob_profit', 'p_itm', 'rr_ratio', 'theo_value', 'ev_per_contract',
                'quality_score', 'event_flag', 'status',
                'exit_premium', 'exit_underlying', 'exit_date', 'realized_pnl'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for _, row in df_picks.iterrows():
                entry_id = f"{datetime.now(timezone.utc).isoformat()}_{str(uuid.uuid4())[:8]}"
                writer.writerow({
                    'entry_id': entry_id,
                    'timestamp': timestamp,
                    'mode': mode,
                    'symbol': row.get('symbol', ''),
                    'type': row.get('type', ''),
                    'strike': row.get('strike', ''),
                    'expiration': row.get('expiration', ''),
                    'entry_premium': row.get('premium', ''),
                    'entry_underlying': row.get('underlying', ''),
                    'delta': row.get('delta', ''),
                    'iv': row.get('impliedVolatility', ''),
                    'hv': row.get('hv_30d', ''),
                    'iv_rank': row.get('iv_rank', ''),
                    'prob_profit': row.get('prob_profit', ''),
                    'p_itm': row.get('p_itm', ''),
                    'rr_ratio': row.get('rr_ratio', ''),
                    'theo_value': row.get('theo_value', ''),
                    'ev_per_contract': row.get('ev_per_contract', ''),
                    'quality_score': row.get('quality_score', ''),
                    'event_flag': row.get('event_flag', ''),
                    'status': 'OPEN',
                    'exit_premium': '',
                    'exit_underlying': '',
                    'exit_date': '',
                    'realized_pnl': ''
                })
        
        print(f"\n  💾 Trade entries logged to {log_file}")
    except Exception as e:
        print(f"Warning: Could not log trades: {e}")


def setup_logging() -> logging.Logger:
    """Configure a simple console logger and JSONL file logger.
    LOG_LEVEL env var controls verbosity (default INFO).
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(message)s")
    logger = logging.getLogger("options_screener")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure logs are always created in the project's root 'logs' directory
    # Get the absolute path of the current script.
    script_path = os.path.abspath(__file__)
    # Navigate up two levels to get to the project root (src -> root).
    project_root = os.path.dirname(os.path.dirname(script_path))

    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.json_path = os.path.join(logs_dir, f"run_{ts}.jsonl")  # type: ignore[attr-defined]
    return logger


def log_picks_json(logger: logging.Logger, picks_df: pd.DataFrame, context: Dict):
    """Append picks to a JSONL log for later evaluation/backtesting."""
    try:
        # Create a copy to avoid modifying the original DataFrame
        log_df = picks_df.copy()

        # Convert any datetime-like columns to ISO 8601 strings
        for col in log_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            log_df[col] = log_df[col].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "picks": log_df.to_dict(orient="records"),
        }
        with open(logger.json_path, "a") as f:  # type: ignore[attr-defined]
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return default if (not val and default is not None) else val


def close_trades():
    """Update trade log with closing prices and realized P/L."""
    log_file = "trades_log/entries.csv"
    
    if not os.path.exists(log_file):
        print("No trade log found. Run the screener first and log some trades.")
        sys.exit(1)
    
    print("=" * 80)
    print("  CLOSE TRADES - Update Trade Log with Realized P/L")
    print("=" * 80)
    
    # Read existing log
    df_trades = pd.read_csv(log_file)
    
    # Filter for OPEN trades
    open_trades = df_trades[df_trades['status'] == 'OPEN'].copy()
    
    if open_trades.empty:
        print("\nNo open trades found in log.")
        sys.exit(0)
    
    print(f"\nFound {len(open_trades)} open trades.")
    print("\nFetching current prices and calculating P/L...\n")
    
    updated_count = 0
    for idx, trade in open_trades.iterrows():
        symbol = trade['symbol']
        exp_date = pd.to_datetime(trade['expiration']).date()
        
        # Check if expired
        if exp_date > datetime.now().date():
            continue  # Skip unexpired trades
        
        print(f"Processing {symbol} {trade['type']} ${trade['strike']} exp {exp_date}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get price at or near expiration
            start_date = exp_date - timedelta(days=3)
            end_date = exp_date + timedelta(days=3)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                print(f"  ⚠️  No price data available")
                continue
            
            # Find closest date to expiration
            hist_dates = hist.index.date
            closest_date = min(hist_dates, key=lambda d: abs((d - exp_date).days))
            exit_price = float(hist[hist.index.date == closest_date]['Close'].iloc[0])
            
            # Calculate intrinsic value at expiration
            strike = float(trade['strike'])
            option_type = trade['type'].lower()
            
            if option_type == 'call':
                intrinsic_value = max(0, exit_price - strike)
            else:  # put
                intrinsic_value = max(0, strike - exit_price)
            
            entry_premium = float(trade['entry_premium'])
            exit_premium = intrinsic_value
            
            # P/L per share
            pnl_per_share = exit_premium - entry_premium
            realized_pnl = pnl_per_share * 100  # Per contract
            
            # Update the dataframe
            df_trades.at[idx, 'exit_premium'] = exit_premium
            df_trades.at[idx, 'exit_underlying'] = exit_price
            df_trades.at[idx, 'exit_date'] = closest_date.strftime('%Y-%m-%d')
            df_trades.at[idx, 'realized_pnl'] = realized_pnl
            df_trades.at[idx, 'status'] = 'CLOSED'
            
            updated_count += 1
            print(f"  ✓ Closed at ${exit_price:.2f} | P/L: ${realized_pnl:.2f}")
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    # Save updated log
    if updated_count > 0:
        df_trades.to_csv(log_file, index=False)
        print(f"\n✓ Updated {updated_count} trades in {log_file}")
    else:
        print("\nNo trades were updated.")
    
    print("\n" + "=" * 80)
    print("  Done!")
    print("=" * 80 + "\n")


def prompt_for_tickers() -> List[str]:
    """
    Prompts the user to select a ticker source and returns a list of tickers.
    """
    print("\nSelect Ticker Source:")
    print("  1. Curated Liquid (default)")
    print("  2. Top Gainers (Finviz)")
    print("  3. High IV Stocks (Finviz)")
    source_choice = prompt_input("Enter 1, 2, or 3", "1")

    if source_choice == "1":
        # Top 100 most liquid options tickers (ordered by typical volume)
        tickers = [
            # Major Indices & ETFs
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "GLD", "SLV", "TLT",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
            # Mega Cap Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "CSCO", "AVGO", "QCOM", "TXN", "AMAT", "MU", "LRCX",
            # Financial
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
            "MA", "PYPL", "SQ", "COIN",
            # Healthcare & Pharma
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "LLY", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "CVS", "MRNA", "BNTX",
            # Consumer & Retail
            "WMT", "HD", "DIS", "NKE", "MCD", "SBUX", "TGT", "COST", "LOW", "TJX",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            # Industrial & Manufacturing
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "LMT", "RTX", "DE",
            # Communication & Media
            "T", "VZ", "CMCSA", "TMUS", "CHTR",
            # Automotive & Transportation
            "F", "GM", "RIVN", "LCID", "NIO", "UBER", "LYFT", "DAL", "UAL", "AAL"
        ]
        return tickers
    else:
        scan_type = "gainers" if source_choice == "2" else "high_iv"
        try:
            max_tickers = int(prompt_input("How many tickers to scan (1-100)", "50"))
            max_tickers = max(1, min(100, max_tickers))
            return get_dynamic_tickers(scan_type, max_tickers=max_tickers)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)


def process_ticker(symbol: str, mode: str, max_expiries: int, min_dte: int, max_dte: int, rfr: float, config: Dict, vix_weights: Dict, trader_profile: str, budget: Optional[float], macro_risk_active: bool, tnx_change_pct: float) -> Dict:
    """
    Process a single ticker: fetch data, enrich, score, and filter.
    Returns a dict with picks, spreads, history, and metadata.
    """
    result = {
        'symbol': symbol,
        'picks': [],
        'credit_spreads': [],
        'iron_condors': [],
        'history': None,
        'success': False,
        'error': None
    }
    
    try:
        # Fetch data
        data_result = fetch_options_yfinance(symbol, max_expiries)
        
        df_chain = data_result["df"]
        history_df = data_result["history_df"]
        context = data_result["context"]
        
        # Cache history for correlation check
        if history_df is not None and not history_df.empty:
            result['history'] = history_df

        # Unpack context
        hv = context.get("hv")
        iv_rank = context.get("iv_rank")
        iv_percentile = context.get("iv_percentile")
        earnings_date = context.get("earnings_date")
        sentiment_score = context.get("sentiment_score")
        seasonal_win_rate = context.get("seasonal_win_rate")
        term_structure_spread = context.get("term_structure_spread")
        sector_perf = context.get("sector_perf", {})
        
        # Build context log
        context_log = []
        if hv: context_log.append(f"HV (30d): {hv:.2%}")
        if iv_rank: context_log.append(f"IV Rank: {iv_rank:.2f}")
        if earnings_date: context_log.append(f"Earnings: {earnings_date.strftime('%Y-%m-%d')}")
        if context.get("rvol"): context_log.append(f"RVOL: {context['rvol']:.2f}x")
        result['context_log'] = context_log

        # Enrich and Score
        df_scored = enrich_and_score(
            df_chain,
            min_dte=min_dte,
            max_dte=max_dte,
            risk_free_rate=rfr,
            config=config,
            vix_regime_weights=vix_weights,
            trader_profile=trader_profile,
            mode=mode,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            earnings_date=earnings_date,
            sentiment_score=sentiment_score,
            seasonal_win_rate=seasonal_win_rate,
            term_structure_spread=term_structure_spread,
            macro_risk_active=macro_risk_active,
            sector_perf=sector_perf,
            tnx_change_pct=tnx_change_pct
        )

        if df_scored.empty:
            result['error'] = "No contracts passed filters"
            return result

        # Validate required columns
        if "symbol" not in df_scored.columns:
            result['error'] = f"'symbol' column missing from {symbol} data"
            return result

        # Apply budget filter if in budget mode
        is_budget_mode = (mode == "Budget scan")
        if is_budget_mode and budget:
            df_scored["contract_cost"] = df_scored["premium"] * 100
            df_scored = df_scored[df_scored["contract_cost"] <= budget].copy()
            if df_scored.empty:
                result['error'] = "No contracts within budget"
                return result

        # Collect results based on mode
        if mode == "Credit Spreads":
            # Vertical Spreads
            spreads = find_credit_spreads(df_scored)
            if not spreads.empty:
                result['credit_spreads'].append(spreads)
                result['success'] = True
        
        elif mode == "Iron Condor":
            # Iron Condors
            condors = find_iron_condors(df_scored)
            if not condors.empty:
                result['iron_condors'] = condors
                result['success'] = True

        elif mode == "Premium Selling":
            # Filter for short puts
            puts = df_scored[df_scored["type"] == "put"].copy()
            if not puts.empty:
                result['picks'].append(puts)
                result['success'] = True

        else:
            # Single stock, Budget, Discovery
            result['picks'].append(df_scored)
            result['success'] = True

    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_scan(mode: str, tickers: List[str], budget: Optional[float], max_expiries: int, min_dte: int, max_dte: int, trader_profile: str, logger: logging.Logger, market_trend: str, volatility_regime: str, macro_risk_active: bool = False, tnx_change_pct: float = 0.0, verbose: bool = True, custom_weights: Optional[Dict] = None):
    # Determine mode booleans for internal logic
    is_budget_mode = (mode == "Budget scan")
    is_discovery_mode = (mode == "Discovery scan")

    # === LOAD CONFIGURATION ===
    if verbose:
        print("\nLoading configuration...")
    config = load_config("config.json")
    
    # Merge custom weights if provided (from UI)
    if custom_weights:
        config['composite_weights'].update(custom_weights)
    
    if verbose:
        print("✓ Configuration loaded")

    # === FETCH VIX FOR ADAPTIVE WEIGHTING ===
    if verbose:
        print("Fetching VIX level for adaptive scoring...")
    vix_level = get_vix_level()
    if verbose:
        if vix_level:
            print(f"✓ VIX Level: {vix_level:.2f}")
        else:
            print("⚠️  Could not fetch VIX, using default weights")

    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    if verbose:
        print(f"✓ Market Regime: {vix_regime.upper()}")

    # Fetch risk-free rate automatically
    if verbose:
        print("Fetching current risk-free rate...")
    rfr = get_risk_free_rate()
    if verbose:
        print(f"Using risk-free rate: {rfr*100:.2f}% (13-week Treasury)")

    # Collect data from all tickers (PARALLEL PROCESSING)
    tickers = list(set(tickers))  # Deduplicate tickers
    all_picks = []
    all_credit_spreads = []
    all_iron_condors = []
    ticker_histories = {} # For Portfolio Protection

    if verbose:
        print(f"\n{'='*80}")
        print(f"  Fetching data for {len(tickers)} ticker(s) in parallel...")
        print(f"{'='*80}\n")

    # Use ThreadPoolExecutor for parallel processing
    # Limit to 8 workers to avoid rate limiting
    max_workers = min(8, len(tickers))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ticker processing jobs
        future_to_symbol = {
            executor.submit(
                process_ticker,
                symbol,
                mode,
                max_expiries,
                min_dte,
                max_dte,
                rfr,
                config,
                vix_weights,
                trader_profile,
                budget,
                macro_risk_active,
                tnx_change_pct
            ): symbol
            for symbol in tickers
        }
        
        # Process results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                
                # Log the ticker processing
                if verbose:
                    print(f"\n>>> Scanning {symbol}...")
                    
                    # Print context logs
                    for log in result.get('context_log', []):
                        print(f"    {log}")
                
                if result['success']:
                    # Store history
                    if result['history'] is not None:
                        ticker_histories[symbol] = result['history']
                    
                    # Aggregate picks
                    for picks_df in result['picks']:
                        all_picks.append(picks_df)
                        if verbose:
                            print(f"    Found {len(picks_df)} contracts.")
                    
                    # Aggregate credit spreads
                    for spreads_df in result['credit_spreads']:
                        all_credit_spreads.append(spreads_df)
                        if verbose:
                            print(f"    Found {len(spreads_df)} Credit Spreads.")
                    
                    # Aggregate iron condors
                    if 'iron_condors' in result and isinstance(result['iron_condors'], pd.DataFrame) and not result['iron_condors'].empty:
                        all_iron_condors.append(result['iron_condors'])
                        if verbose:
                            print(f"    Found {len(result['iron_condors'])} Iron Condors.")
                else:
                    if verbose and result['error']:
                        print(f"    {result['error']}")
                
            except Exception as e:
                if verbose:
                    print(f"\n>>> Scanning {symbol}...")
                    print(f"    ⚠️  Error: {e}")

    # --- Portfolio Protection: Correlation Warning ---
    if verbose and len(ticker_histories) > 1:
        print("\n🔎 Checking Portfolio Correlation...")
        try:
            # Create a combined DF of 'Close' prices
            price_data = {}
            for t, h in ticker_histories.items():
                if not h.empty and "Close" in h.columns:
                    # Use last 30 days
                    price_data[t] = h["Close"].tail(30)
            
            if len(price_data) > 1:
                prices_df = pd.DataFrame(price_data)
                # Forward fill / Drop NA (using newer pandas syntax)
                prices_df = prices_df.ffill().dropna()
                
                if not prices_df.empty and len(prices_df.columns) > 1:
                    corr_matrix = prices_df.corr()
                    
                    # Check for high correlation (> 0.80)
                    # Iterate upper triangle
                    high_corr_pairs = []
                    cols = corr_matrix.columns
                    for i in range(len(cols)):
                        for j in range(i+1, len(cols)):
                            c = corr_matrix.iloc[i, j]
                            if c > 0.80:
                                high_corr_pairs.append((cols[i], cols[j], c))
                    
                    if high_corr_pairs:
                        print("\n⚠️  PORTFOLIO PROTECTION WARNING: You are making the same bet twice!")
                        for t1, t2, c in high_corr_pairs:
                            print(f"  - {t1} and {t2} are highly correlated ({c:.2f})")
                    else:
                        print("✓ Portfolio correlation looks healthy (no pairs > 0.80).")
        except Exception as e:
            print(f"⚠️  Could not compute portfolio correlation: {e}")

    # Consolidate picks and determine underlying price
    picks = pd.DataFrame()
    credit_spreads_df = pd.DataFrame()
    iron_condors_df = pd.DataFrame()
    
    if all_picks:
        non_empty_picks = [df for df in all_picks if not df.empty]
        if non_empty_picks:
            picks = pd.concat(non_empty_picks, ignore_index=True)
    
    if all_credit_spreads:
        credit_spreads_df = pd.concat(all_credit_spreads, ignore_index=True)
    
    if all_iron_condors:
        iron_condors_df = pd.concat(all_iron_condors, ignore_index=True)
    
    underlying_price = 0.0
    if not picks.empty and "underlying" in picks.columns:
        underlying_price = picks.iloc[0]["underlying"]

    # Generate Final Reports
    if mode == "Budget scan":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=budget)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, budget=budget, market_trend=market_trend, volatility_regime=volatility_regime)
        elif verbose:
            print("\nNo options found within budget.")

    elif mode == "Discovery scan":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=None)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
        elif verbose:
            print("\nNo discovery picks found.")
            
    elif mode == "Credit Spreads":
        if not credit_spreads_df.empty:
            final_spreads = credit_spreads_df.sort_values("quality_score", ascending=False)
            if verbose:
                print_credit_spreads_report(final_spreads)
        elif verbose:
            print("\nNo credit spreads found.")
    
    elif mode == "Iron Condor":
        if not iron_condors_df.empty:
            final_condors = iron_condors_df.sort_values("return_on_risk", ascending=False)
            if verbose:
                print_iron_condor_report(final_condors)
        elif verbose:
            print("\nNo iron condors found.")

    elif mode == "Premium Selling":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=None)
            if verbose:
                print_report(final_df.head(10), underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
        elif verbose:
            print("\nNo premium selling candidates found.")

    else:
        # Single stock mode
        if not picks.empty:
            final_df = picks.copy()
            final_df = categorize_by_premium(final_df, budget=None)
            if verbose:
                print_report(final_df, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
        elif verbose:
            print("\nNo suitable options found.")

    top_pick = None
    if not picks.empty:
        picks["overall_score"] = picks["quality_score"]
        top_pick = picks.sort_values("overall_score", ascending=False).iloc[0]
    elif not credit_spreads_df.empty:
         top_pick = credit_spreads_df.sort_values("quality_score", ascending=False).iloc[0]
    elif not iron_condors_df.empty:
         top_pick = iron_condors_df.sort_values("return_on_risk", ascending=False).iloc[0]

    chain_iv_median = 0.0
    if not picks.empty and "impliedVolatility" in picks.columns:
        chain_iv_median = picks["impliedVolatility"].median()

    return {
        'picks': picks,
        'spreads': pd.DataFrame(),
        'credit_spreads': credit_spreads_df,
        'iron_condors': iron_condors_df,
        'top_pick': top_pick,
        'underlying_price': underlying_price,
        'rfr': rfr,
        'chain_iv_median': chain_iv_median,
        'timestamp': datetime.now().isoformat(),
        'market_context': {
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'market_trend': market_trend,
            'volatility_regime': volatility_regime,
            'macro_risk_active': macro_risk_active,
            'tnx_change_pct': tnx_change_pct
        }
    }

def select_trades_to_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive helper to let the user select specific trades to log.
    Returns a DataFrame containing only the selected rows.
    """
    if df.empty:
        print("No trades to select.")
        return pd.DataFrame()

    if "quality_score" in df.columns:
        df_sorted = df.sort_values("quality_score", ascending=False).reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    top_n = df_sorted.head(50)

    print("\n" + "="*60)
    print("  SELECT TRADES TO LOG")
    print("="*60)
    
    for i, row in top_n.iterrows():
        symbol = row.get('symbol', 'N/A')
        type_ = row.get('type', 'N/A').upper()
        strike = row.get('strike', 0.0)
        exp = row.get('expiration', 'N/A')
        if isinstance(exp, str):
            exp = exp.split("T")[0]
        
        premium = row.get('premium', 0.0)
        quality = row.get('quality_score', 0.0)
        
        print(f"  [{i+1}] {symbol:<5} {type_:<4} {strike:>7.2f} {exp} | Prem: ${premium:>6.2f} | Qual: {quality:.2f}")

    print("="*60)
    print("Enter the numbers of the trades you want to log, separated by commas.")
    print("Example: 1, 3, 5 (or 'all' for all listed, 'q' to cancel)")
    
    selection = prompt_input("Selection", "").strip().lower()
    
    if not selection or selection == 'q':
        print("Selection cancelled.")
        return pd.DataFrame()
    
    if selection == 'all':
        return top_n

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        valid_indices = [i for i in indices if 0 <= i < len(top_n)]
        
        if not valid_indices:
            print("No valid selections made.")
            return pd.DataFrame()
            
        selected_df = top_n.iloc[valid_indices].copy()
        print(f"Selected {len(selected_df)} trades.")
        return selected_df
        
    except Exception as e:
        print(f"Error parsing selection: {e}")
        return pd.DataFrame()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--close-trades":
            close_trades()
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python options_screener.py [--close-trades]")
            sys.exit(1)
    
    config = load_config("config.json")
    
    # Initialize PaperManager and update positions
    pm = PaperManager(db_path="paper_trades.db", config_path="config.json")
    print("\nChecking existing paper trade positions...")
    pm.update_positions()

    print("Options Screener (yfinance)")
    print("Note: For personal/informational use only. Review data provider terms.")
    print("\nModes:")
    print("  1. Enter a ticker (e.g., AAPL) for single-stock analysis")
    print("  2. Enter 'ALL' for budget-based multi-stock scan")
    print("  3. Enter 'DISCOVER' to scan top 100 most-traded tickers (no budget limit)")
    print("  4. Enter 'SELL' for Premium Selling analysis (short puts)")
    print("  5. Enter 'SPREADS' for Credit Spread analysis")
    print("  6. Enter 'IRON' for Iron Condor analysis")
    print("  7. Enter 'PORTFOLIO' to view open position P/L\n")

    symbol_input = prompt_input("Enter stock ticker or command", "DISCOVER").upper()

    if symbol_input == "PORTFOLIO":
        from .check_pnl import view_portfolio
        view_portfolio()
        sys.exit(0)

    is_budget_mode = (symbol_input == "ALL")
    is_discovery_mode = (symbol_input == "DISCOVER" or symbol_input == "")
    is_premium_selling_mode = (symbol_input == "SELL")
    is_credit_spread_mode = (symbol_input == "SPREADS")
    is_iron_condor_mode = (symbol_input == "IRON")

    if is_discovery_mode: mode = "Discovery scan"
    elif is_budget_mode: mode = "Budget scan"
    elif is_premium_selling_mode: mode = "Premium Selling"
    elif is_credit_spread_mode: mode = "Credit Spreads"
    elif is_iron_condor_mode: mode = "Iron Condor"
    else: mode = "Single-stock"

    budget = None
    tickers = []

    if is_discovery_mode or is_premium_selling_mode or is_credit_spread_mode or is_iron_condor_mode:
        tickers = prompt_for_tickers()
        print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_budget_mode:
        try:
            budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
        except Exception:
            print("Invalid budget amount."); sys.exit(1)
        scan_type = prompt_input("Enter 1 for TARGETED or 2 for DISCOVERY", "1")
        if scan_type == "2": tickers = prompt_for_tickers()
        else:
            default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
            tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        if not symbol_input.isalnum():
            print("Please enter a valid alphanumeric ticker."); sys.exit(1)
        tickers = [symbol_input]
    
    logger = setup_logging()
    print("\nFetching market context (SPY/VIX)...")
    market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
    print(f"✓ Market Trend: {market_trend} | Volatility: {volatility_regime}")

    try:
        max_expiries = int(prompt_input("How many nearest expirations to scan", "4"))
    except Exception:
        print("Invalid number for expirations."); sys.exit(1)

    f_config = config.get("filters", {})
    if is_iron_condor_mode:
        default_min_dte = str(f_config.get("min_days_to_expiration_iron", 30))
        default_max_dte = str(f_config.get("max_days_to_expiration_iron", 60))
    else:
        default_min_dte = str(f_config.get("min_days_to_expiration", 7))
        default_max_dte = str(f_config.get("max_days_to_expiration", 45))

    try:
        min_dte = int(prompt_input("Minimum days to expiration (DTE)", default_min_dte))
        max_dte = int(prompt_input("Maximum days to expiration (DTE)", default_max_dte))
    except Exception:
        print("Invalid DTE inputs."); sys.exit(1)

    profile_choice = prompt_input("Enter 1 for Swing or 2 for Day trader", "1").strip()
    trader_profile = "day" if profile_choice == "2" else "swing"

    try:
        scan_results = run_scan(mode=mode, tickers=tickers, budget=budget, max_expiries=max_expiries, min_dte=min_dte, max_dte=max_dte, trader_profile=trader_profile, logger=logger, market_trend=market_trend, volatility_regime=volatility_regime, macro_risk_active=macro_risk_active, tnx_change_pct=tnx_change_pct)
        if scan_results is None: sys.exit(0)

        picks = scan_results['picks']
        rfr = scan_results['rfr']
        chain_iv_median = scan_results['chain_iv_median']
        
        # === PAPER TRADING INTERACTIVE PROMPT ===
        if not picks.empty and mode != "Credit Spreads" and mode != "Iron Condor":
            pt_choice = prompt_input("Would you like to paper trade the top 5 picks? (y/n)", "n").lower()
            if pt_choice == "y":
                top_5_pt = picks.sort_values("quality_score", ascending=False).head(5)
                today_str = datetime.now().strftime("%Y-%m-%d")
                for _, row in top_5_pt.iterrows():
                    trade_dict = {
                        "date": today_str,
                        "ticker": row["symbol"],
                        "expiration": row["expiration"],
                        "strike": row["strike"],
                        "type": str(row["type"]).capitalize(),
                        "entry_price": safe_float(row.get("ask") or None, row["lastPrice"]),
                        "quality_score": row["quality_score"],
                        "strategy_name": f"Long {str(row['type']).capitalize()}"
                    }
                    pm.log_trade(trade_dict)
                print(f"\n  ✅ Successfully logged {len(top_5_pt)} paper trades.")
        
        export_choice = prompt_input("Export results to CSV? (y/n)", "n").lower()
        if export_choice == "y":
            csv_file = export_to_csv(picks, mode, budget)
            if csv_file: print(f"\n  📄 Results exported to: {csv_file}")
        
        if HAS_VISUALIZATION:
            viz_choice = prompt_input("Generate visualization charts? (y/n)", "n").lower()
            if viz_choice == "y": create_visualizations(picks, mode, output_dir="reports")
        
        log_choice = prompt_input("Log trades for P/L tracking? (y/n)", "n").lower()
        if log_choice == "y":
            mode_choice = prompt_input("Log (A)ll or (S)elect specific?", "s").lower()
            if mode_choice == "s":
                picks_to_log = select_trades_to_log(picks)
                if not picks_to_log.empty: log_trade_entry(picks_to_log, mode)
            else: log_trade_entry(picks, mode)
        
        print("\n👋 Done! Happy trading!\n")
        
    except KeyboardInterrupt: print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    if "--ui" in sys.argv:
        import subprocess
        print("Launching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "src/dashboard.py"])
    else: main()
