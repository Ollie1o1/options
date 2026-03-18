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
from .types import ScanResult
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
import functools
import random
import argparse
import shutil
import warnings
import contextlib
import io


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
    get_dynamic_tickers,
    batch_fetch,
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
    bs_charm,
    bs_vanna,
    _d1d2,
    format_pct,
    format_money,
    determine_moneyness,
)
from .filters import (
    filter_options,
    filter_iv_smile_outliers,
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
        format_trade_plan,
        explain_quality_score,
        format_risk_alerts,
        build_scenario_table,
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

try:
    from .vol_analytics import print_vol_cone, print_iv_surface, classify_vol_regime, print_regime_summary
    from .backtester import print_paper_trade_ic
    HAS_VOL_ANALYTICS = True
except ImportError:
    HAS_VOL_ANALYTICS = False





from .cli_display import (
    get_display_width, format_analysis_row, format_mechanics_row,
    _format_breakeven_line, _print_strategy_panel, print_executive_summary,
    print_best_setup_callout, print_report, print_news_panel,
    print_spreads_report, print_credit_spreads_report, print_iron_condor_report,
)
from .watchlist import (
    _WATCHLIST_PATH, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
)
from .oi_snapshot import _OI_SNAPSHOT_PATH, load_oi_snapshot, save_oi_snapshot


@contextlib.contextmanager
def _suppress_scan_noise():
    """Suppress noisy third-party logging/warnings during parallel scan."""
    _noisy = ['yfinance', 'urllib3', 'peewee', 'charset_normalizer',
              'requests', 'asyncio', 'httpx', 'httpcore']
    _saved = {}
    for name in _noisy:
        lg = logging.getLogger(name)
        _saved[name] = lg.level
        lg.setLevel(logging.CRITICAL)
    # Also silence the root logger's stderr handler temporarily
    _root = logging.getLogger()
    _saved_root = _root.level
    # Capture and discard warnings from third-party libs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            yield
        finally:
            for name, level in _saved.items():
                logging.getLogger(name).setLevel(level)
            _root.setLevel(_saved_root)


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file with fallback defaults."""
    default_config = {
        # Composite quality score weights (can be overridden in config.json)
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


_IC_WEIGHTS_CACHE: dict | None = None


def load_ic_adjusted_weights(config: Dict, cache_path: str = "ic_weights_cache.json") -> Dict:
    """Blend config composite weights with IC-derived weights from paper trade analysis.

    Blending formula: final_weight = 0.7 * config_weight + 0.3 * ic_weight
    where ic_weight is the raw IC value (floored at 0) normalized to sum to 1.
    Returns plain config weights on any failure.
    """
    global _IC_WEIGHTS_CACHE
    if _IC_WEIGHTS_CACHE is not None:
        return _IC_WEIGHTS_CACHE
    base_weights = config.get("composite_weights", {}) or {}
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        component_ic = cache.get("component_ic", {})
        if not component_ic:
            _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        # Map component_ic keys (e.g. "pop_score") to weight keys (e.g. "pop")
        key_map = {
            "pop_score": "pop", "ev_score": "ev", "rr_score": "rr",
            "liquidity_score": "liquidity", "momentum_score": "momentum",
            "iv_rank_score": "iv_rank", "theta_score": "theta",
        }
        ic_vals = {}
        for ic_key, w_key in key_map.items():
            ic_raw = component_ic.get(ic_key)
            if ic_raw is not None and isinstance(ic_raw, (int, float)):
                ic_vals[w_key] = max(0.0, float(ic_raw))
        if not ic_vals:
            _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        ic_total = sum(ic_vals.values()) or 1.0
        blended = dict(base_weights)
        for w_key, ic_raw in ic_vals.items():
            if w_key in blended:
                ic_norm = ic_raw / ic_total
                blended[w_key] = 0.7 * float(blended[w_key]) + 0.3 * ic_norm
        _IC_WEIGHTS_CACHE = blended
        return _IC_WEIGHTS_CACHE
    except Exception:
        _IC_WEIGHTS_CACHE = base_weights
        return _IC_WEIGHTS_CACHE


def _invalidate_ic_weights_cache() -> None:
    global _IC_WEIGHTS_CACHE
    _IC_WEIGHTS_CACHE = None




def calculate_probability_of_profit(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray], premium: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate probability of profit at expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        premium = np.asanyarray(premium)
        # Clip T to 1 hour minimum to prevent division-by-zero on expiration day
        T = np.maximum(T, 1.0 / (365.0 * 24.0))

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
        # Clip T to 1 hour minimum to prevent division-by-zero on expiration day
        T = np.maximum(T, 1.0 / (365.0 * 24.0))

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
    tnx_change_pct: float,
    short_interest: Optional[float] = None,
    next_ex_div: Optional[object] = None,
    earnings_move_data: Optional[dict] = None,
    mode: str = "Single-stock",
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
    _now_utc = datetime.now(timezone.utc)
    if earnings_date and earnings_date > _now_utc:   # only flag future earnings
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
    df["charm"] = bs_charm(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    df["vanna"] = bs_vanna(S_vals, K_vals, T_vals, risk_free_rate, IV_vals)

    # --- PCR per Expiration ---
    is_call = np.char.lower(types_vals.astype(str)) == "call"
    try:
        pcr_map = {}
        for exp, grp in df.groupby("expiration"):
            call_vol = grp.loc[grp["type"] == "call", "volume"].sum()
            put_vol = grp.loc[grp["type"] == "put", "volume"].sum()
            pcr_val = float(put_vol) / float(call_vol) if call_vol > 0 else np.nan
            pcr_map[exp] = pcr_val
        df["pcr"] = df["expiration"].map(pcr_map)
        def _pcr_signal(v):
            if pd.isna(v):
                return ""
            if v > 1.5:
                return "HEAVY HEDGING"
            if v < 0.5:
                return "BULLISH FLOW"
            return ""
        df["pcr_signal"] = df["pcr"].apply(_pcr_signal)
    except Exception:
        df["pcr"] = np.nan
        df["pcr_signal"] = ""

    # --- GEX (Gamma Exposure) by Strike ---
    try:
        gex_per_contract = df["gamma"].values * df["openInterest"].values * 100.0 * S_vals ** 2
        df["gex"] = np.where(is_call, gex_per_contract, -gex_per_contract)
        gex_by_strike = df.groupby("strike")["gex"].sum().sort_index()
        cumulative_gex = gex_by_strike.cumsum()
        negative_strikes = cumulative_gex[cumulative_gex < 0]
        gex_flip = float(negative_strikes.index[0]) if not negative_strikes.empty else None
        df["gex_flip_price"] = gex_flip
    except Exception:
        df["gex"] = 0.0
        df["gex_flip_price"] = None

    # --- OI Change (Day-over-Day) ---
    _oi_prev = load_oi_snapshot()
    if _oi_prev:
        def _oi_delta(row):
            key = f"{row.get('symbol','')}_{row.get('strike','')}_{row.get('expiration','')}_{row.get('type','')}"
            prev = _oi_prev.get(key)
            if prev is not None:
                return int(row.get("openInterest", 0)) - prev
            return 0
        df["oi_change"] = df.apply(_oi_delta, axis=1)
    else:
        df["oi_change"] = 0

    # --- Short Interest ---
    df["short_interest"] = short_interest if short_interest is not None else pd.NA

    # --- Dividend Warning ---
    df["div_warning"] = ""
    if next_ex_div is not None:
        for idx, row in df.iterrows():
            if row.get("type") == "call" and row.get("abs_delta", 0) > 0.70:
                try:
                    exp_date = row["exp_dt"].replace(tzinfo=None).date()
                    if exp_date >= next_ex_div:
                        df.at[idx, "div_warning"] = f"EX-DIV {next_ex_div}"
                except Exception:
                    pass

    # --- Earnings Implied Move vs Historical ---
    df["implied_earnings_move"] = pd.NA
    df["hist_earnings_move"] = pd.NA
    df["earnings_beat_rate"] = pd.NA
    df["earnings_iv_cheap"] = pd.NA
    if earnings_move_data:
        emd = earnings_move_data
        df["implied_earnings_move"] = emd.get("implied_move_pct")
        df["hist_earnings_move"] = emd.get("hist_avg_move")
        df["earnings_beat_rate"] = emd.get("hist_beat_rate")
        df["earnings_iv_cheap"] = emd.get("is_cheap")

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
    
    # --- Risk Checks (OI wall + gamma ramp, delegated to risk_engine) ---
    from .risk_engine import run_risk_checks
    current_price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
    df = run_risk_checks(df, current_price=current_price_scalar, config=config)

    # IV Skew — 25-delta risk reversal (standard vol surface signal)
    # Finds the call and put closest to 0.25 abs_delta per expiration,
    # then computes skew = Put_25Δ_IV − Call_25Δ_IV.
    # Positive = market paying more for downside protection (fear/hedging demand).
    # Negative = call skew (momentum/squeeze regime).
    df["iv_skew"] = 0.0
    try:
        for exp, exp_grp in df.groupby("expiration"):
            calls_exp = exp_grp[exp_grp["type"] == "call"]
            puts_exp  = exp_grp[exp_grp["type"] == "put"]
            if calls_exp.empty or puts_exp.empty:
                continue
            call_25d = calls_exp.iloc[(calls_exp["abs_delta"] - 0.25).abs().argsort()[:1]]
            put_25d  = puts_exp.iloc[(puts_exp["abs_delta"] - 0.25).abs().argsort()[:1]]
            if call_25d.empty or put_25d.empty:
                continue
            skew_val = (float(put_25d["impliedVolatility"].iloc[0])
                        - float(call_25d["impliedVolatility"].iloc[0]))
            df.loc[exp_grp.index, "iv_skew"] = skew_val
    except Exception:
        pass
    df["iv_skew"] = df["iv_skew"].fillna(0.0)

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

    # --- Vega Dollar Exposure ---
    # Dollar P&L change per 1 volatility point (1%) move in IV, per contract (100 shares).
    # A vega_dollar of $50 means IV moving +1% adds $50 to the position value.
    df["vega_dollar"] = np.abs(df["vega"].values) * 100.0

    # --- Breakeven Distance % ---
    # What % move in the underlying is required to reach breakeven at expiration.
    # Low = more achievable; >1x expected move = structurally difficult.
    with np.errstate(divide='ignore', invalid='ignore'):
        be_dist = np.where(
            df["underlying"].values > 0,
            df["required_move"].values / df["underlying"].values * 100.0,
            np.nan
        )
    df["be_dist_pct"] = np.where(np.isfinite(be_dist), be_dist, np.nan)

    # --- Annualized Return (premium selling context) ---
    # Annualizes the premium collected relative to the strike price.
    # Standard metric for cash-secured puts / covered calls: (premium/strike) * (365/DTE).
    with np.errstate(divide='ignore', invalid='ignore'):
        ann_ret = np.where(
            (df["strike"].values > 0) & (df["T_years"].values > 0),
            (df["premium"].values / df["strike"].values) * (1.0 / df["T_years"].values),
            np.nan
        )
    df["annualized_return"] = np.where(np.isfinite(ann_ret), ann_ret, np.nan)
    
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

    # For Premium Selling, flip PoP to reflect the SELLER's perspective.
    # calculate_probability_of_profit() returns the BUYER's PoP (P option expires ITM).
    # Seller profits when that same option expires worthless, so seller's PoP = 1 − buyer's PoP.
    # e.g. OTM put buyer: 30% PoP → seller: 70% PoP (which is what we want to score highly).
    if mode == "Premium Selling":
        df["prob_profit"] = (1.0 - df["prob_profit"]).clip(0.0, 1.0)

    # Theoretical value and P(ITM) using market IV (for display/reference)
    d1, d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, IV_vals)
    p_itm = np.where(is_call, norm_cdf(d2), norm_cdf(-d2))
    disc = np.exp(-risk_free_rate * T_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        theo_payoff = np.where(is_call,
            S_vals * norm_cdf(d1) - K_vals * disc * norm_cdf(d2),
            K_vals * disc * norm_cdf(-d2) - S_vals * norm_cdf(-d1))
    df["p_itm"], df["theo_value"] = p_itm, theo_payoff

    # HV-adjusted EV: BS(realized_vol) - market_price
    # Positive = options cheap vs realized vol (edge for buyers)
    # Negative = options expensive vs realized vol (edge for sellers)
    # Prefer EWMA vol for EV (more responsive to recent moves); fall back to 30d HV then IV
    hv_for_ev = df["hv_ewma"] if "hv_ewma" in df.columns else df["hv_30d"]
    hv_arr = np.maximum(hv_for_ev.fillna(df["hv_30d"]).fillna(df["impliedVolatility"]).values, 1e-9)
    hv_d1, hv_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, hv_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        hv_payoff = np.where(is_call,
            S_vals * norm_cdf(hv_d1) - K_vals * disc * norm_cdf(hv_d2),
            K_vals * disc * norm_cdf(-hv_d2) - S_vals * norm_cdf(-hv_d1))
    df["ev_per_contract"] = 100.0 * (hv_payoff - prem_vals) - (100.0 * prem_vals * df["spread_pct"].fillna(0.0).values)

    # Earnings-adjusted EV for earnings plays
    df["ev_earnings"] = pd.NA
    _earn_mask_ev = df.get("Earnings Play", pd.Series("NO", index=df.index)) == "YES"
    if _earn_mask_ev.any():
        try:
            emd_hist = pd.to_numeric(
                df.get("hist_earnings_move", pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            emd_impl = pd.to_numeric(
                df.get("implied_earnings_move", pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            eff_sigma = emd_hist.where(emd_hist.notna(), emd_impl * 1.2)
            valid_sigma = _earn_mask_ev & eff_sigma.notna() & (eff_sigma > 0)
            if valid_sigma.any():
                ev_sig_full = np.where(
                    valid_sigma.values,
                    np.maximum(eff_sigma.fillna(0).values, 1e-9),
                    hv_arr,
                )
                ev_d1, ev_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, ev_sig_full)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ev_earn_payoff = np.where(
                        is_call,
                        S_vals * norm_cdf(ev_d1) - K_vals * disc * norm_cdf(ev_d2),
                        K_vals * disc * norm_cdf(-ev_d2) - S_vals * norm_cdf(-ev_d1),
                    )
                ev_earn_raw = 100.0 * (ev_earn_payoff - prem_vals)
                df.loc[valid_sigma, "ev_earnings"] = ev_earn_raw[valid_sigma.values]
        except Exception as _ev_exc:
            logging.getLogger(__name__).warning("ev_earnings computation failed: %s", _ev_exc)

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
    sp = pd.to_numeric(df["spread_pct"], errors="coerce").fillna(float("inf")).clip(lower=0, upper=sp_cap)
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
    # Blend ev_score with ev_earnings_score for earnings plays (Improvement 6)
    if "ev_earnings" in df.columns and "Earnings Play" in df.columns:
        try:
            _earn_play_mask = df["Earnings Play"] == "YES"
            _ev_earn_num = pd.to_numeric(df["ev_earnings"], errors="coerce")
            _ev_earn_valid = _earn_play_mask & _ev_earn_num.notna()
            if _ev_earn_valid.any():
                ev_earnings_score = rank_norm(_ev_earn_num.fillna(df["ev_per_contract"].median()))
                ev_score = ev_score.copy()
                ev_score.loc[_ev_earn_valid] = (
                    0.5 * ev_score.loc[_ev_earn_valid]
                    + 0.5 * ev_earnings_score.loc[_ev_earn_valid]
                )
        except Exception:
            pass
    em_realism_score = pd.to_numeric(df["em_realism_score"], errors='coerce').fillna(0.5).clip(0, 1)
    theta_raw = df["theta_decay_pressure"].replace([pd.NA, pd.NaT], np.nan)
    theta_score = (1.0 - rank_norm(theta_raw.fillna(theta_raw.median()))).clip(0, 1)
    theta_score = theta_score.where((df["T_years"] * 365.0) > 7, theta_score * 0.7)
    
    ret_score = rank_norm(pd.to_numeric(df.get("ret_5d", pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0))
    rsi_vals = pd.to_numeric(df.get("rsi_14", pd.Series(np.nan, index=df.index)), errors="coerce")
    rsi_score = (1.0 - (abs((rsi_vals - 50.0) / 50.0))).clip(0, 1)
    atr_score = rank_norm(pd.to_numeric(df.get("atr_trend", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0))
    momentum_score = 0.4 * ret_score.fillna(0.5) + 0.3 * rsi_score.fillna(0.5) + 0.3 * atr_score.fillna(0.5)
    
    # Blend 30-day and 90-day IV percentile for a more stable IV rank signal.
    # 90-day context prevents over-reacting to short-term vol spikes.
    iv_pct_30 = pd.to_numeric(df.get("iv_percentile_30", df.get("iv_percentile", pd.Series(np.nan, index=df.index))), errors="coerce")
    iv_pct_90 = pd.to_numeric(df.get("iv_percentile_90", pd.Series(np.nan, index=df.index)), errors="coerce")
    # Where 90-day is available, blend 60/40 (30d/90d); otherwise fall back to 30d alone
    iv_pct_series = iv_pct_30.where(iv_pct_90.isna(), 0.6 * iv_pct_30 + 0.4 * iv_pct_90)
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
        try:
            _cdf = pd.DataFrame({"PoP": w["pop"]*pop_score, "RoR": w["return_on_risk"]*ror_score,
                                  "IV rank": w["iv_rank"]*iv_rank_score, "Liq": w["liquidity"]*liquidity,
                                  "Theta": w["theta"]*theta_score, "EV": w["ev"]*ev_score}, index=df.index)
            df["score_drivers"] = _cdf.apply(lambda r: " · ".join(r.nlargest(3).index.tolist()), axis=1)
        except Exception:
            df["score_drivers"] = ""
    else:
        # PCR score
        pcr_vals = pd.to_numeric(df.get("pcr", pd.Series(np.nan, index=df.index)), errors="coerce")
        is_call_series = df["type"].str.lower() == "call"
        pcr_score_call = (1 - np.clip(pcr_vals / 2.0, 0, 1)).fillna(0.5)
        pcr_score_put = np.clip(pcr_vals / 2.0, 0, 1).fillna(0.5)
        pcr_score = pd.Series(np.where(is_call_series, pcr_score_call, pcr_score_put), index=df.index)

        # GEX score
        gex_flip = pd.to_numeric(df.get("gex_flip_price", pd.Series(np.nan, index=df.index)), errors="coerce")
        underlying_s = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce")
        gex_flip_valid = gex_flip.notna()
        gex_score_call = pd.Series(0.5, index=df.index)
        gex_score_put = pd.Series(0.5, index=df.index)
        gex_score_call[gex_flip_valid & (gex_flip > underlying_s)] = 0.7
        gex_score_call[gex_flip_valid & (gex_flip <= underlying_s)] = 0.3
        gex_score_put[gex_flip_valid & (gex_flip < underlying_s)] = 0.7
        gex_score_put[gex_flip_valid & (gex_flip >= underlying_s)] = 0.3
        gex_score = pd.Series(np.where(is_call_series, gex_score_call, gex_score_put), index=df.index)

        # OI change score
        oi_chg = pd.to_numeric(df.get("oi_change", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        oi_change_score = rank_norm(oi_chg)

        # Sentiment score: bullish sentiment helps calls, hurts puts
        raw_sent = pd.to_numeric(
            df.get("sentiment_score", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0).clip(-1.0, 1.0)
        sent_call = ((raw_sent + 0.5) / 1.0).clip(0, 1)
        sent_put = ((0.5 - raw_sent) / 1.0).clip(0, 1)
        sentiment_score_component = pd.Series(
            np.where(is_call_series, sent_call, sent_put), index=df.index
        )

        dw = {
            "pop": 0.20, "em_realism": 0.10, "rr": 0.12, "momentum": 0.08,
            "iv_rank": 0.07, "liquidity": 0.12, "catalyst": 0.04, "theta": 0.08,
            "ev": 0.06, "trader_pref": 0.07, "iv_edge": 0.08, "skew_align": 0.05,
            "gamma_theta": 0.04, "pcr": 0.05, "gex": 0.04, "oi_change": 0.04,
            "sentiment": 0.04,
        }
        cw = load_ic_adjusted_weights(config)
        w = {k: cw.get(k, dw[k]) for k in dw}
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (
            w["pop"]*pop_score + w["em_realism"]*em_realism_score + w["rr"]*rr_score
            + w["momentum"]*momentum_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity
            + w["catalyst"]*catalyst_score + w["theta"]*theta_score + w["ev"]*ev_score
            + w["trader_pref"]*trader_pref_score + w["iv_edge"]*iv_edge_score
            + w["skew_align"]*skew_align_score + w["gamma_theta"]*gamma_theta_score
            + w["pcr"]*pcr_score + w["gex"]*gex_score + w["oi_change"]*oi_change_score
            + w["sentiment"]*sentiment_score_component
        ) / w_sum
        try:
            _cdf = pd.DataFrame({
                "PoP": w["pop"]*pop_score, "EV": w["ev"]*ev_score,
                "RR": w["rr"]*rr_score, "IV edge": w["iv_edge"]*iv_edge_score,
                "Liq": w["liquidity"]*liquidity, "Theta": w["theta"]*theta_score,
                "Mom": w["momentum"]*momentum_score, "Skew": w["skew_align"]*skew_align_score,
                "Sent": w["sentiment"]*sentiment_score_component,
            }, index=df.index)
            df["score_drivers"] = _cdf.apply(lambda r: " · ".join(r.nlargest(3).index.tolist()), axis=1)
        except Exception:
            df["score_drivers"] = ""

    # Adjustments
    df.loc[df["event_flag"] == "EARNINGS_NEARBY", "quality_score"] -= 0.05
    # Reward earnings plays where IV is actually underpriced vs realized vol
    if "Earnings Play" in df.columns and "is_underpriced" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["is_underpriced"] == True), "quality_score"] += 0.08
    df.loc[df["Trend_Aligned"] == True, "quality_score"] += 0.05
    df.loc[df["decay_warning"] == True, "quality_score"] -= 0.20
    # Gamma ramp: near-expiry gamma explosion is a structural risk — penalise hard
    if "gamma_ramp" in df.columns:
        df.loc[df["gamma_ramp"] == True, "quality_score"] -= 0.15
    df.loc[df["sr_warning"] != "", "quality_score"] -= 0.10
    if "seasonal_win_rate" in df.columns:
        df.loc[df["seasonal_win_rate"] >= 0.8, "quality_score"] += 0.10
        df.loc[df["seasonal_win_rate"] <= 0.2, "quality_score"] -= 0.10
    df.loc[df["oi_wall_warning"] != "", "quality_score"] -= 0.10
    df["squeeze_play"] = (df.get("is_squeezing", pd.Series(False, index=df.index)) == True) & (df.get("Unusual_Whale", pd.Series(False, index=df.index)) == True)
    df.loc[df["squeeze_play"], "quality_score"] += 0.25
    df.loc[df["macro_warning"].str.contains("MACRO RISK", na=False), "quality_score"] -= 0.10

    # Short interest squeeze potential
    if "short_interest" in df.columns:
        df.loc[pd.to_numeric(df["short_interest"], errors="coerce").fillna(0) > 0.20, "quality_score"] += 0.05

    # Dividend early exercise warning
    if "div_warning" in df.columns:
        df.loc[df["div_warning"] != "", "quality_score"] -= 0.08

    # Earnings implied move: if IV is cheap vs historical, boost earnings plays
    if "earnings_iv_cheap" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["earnings_iv_cheap"] == True), "quality_score"] += 0.06
        df.loc[(df["Earnings Play"] == "YES") & (df["earnings_iv_cheap"] == False), "quality_score"] -= 0.06

    # --- Charm / Vanna Greek Adjustments ---
    greek_adj = config.get("greek_adjustments", {})

    # Charm penalty: near-expiry OTM options with rapid delta decay
    charm_thresh  = greek_adj.get("charm_penalty_threshold", -0.05)
    charm_penalty = greek_adj.get("charm_penalty_value", -0.05)
    if "charm" in df.columns and "dte" not in df.columns:
        df["dte"] = df["T_years"] * 365.0
    if "charm" in df.columns:
        charm_mask = (df["dte"] < 7) & (pd.to_numeric(df["charm"], errors="coerce").fillna(0) < charm_thresh)
        df.loc[charm_mask, "quality_score"] += charm_penalty

    # Vanna reward: positive vanna in rising IV environment
    vanna_iv_min = greek_adj.get("vanna_reward_iv_rank_min", 0.50)
    vanna_reward  = greek_adj.get("vanna_reward_value", 0.03)
    if "vanna" in df.columns:
        iv_rank_col = df.get("iv_rank_30", pd.Series(np.nan, index=df.index))
        vanna_mask = (
            (pd.to_numeric(df["vanna"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(iv_rank_col, errors="coerce").fillna(0) > vanna_iv_min)
        )
        df.loc[vanna_mask, "quality_score"] += vanna_reward

    # Macro event penalty
    try:
        from .macro_analyzer import get_macro_penalty
        _macro_pen, _macro_active, _macro_desc = get_macro_penalty(config)
        if _macro_active and _macro_pen != 0.0:
            df["quality_score"] += _macro_pen
            df["macro_event_flag"] = _macro_desc
    except Exception:
        pass

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
    short_interest: Optional[float] = None,
    next_ex_div: Optional[object] = None,
    earnings_move_data: Optional[dict] = None,
    hv_ewma: Optional[float] = None,
    news_data=None,
) -> pd.DataFrame:
    # Use richer multi-source news sentiment when available
    if news_data is not None and hasattr(news_data, "aggregate_sentiment"):
        sentiment_score = news_data.aggregate_sentiment

    # Prepare
    now = datetime.now(timezone.utc)
    df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
    df = df[df["exp_dt"].notna()].copy()
    df["T_years"] = (df["exp_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    df = df[(df["T_years"] >= min_dte / 365.0) & (df["T_years"] <= max_dte / 365.0)].copy()

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

    # IV Smile Outlier Filter: remove bad-print IV rows before enrichment
    df = filter_iv_smile_outliers(
        df,
        iv_threshold=config.get("iv_outlier_threshold", 0.30),
        min_volume=config.get("iv_outlier_min_volume", 10),
    )
    if df.empty: return df

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors='coerce')
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    ov_iv_m = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(ov_iv_m if pd.notna(ov_iv_m) else 0.25)

    # Attach EWMA vol column if provided
    if hv_ewma is not None and "hv_ewma" not in df.columns:
        df["hv_ewma"] = hv_ewma

    # 1. Call Helper: Metrics
    df = calculate_metrics(
        df, risk_free_rate, earnings_date, config, iv_rank, iv_percentile,
        sentiment_score, macro_risk_active, sector_perf, tnx_change_pct,
        short_interest=short_interest, next_ex_div=next_ex_div,
        earnings_move_data=earnings_move_data, mode=mode,
    )

    # 2. Call Helper: Scores
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte)

    # Final Filters
    if mode == "Premium Selling":
        d_min = fc.get("premium_selling_delta_min", 0.15)
        d_max = fc.get("premium_selling_delta_max", 0.40)
    else:
        d_min = fc.get("delta_min", 0.15)
        d_max = fc.get("delta_max", 0.35)
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
            "theo_value", "ev_per_contract", "ev_earnings", "ev_score",
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
    if HAS_ENHANCED_CLI:
        colored_prompt = fmt.colorize(prompt, fmt.Colors.BRIGHT_CYAN)
        if default is not None:
            colored_default = fmt.colorize(f"[{default}]", fmt.Colors.DIM)
            val = input(f"{colored_prompt} {colored_default}: ").strip()
        else:
            val = input(f"{colored_prompt}: ").strip()
    else:
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



def _score_fetched_data(
    symbol: str, data_result: dict, mode: str, min_dte: int, max_dte: int,
    rfr: float, config: dict, vix_weights: dict, trader_profile: str,
    budget=None, macro_risk_active: bool = False, tnx_change_pct: float = 0.0,
) -> dict:
    """Score and filter already-fetched options data for one symbol."""
    result = {
        "symbol": symbol,
        "picks": [],
        "credit_spreads": [],
        "iron_condors": [],
        "history": None,
        "success": False,
        "error": None,
    }
    try:
        df_chain = data_result["df"]
        history_df = data_result["history_df"]
        context = data_result["context"]
        result["context"] = context

        if history_df is not None and not history_df.empty:
            result["history"] = history_df

        hv = context.get("hv")
        hv_ewma = context.get("hv_ewma")
        iv_rank = context.get("iv_rank")
        iv_percentile = context.get("iv_percentile")
        earnings_date = context.get("earnings_date")
        earnings_move_data = context.get("earnings_move_data")
        sentiment_score = context.get("sentiment_score")
        seasonal_win_rate = context.get("seasonal_win_rate")
        term_structure_spread = context.get("term_structure_spread")
        sector_perf = context.get("sector_perf", {})
        short_interest = context.get("short_interest")
        next_ex_div = context.get("next_ex_div")
        news_data = context.get("news_data")

        context_log = []
        if hv: context_log.append(f"HV (30d): {hv:.2%}")
        if iv_rank: context_log.append(f"IV Rank: {iv_rank:.2f}")
        if earnings_date: context_log.append(f"Earnings: {earnings_date.strftime('%Y-%m-%d')}")
        if context.get("rvol"): context_log.append(f"RVOL: {context['rvol']:.2f}x")
        result["context_log"] = context_log
        result["news_data"] = news_data

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
            tnx_change_pct=tnx_change_pct,
            short_interest=short_interest,
            next_ex_div=next_ex_div,
            earnings_move_data=earnings_move_data,
            hv_ewma=hv_ewma,
            news_data=news_data,
        )

        if df_scored.empty:
            result["error"] = "No contracts passed filters"
            return result

        if "symbol" not in df_scored.columns:
            result["error"] = f"'symbol' column missing from {symbol} data"
            return result

        is_budget_mode = (mode == "Budget scan")
        if is_budget_mode and budget:
            df_scored["contract_cost"] = df_scored["premium"] * 100
            df_scored = df_scored[df_scored["contract_cost"] <= budget].copy()
            if df_scored.empty:
                result["error"] = "No contracts within budget"
                return result

        if mode == "Credit Spreads":
            spreads = find_credit_spreads(df_scored)
            if not spreads.empty:
                result["credit_spreads"].append(spreads)
                result["success"] = True
        elif mode == "Iron Condor":
            condors = find_iron_condors(df_scored)
            if not condors.empty:
                result["iron_condors"] = condors
                result["success"] = True
        elif mode == "Premium Selling":
            puts = df_scored[df_scored["type"] == "put"].copy()
            if not puts.empty:
                result["picks"].append(puts)
                result["success"] = True
        else:
            result["picks"].append(df_scored)
            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def process_ticker(symbol: str, mode: str, max_expiries: int, min_dte: int, max_dte: int,
                   rfr: float, config: dict, vix_weights: dict, trader_profile: str,
                   budget=None, macro_risk_active: bool = False, tnx_change_pct: float = 0.0) -> dict:
    """Thin wrapper: fetch data then score it."""
    try:
        data_result = fetch_options_yfinance(symbol, max_expiries)
        return _score_fetched_data(symbol, data_result, mode, min_dte, max_dte, rfr,
                                   config, vix_weights, trader_profile, budget,
                                   macro_risk_active, tnx_change_pct)
    except Exception as e:
        return {"symbol": symbol, "picks": [], "credit_spreads": [], "iron_condors": [],
                "history": None, "success": False, "error": str(e)}


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

    WIDTH = get_display_width()
    if verbose:
        if HAS_ENHANCED_CLI:
            print("\n" + fmt.draw_box(f"Scanning {len(tickers)} ticker(s)", WIDTH))
        else:
            print(f"\n{'='*WIDTH}")
            print(f"  Fetching data for {len(tickers)} ticker(s)...")
            print(f"{'='*WIDTH}\n")

    # ── Pre-scan active filter summary ───────────────────────────────────────
    if verbose:
        _fc = config.get("filters", {})
        _d_min = _fc.get("delta_min", 0.15)
        _d_max = _fc.get("delta_max", 0.35)
        _spread_cap = _fc.get("max_bid_ask_spread_pct", 0.40)
        _min_vol = _fc.get("min_volume", 50)
        _iv_pct_min = _fc.get("min_iv_percentile", 20)
        _filter_line = (
            f"  Filters  DTE: {min_dte}\u2013{max_dte}d"
            f"  |  \u0394: {_d_min:.2f}\u2013{_d_max:.2f}"
            f"  |  Spread \u2264{_spread_cap*100:.0f}%"
            f"  |  Vol \u2265{_min_vol}"
            f"  |  IV%ile \u2265{_iv_pct_min}"
        )
        print(fmt.colorize(_filter_line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else _filter_line)

    # IV history coverage summary — print once before fetching starts
    if verbose:
        try:
            from .data_fetching import iv_history_coverage as _iv_cov
            _cov_parts = []
            for _sym in tickers:
                try:
                    _cov = _iv_cov(_sym)
                    _cov_parts.append(f"{_sym}: {_cov['days']}d ({_cov['confidence']})")
                except Exception:
                    pass
            if _cov_parts:
                _cov_line = "  IV history: " + "  |  ".join(_cov_parts)
                print(fmt.colorize(_cov_line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else _cov_line)
        except Exception:
            pass

    results_buffer: Dict[str, Any] = {}

    # Phase 1 — Pre-fetch all chains in parallel (ThreadPoolExecutor, respects max_expiries)
    def _fetch_one(sym: str):
        try:
            return sym, fetch_options_yfinance(sym, max_expiries)
        except Exception as exc:
            return sym, {"error": str(exc)}

    raw_results: Dict[str, Any] = {}
    with _suppress_scan_noise():
        with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
            for sym, result in executor.map(_fetch_one, tickers):
                raw_results[sym] = result

    # Phase 2 — Score each fetched result (with optional tqdm progress bar)
    if HAS_ENHANCED_CLI and verbose:
        bar_fmt = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        score_iter = tqdm(
            tickers,
            total=len(tickers),
            desc="  Scanning",
            unit="",
            leave=False,
            dynamic_ncols=True,
            bar_format=bar_fmt,
            file=sys.stderr,
        )
    else:
        score_iter = tickers

    for symbol in score_iter:
        data_result = raw_results.get(symbol)
        if data_result is None or "error" in data_result:
            err_msg = (data_result or {}).get("error", "fetch returned no data")
            results_buffer[symbol] = {
                'success': False, 'error': err_msg,
                'context_log': [], 'picks': [],
                'credit_spreads': [], 'iron_condors': pd.DataFrame(),
                'history': None
            }
            continue
        results_buffer[symbol] = _score_fetched_data(
            symbol, data_result, mode, min_dte, max_dte,
            rfr, config, vix_weights, trader_profile,
            budget, macro_risk_active, tnx_change_pct
        )


    # Print per-ticker summary after all futures complete
    if verbose:
        ok, fail = [], []
        for symbol in tickers:
            result = results_buffer.get(symbol, {})
            if result.get('success'):
                n = (sum(len(p) for p in result.get('picks', []))
                     + sum(len(s) for s in result.get('credit_spreads', []))
                     + (len(result['iron_condors']) if isinstance(result.get('iron_condors'), pd.DataFrame) and not result['iron_condors'].empty else 0))
                ok.append((symbol, n))
            else:
                fail.append((symbol, result.get('error', 'no contracts passed filters')))

        if ok or fail:
            sep = fmt.draw_separator(WIDTH) if HAS_ENHANCED_CLI else "-" * WIDTH
            print(sep)
            for sym, n in ok:
                cov_str = ""
                try:
                    from .data_fetching import iv_history_coverage as _iv_cov
                    _cov = _iv_cov(sym)
                    cov_str = f"  IV: {_cov['days']}d ({_cov['confidence']})"
                except Exception:
                    pass
                line = f"  \u2713 {sym:<6}  {n} contract(s){cov_str}"
                print(fmt.colorize(line, fmt.Colors.GREEN) if HAS_ENHANCED_CLI else line)
            for sym, err in fail:
                # Only show brief reason, not the full stack trace
                short_err = str(err).split('\n')[0][:60]
                line = f"  \u2013 {sym:<6}  {short_err}"
                print(fmt.colorize(line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else line)
            print(sep)
            print()

    # Aggregate buffered results — also collect news_data per ticker
    news_map: Dict[str, Any] = {}
    for symbol, result in results_buffer.items():
        if result.get('success'):
            if result.get('history') is not None:
                ticker_histories[symbol] = result['history']
            for picks_df in result.get('picks', []):
                all_picks.append(picks_df)
            for spreads_df in result.get('credit_spreads', []):
                all_credit_spreads.append(spreads_df)
            condors = result.get('iron_condors')
            if isinstance(condors, pd.DataFrame) and not condors.empty:
                all_iron_condors.append(condors)
        if result.get('news_data') is not None:
            news_map[symbol] = result['news_data']

    ticker_contexts: dict = {}
    for symbol, result in results_buffer.items():
        if result.get("success") and result.get("context"):
            ticker_contexts[symbol] = result["context"]

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

    # --- Portfolio GEX Gate ---
    try:
        from .portfolio_risk import RiskAggregator
        _risk = RiskAggregator(config=config)
        _risk_off, _risk_reason = _risk.is_risk_off_required(config)
        if _risk_off and verbose:
            _warn_msg = f"RISK-OFF MODE: {_risk_reason}"
            if HAS_ENHANCED_CLI:
                print(fmt.format_warning(_warn_msg))
            else:
                print(f"  ⚠️  {_warn_msg}")
        if _risk_off and not picks.empty and "abs_delta" in picks.columns:
            picks = picks[picks["abs_delta"] < 0.30].copy()
    except Exception:
        pass

    # Concentration warning across scan results
    if verbose and not picks.empty and len(picks) >= 5:
        call_count = (picks["type"].str.lower() == "call").sum()
        put_count = (picks["type"].str.lower() == "put").sum()
        total = len(picks)
        if call_count / total > 0.80:
            msg = f"Concentration warning: {call_count}/{total} picks are CALLS \u2014 portfolio skews heavily bullish"
            print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")
        elif put_count / total > 0.80:
            msg = f"Concentration warning: {put_count}/{total} picks are PUTS \u2014 portfolio skews heavily bearish"
            print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")
        if "symbol" in picks.columns:
            symbol_counts = picks["symbol"].value_counts()
            if not symbol_counts.empty:
                dominant = symbol_counts.index[0]
                dominant_count = symbol_counts.iloc[0]
                if dominant_count >= 5:
                    msg = f"Concentration warning: {dominant_count} picks from {dominant} \u2014 consider diversifying"
                    print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")

    # Generate Final Reports
    if mode == "Budget scan":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=budget)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, budget=budget, market_trend=market_trend, volatility_regime=volatility_regime, config=config)
        elif verbose:
            print("\nNo options found within budget.")

    elif mode == "Discovery scan":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=None)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config)
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
                print_report(final_df.head(10), underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config)
        elif verbose:
            print("\nNo premium selling candidates found.")

    else:
        # Single stock mode
        if not picks.empty:
            final_df = picks.copy()
            final_df = categorize_by_premium(final_df, budget=None)
            if verbose:
                print_report(final_df, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config)
        elif verbose:
            print("\nNo suitable options found.")

        # Vol analytics for single-ticker scans
        if verbose and HAS_VOL_ANALYTICS and len(tickers) == 1:
            try:
                _ticker_sym = tickers[0]
                _current_iv = float(picks["impliedVolatility"].median()) if not picks.empty and "impliedVolatility" in picks.columns else None
                _current_price = underlying_price if underlying_price and underlying_price > 0 else None
                print_vol_cone(_ticker_sym, current_iv=_current_iv, width=WIDTH)
                print_iv_surface(_ticker_sym, spot=_current_price, width=WIDTH)
                print_regime_summary(_ticker_sym, current_iv=_current_iv, width=WIDTH)
            except Exception:
                pass

    # Phase 4: Executive Summary
    if verbose and HAS_ENHANCED_CLI and not picks.empty:
        print_executive_summary(
            picks,
            config,
            mode=mode,
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            macro_risk=macro_risk_active,
            num_tickers=len(tickers)
        )

    # Phase 5: News & Events digest — shown after picks so it doesn't interrupt the report flow
    if verbose and news_map and not picks.empty:
        print_news_panel(news_map, picks, width=WIDTH)

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

    return ScanResult(
        picks=picks,
        spreads=pd.DataFrame(),
        credit_spreads=credit_spreads_df,
        iron_condors=iron_condors_df,
        top_pick=top_pick,
        underlying_price=underlying_price,
        rfr=rfr,
        chain_iv_median=chain_iv_median,
        timestamp=datetime.now().isoformat(),
        ticker_contexts=ticker_contexts,
        market_context={
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'market_trend': market_trend,
            'volatility_regime': volatility_regime,
            'macro_risk_active': macro_risk_active,
            'tnx_change_pct': tnx_change_pct
        },
    )

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


def _check_market_hours() -> tuple:
    """
    Returns (is_open: bool, message: str) in US Eastern time.
    Options are liquid 09:30–16:00 ET, Mon–Fri. Outside this window,
    yfinance data is stale and bid-ask spreads are unreliable.
    """
    return True, "Market hours check unavailable"
    try:
        from zoneinfo import ZoneInfo
        et_zone = ZoneInfo("America/New_York")
        now_et = datetime.now(et_zone)
    except Exception:
        # Fallback: rough UTC-4 (EDT) — acceptable for a warning-only check
        now_et = datetime.now(timezone(timedelta(hours=-4)))

    weekday = now_et.weekday()  # 0=Mon … 6=Sun
    hhmm = now_et.hour * 100 + now_et.minute
    time_str = now_et.strftime("%H:%M ET")

    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets closed — it's {day_name} ({time_str}). Data is stale."

    if hhmm < 930:
        return False, f"Pre-market ({time_str}). Options open at 09:30 ET — bid/ask spreads not reliable yet."

    if hhmm >= 1600:
        return False, f"After-hours ({time_str}). Options closed at 16:00 ET — quotes are stale."

    return True, f"Market open ({time_str})"


def _run_ai_pipeline(picks: "pd.DataFrame", volatility_regime: str, verbose: bool = True) -> "Optional[pd.DataFrame]":
    """Thin wrapper: delegates to ai_rank pipeline so CLI and ai_rank.py share one code path."""
    try:
        from ai_rank import score_and_rank
        from src.ranking import print_ranked_table
        from src.config_ai import AI_CONFIG
        from src.data_fetching import _CHAIN_CACHE

        vix_map = {"Low": "low", "Normal": "normal", "High": "high"}
        vix_regime = vix_map.get(str(volatility_regime), "normal")

        candidates = picks.sort_values("quality_score", ascending=False).head(20).copy()

        ticker_contexts = {}
        if AI_CONFIG.get("two_pass_enabled", True):
            for sym in candidates["symbol"].unique():
                if sym in _CHAIN_CACHE:
                    ticker_contexts[sym] = _CHAIN_CACHE[sym].get("context", {})

        ranked = score_and_rank(candidates, ticker_contexts, vix_regime)
        if verbose:
            print_ranked_table(ranked, top_n=10)
        return ranked
    except Exception as exc:
        print(f"  AI scoring unavailable: {exc}")
        return None

def main():
    # ── CLI argument parsing (Phase 7) ───────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--help", "-h", action="store_true", help="Show help and exit")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--close-trades", action="store_true", help="Update trade log with closing P/L")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI analysis after scan")
    args, _ = parser.parse_known_args()

    if args.no_color and HAS_ENHANCED_CLI:
        fmt.set_color_enabled(False)

    if args.version:
        print("Options Screener v1.0.0")
        sys.exit(0)

    WIDTH = get_display_width()

    if args.help:
        if HAS_ENHANCED_CLI:
            print(fmt.draw_box("OPTIONS SCREENER  \u2014  HELP", WIDTH, double=True))
            print(fmt.colorize("\nUsage:", fmt.Colors.BRIGHT_CYAN, bold=True))
            print("  python -m src.options_screener [OPTIONS]\n")
            print(fmt.colorize("Options:", fmt.Colors.BRIGHT_CYAN, bold=True))
            for flag, desc in [
                ("--no-color",     "Disable colored output"),
                ("-h, --help",     "Show this help and exit"),
                ("--version",      "Show version string and exit"),
                ("--close-trades", "Update trade log with closing P/L"),
                ("--ui",           "Launch the Streamlit dashboard"),
            ]:
                print(f"  {fmt.colorize(f'{flag:<18}', fmt.Colors.BRIGHT_YELLOW)} {desc}")
        else:
            print("Options Screener v1.0.0")
            print("Usage: python -m src.options_screener [--no-color] [-h/--help] [--version] [--close-trades] [--ui]")
        sys.exit(0)

    if args.close_trades:
        close_trades()
        sys.exit(0)

    if args.ui:
        import subprocess
        print("Launching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "src/dashboard.py"])
        sys.exit(0)

    config = load_config("config.json")

    # ── Startup Banner (Phase 1) ─────────────────────────────────────────────
    now_str = datetime.now().strftime("%a %d %b %Y  %H:%M")
    if HAS_ENHANCED_CLI:
        print("\n" + fmt.draw_box("OPTIONS SCREENER  \u2022  Pro Edition", WIDTH, double=True))
        print(fmt.colorize(f"  {now_str}", fmt.Colors.DIM))
    else:
        print("\n" + "=" * WIDTH)
        print("  OPTIONS SCREENER  \u2022  Pro Edition")
        print(f"  {now_str}")
        print("=" * WIDTH)

    print(fmt.colorize("  Note: For personal/informational use only. Review data provider terms.", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "  Note: For personal/informational use only. Review data provider terms.")

    # ── Regime Dashboard ─────────────────────────────────────────────────────
    try:
        from .regime_dashboard import print_regime_dashboard
        print_regime_dashboard(WIDTH)
    except Exception:
        pass

    # ── Market Hours Check ───────────────────────────────────────────────────
    is_open, mkt_msg = _check_market_hours()
    if not is_open:
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(mkt_msg))
            print(fmt.colorize("  Quotes are 15+ min delayed. Use results for planning, not live execution.", fmt.Colors.DIM))
        else:
            print(f"⚠  {mkt_msg}")
            print("  Quotes are 15+ min delayed. Use results for planning, not live execution.")
    print()

    # Initialize PaperManager and silently auto-close any TP/SL hits
    pm = PaperManager(db_path="paper_trades.db", config_path="config.json")
    pm.update_positions()

    # ── Mode Menu (Phase 1) ──────────────────────────────────────────────────
    _wl = load_watchlist()
    _wl_desc = f"Scan your {len(_wl)} saved ticker(s)" if _wl else "(empty \u2014 type ADD AAPL to begin)"
    if HAS_ENHANCED_CLI:
        print("\n" + fmt.draw_separator(WIDTH))
        modes = [
            ("1", "TICKER",    "Single-stock deep analysis (e.g. AAPL)"),
            ("2", "ALL",       "Budget-based multi-stock scan"),
            ("3", "DISCOVER",  "Top 100 most-traded tickers \u2014 no budget limit"),
            ("4", "SELL",      "Premium Selling \u2014 income via short puts"),
            ("5", "SPREADS",   "Credit Spread analysis"),
            ("6", "IRON",      "Iron Condor analysis \u2014 range-bound"),
            ("7", "PORTFOLIO", "View open position P/L"),
            ("8", "MY LIST",   _wl_desc),
        ]
        for num, cmd, desc in modes:
            n = fmt.colorize(f"[{num}]", fmt.Colors.BRIGHT_YELLOW)
            c = fmt.colorize(f"{cmd:<10}", fmt.Colors.BRIGHT_WHITE, bold=True)
            d = fmt.colorize(f"\u2014 {desc}", fmt.Colors.DIM)
            print(f"  {n} {c} {d}")
        print(fmt.draw_separator(WIDTH))
    else:
        print("\nModes:")
        print("  [1] TICKER     \u2014 Single-stock deep analysis (e.g. AAPL)")
        print("  [2] ALL        \u2014 Budget-based multi-stock scan")
        print("  [3] DISCOVER   \u2014 Top 100 most-traded tickers (no budget limit)")
        print("  [4] SELL       \u2014 Premium Selling analysis (short puts)")
        print("  [5] SPREADS    \u2014 Credit Spread analysis")
        print("  [6] IRON       \u2014 Iron Condor analysis")
        print("  [7] PORTFOLIO  \u2014 View open position P/L")
        print(f"  [8] MY LIST    \u2014 {_wl_desc}")
    print()

    symbol_input = prompt_input("Enter number, ticker, or command (default: 3)", "3").upper()

    # ── Watchlist commands ────────────────────────────────────────────────────
    if symbol_input.startswith("ADD "):
        add_to_watchlist(symbol_input[4:].strip())
        sys.exit(0)
    if symbol_input.startswith("REMOVE "):
        remove_from_watchlist(symbol_input[7:].strip())
        sys.exit(0)
    if symbol_input in ("SHOW LIST", "SHOW"):
        wl_cur = load_watchlist()
        if wl_cur:
            print(f"  Your watchlist ({len(wl_cur)} tickers): " + ", ".join(wl_cur))
        else:
            print("  Watchlist is empty. Type ADD AAPL to begin.")
        sys.exit(0)

    # ── Number → command mapping ──────────────────────────────────────────────
    _num_map = {"1": "TICKER", "2": "ALL", "3": "DISCOVER", "4": "SELL",
                "5": "SPREADS", "6": "IRON", "7": "PORTFOLIO", "8": "MY LIST"}
    if symbol_input in _num_map:
        symbol_input = _num_map[symbol_input]

    if symbol_input == "PORTFOLIO":
        from .check_pnl import view_portfolio
        view_portfolio()
        sys.exit(0)

    # ── MY LIST mode ──────────────────────────────────────────────────────────
    is_my_list_mode = (symbol_input in ("MY LIST", "MYLIST"))
    if is_my_list_mode:
        _wl_tickers = load_watchlist()
        if not _wl_tickers:
            print("  Your watchlist is empty. Type ADD AAPL to add a ticker first.")
            sys.exit(0)
        symbol_input = "DISCOVER"  # reuse discovery flow with custom ticker list

    is_budget_mode = (symbol_input == "ALL")
    is_discovery_mode = (symbol_input in ("DISCOVER", "")) or is_my_list_mode
    is_ticker_mode = (symbol_input == "TICKER")  # user chose [1] — will prompt for symbol
    is_premium_selling_mode = (symbol_input == "SELL")
    is_credit_spread_mode = (symbol_input == "SPREADS")
    is_iron_condor_mode = (symbol_input == "IRON")

    if is_my_list_mode: mode = "Discovery scan"
    elif is_discovery_mode: mode = "Discovery scan"
    elif is_budget_mode: mode = "Budget scan"
    elif is_premium_selling_mode: mode = "Premium Selling"
    elif is_credit_spread_mode: mode = "Credit Spreads"
    elif is_iron_condor_mode: mode = "Iron Condor"
    else: mode = "Single-stock"

    budget = None
    tickers = []

    if is_my_list_mode:
        tickers = _wl_tickers
        print(f"  Scanning your watchlist: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_discovery_mode or is_premium_selling_mode or is_credit_spread_mode or is_iron_condor_mode:
        tickers = prompt_for_tickers()
        print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_budget_mode:
        try:
            budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
            if budget <= 0:
                print("Budget must be greater than 0."); sys.exit(1)
        except Exception:
            print("Invalid budget amount."); sys.exit(1)
        scan_type = prompt_input("Enter 1 for TARGETED or 2 for DISCOVERY", "1")
        if scan_type == "2": tickers = prompt_for_tickers()
        else:
            default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
            tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    elif is_ticker_mode:
        ticker_sym = prompt_input("Enter stock ticker symbol", "AAPL").upper()
        if not ticker_sym.isalnum():
            print("Please enter a valid alphanumeric ticker."); sys.exit(1)
        tickers = [ticker_sym]
    else:
        if not symbol_input.isalnum():
            print("Please enter a valid alphanumeric ticker."); sys.exit(1)
        tickers = [symbol_input]
    
    logger = setup_logging()
    print("\nFetching market context (SPY/VIX)...")
    market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
    if HAS_ENHANCED_CLI:
        trend_color = fmt.Colors.GREEN if market_trend == "Bullish" else (fmt.Colors.RED if market_trend == "Bearish" else fmt.Colors.YELLOW)
        vix_color = fmt.Colors.GREEN if volatility_regime == "Low" else (fmt.Colors.RED if volatility_regime == "High" else fmt.Colors.YELLOW)
        trend_str = fmt.colorize(market_trend, trend_color, bold=True)
        vol_str = fmt.colorize(volatility_regime, vix_color)
        print(f"\u2713 Market Trend: {trend_str} | Volatility: {vol_str}")
        if macro_risk_active:
            print(fmt.format_warning("Macro risk active \u2014 elevated market uncertainty"))
    else:
        print(f"\u2713 Market Trend: {market_trend} | Volatility: {volatility_regime}")

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

    _is_single_stock = (mode == "Single-stock")
    _repeat_count = 0

    try:
        while True:
            scan_results = run_scan(mode=mode, tickers=tickers, budget=budget, max_expiries=max_expiries, min_dte=min_dte, max_dte=max_dte, trader_profile=trader_profile, logger=logger, market_trend=market_trend, volatility_regime=volatility_regime, macro_risk_active=macro_risk_active, tnx_change_pct=tnx_change_pct)
            if scan_results is None: sys.exit(0)

            picks = scan_results.picks
            rfr = scan_results.rfr
            chain_iv_median = scan_results.chain_iv_median

            # ── AI Analysis ────────────────────────────────────────────────
            _ai_ranked = None
            if not picks.empty and not getattr(args, 'no_ai', False):
                _ai_ranked = _run_ai_pipeline(picks, volatility_regime, verbose=True)

            # Pull spread/condor results for the save menu
            _credit_spreads = scan_results.credit_spreads
            _iron_condors   = scan_results.iron_condors
            _has_results = (
                not picks.empty
                or (isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty)
                or (isinstance(_iron_condors,   pd.DataFrame) and not _iron_condors.empty)
            )

            # ── Collapsed post-scan prompt (always shown BEFORE scan-another) ──
            if _has_results:
                if HAS_ENHANCED_CLI:
                    save_label = fmt.colorize("Save/Export?", fmt.Colors.BRIGHT_CYAN)
                    p_opt = fmt.colorize("[P]", fmt.Colors.BRIGHT_YELLOW) + " Paper trade top pick"
                    c_opt = fmt.colorize("[C]", fmt.Colors.BRIGHT_YELLOW) + " CSV"
                    l_opt = fmt.colorize("[L]", fmt.Colors.BRIGHT_YELLOW) + " Log trades"
                    skip_opt = fmt.colorize("[Enter]", fmt.Colors.DIM) + " Skip"
                    print(f"\n  {save_label}  {p_opt}  \u00b7  {c_opt}  \u00b7  {l_opt}  \u00b7  {skip_opt}")
                else:
                    print("\n  Save/Export?  [P] Paper trade top pick  [C] CSV  [L] Log trades  [Enter] Skip")
                save_choice = prompt_input("Choice", "").strip().upper()

                if save_choice == "P":
                    if mode in ("Credit Spreads", "Iron Condor"):
                        msg = "Paper trading for spreads/condors is not supported — use [L] Log trades instead."
                        print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  \u26a0  {msg}")
                    elif not picks.empty:
                        # Use AI-ranked top pick when available, otherwise fall back to quality_score
                        if _ai_ranked is not None and not _ai_ranked.empty and "final_score" in _ai_ranked.columns:
                            _best_idx = _ai_ranked.sort_values("final_score", ascending=False).index[0]
                            top_pick_row = picks.loc[_best_idx] if _best_idx in picks.index \
                                else picks.sort_values("quality_score", ascending=False).iloc[0]
                        else:
                            top_pick_row = picks.sort_values("quality_score", ascending=False).iloc[0]
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        trade_dict = {
                            "date": today_str,
                            "ticker": top_pick_row["symbol"],
                            "expiration": top_pick_row["expiration"],
                            "strike": top_pick_row["strike"],
                            "type": str(top_pick_row["type"]).capitalize(),
                            "entry_price": (
                                safe_float(top_pick_row.get("ask") or None)
                                or safe_float(top_pick_row.get("lastPrice"))
                                or safe_float(top_pick_row.get("premium"), 0.0)
                            ),
                            "quality_score": top_pick_row["quality_score"],
                            "strategy_name": f"Long {str(top_pick_row['type']).capitalize()}"
                        }
                        pm.log_trade(trade_dict)
                        msg = f"Paper trade logged: {top_pick_row['symbol']} {str(top_pick_row['type']).upper()} ${top_pick_row['strike']:.0f}"
                        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")

                elif save_choice == "C":
                    # Export best available data: AI-ranked picks > raw picks > spreads > condors
                    if _ai_ranked is not None and not _ai_ranked.empty:
                        export_df = _ai_ranked
                    elif not picks.empty:
                        export_df = picks
                    elif isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty:
                        export_df = _credit_spreads
                    else:
                        export_df = _iron_condors
                    csv_file = export_to_csv(export_df, mode, budget)
                    if csv_file:
                        msg = f"Results exported to: {csv_file}"
                        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"\n  \U0001f4c4 {msg}")

                elif save_choice == "L":
                    log_src = picks if not picks.empty else (
                        _credit_spreads if isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty
                        else _iron_condors
                    )
                    if isinstance(log_src, pd.DataFrame) and not log_src.empty:
                        picks_to_log = select_trades_to_log(log_src)
                        if not picks_to_log.empty:
                            log_trade_entry(picks_to_log, mode)
                            msg = f"Logged {len(picks_to_log)} trades."
                            print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2705 {msg}")

            # ── Scan-another shortcut (single-stock only, AFTER save menu) ──
            if _is_single_stock and _repeat_count < 5:
                _another = prompt_input("Scan another ticker? (enter symbol or Enter to quit)", "").upper().strip()
                if _another and _another.isalnum() and 1 <= len(_another) <= 6:
                    tickers = [_another]
                    _repeat_count += 1
                    continue  # loop back

            # Done message
            if HAS_ENHANCED_CLI:
                WIDTH = get_display_width()
                print("\n" + fmt.draw_separator(WIDTH, fmt.BoxChars.D_HORIZONTAL))
                print(fmt.colorize("  \U0001f44b  Done! Happy trading!", fmt.Colors.BRIGHT_GREEN, bold=True))
                print(fmt.draw_separator(WIDTH, fmt.BoxChars.D_HORIZONTAL) + "\n")
            else:
                print("\n\U0001f44b Done! Happy trading!\n")
            break

    except KeyboardInterrupt: print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
