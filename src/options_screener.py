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
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
from finvizfinance.screener.performance import Performance
import functools
import random

# Default liquid tickers for fallback
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL",
    "META", "NFLX", "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "UNH", "HD"
]

def retry_with_backoff(retries=3, backoff_in_seconds=1, error_types=(Exception,)):
    """
    Decorator to retry a function with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if x == retries:
                        logging.warning(f"Function {func.__name__} failed after {retries} retries: {e}")
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                    logging.info(f"Retrying {func.__name__} in {sleep:.2f}s due to error: {e}")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator


@retry_with_backoff(retries=3, backoff_in_seconds=2, error_types=(RuntimeError, URLError, ConnectionError, OSError))
def get_dynamic_tickers(scan_type: str, max_tickers: int = 50) -> List[str]:
    """
    Fetches a list of tickers from Finviz based on a given scan type.
    Falls back to DEFAULT_TICKERS if fetching fails.
    """
    filters_dict = {
        'Option/Short': 'Optionable',
        'Average Volume': 'Over 500K',
        'Country': 'USA',
    }
    order = 'Change'  # Default for gainers
    if scan_type == 'high_iv':
        order = 'Volatility (Month)'

    try:
        fperformance = Performance()
        fperformance.set_filter(filters_dict=filters_dict)
        # Suppress print output from finvizfinance if possible, or just let it be.
        # finvizfinance often prints "Scraping..." which we can't easily silence without redirecting stdout.
        df = fperformance.screener_view(order=order, limit=max_tickers, verbose=0)

        if df.empty:
            logging.warning("Finviz returned empty dataframe. Using default tickers.")
            return DEFAULT_TICKERS[:max_tickers]

        tickers = df['Ticker'].tolist()
        return tickers
    except Exception as e:
        logging.warning(f"Could not fetch '{scan_type}' from Finviz: {e}. Using default tickers.")
        return DEFAULT_TICKERS[:max_tickers]


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
    safe_float
)

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
        "monte_carlo_simulations": 10000
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





def norm_cdf(x: float) -> float:
    # Standard normal CDF using erf to avoid external deps
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """
    Black-Scholes delta. Returns:
      call: N(d1)
      put:  N(d1) - 1
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        if option_type.lower() == "call":
            return norm_cdf(d1)
        else:
            return norm_cdf(d1) - 1.0
    except Exception:
        return None


def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes theoretical option price (using N(d1), N(d2))."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type.lower() == "call":
            return S * math.exp(r * T) * norm_cdf(d1) - K * norm_cdf(d2)
        else:
            return K * norm_cdf(-d2) - S * math.exp(r * T) * norm_cdf(-d1)
    except Exception:
        return None


def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute d1, d2 for Black-Scholes; returns (d1, d2) or (None, None)."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None, None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except Exception:
        return None, None


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / (math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * x * x)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes gamma: N'(d1) / (S * sigma * sqrt(T))."""
    try:
        d1, _ = _d1d2(S, K, T, r, sigma)
        if d1 is None:
            return None
        return norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        return None


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes vega: S * N'(d1) * sqrt(T)."""
    try:
        d1, _ = _d1d2(S, K, T, r, sigma)
        if d1 is None:
            return None
        return S * norm_pdf(d1) * math.sqrt(T) / 100 # per 1% change in IV
    except Exception:
        return None


def bs_theta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes theta (per day)."""
    try:
        d1, d2 = _d1d2(S, K, T, r, sigma)
        if d1 is None or d2 is None:
            return None

        p1 = - (S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        if option_type.lower() == "call":
            p2 = r * K * math.exp(-r * T) * norm_cdf(d2)
            theta = (p1 - p2) / 365.0
        else: # put
            p2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
            theta = (p1 + p2) / 365.0

        return theta
    except Exception:
        return None


def calculate_probability_of_profit(option_type: str, S: float, K: float, T: float, sigma: float, premium: float) -> Optional[float]:
    """Calculate probability of profit at expiration."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or premium <= 0:
            return None
        
        # Break-even point
        if option_type.lower() == "call":
            breakeven = K + premium
        else:  # put
            breakeven = K - premium
        
        # Probability that stock will be beyond break-even
        d = (math.log(S / breakeven) - (0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower() == "call":
            return norm_cdf(d)
        else:
            return 1.0 - norm_cdf(d)
    except Exception:
        return None


def calculate_expected_move(S: float, sigma: float, T: float) -> Optional[float]:
    """Calculate expected move (1 standard deviation) until expiration."""
    try:
        if S <= 0 or sigma <= 0 or T <= 0:
            return None
        return S * sigma * math.sqrt(T)
    except Exception:
        return None


def calculate_probability_of_touch(option_type: str, S: float, K: float, T: float, sigma: float) -> Optional[float]:
    """Calculate probability that option will touch the strike price before expiration."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        
        # Probability of touching is approximately 2 * delta for ATM options
        # More precise: P(touch) ≈ 2 * N(d2)
        d2 = (math.log(S / K) - (0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower() == "call" and K > S:
            return 2 * norm_cdf(d2)
        elif option_type.lower() == "put" and K < S:
            return 2 * (1.0 - norm_cdf(d2))
        else:
            # ITM already touched
            return 1.0
    except Exception:
        return None


def calculate_risk_reward(
    option_type: str,
    premium: float,
    S: float,
    K: float,
    expected_move: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate max loss, break-even, and risk/reward ratio.

    Uses the prompt's definition:
      - target_price = stock_price ± 0.75 * EM
      - RR = max_gain_if_target_hit / premium
    where gains and premium are measured per share.
    """
    try:
        if premium <= 0 or S <= 0 or K <= 0:
            return None, None, None

        max_loss = premium * 100  # Per contract
        otype = option_type.lower()

        # Break-even price
        if otype == "call":
            breakeven = K + premium
        else:  # put
            breakeven = K - premium

        # Compute max gain at target using expected move when available
        if expected_move is not None and expected_move > 0:
            if otype == "call":
                target_price = S + 0.75 * expected_move
                payoff_per_share = max(0.0, target_price - K)
            else:
                target_price = S - 0.75 * expected_move
                payoff_per_share = max(0.0, K - target_price)
        else:
            # Fallback: simple heuristic target if EM is unavailable
            if otype == "call":
                target_price = S * 1.5
                payoff_per_share = max(0.0, target_price - K)
            else:
                target_price = S * 0.5
                payoff_per_share = max(0.0, K - target_price)

        max_gain_per_share = max(0.0, payoff_per_share - premium)
        risk_reward_ratio = max_gain_per_share / premium if premium > 0 else 0.0

        return max_loss, breakeven, risk_reward_ratio
    except Exception:
        return None, None, None





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
    # Prepare and filter
    now = datetime.now(timezone.utc)

    # expiration to dt
    df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
    df = df[df["exp_dt"].notna()].copy()
    df["T_years"] = (df["exp_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    # filter by DTE bounds
    df = df[(df["T_years"] > min_dte / 365.0) & (df["T_years"] < max_dte / 365.0)].copy()

    # Numerics
    for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "underlying"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # === STRIKE FILTERING (Clean Moneyness Bands) ===
    moneyness_band = config.get("moneyness_band", 0.15)
    if "underlying" in df.columns and "strike" in df.columns:
        df = df[
            (df["strike"] >= df["underlying"] * (1 - moneyness_band)) &
            (df["strike"] <= df["underlying"] * (1 + moneyness_band))
        ].copy()

    # Premium as mid if possible, else last
    df["bid"] = df["bid"].fillna(0.0)
    df["ask"] = df["ask"].fillna(0.0)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["premium"] = df["mid"].where(df["mid"] > 0.0, df["lastPrice"])

    # === PREMIUM SELLING MODE LOGIC ===
    if mode == "Premium Selling":
        # a. Filter the DataFrame to only include type == 'put'.
        df = df[df['type'] == 'put'].copy()
        if df.empty:
            return df
        # d. Add a new metric: return_on_risk = df['premium'] / df['strike'].
        df['return_on_risk'] = df['premium'] / df['strike']

    # Drop where we have no usable premium
    df = df[(df["premium"].notna()) & (df["premium"] > 0)].copy()

    # Spread pct (relative to mid)
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df.loc[~df["spread_pct"].replace([pd.NA, pd.NaT], pd.NA).apply(lambda x: pd.notna(x) and math.isfinite(x)), "spread_pct"] = float("inf")

    # Hard filter for wide spreads
    df = df[df["spread_pct"] <= 0.20].copy()

    # Liquidity filters: remove totally dead contracts
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["openInterest"] = df["openInterest"].fillna(0).astype(float)
    df = df[(df["volume"] > 0) | (df["openInterest"] > 0)].copy()

    if df.empty:
        return df

    # --- Institutional Flow & Sentiment ---
    df["Vol_OI_Ratio"] = df["volume"] / df["openInterest"].replace(0, pd.NA)
    df["Unusual_Whale"] = (df["Vol_OI_Ratio"] > 1.5) & (df["volume"] > 500)

    # Sentiment Tag
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

    df["is_underpriced"] = pd.NA
    df.loc[df["Earnings Play"] == "YES", "is_underpriced"] = df["impliedVolatility"] < df["hv_30d"]

    # --- Trend Alignment Filter ---
    df["Trend_Aligned"] = False
    df.loc[(df["type"] == "call") & (df["underlying"] > df["sma_20"]), "Trend_Aligned"] = True
    df.loc[(df["type"] == "put") & (df["underlying"] < df["sma_20"]), "Trend_Aligned"] = True

    # Fill missing IV with chain median per expiration + type to avoid skew
    df["impliedVolatility"] = df["impliedVolatility"].astype(float)
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    overall_iv_median = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(overall_iv_median)

    # Compute delta
    def _row_delta(row):
        d = bs_delta(
            option_type=row["type"],
            S=safe_float(row["underlying"], 0.0) or 0.0,
            K=safe_float(row["strike"], 0.0) or 0.0,
            T=safe_float(row["T_years"], 0.0) or 0.0,
            r=risk_free_rate,
            sigma=max(1e-9, safe_float(row["impliedVolatility"], 0.0) or 0.0),
        )
        return float("nan") if d is None else d

    df["delta"] = df.apply(_row_delta, axis=1)
    df["abs_delta"] = df["delta"].abs()

    # Hard filter for delta
    if mode == "Premium Selling":
        # b. Change the delta filter to target abs_delta between 0.20 and 0.40.
        df = df[(df["abs_delta"] >= 0.20) & (df["abs_delta"] <= 0.40)].copy()
    else:
        df = df[df["abs_delta"] >= 0.30].copy()

    # Hard filter for delta
    if mode == "Premium Selling":
        # b. Change the delta filter to target abs_delta between 0.20 and 0.40.
        df = df[(df["abs_delta"] >= 0.20) & (df["abs_delta"] <= 0.40)].copy()
    else:
        df = df[df["abs_delta"] >= 0.30].copy()

    # !!! ADD THIS CHECK !!!
    if df.empty:
        return df  # Return the empty frame cleanly if no options survived the delta filter

    # === NEW ADVANCED METRICS ===

    # === NEW ADVANCED METRICS ===

    # Expected Move (1-sigma price move until expiration)
    def _calc_exp_move(row):
        return calculate_expected_move(
            safe_float(row["underlying"]),
            safe_float(row["impliedVolatility"]),
            safe_float(row["T_years"]),
        )

    df["expected_move"] = df.apply(_calc_exp_move, axis=1)

    # Probability of Profit using delta approximation and expected-move adjustments
    def _calc_pop_delta(row):
        try:
            delta_val = safe_float(row["delta"])
            S = safe_float(row["underlying"])
            K = safe_float(row["strike"])
            em = safe_float(row["expected_move"])
            if delta_val is None or S is None or K is None:
                return None
            otm_delta = abs(delta_val)
            pop = 1.0 - otm_delta
            # Adjust PoP if strike sits outside expected move band
            if em is not None and em > 0:
                upper = S + em
                lower = S - em
                opt_type = str(row["type"]).lower()
                outside_em = (opt_type == "call" and K > upper) or (opt_type == "put" and K < lower)
                if outside_em:
                    pop *= 0.7
            return max(0.0, min(1.0, pop))
        except Exception:
            return None

    df["prob_profit"] = df.apply(_calc_pop_delta, axis=1)

    # Probability of Touch (keep BS-based approximation)
    def _calc_pot(row):
        return calculate_probability_of_touch(
            row["type"],
            safe_float(row["underlying"]),
            safe_float(row["strike"]),
            safe_float(row["T_years"]),
            safe_float(row["impliedVolatility"]),
        )

    df["prob_touch"] = df.apply(_calc_pot, axis=1)

    if mode != "Premium Selling":
        # Risk/Reward Analysis with EM-based target price
        def _calc_rr(row):
            max_loss, breakeven, rr_ratio = calculate_risk_reward(
                row["type"],
                safe_float(row["premium"]),
                safe_float(row["underlying"]),
                safe_float(row["strike"]),
                safe_float(row["expected_move"]),
            )
            return pd.Series(
                {"max_loss": max_loss, "breakeven": breakeven, "rr_ratio": rr_ratio}
            )

        rr_data = df.apply(_calc_rr, axis=1)
        df["max_loss"] = rr_data["max_loss"]
        df["breakeven"] = rr_data["breakeven"]
        df["rr_ratio"] = rr_data["rr_ratio"]
        df = df[df["rr_ratio"] >= 0.50].copy()

        if df.empty:
            return df

        # Break-even realism: required move vs expected move
        def _calc_em_realism(row):
            S = safe_float(row["underlying"])
            K = safe_float(row["strike"])
            premium = safe_float(row["premium"])
            em = safe_float(row["expected_move"])
            opt_type = str(row["type"]).lower()
            if S is None or K is None or premium is None or em is None or em <= 0:
                return pd.Series({"required_move": None, "em_realism_score": None})
            if opt_type == "call":
                breakeven = K + premium
                required_move = max(0.0, breakeven - S)
            else:  # put
                breakeven = K - premium
                required_move = max(0.0, S - breakeven)
            ratio = required_move / em if em > 0 else None
            if ratio is None:
                score = None
            elif ratio <= 0.5:
                score = 1.0
            elif ratio <= 1.0:
                score = 0.7
            else:
                # Penalize contracts that need a move beyond EM
                score = max(0.1, em / (required_move + 1e-9))
            return pd.Series({"required_move": required_move, "em_realism_score": score})

        em_data = df.apply(_calc_em_realism, axis=1)
        df["required_move"] = em_data["required_move"]
        df["em_realism_score"] = em_data["em_realism_score"]
    else:
        # For premium selling, these metrics are not used. Set defaults.
        df["max_loss"] = None
        df["breakeven"] = None
        df["rr_ratio"] = None
        df["required_move"] = None
        df["em_realism_score"] = None

    # Theta Decay Pressure (premium per day relative to delta)
    def _calc_theta_pressure(row):
        premium = safe_float(row["premium"])
        T_yrs = safe_float(row["T_years"])
        abs_delta = abs(safe_float(row["delta"]) or 0.0)
        if premium is None or T_yrs is None or T_yrs <= 0:
            return None
        dte = max(int(T_yrs * 365), 1)
        tdp_raw = (premium * 100.0) / dte
        # Normalize by delta so low-delta, high-premium short-term options are penalized more
        effective_tdp = tdp_raw / max(abs_delta, 0.1)
        return effective_tdp

    df["theta_decay_pressure"] = df.apply(_calc_theta_pressure, axis=1)

    # IV vs HV comparison (IV advantage)
    if "hv_30d" in df.columns and df["hv_30d"].notna().any():
        df["iv_vs_hv"] = df["impliedVolatility"] - df["hv_30d"]
        df["iv_hv_ratio"] = df["impliedVolatility"] / df["hv_30d"].replace(0, float('nan'))
    else:
        df["iv_vs_hv"] = 0.0
        df["iv_hv_ratio"] = 1.0
    
    # --- OI Wall Detection ---
    df["oi_wall_warning"] = ""
    for expiry in df["expiration"].unique():
        expiry_df = df[df["expiration"] == expiry]

        # Find Call Wall
        calls = expiry_df[expiry_df["type"] == "call"]
        if not calls.empty:
            call_wall_strike_idx = calls["openInterest"].idxmax()
            if pd.notna(call_wall_strike_idx):
                call_wall = expiry_df.loc[call_wall_strike_idx]["strike"]
                # Get the strike immediately below the wall
                strikes_below = sorted([s for s in calls[calls["strike"] < call_wall]["strike"].unique()], reverse=True)
                strike_below_wall = strikes_below[0] if strikes_below else None

                df.loc[(df["expiration"] == expiry) & (df["type"] == "call") & ((df["strike"] == call_wall) | (df["strike"] == strike_below_wall)), "oi_wall_warning"] = "LIMITED UPSIDE"

        # Find Put Wall
        puts = expiry_df[expiry_df["type"] == "put"]
        if not puts.empty:
            put_wall_strike_idx = puts["openInterest"].idxmax()
            if pd.notna(put_wall_strike_idx):
                put_wall = expiry_df.loc[put_wall_strike_idx]["strike"]
                # Get the strike immediately above the wall
                strikes_above = sorted([s for s in puts[puts["strike"] > put_wall]["strike"].unique()])
                strike_above_wall = strikes_above[0] if strikes_above else None

                df.loc[(df["expiration"] == expiry) & (df["type"] == "put") & ((df["strike"] == put_wall) | (df["strike"] == strike_above_wall)), "oi_wall_warning"] = "LIMITED DOWNSIDE"

    # IV Skew (calls vs puts at same strike/expiry)
    df["iv_skew"] = np.nan  # start as NaN, then fill for stability
    for (exp, strike), group in df.groupby(["expiration", "strike"]):
        if len(group) == 2:  # Has both call and put
            call_iv = group[group["type"] == "call"]["impliedVolatility"].values
            put_iv = group[group["type"] == "put"]["impliedVolatility"].values
            if len(call_iv) > 0 and len(put_iv) > 0:
                skew = put_iv[0] - call_iv[0]
                df.loc[group.index, "iv_skew"] = skew
    # Forward/backward fill any remaining NaNs so downstream summaries don't break
    df["iv_skew"] = df["iv_skew"].ffill().bfill().fillna(0.0)
    
    # Liquidity Quality Flags
    df["liquidity_flag"] = "GOOD"
    df.loc[(df["volume"] < 10) & (df["openInterest"] < 100), "liquidity_flag"] = "POOR"
    df.loc[(df["volume"] >= 10) & (df["volume"] < 50) & (df["openInterest"] >= 100) & (df["openInterest"] < 500), "liquidity_flag"] = "FAIR"
    
    # Wide Spread Flag
    df["spread_flag"] = "OK"
    df.loc[df["spread_pct"] > 0.10, "spread_flag"] = "WIDE"
    df.loc[df["spread_pct"] > 0.20, "spread_flag"] = "VERY_WIDE"
    
    # === IV RANK AND PERCENTILE ===
    df["iv_rank"] = iv_rank if iv_rank is not None else pd.NA
    df["iv_percentile"] = iv_percentile if iv_percentile is not None else pd.NA
    
    # === EARNINGS AWARENESS ===
    df["event_flag"] = "OK"
    if earnings_date is not None:
        earnings_buffer_days = config.get("earnings_buffer_days", 5)
        for idx, row in df.iterrows():
            exp_dt = row["exp_dt"]
            if pd.notna(exp_dt):
                # Check if expiration is within buffer of earnings
                days_to_earnings = abs((exp_dt.replace(tzinfo=None) - earnings_date.replace(tzinfo=None)).days)
                if days_to_earnings <= earnings_buffer_days:
                    df.at[idx, "event_flag"] = "EARNINGS_NEARBY"
    
    # === MONTE CARLO PROBABILITY SIMULATION ===
    if HAS_SIMULATION:
        n_sims = config.get("monte_carlo_simulations", 10000)
        
        def _calc_mc_pop(row):
            pop_sim, pot_sim = monte_carlo_pop(
                S=safe_float(row["underlying"]),
                K=safe_float(row["strike"]),
                T=safe_float(row["T_years"]),
                sigma=safe_float(row["impliedVolatility"]),
                r=risk_free_rate,
                premium=safe_float(row["premium"]),
                option_type=row["type"],
                n_simulations=n_sims
            )
            return pd.Series({"pop_sim": pop_sim, "pot_sim": pot_sim})
        
        mc_results = df.apply(_calc_mc_pop, axis=1)
        df["pop_sim"] = mc_results["pop_sim"]
        df["pot_sim"] = mc_results["pot_sim"]
    else:
        df["pop_sim"] = pd.NA
        df["pot_sim"] = pd.NA
    
    # === END NEW METRICS ===

    # Expected value (EV) and probability ITM using Black-Scholes
    def _calc_ev(row):
        S = safe_float(row["underlying"], 0.0) or 0.0
        K = safe_float(row["strike"], 0.0) or 0.0
        T = safe_float(row["T_years"], 0.0) or 0.0
        sigma = max(1e-9, safe_float(row["impliedVolatility"], 0.0) or 0.0)
        opt_type = row["type"].lower()
        d1, d2 = _d1d2(S, K, T, risk_free_rate, sigma)

        gamma = bs_gamma(S, K, T, risk_free_rate, sigma)
        vega = bs_vega(S, K, T, risk_free_rate, sigma)
        theta = bs_theta(opt_type, S, K, T, risk_free_rate, sigma)

        if d1 is None:
            return pd.Series({"p_itm": None, "theo_value": None, "ev_per_contract": None, "gamma": None, "vega": None, "theta": None})
        # Probability of expiring ITM under risk-neutral measure
        p_itm = norm_cdf(d2) if opt_type == "call" else norm_cdf(-d2)
        # Expected payoff at expiration (risk-neutral)
        if opt_type == "call":
            expected_payoff = S * math.exp(risk_free_rate * T) * norm_cdf(d1) - K * norm_cdf(d2)
        else:
            expected_payoff = K * norm_cdf(-d2) - S * math.exp(risk_free_rate * T) * norm_cdf(-d1)
        theo_value = expected_payoff  # theoretical RN value
        premium = safe_float(row["premium"], 0.0) or 0.0
        # Approximate round-trip half-spread cost (enter+exit ~ 1 spread)
        spread_pct = row.get("spread_pct", 0.0)
        spread_cost = 100.0 * premium * float(spread_pct if pd.notna(spread_pct) else 0.0)
        ev = 100.0 * (expected_payoff - premium) - spread_cost
        return pd.Series({"p_itm": p_itm, "theo_value": theo_value, "ev_per_contract": ev, "gamma": gamma, "vega": vega, "theta": theta})

    ev_data = df.apply(_calc_ev, axis=1)
    df["p_itm"] = ev_data["p_itm"]
    df["theo_value"] = ev_data["theo_value"]
    df["ev_per_contract"] = ev_data["ev_per_contract"]
    df["gamma"] = ev_data["gamma"]
    df["vega"] = ev_data["vega"]
    df["theta"] = ev_data["theta"]

    # --- Theta Safety Check ---
    df["Theta_Burn_Rate"] = df.apply(lambda row: abs(row["theta"]) / row["premium"] if row["premium"] > 0 and pd.notna(row["theta"]) else 0, axis=1)
    df["decay_warning"] = df["Theta_Burn_Rate"] > 0.06

    # --- Support/Resistance Warnings ---
    df["sr_warning"] = ""
    df.loc[(df["type"] == "call") & (df["underlying"] > df["high_20"] * 0.98), "sr_warning"] = "NEAR RESISTANCE"
    df.loc[(df["type"] == "put") & (df["underlying"] < df["low_20"] * 1.02), "sr_warning"] = "NEAR SUPPORT"


    # Normalize features using ranks to reduce outlier impact
    def rank_norm(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1:
            return pd.Series([0.5] * n, index=s.index)
        r = s.rank(method="average", na_option="keep")
        return (r - 1.0) / (n - 1.0)

    vol_n = rank_norm(df["volume"].fillna(0))
    oi_n = rank_norm(df["openInterest"].fillna(0))

    # Spread score: 1 for very tight spreads, 0 for very wide
    # Cap spread at 25% of mid; beyond that is treated equally poor
    sp = df["spread_pct"].replace([pd.NA, pd.NaT], float("inf"))
    sp = sp.clip(lower=0, upper=0.25)
    spread_score = 1.0 - (sp / 0.25)

    # Delta quality: target around 0.4 absolute delta
    delta_target = 0.40
    delta_quality = 1.0 - (df["abs_delta"] - delta_target).abs() / max(delta_target, 1e-6)
    delta_quality = delta_quality.clip(lower=0.0, upper=1.0)

    # IV quality: prefer moderate IV vs chain (avoid extremes)
    iv_n = rank_norm(df["impliedVolatility"].fillna(df["impliedVolatility"].median()))
    iv_quality = 1.0 - (2.0 * (iv_n - 0.5).abs())  # 1 at mid, 0 at edges

    # Liquidity (volume+OI)
    liquidity = 0.5 * (vol_n + oi_n)

    # IV Advantage Score (keep for diagnostics)
    iv_advantage = df["iv_vs_hv"].fillna(0).clip(lower=-0.2, upper=0.2)
    iv_advantage_score = (iv_advantage + 0.2) / 0.4  # Normalize to 0-1

    # Probability of Profit Score
    pop_score = df["prob_profit"].fillna(0.5).clip(lower=0, upper=1)

    # Risk/Reward Score per prompt thresholds
    rr_raw = pd.to_numeric(df["rr_ratio"], errors='coerce').fillna(0.0)
    rr_score = pd.Series(0.2, index=df.index)
    rr_score = rr_score.mask(rr_raw >= 2.0, 0.5)
    rr_score = rr_score.mask(rr_raw >= 3.0, 0.8)
    rr_score = rr_score.mask(rr_raw >= 4.0, 1.0)

    # EV score (rank-normalized expected value per contract)
    ev_score = rank_norm(df["ev_per_contract"].fillna(df["ev_per_contract"].median()))

    # EM realism score (already 0-1, fallback to neutral 0.5)
    em_realism_score = pd.to_numeric(df["em_realism_score"], errors='coerce').fillna(0.5).clip(lower=0.0, upper=1.0)

    # Theta decay pressure => score where lower pressure is better
    theta_raw = df["theta_decay_pressure"].replace([pd.NA, pd.NaT], np.nan)
    theta_rank = rank_norm(theta_raw.fillna(theta_raw.median()))
    theta_score = (1.0 - theta_rank).clip(lower=0.0, upper=1.0)
    # For very short-dated options, weight theta risk more heavily by slightly
    # compressing high scores (i.e., making high TDP hurt more)
    short_dte_mask = (df["T_years"] * 365.0) <= 7
    theta_score = theta_score.where(~short_dte_mask, theta_score * 0.7)

    # Momentum score combining 5d return, RSI distance from 50, and ATR trend
    ret_score = rank_norm(pd.to_numeric(df.get("ret_5d", pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0))
    rsi_vals = pd.to_numeric(df.get("rsi_14", pd.Series(np.nan, index=df.index)), errors="coerce")
    rsi_score = 1.0 - (abs((rsi_vals - 50.0) / 50.0)).clip(lower=0.0, upper=1.0)
    atr_trend_vals = pd.to_numeric(df.get("atr_trend", pd.Series(0.0, index=df.index)), errors="coerce")
    atr_score = rank_norm(atr_trend_vals.fillna(0.0))
    momentum_score = (
        0.4 * ret_score.fillna(0.5)
        + 0.3 * rsi_score.fillna(0.5)
        + 0.3 * atr_score.fillna(0.5)
    )

    # IV rank score: favor cheaper IV (lower percentile) for buyers
    iv_pct_series = pd.to_numeric(
        df.get("iv_percentile_30", df.get("iv_percentile", pd.Series(np.nan, index=df.index))),
        errors="coerce",
    )
    if mode == "Premium Selling":
        # c. Invert the iv_rank_score calculation.
        iv_rank_score = iv_pct_series.clip(lower=0.0, upper=1.0).fillna(0.5)
    else:
        iv_rank_score = (1.0 - iv_pct_series.clip(lower=0.0, upper=1.0)).fillna(0.5)

    # Catalyst strength from event flags (earnings proximity, etc.)
    catalyst_score = pd.Series(0.3, index=df.index)
    catalyst_score = catalyst_score.mask(df["event_flag"] == "EARNINGS_NEARBY", 0.8)

    # Trader profile adjustment: day trader vs swing trader preferences
    dte_days = df["T_years"] * 365.0
    # Normalize DTE between the configured bounds
    dte_norm = ((dte_days - min_dte) / max(1, (max_dte - min_dte))).clip(lower=0.0, upper=1.0)
    if trader_profile.lower().startswith("day"):
        trader_pref_score = 0.6 * liquidity + 0.4 * spread_score
    else:  # swing trader (default)
        trader_pref_score = 0.5 * delta_quality + 0.5 * dte_norm

    # Composite quality score redesign (0-1)
    if mode == "Premium Selling":
        composite_weights = config.get("premium_selling_weights", {})
        return_on_risk_score = rank_norm(df["return_on_risk"].fillna(df["return_on_risk"].median()))
        w_pop = composite_weights.get("pop", 0.0)
        w_ror = composite_weights.get("return_on_risk", 0.0)
        w_iv = composite_weights.get("iv_rank", 0.0)
        w_liq = composite_weights.get("liquidity", 0.0)
        w_theta = composite_weights.get("theta", 0.0)
        w_ev = composite_weights.get("ev", 0.0)
        w_tp = composite_weights.get("trader_pref", 0.0)

        # Normalize weights to sum to 1.0
        w_sum = w_pop + w_ror + w_iv + w_liq + w_theta + w_ev + w_tp
        if w_sum <= 0:
            w_sum = 1.0

        df["quality_score"] = (
            w_pop * pop_score
            + w_ror * return_on_risk_score
            + w_iv * iv_rank_score
            + w_liq * liquidity
            + w_theta * theta_score
            + w_ev * ev_score
            + w_tp * trader_pref_score
        ) / w_sum
    else:
        default_weights = {
            "pop": 0.18,
            "em_realism": 0.12,
            "rr": 0.15,
            "momentum": 0.10,
            "iv_rank": 0.10,
            "liquidity": 0.15,
            "catalyst": 0.05,
            "theta": 0.10,
            "ev": 0.05,
            "trader_pref": 0.10,
        }
        composite_weights = config.get("composite_weights", {}) or {}

        def _w(key: str) -> float:
            return float(composite_weights.get(key, default_weights[key]))

        w_pop = _w("pop")
        w_em = _w("em_realism")
        w_rr = _w("rr")
        w_mom = _w("momentum")
        w_iv = _w("iv_rank")
        w_liq = _w("liquidity")
        w_cat = _w("catalyst")
        w_theta = _w("theta")
        w_ev = _w("ev")
        w_tp = _w("trader_pref")

        # Normalize weights to sum to 1.0
        w_sum = w_pop + w_em + w_rr + w_mom + w_iv + w_liq + w_cat + w_theta + w_ev + w_tp
        if w_sum <= 0:
            w_sum = 1.0

        df["quality_score"] = (
            w_pop * pop_score
            + w_em * em_realism_score
            + w_rr * rr_score
            + w_mom * momentum_score
            + w_iv * iv_rank_score
            + w_liq * liquidity
            + w_cat * catalyst_score
            + w_theta * theta_score
            + w_ev * ev_score
            + w_tp * trader_pref_score
        ) / w_sum

    # Earnings penalty: modestly reduce score if very close to earnings
    df.loc[df["event_flag"] == "EARNINGS_NEARBY", "quality_score"] -= 0.05

    # --- Safety Filter Score Adjustments ---
    df.loc[df["Trend_Aligned"] == True, "quality_score"] += 0.15
    df.loc[df["decay_warning"] == True, "quality_score"] -= 0.20
    df.loc[df["sr_warning"] != "", "quality_score"] -= 0.10

    # --- Probability Enhancer Score Adjustments ---
    if "seasonal_win_rate" in df.columns:
        df.loc[df["seasonal_win_rate"] >= 0.8, "quality_score"] += 0.10
        df.loc[df["seasonal_win_rate"] <= 0.2, "quality_score"] -= 0.10

    df.loc[df["oi_wall_warning"] != "", "quality_score"] -= 0.10

    df["squeeze_play"] = (df["is_squeezing"] == True) & (df["Unusual_Whale"] == True)
    df.loc[df["squeeze_play"], "quality_score"] += 0.25

    # --- Term Structure Score Adjustment ---
    if term_structure_spread is not None and term_structure_spread < 0: # Backwardation
        # Penalize long calls, boost short puts
        df.loc[df["type"] == "call", "quality_score"] -= 0.10
        df.loc[df["type"] == "put", "quality_score"] += 0.10


    # --- INVISIBLE FILTERS (Professional Edition) ---

    # 1. Macro Risk
    df["macro_warning"] = ""
    if macro_risk_active:
        df["macro_warning"] = "⛔ MACRO RISK"
        # Optional: penalize score slightly? Prompt didn't specify score penalty, just "Red Light" string.
        # But let's be safe and penalize slightly to bubble up safer plays.
        df["quality_score"] -= 0.10

    # 2. Sector Relative Strength
    # Logic:
    # If (Stock > 0%) AND (Sector < -1.5%): "Fakeout Divergence". Penalize 0.15.
    # If (Stock > 0%) AND (Sector > 0%): Boost 0.05 (Aligned).
    if sector_perf:
        stock_ret = sector_perf.get("ticker_return", 0.0)
        sector_ret = sector_perf.get("sector_return", 0.0)
        def _check_max_pain(row):
            mp = safe_float(row.get("max_pain"))
            und = safe_float(row.get("underlying"))
            dte = safe_float(row.get("T_years")) * 365.0
            
            if mp and und and dte < 3:
                diff_pct = abs(und - mp) / mp
                if diff_pct > 0.05:
                    return "⚠️ FIGHTING MAX PAIN"
            return ""
        
        df["max_pain_warning"] = df.apply(_check_max_pain, axis=1)

    # 4. Yield Spike Guard
    # Logic: If ^TNX up > 2.5% today, penalize Tech/High Beta calls by 0.20 and add tag.
    df["yield_warning"] = ""
    if tnx_change_pct > 0.025:
        high_beta_tickers = ["QQQ", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX"]
        # Check if current ticker is in high beta list
        # Since df contains one ticker usually, we can check the first row's symbol or pass it in.
        # But df has 'symbol' column.
        
        mask_tech = df["symbol"].isin(high_beta_tickers)
        mask_calls = df["type"] == "call"
        
        # Apply penalty
        df.loc[mask_tech & mask_calls, "quality_score"] -= 0.20
        df.loc[mask_tech & mask_calls, "yield_warning"] = "📉 RATES UP"


    # Ensure quality_score stays in [0, 1]
    df["quality_score"] = df["quality_score"].clip(lower=0, upper=1)
    df["ev_score"] = ev_score

    # Keep helpful computed columns
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"] = liquidity
    df["delta_quality"] = delta_quality
    df["iv_quality"] = iv_quality
    df["spread_score"] = spread_score
    df["theta_score"] = theta_score
    df["momentum_score"] = momentum_score
    df["iv_rank_score"] = iv_rank_score
    df["catalyst_score"] = catalyst_score
    df["iv_advantage_score"] = iv_advantage_score

    # Basic sanity ordering hints
    df = df.sort_values(["Unusual_Whale", "quality_score", "volume", "openInterest"], ascending=[False, False, False, False]).reset_index(drop=True)
    return df


def categorize_by_premium(df: pd.DataFrame, budget: Optional[float] = None) -> pd.DataFrame:
    """Categorize by premium using quantiles (single-stock) or budget-based (multi-stock)."""
    if df.empty:
        return df
    
    # Calculate contract cost
    df["contract_cost"] = df["premium"] * 100
    
    if budget is not None:
        # Budget mode: categorize based on % of budget
        # LOW: 0-33% of budget, MEDIUM: 33-66%, HIGH: 66-100%
        def cat_budget(cost):
            pct = cost / budget
            if pct <= 0.33:
                return "LOW"
            elif pct <= 0.66:
                return "MEDIUM"
            else:
                return "HIGH"
        df["price_bucket"] = df["contract_cost"].apply(cat_budget)
    else:
        # Single-stock mode: use quantiles
        premiums = df["premium"].astype(float)
        q1 = premiums.quantile(1/3)
        q2 = premiums.quantile(2/3)

        def cat(p):
            if p <= q1:
                return "LOW"
            elif p <= q2:
                return "MEDIUM"
            else:
                return "HIGH"

        df["price_bucket"] = premiums.apply(cat)
    
    return df


def pick_top_per_bucket(df: pd.DataFrame, per_bucket: int = 5, diversify_tickers: bool = False) -> pd.DataFrame:
    """Pick top options per bucket, optionally diversifying across tickers."""
    picks = []
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["price_bucket"] == bucket].copy()
        if sub.empty:
            continue
        
        if diversify_tickers and "symbol" in sub.columns:
            # Try to get diverse tickers in budget mode
            # Sort by quality first (now includes EV)
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            
            # Pick best from each ticker, then fill remaining slots
            selected = []
            tickers_used = set()
            
            # First pass: best option from each unique ticker
            for _, row in sub.iterrows():
                if row["symbol"] not in tickers_used:
                    selected.append(row)
                    tickers_used.add(row["symbol"])
                    if len(selected) >= per_bucket:
                        break
            
            # Second pass: fill remaining slots with next best regardless of ticker
            if len(selected) < per_bucket:
                for _, row in sub.iterrows():
                    if len(selected) >= per_bucket:
                        break
                    # Check if this exact row is already selected
                    is_duplicate = any(
                        (s["symbol"] == row["symbol"] and 
                         s["strike"] == row["strike"] and 
                         s["expiration"] == row["expiration"] and
                         s["type"] == row["type"]) 
                        for s in selected
                    )
                    if not is_duplicate:
                        selected.append(row)
            
            picks.append(pd.DataFrame(selected))
        else:
            # Standard sorting for single-stock mode
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            picks.append(sub.head(per_bucket))
    
    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True)
    return out


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
            print(f"    DEBUG: {symbol} {exp} - Credit ${total_credit:.2f} < Min ${min_credit:.2f} (20% of ${max_width:.2f})")
            continue
        
        # Delta Neutrality Check: abs(short_put_delta + short_call_delta) < 0.10
        short_put_delta = best_put_spread['short_put']['delta']
        short_call_delta = best_call_spread['short_call']['delta']
        net_delta = short_put_delta + short_call_delta
        
        if abs(net_delta) >= 0.10:
            print(f"    DEBUG: {symbol} {exp} - Net Delta {net_delta:.3f} too directional (abs must be < 0.10)")
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


def format_pct(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x):
            return "-"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "-"


def format_money(x: Optional[float]) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"


def determine_moneyness(row: pd.Series) -> str:
    """Determine if option is ITM or OTM based on strike vs underlying."""
    try:
        strike = float(row["strike"])
        underlying = float(row["underlying"])
        opt_type = row["type"].lower()
        
        if opt_type == "call":
            return "ITM" if strike < underlying else "OTM"
        else:  # put
            return "ITM" if strike > underlying else "OTM"
    except Exception:
        return "---"


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
            "delta", "gamma", "vega", "theta", "impliedVolatility", "hv_30d", "iv_vs_hv", "iv_rank", "iv_percentile",
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


def run_scan(mode: str, tickers: List[str], budget: Optional[float], max_expiries: int, min_dte: int, max_dte: int, trader_profile: str, logger: logging.Logger, market_trend: str, volatility_regime: str, macro_risk_active: bool = False, tnx_change_pct: float = 0.0):
    # Determine mode booleans for internal logic
    is_budget_mode = (mode == "Budget scan")
    is_discovery_mode = (mode == "Discovery scan")

    # === LOAD CONFIGURATION ===
    print("\nLoading configuration...")
    config = load_config("config.json")
    print("✓ Configuration loaded")

    # === FETCH VIX FOR ADAPTIVE WEIGHTING ===
    print("Fetching VIX level for adaptive scoring...")
    vix_level = get_vix_level()
    if vix_level:
        print(f"✓ VIX Level: {vix_level:.2f}")
    else:
        print("⚠️  Could not fetch VIX, using default weights")

    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    print(f"✓ Market Regime: {vix_regime.upper()}")

    # Fetch risk-free rate automatically
    print("Fetching current risk-free rate...")
    rfr = get_risk_free_rate()
    print(f"Using risk-free rate: {rfr*100:.2f}% (13-week Treasury)")

    # Collect data from all tickers (PARALLEL PROCESSING)
    tickers = list(set(tickers))  # Deduplicate tickers
    all_picks = []
    all_credit_spreads = []
    all_iron_condors = []
    ticker_histories = {} # For Portfolio Protection

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
                        print(f"    Found {len(picks_df)} contracts.")
                    
                    # Aggregate credit spreads
                    for spreads_df in result['credit_spreads']:
                        all_credit_spreads.append(spreads_df)
                        print(f"    Found {len(spreads_df)} Credit Spreads.")
                    
                    # Aggregate iron condors
                    if 'iron_condors' in result and not result['iron_condors'].empty:
                        all_iron_condors.append(result['iron_condors'])
                        print(f"    Found {len(result['iron_condors'])} Iron Condors.")
                else:
                    if result['error']:
                        print(f"    {result['error']}")
                
            except Exception as e:
                print(f"\n>>> Scanning {symbol}...")
                print(f"    ⚠️  Error: {e}")

    # --- Portfolio Protection: Correlation Warning ---
    if len(ticker_histories) > 1:
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

    # Generate Final Reports
    if mode == "Budget scan":
        if all_picks:
            # Filter out empty DataFrames to avoid FutureWarning
            non_empty_picks = [df for df in all_picks if not df.empty]
            if non_empty_picks:
                final_df = pd.concat(non_empty_picks, ignore_index=True)
                # Re-sort by quality score
                final_df = final_df.sort_values("quality_score", ascending=False)
                # Categorize by premium
                final_df = categorize_by_premium(final_df, budget=budget)
                # Pick top unique tickers
                top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
                print_report(top_picks, 0.0, rfr, max_expiries, min_dte, max_dte, mode=mode, budget=budget, market_trend=market_trend, volatility_regime=volatility_regime)
            else:
                print("\nNo options found within budget.")
        else:
            print("\nNo options found within budget.")

    elif mode == "Discovery scan":
        if all_picks:
            non_empty_picks = [df for df in all_picks if not df.empty]
            if non_empty_picks:
                final_df = pd.concat(non_empty_picks, ignore_index=True)
                final_df = final_df.sort_values("quality_score", ascending=False)
                # Categorize by premium
                final_df = categorize_by_premium(final_df, budget=None)
                # Show top 10 overall
                print_report(final_df.head(10), 0.0, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
            else:
                print("\nNo discovery picks found.")
        else:
            print("\nNo discovery picks found.")
            
    elif mode == "Credit Spreads":
        if all_credit_spreads:
            final_spreads = pd.concat(all_credit_spreads, ignore_index=True)
            final_spreads = final_spreads.sort_values("quality_score", ascending=False)
            print_credit_spreads_report(final_spreads)
        else:
            print("\nNo credit spreads found.")
    
    elif mode == "Iron Condor":
        if all_iron_condors:
            final_condors = pd.concat(all_iron_condors, ignore_index=True)
            final_condors = final_condors.sort_values("return_on_risk", ascending=False)
            print_iron_condor_report(final_condors)
        else:
            print("\nNo iron condors found.")

    elif mode == "Premium Selling":
        if all_picks:
            non_empty_picks = [df for df in all_picks if not df.empty]
            if non_empty_picks:
                final_df = pd.concat(non_empty_picks, ignore_index=True)
                final_df = final_df.sort_values("quality_score", ascending=False)
                # Categorize by premium
                final_df = categorize_by_premium(final_df, budget=None)
                print_report(final_df.head(10), 0.0, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
            else:
                print("\nNo premium selling candidates found.")
        else:
            print("\nNo premium selling candidates found.")

    else:
        # Single stock mode
        if all_picks:
            non_empty_picks = [df for df in all_picks if not df.empty]
            if non_empty_picks:
                final_df = pd.concat(non_empty_picks, ignore_index=True)
                # Categorize by premium
                final_df = categorize_by_premium(final_df, budget=None)
                print_report(final_df, 0.0, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime)
            else:
                print("\nNo suitable options found.")
        else:
            print("\nNo suitable options found.")

    # Return results for main() to use if needed (though we printed reports already)
    # We need to construct a return dict to match what main() expects, 
    # OR we can just return None and let main exit if we handled printing.
    # But main() has logic to print "TOP OVERALL PICK" after run_scan returns.
    # So we should return the best pick.
    
    # Consolidate for top pick selection
    picks = pd.DataFrame()
    if all_picks:
        # Filter out empty DataFrames to avoid FutureWarning
        non_empty_picks = [df for df in all_picks if not df.empty]
        if non_empty_picks:
            picks = pd.concat(non_empty_picks, ignore_index=True)
    
    top_pick = None
    if not picks.empty:
        picks["overall_score"] = picks["quality_score"] # Simplified
        top_pick = picks.sort_values("overall_score", ascending=False).iloc[0]
    elif all_credit_spreads:
         # If credit spreads mode, pick top spread
         all_spreads = pd.concat(all_credit_spreads, ignore_index=True)
         if not all_spreads.empty:
             top_pick = all_spreads.sort_values("quality_score", ascending=False).iloc[0]

    return {
        'picks': picks,
        'top_pick': top_pick,
        'spreads': pd.DataFrame(), # Placeholder
        'credit_spreads': pd.concat(all_credit_spreads, ignore_index=True) if all_credit_spreads else pd.DataFrame(),
        'rfr': rfr,
        'chain_iv_median': 0.0, # Placeholder
        'underlying_price': 0.0, # Placeholder
        'num_expiries': max_expiries,
        'min_dte': min_dte,
        'max_dte': max_dte,
        'mode': mode,
        'budget': budget,
    }

def main():
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--close-trades":
            close_trades()
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python options_screener.py [--close-trades]")
            sys.exit(1)
    
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

    # Handle portfolio viewer
    if symbol_input == "PORTFOLIO":
        from .check_pnl import view_portfolio
        view_portfolio()
        sys.exit(0)

    # Determine mode
    is_budget_mode = (symbol_input == "ALL")
    is_discovery_mode = (symbol_input == "DISCOVER" or symbol_input == "")
    is_premium_selling_mode = (symbol_input == "SELL")
    is_credit_spread_mode = (symbol_input == "SPREADS")
    is_iron_condor_mode = (symbol_input == "IRON")

    if is_discovery_mode:
        mode = "Discovery scan"
    elif is_budget_mode:
        mode = "Budget scan"
    elif is_premium_selling_mode:
        mode = "Premium Selling"
    elif is_credit_spread_mode:
        mode = "Credit Spreads"
    elif is_iron_condor_mode:
        mode = "Iron Condor"
    else:
        mode = "Single-stock"

    budget = None
    tickers = []

    if is_discovery_mode or is_premium_selling_mode or is_credit_spread_mode or is_iron_condor_mode:
        # Discovery mode: scan top 100 most-traded options tickers
        if is_premium_selling_mode:
            print("\n=== PREMIUM SELLING MODE ===")
            print("Scanning top tickers for short put opportunities...")
        elif is_credit_spread_mode:
            print("\n=== CREDIT SPREAD MODE ===")
            print("Scanning top tickers for credit spread opportunities...")
        elif is_iron_condor_mode:
            print("\n=== IRON CONDOR MODE ===")
            print("Scanning top tickers for iron condor opportunities...")
        else:
            print("\n=== DISCOVERY MODE ===")
            print("Scanning top tickers for best opportunities...")

        tickers = prompt_for_tickers()
        print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        
    elif is_budget_mode:
        # Budget mode setup
        print("\n=== BUDGET MODE ===")
        print("Find options within your budget constraint.\n")
        
        try:
            budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
            if budget <= 0:
                print("Budget must be positive.")
                sys.exit(1)
        except Exception:
            print("Invalid budget amount.")
            sys.exit(1)
        
        print("\nSelect scan type:")
        print("  1. TARGETED - Scan specific tickers you choose")
        print("  2. DISCOVERY - Scan many tickers to see what your budget can get you")
        
        scan_type = prompt_input("Enter 1 for TARGETED or 2 for DISCOVERY", "1")
        
        if scan_type == "2":
            # Discovery-style budget scan
            print("\n=== BUDGET DISCOVERY SCAN ===")
            print(f"Scanning market with ${budget:.2f} budget constraint...")
            
            tickers = prompt_for_tickers()
            
            print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
            print(f"Budget: ${budget:.2f} per contract (premium × 100)\n")
        else:
            # Targeted budget scan
            print("\n=== BUDGET TARGETED SCAN ===")
            default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
            tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            
            if not tickers:
                print("No valid tickers provided.")
                sys.exit(1)
            
            print(f"\nScanning {len(tickers)} tickers with ${budget:.2f} budget...")
    else:
        # Single-stock mode
        if not symbol_input.isalnum():
            print("Please enter a valid alphanumeric ticker (e.g., AAPL).")
            sys.exit(1)
        tickers = [symbol_input]
        print(f"\nSingle-Stock Mode: Analyzing {symbol_input}...")
    
    logger = setup_logging()

    # === GET MARKET CONTEXT ===
    print("\nFetching market context (SPY/VIX)...")
    market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
    print(f"✓ Market Trend: {market_trend} | Volatility: {volatility_regime}")
    if macro_risk_active:
        print("⚠️  MACRO RISK DETECTED (Forex Volatility)")
    if tnx_change_pct > 0.025:
        print(f"⚠️  YIELD SPIKE DETECTED (^TNX +{tnx_change_pct:.1%})")

    try:
        max_expiries = int(prompt_input("How many nearest expirations to scan", "4"))
        if max_expiries <= 0 or max_expiries > 12:
            print("Please choose between 1 and 12 expirations.")
            sys.exit(1)
    except Exception:
        print("Invalid number for expirations.")
        sys.exit(1)

    # DTE defaults depend on mode - Iron Condors need longer expiration for theta decay
    if is_iron_condor_mode:
        default_min_dte = "30"
        default_max_dte = "60"
        print("\n💡 Iron Condors typically perform best with 30-60 DTE for optimal theta decay.")
    else:
        default_min_dte = "7"
        default_max_dte = "120"

    try:
        min_dte = int(prompt_input("Minimum days to expiration (DTE)", default_min_dte))
        max_dte = int(prompt_input("Maximum days to expiration (DTE)", default_max_dte))
        if min_dte < 0 or max_dte <= min_dte:
            print("DTE bounds invalid. Ensure 0 <= min < max.")
            sys.exit(1)
    except Exception:
        print("Invalid DTE inputs.")
        sys.exit(1)

    # Trading style profile (day trader vs swing trader)
    print("\nSelect trading style profile:")
    print("  1. Swing trader (default) - balanced delta + more DTE")
    print("  2. Day / short-term trader - prioritize liquidity & tight spreads")
    profile_choice = prompt_input("Enter 1 for Swing or 2 for Day trader", "1").strip()
    trader_profile = "day" if profile_choice == "2" else "swing"

    try:
        scan_results = run_scan(
            mode=mode,
            tickers=tickers,
            budget=budget,
            max_expiries=max_expiries,
            min_dte=min_dte,
            max_dte=max_dte,
            trader_profile=trader_profile,
            logger=logger,
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            macro_risk_active=macro_risk_active,
            tnx_change_pct=tnx_change_pct,
        )
        if scan_results is None:
            sys.exit(0)

        # Unpack results
        picks = scan_results['picks']
        top_pick = scan_results['top_pick']
        spreads = scan_results['spreads']
        credit_spreads = scan_results['credit_spreads']
        rfr = scan_results['rfr']
        chain_iv_median = scan_results['chain_iv_median']
        underlying_price = scan_results['underlying_price']
        
        if mode == "Credit Spreads":
            pass # Already printed in run_scan
        else:
            # Print spreads report if any found
            print_spreads_report(spreads)

        if top_pick is not None:
            # Compute and display top overall pick
            print("\n" + "="*80)
            print("  ⭐ TOP OVERALL PICK")
            print("="*80)
            
            exp = pd.to_datetime(top_pick["expiration"]).date()
            
            if mode == "Credit Spreads":
                # Spread-specific display
                print(
                    f"\n  {top_pick['symbol']} {top_pick['type'].upper()} | "
                    f"Short ${top_pick['short_strike']:.2f} / Long ${top_pick['long_strike']:.2f} | Exp {exp}\n"
                )
                print(f"  Net Credit: {format_money(top_pick['net_credit'])} (Max Profit)")
                print(f"  Max Risk:   {format_money(top_pick['max_loss'])}")
                print(f"  Score:      {top_pick['quality_score']:.2f}")
                
                print(f"\n  💡 Rationale: High probability income strategy. Defined risk with {format_money(top_pick['net_credit'])} credit collected upfront.")
            
            else:
                # Standard single-leg display
                moneyness = determine_moneyness(top_pick)
                dte = int(top_pick["T_years"] * 365)
                
                print(
                    f"\n  {top_pick['symbol']} {top_pick['type'].upper()} | "
                    f"Strike ${top_pick['strike']:.2f} | Exp {exp} ({dte}d) | {moneyness}\n"
                )
                if mode in ["Budget scan", "Discovery scan"]:
                    print(f"  Stock Price: ${top_pick['underlying']:.2f}")
                print(f"  Premium: {format_money(top_pick['premium'])}")
                if mode == "Budget scan":
                    contract_cost = top_pick['premium'] * 100
                    print(f"  Contract Cost: ${contract_cost:.2f} (within ${budget:.2f} budget)")
                elif mode == "Discovery scan":
                    contract_cost = top_pick['premium'] * 100
                    print(f"  Contract Cost: ${contract_cost:.2f}")
                print(f"  IV: {format_pct(top_pick['impliedVolatility'])} | Delta: {top_pick['delta']:+.2f} | Quality: {top_pick['quality_score']:.2f}")
                print(f"  Volume: {int(top_pick['volume'])} | OI: {int(top_pick['openInterest'])} | Spread: {format_pct(top_pick['spread_pct'])}")
                
                # Generate justification
                justification_parts = []
                
                # Liquidity assessment (only if picks is not empty)
                if not picks.empty and "volume" in picks.columns:
                    if top_pick["volume"] > picks["volume"].quantile(0.75):
                        justification_parts.append("excellent liquidity")
                    elif top_pick["volume"] > picks["volume"].median():
                        justification_parts.append("good liquidity")
                
                # IV assessment
                iv_diff = abs(top_pick["impliedVolatility"] - chain_iv_median)
                if iv_diff <= 0.05:
                    justification_parts.append("balanced IV near chain median")
                elif top_pick["impliedVolatility"] < chain_iv_median:
                    justification_parts.append("favorable IV below median")
                
                # Spread assessment
                if pd.notna(top_pick["spread_pct"]) and top_pick["spread_pct"] < 0.05:
                    justification_parts.append("tight bid-ask spread")
                
                # Delta assessment
                if 0.35 <= abs(top_pick["delta"]) <= 0.50:
                    justification_parts.append("optimal delta range")
                
                # DTE assessment
                if dte <= 30:
                    justification_parts.append("short-term play")
                elif dte <= 60:
                    justification_parts.append("medium-term opportunity")
                else:
                    justification_parts.append("longer-dated position")
                
                justification = "Chosen for " + ", ".join(justification_parts[:3]) + "."
                if len(justification_parts) > 3:
                    justification += f" Also offers {', '.join(justification_parts[3:])}." 
                
                print(f"\n  💡 Rationale: {justification}")
        
        # Summary footer
        print("\n" + "="*80)
        print("  SCAN SUMMARY")
        print("="*80)
        print(f"  Total Picks Displayed: {len(picks)}")
        if mode in ["Budget scan", "Discovery scan"]:
            unique_tickers = picks["symbol"].nunique() if not picks.empty and "symbol" in picks.columns else 0
            print(f"  Tickers Covered: {unique_tickers}")
        if mode == "Budget scan":
            print(f"  Budget Constraint: ${budget:.2f} per contract")
        print(f"  Chain Median IV: {format_pct(chain_iv_median)}")
        print(f"  Expirations Scanned: {max_expiries}")
        print(f"  Risk-Free Rate Used: {rfr*100:.2f}%")
        print(f"  DTE Filter: {min_dte}-{max_dte} days")
        print(f"  Mode: {mode}")
        print("="*80)
        print("\n  ⚠️  Not financial advice. Verify all data before trading.")
        print("="*80 + "\n")
        
        # === EXPORT AND LOGGING ===
        export_choice = prompt_input("Export results to CSV? (y/n)", "n").lower()
        if export_choice == "y":
            csv_file = export_to_csv(picks, mode, budget)
            if csv_file:
                print(f"\n  📄 Results exported to: {csv_file}")
        
        # === VISUALIZATION ===
        if HAS_VISUALIZATION:
            viz_choice = prompt_input("Generate visualization charts? (y/n)", "n").lower()
            if viz_choice == "y":
                create_visualizations(picks, mode, output_dir="reports")
        
        log_choice = prompt_input("Log trades for P/L tracking? (y/n)", "n").lower()
        if log_choice == "y":
            log_trade_entry(picks, mode)
        
        print("\n👋 Done! Happy trading!\n")
        
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
