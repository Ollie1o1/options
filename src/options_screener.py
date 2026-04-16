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
import os
import csv
import json
import logging
import uuid
import time
import threading as _threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Union, Any
from .schemas import ScanResult
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import warnings
import contextlib


# Dependency checks
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import numpy as np
except Exception:
    missing.append("numpy")
if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

from .data_fetching import (
    get_risk_free_rate,
    get_vix_level,
    determine_vix_regime,
    get_market_context,
    fetch_options_yfinance,
    get_dynamic_tickers,
)
from .utils import (
    safe_float,
    norm_cdf,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    bs_charm,
    bs_vanna,
    early_exercise_premium,
    _d1d2,
)
from .filters import (
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

# Scan-level warning counter (incremented in except blocks, reported at end of scan)
_SCAN_WARNINGS = [0]

# Optional imports (relative to this package)
try:
    from .simulation import monte_carlo_pop, batch_monte_carlo_pop
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
    get_display_width, print_executive_summary,
    print_report, print_news_panel,
    print_credit_spreads_report, print_iron_condor_report,
)
from .watchlist import (
    load_watchlist, add_to_watchlist, remove_from_watchlist,
)
from .oi_snapshot import load_oi_snapshot
from .utils import safe_float


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
_IC_RECALIB_RUNNING: bool = False
_IC_RECALIB_LOCK = _threading.Lock()
_CACHE_MAX_AGE_DAYS = 7


def _maybe_trigger_recalib(cache_path: str) -> None:
    """Fire-and-forget: recalibrate IC weights in background if cache is stale and ≥30 closed trades exist."""
    global _IC_RECALIB_RUNNING
    if _IC_RECALIB_RUNNING:
        return
    cache_stale = True
    try:
        mtime = os.path.getmtime(cache_path)
        cache_stale = (time.time() - mtime) > (_CACHE_MAX_AGE_DAYS * 86400)
    except OSError:
        cache_stale = True
    if not cache_stale:
        return

    def _run():
        global _IC_RECALIB_RUNNING, _IC_WEIGHTS_CACHE
        with _IC_RECALIB_LOCK:
            _IC_RECALIB_RUNNING = True
            try:
                import sqlite3 as _sqlite3
                with _sqlite3.connect("paper_trades.db") as _conn:
                    n = _conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL"
                    ).fetchone()[0]
                if n < 30:
                    return
                from .backtester import run_paper_trade_ic
                ic_data = run_paper_trade_ic()
                if ic_data.get("n_trades", 0) >= 30:
                    with open(cache_path, "w") as _f:
                        json.dump(ic_data, _f, indent=2)
                    _IC_WEIGHTS_CACHE = None  # invalidate in-memory cache so next call re-reads
                    logging.getLogger(__name__).info(
                        "IC weights auto-recalibrated from %d trades", ic_data["n_trades"]
                    )
            except Exception as _e:
                logging.getLogger(__name__).debug("IC auto-recalib failed: %s", _e)
            finally:
                _IC_RECALIB_RUNNING = False

    t = _threading.Thread(target=_run, daemon=True)
    t.start()


def load_ic_adjusted_weights(config: Dict, cache_path: str = "ic_weights_cache.json") -> Dict:
    """Blend config composite weights with IC-derived weights from paper trade analysis.

    Blending formula: final_weight = 0.7 * config_weight + 0.3 * ic_weight
    where ic_weight is the raw IC value (floored at 0) normalized to sum to 1.
    Returns plain config weights on any failure.
    """
    global _IC_WEIGHTS_CACHE
    if _IC_WEIGHTS_CACHE is not None:
        return _IC_WEIGHTS_CACHE
    _maybe_trigger_recalib(cache_path)
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
        component_pvalues = cache.get("component_pvalues", {})
        ic_vals = {}
        for ic_key, w_key in key_map.items():
            ic_raw = component_ic.get(ic_key)
            p_val = component_pvalues.get(ic_key, 1.0)
            if ic_raw is not None and isinstance(ic_raw, (int, float)) and p_val < 0.10:
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




def calculate_probability_of_profit(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray], premium: Union[float, np.ndarray], r: Union[float, np.ndarray] = 0.0, q: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray, None]:
    """Calculate probability of profit at expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        premium = np.asanyarray(premium)
        r = np.asanyarray(r, dtype=float)
        q = np.asanyarray(q, dtype=float)
        # Clip T to 1 hour minimum to prevent division-by-zero on expiration day
        T = np.maximum(T, 1.0 / (365.0 * 24.0))

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        # Break-even point
        breakeven = np.where(is_call, K + premium, K - premium)

        # Forward price: F = S * exp((r - q) * T)
        F = S * np.exp((r - q) * T)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (np.log(F / breakeven) - (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

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
    em_factor: float = 0.68,
) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
    """Calculate max loss, break-even, and risk/reward ratio (Vectorized).

    Uses:
      - target_price = stock_price ± em_factor * EM  (default 0.68 = 1σ)
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
            target_price = np.where(is_call, S + em_factor * expected_move, S - em_factor * expected_move)
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
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Calculates all objective mathematical metrics and merges external data."""
    
    # --- Institutional Flow & Sentiment ---
    df["Vol_OI_Ratio"] = (df["volume"] / df["openInterest"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
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

    _q = float(dividend_yield) if dividend_yield else 0.0
    df["delta"] = bs_delta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["abs_delta"] = np.abs(df["delta"].values)
    df["gamma"] = bs_gamma(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["vega"] = bs_vega(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["theta"] = bs_theta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["rho"] = bs_rho(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["charm"] = bs_charm(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["vanna"] = bs_vanna(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)

    # --- Early Exercise Premium (American vs European put) ---
    # For puts where early exercise is materially valuable, flag the contract.
    # Threshold: early exercise premium > 3% of market premium → flag "EARLY_EX".
    # This warns that BS Greeks understate true option value for these rows.
    try:
        _put_mask = (df["type"].str.lower() == "put") & df["underlying"].notna() & df["impliedVolatility"].notna() & df["premium"].notna() & (df["T_years"] > 0) & (df["impliedVolatility"] > 0)
        df["early_exercise_flag"] = ""
        if _put_mask.any():
            _ee_func = np.vectorize(
                lambda S, K, T, IV: early_exercise_premium("put", float(S), float(K), float(T), risk_free_rate, float(IV), _q),
                otypes=[float],
            )
            _sub = df.loc[_put_mask]
            _ee_vals = _ee_func(_sub["underlying"].values, _sub["strike"].values, _sub["T_years"].values, _sub["impliedVolatility"].values)
            df.loc[_put_mask, "early_exercise_flag"] = np.where(
                _ee_vals > _sub["premium"].values * 0.03, "EARLY_EX", ""
            )
    except Exception:
        df["early_exercise_flag"] = ""

    # --- PCR per Expiration ---
    is_call = np.char.lower(types_vals.astype(str)) == "call"
    try:
        pcr_map = {}
        for exp, grp in df.groupby("expiration"):
            call_vol = grp.loc[grp["type"] == "call", "volume"].sum()
            put_vol = grp.loc[grp["type"] == "put", "volume"].sum()
            pcr_val = float(put_vol) / float(call_vol) if pd.notna(call_vol) and call_vol > 0 else np.nan
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
        # Max gamma strike pinning
        try:
            gex_by_strike_abs = df.groupby("strike")["gex"].apply(lambda x: x.abs().sum())
            max_gamma_strike = float(gex_by_strike_abs.idxmax()) if not gex_by_strike_abs.empty else None
            df["max_gamma_strike"] = max_gamma_strike
            _price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
            if max_gamma_strike and _price_scalar > 0:
                df["gamma_pin_dist_pct"] = abs(max_gamma_strike - _price_scalar) / _price_scalar * 100
            else:
                df["gamma_pin_dist_pct"] = pd.NA
        except Exception:
            df["max_gamma_strike"] = pd.NA
            df["gamma_pin_dist_pct"] = pd.NA
    except Exception:
        df["gex"] = 0.0
        df["gex_flip_price"] = None
        df["max_gamma_strike"] = pd.NA
        df["gamma_pin_dist_pct"] = pd.NA

    # Max pain distance from current price
    if "max_pain_strike" not in df.columns:
        df["max_pain_strike"] = pd.NA
    _mp_price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
    if df["max_pain_strike"].notna().any() and _mp_price_scalar > 0:
        _mp_val = pd.to_numeric(df["max_pain_strike"], errors="coerce")
        df["max_pain_dist_pct"] = ((_mp_val - _mp_price_scalar) / _mp_price_scalar * 100).abs()
    else:
        df["max_pain_dist_pct"] = pd.NA

    # --- Option RVOL unusual activity flag ---
    if "option_rvol" in df.columns:
        df["unusual_options_activity"] = df["option_rvol"] > 5.0
    else:
        df["unusual_options_activity"] = False

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
        try:
            _exp_dates = df["exp_dt"].dt.tz_localize(None).dt.date
            _div_mask = (
                (df["type"] == "call")
                & (df["abs_delta"] > 0.70)
                & (_exp_dates >= next_ex_div)
            )
            df.loc[_div_mask, "div_warning"] = f"EX-DIV {next_ex_div}"
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

    # --- Earnings IV Crush Prediction ---
    df["predicted_iv_crush"] = pd.NA
    df["crush_confidence"] = ""
    if earnings_move_data:
        df["predicted_iv_crush"] = earnings_move_data.get("predicted_iv_crush")
        df["crush_confidence"] = earnings_move_data.get("crush_confidence", "")

    # --- ADVANCED METRICS ---
    _em_result = calculate_expected_move(S_vals, IV_vals, T_vals)
    if _em_result is None:
        _em_result = S_vals * IV_vals * np.sqrt(T_vals)
    df["expected_move"] = _em_result
    is_call = np.char.lower(types_vals.astype(str)) == "call"

    # Probability of Profit: breakeven-based formula P(S_T > K+prem) for calls,
    # P(S_T < K-prem) for puts — correctly accounts for premium cost unlike 1-delta.
    prem_vals = df["premium"].values
    pop_arr = calculate_probability_of_profit(types_vals, S_vals, K_vals, T_vals, IV_vals, prem_vals, r=risk_free_rate, q=_q)
    if pop_arr is None:
        pop_arr = np.where(is_call, 1.0 - df["delta"].values, 1.0 + df["delta"].values)
    df["prob_profit"] = np.clip(pop_arr, 0.0, 1.0)

    _pot_result = calculate_probability_of_touch(types_vals, S_vals, K_vals, T_vals, IV_vals)
    df["prob_touch"] = _pot_result if _pot_result is not None else np.nan
    _em_factor = config.get("rr_expected_move_factor", 0.68)
    max_loss, breakeven, rr_ratio = calculate_risk_reward(types_vals, prem_vals, S_vals, K_vals, df["expected_move"].values, em_factor=_em_factor)
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

    # Theta Decay Pressure: use actual Greek theta as fraction of premium
    # (replaces old premium/DTE/delta formula which inflated OTM options)
    _theta_abs = np.abs(df["theta"].values)
    _prem_safe = np.nan_to_num(df["premium"].values, nan=0.0)
    df["theta_decay_pressure"] = _theta_abs / np.maximum(_prem_safe, 0.01)

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
            put_iv = put_25d["impliedVolatility"].iloc[0]
            call_iv = call_25d["impliedVolatility"].iloc[0]
            if pd.isna(put_iv) or pd.isna(call_iv):
                continue
            skew_val = float(put_iv) - float(call_iv)
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
                if days_to_e <= eb_days:
                    df.at[idx, "event_flag"] = "EARNINGS_NEARBY"
    
    # Monte Carlo
    if HAS_SIMULATION:
        n_sims = config.get("monte_carlo_simulations", 10000)
        _short_modes = {"Premium Selling", "Credit Spreads", "Iron Condor"}
        _is_short_mode = mode in _short_modes
        # Deterministic seed for reproducible PoP across runs on the same date
        _mc_seed_input = (
            tuple(df["underlying"].to_numpy(dtype=float, na_value=0.0)[:5]),
            tuple(df["strike"].to_numpy(dtype=float, na_value=0.0)[:5]),
            len(df),
            datetime.now().strftime("%Y-%m-%d"),
        )
        _mc_seed = hash(_mc_seed_input) % (2**31)
        pop_arr, pot_arr = batch_monte_carlo_pop(
            S_arr=df["underlying"].to_numpy(dtype=float, na_value=0.0),
            K_arr=df["strike"].to_numpy(dtype=float, na_value=0.0),
            T_arr=df["T_years"].to_numpy(dtype=float, na_value=0.0),
            sigma_arr=df["impliedVolatility"].to_numpy(dtype=float, na_value=0.0),
            r=risk_free_rate,
            premium_arr=df["premium"].to_numpy(dtype=float, na_value=0.0),
            option_types=df["type"].to_numpy(),
            n_simulations=n_sims,
            is_short=_is_short_mode,
            random_seed=_mc_seed,
            q_arr=df["dividend_yield"].to_numpy(dtype=float, na_value=0.0) if "dividend_yield" in df.columns else None,
        )
        df["pop_sim"] = pop_arr
        df["pot_sim"] = pot_arr
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

    # Earnings IV crush: recompute PoP with crush-adjusted IV for earnings plays
    # (bakes crush into probability instead of relying solely on post-hoc score penalty)
    _earn_pop_mask = df.get("Earnings Play", pd.Series("NO", index=df.index)) == "YES"
    if _earn_pop_mask.any() and "predicted_iv_crush" in df.columns:
        _crush_raw = pd.to_numeric(df["predicted_iv_crush"], errors="coerce").fillna(0.0)
        _crush_valid = _earn_pop_mask & (_crush_raw > 0.01)
        if _crush_valid.any():
            _crush_adj_iv = IV_vals[_crush_valid] * (1.0 - _crush_raw.values[_crush_valid])
            _crush_adj_iv = np.maximum(_crush_adj_iv, 1e-9)
            _pop_crush = calculate_probability_of_profit(
                types_vals[_crush_valid], S_vals[_crush_valid], K_vals[_crush_valid],
                T_vals[_crush_valid], _crush_adj_iv, prem_vals[_crush_valid], r=risk_free_rate, q=_q,
            )
            if _pop_crush is not None:
                # Blend: 70% crush-adjusted, 30% raw (uncertainty in crush magnitude)
                df.loc[_crush_valid, "prob_profit"] = np.clip(
                    0.7 * _pop_crush + 0.3 * df.loc[_crush_valid, "prob_profit"].values, 0.0, 1.0
                )

    # For Premium Selling, flip PoP to reflect the SELLER's perspective.
    # calculate_probability_of_profit() returns the BUYER's PoP (P option expires ITM).
    # Seller profits when that same option expires worthless, so seller's PoP = 1 − buyer's PoP.
    # e.g. OTM put buyer: 30% PoP → seller: 70% PoP (which is what we want to score highly).
    if mode == "Premium Selling":
        df["prob_profit"] = (1.0 - df["prob_profit"]).clip(0.0, 1.0)

    # Theoretical value and P(ITM) using market IV (for display/reference)
    d1, d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, q=_q)
    p_itm = np.where(is_call, norm_cdf(d2), norm_cdf(-d2))
    disc = np.exp(-risk_free_rate * T_vals)
    _q_disc_theo = np.exp(-_q * T_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        theo_payoff = np.where(is_call,
            S_vals * _q_disc_theo * norm_cdf(d1) - K_vals * disc * norm_cdf(d2),
            K_vals * disc * norm_cdf(-d2) - S_vals * _q_disc_theo * norm_cdf(-d1))
    df["p_itm"], df["theo_value"] = p_itm, theo_payoff

    # HV-adjusted EV: BS(realized_vol) - market_price
    # Positive = options cheap vs realized vol (edge for buyers)
    # Negative = options expensive vs realized vol (edge for sellers)
    # Prefer EWMA vol for EV (more responsive to recent moves); fall back to 30d HV
    # If HV unavailable, flag as unreliable instead of silently using IV (which defeats EV purpose)
    hv_for_ev = df["hv_ewma"] if "hv_ewma" in df.columns else df["hv_30d"]
    _hv_raw = hv_for_ev.fillna(df["hv_30d"])
    _hv_fallback_mask = _hv_raw.isna()
    hv_arr = np.maximum(_hv_raw.fillna(df["impliedVolatility"]).values, 1e-9)

    # Forward-looking adjustment: IV term structure slope signals whether market
    # expects vol to rise (contango) or fall (backwardation). Adjust HV used for
    # EV by ±5% based on term structure direction.
    try:
        if "expiration" in df.columns and "impliedVolatility" in df.columns:
            exp_iv_mean = df.groupby("expiration")["impliedVolatility"].transform("mean")
            exp_dte = (df["T_years"] * 365).fillna(30)
            # Per-row slope: compare this expiration's avg IV to the chain mean
            chain_iv_mean = df["impliedVolatility"].mean()
            ts_signal = np.where(
                (exp_iv_mean > chain_iv_mean * 1.02) & (exp_dte > 20), 1.05,
                np.where(
                    (exp_iv_mean < chain_iv_mean * 0.98) & (exp_dte > 20), 0.95,
                    1.0
                )
            )
            hv_arr = hv_arr * ts_signal
    except Exception as _ts_exc:
        _SCAN_WARNINGS[0] += 1
        logging.getLogger(__name__).debug("IV term structure adjustment failed: %s", _ts_exc)
    hv_d1, hv_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, hv_arr, q=_q)
    _q_disc = np.exp(-_q * T_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        hv_payoff = np.where(is_call,
            S_vals * _q_disc * norm_cdf(hv_d1) - K_vals * disc * norm_cdf(hv_d2),
            K_vals * disc * norm_cdf(-hv_d2) - S_vals * _q_disc * norm_cdf(-hv_d1))
    df["ev_per_contract"] = 100.0 * (hv_payoff - prem_vals) - (50.0 * prem_vals * df["spread_pct"].fillna(0.0).values)
    # Null out EV where HV was missing (IV fallback produces meaningless ~0 values)
    df["ev_hv_fallback"] = _hv_fallback_mask
    df.loc[_hv_fallback_mask, "ev_per_contract"] = np.nan

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
                ev_d1, ev_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, ev_sig_full, q=_q)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ev_earn_payoff = np.where(
                        is_call,
                        S_vals * _q_disc * norm_cdf(ev_d1) - K_vals * disc * norm_cdf(ev_d2),
                        K_vals * disc * norm_cdf(-ev_d2) - S_vals * _q_disc * norm_cdf(-ev_d1),
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
    RATE_SENSITIVE = set(config.get("rate_sensitive_tickers", ["QQQ", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX"]))
    if tnx_change_pct > 0.025:
        df["yield_warning"] = np.where(df["symbol"].isin(RATE_SENSITIVE), "📉 RATES UP", "")
    else:
        df["yield_warning"] = ""

    # SVI IV surface fitting — detect mispriced contracts vs fitted vol smile
    try:
        from .iv_surface import fit_svi_surface
        df = fit_svi_surface(df)
    except Exception as _svi_exc:
        _SCAN_WARNINGS[0] += 1
        logging.getLogger(__name__).debug("SVI surface fit failed: %s", _svi_exc)
        df["iv_surface_residual"] = np.nan
        df["iv_surface_confidence"] = 0.0
        df["iv_surface_fitted"] = False

    # NaN gate: drop contracts where ALL key columns are NaN (no usable data)
    _nan_key_cols = ["impliedVolatility", "volume", "openInterest"]
    _nan_present = [c for c in _nan_key_cols if c in df.columns]
    if _nan_present:
        _nan_count = df[_nan_present].isna().sum(axis=1)
        _nan_drop = _nan_count >= len(_nan_present)
        if _nan_drop.any():
            logging.getLogger(__name__).info(
                "NaN gate: dropped %d/%d contracts (all key columns NaN)",
                _nan_drop.sum(), len(df),
            )
            df = df[~_nan_drop].copy()

    return df


def _cross_section_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw quality_score to the display [0, 1] scale using a fixed absolute reference.

    The raw quality_score is a weighted average of 23+ components compressed into
    roughly the 0.28–0.82 range. This function stretches that to [0, 1] using a
    fixed linear mapping — no batch effects, no relative ranking, no caps.

    A contract with raw 0.55 always maps to the same display score regardless of
    what else is in the scan. Weak market days honestly show 2-star setups;
    strong market days show 3–5 star setups.

    Reference:
        raw 0.28 → 0.00  (near-minimum: all signals weak + penalties)
        raw 0.40 → 0.30  (2-star threshold)
        raw 0.50 → 0.52  (3-star threshold)
        raw 0.60 → 0.70  (4-star threshold)
        raw 0.70 → 0.85  (5-star threshold)
        raw 0.82 → 1.00  (near-maximum: all signals strong + bonuses)
    """
    n = len(df)
    if n <= 1:
        return df

    raw = df["quality_score"].copy()

    # Wider range [0.28, 0.82] (span=0.54) accommodates penalty accumulation.
    # Power curve (x^0.65) expands the middle range where scores cluster.
    base = ((raw - 0.28) / 0.54).clip(0, 1)
    normalized = base ** 0.65

    # EV tiebreaker: contracts with the same raw quality rank by expected value.
    # Centered at 0 (median EV gets no bonus/penalty). Max effect: ±0.015.
    if "ev" in df.columns:
        ev_rank = df["ev"].rank(pct=True).fillna(0.5)
        normalized = (normalized + (ev_rank - 0.5) * 0.03).clip(0, 1)

    df["quality_score"] = normalized.round(4)
    return df


def calculate_scores(
    df: pd.DataFrame,
    config: Dict,
    vix_regime_weights: Dict,
    trader_profile: str,
    mode: str,
    min_dte: int,
    max_dte: int,
    sector_etf: Optional[str] = None,
) -> pd.DataFrame:
    """Calculates subjective quality scores using normalization and weights."""
    
    def rank_norm(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1:
            return pd.Series([0.5] * n, index=s.index)
        r = s.rank(method="average", na_option="keep")
        return (r - 1.0) / (n - 1.0)

    def _sigmoid_scale(x, center: float = 0.0, scale: float = 12.0):
        """Smooth [0,1] mapping that preserves information at extremes."""
        return 1.0 / (1.0 + np.exp(-scale * (x - center)))

    # Base features — absolute liquidity scoring via sigmoid (cross-ticker comparable)
    _liq_cfg = config.get("liquidity_thresholds", {})
    _vol_center = _liq_cfg.get("volume_center", 500)
    _vol_scale = _liq_cfg.get("volume_scale", 0.005)
    _oi_center = _liq_cfg.get("oi_center", 2000)
    _oi_scale = _liq_cfg.get("oi_scale", 0.001)
    vol_raw = df["volume"].fillna(0).astype(float)
    oi_raw = df["openInterest"].fillna(0).astype(float)
    vol_n = pd.Series(1.0 / (1.0 + np.exp(-_vol_scale * (vol_raw.values - _vol_center))), index=df.index)
    oi_n = pd.Series(1.0 / (1.0 + np.exp(-_oi_scale * (oi_raw.values - _oi_center))), index=df.index)
    sp_cap = max(config.get("spread_score_cap", 0.25), 0.01)
    sp = pd.to_numeric(df["spread_pct"], errors="coerce").fillna(1.0).clip(lower=0)
    spread_score = pd.Series(
        1.0 / (1.0 + np.exp(20.0 * (sp.values / sp_cap - 0.7))),
        index=df.index,
    ).clip(0, 1)
    d_target = config.get("target_delta", 0.40)
    delta_quality = (1.0 - (df["abs_delta"] - d_target).abs() / max(d_target, 1e-6)).clip(0, 1)
    iv_n = rank_norm(df["impliedVolatility"].fillna(df["impliedVolatility"].median()))
    iv_quality = 1.0 - (2.0 * (iv_n - 0.5).abs())
    liquidity = 0.5 * (vol_n + oi_n)
    pop_score = df["prob_profit"].fillna(0.5).clip(0, 1)
    rr_raw = pd.to_numeric(df["rr_ratio"], errors='coerce').fillna(0.0)
    # Smooth linear mapping [0.5 → 0, 4.0 → 1] instead of hard step thresholds
    rr_score = np.clip((rr_raw - 0.5) / 3.5, 0.0, 1.0)
    _ev_for_rank = df["ev_per_contract"].copy()
    if mode == "Premium Selling":
        _ev_for_rank = -_ev_for_rank  # seller's edge = prem_vals - hv_payoff
    ev_score = rank_norm(_ev_for_rank.fillna(_ev_for_rank.median()))
    # Blend ev_score with ev_earnings_score for earnings plays (Improvement 6)
    if "ev_earnings" in df.columns and "Earnings Play" in df.columns:
        try:
            _earn_play_mask = df["Earnings Play"] == "YES"
            _ev_earn_num = pd.to_numeric(df["ev_earnings"], errors="coerce")
            _ev_earn_valid = _earn_play_mask & _ev_earn_num.notna()
            if _ev_earn_valid.any():
                _ev_earn_for_rank = _ev_earn_num.copy()
                if mode == "Premium Selling":
                    _ev_earn_for_rank = -_ev_earn_for_rank
                ev_earnings_score = rank_norm(_ev_earn_for_rank.fillna(_ev_for_rank.median()))
                ev_score = ev_score.copy()
                ev_score.loc[_ev_earn_valid] = (
                    0.5 * ev_score.loc[_ev_earn_valid]
                    + 0.5 * ev_earnings_score.loc[_ev_earn_valid]
                )
        except Exception as _ev_blend_exc:
            _SCAN_WARNINGS[0] += 1
            logging.getLogger(__name__).debug("Earnings EV blend failed: %s", _ev_blend_exc)
    em_realism_score = pd.to_numeric(df["em_realism_score"], errors='coerce').fillna(0.5).clip(0, 1)
    theta_raw = df["theta_decay_pressure"].replace([pd.NA, pd.NaT], np.nan)
    theta_ranked = rank_norm(theta_raw.fillna(theta_raw.median()))
    if mode == "Premium Selling":
        theta_score = theta_ranked.clip(0, 1)        # fast decay = good for sellers
    else:
        theta_score = (1.0 - theta_ranked).clip(0, 1) # fast decay = bad for buyers
    theta_score = theta_score.where((df["T_years"] * 365.0) > 7, theta_score * 0.7)
    
    # Multi-timeframe momentum confluence (replaces simple momentum_score)
    ret5 = pd.to_numeric(df.get("ret_5d", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    rsi_vals = pd.to_numeric(df.get("rsi_14", pd.Series(50.0, index=df.index)), errors="coerce").fillna(50.0)
    price_vs_sma20 = (df.get("underlying", pd.Series(0.0, index=df.index)).values /
                      df.get("sma_20", pd.Series(1.0, index=df.index)).replace(0, np.nan).values) - 1.0
    price_vs_vwap = (df.get("underlying", pd.Series(0.0, index=df.index)).values /
                     df.get("vwap", pd.Series(1.0, index=df.index)).replace(0, np.nan).values) - 1.0

    is_call_mom = df["type"].str.lower() == "call"

    # Each signal returns 1 (aligned) or 0 (not aligned) for the option's direction
    # For calls: want momentum UP, RSI not overbought, price above SMA/VWAP
    # For puts: want momentum DOWN, RSI not oversold, price below SMA/VWAP
    sig_ret5 = np.where(is_call_mom, (ret5 > 0).astype(float), (ret5 < 0).astype(float))
    # ADX trend-strength gate: reduce RSI penalty by 70% when ADX > 25 (strong trend)
    # prevents penalizing overbought momentum in trending markets
    adx_vals = pd.to_numeric(df.get("adx_14", pd.Series(20.0, index=df.index)), errors="coerce").fillna(20.0)
    _rsi_penalty_mult = np.where(adx_vals > 25, 0.3, 1.0)
    sig_rsi  = np.where(is_call_mom,
                        np.clip(1.0 - _rsi_penalty_mult * (rsi_vals - 50.0).clip(0, 30) / 30.0, 0.0, 1.0),
                        np.clip(1.0 - _rsi_penalty_mult * (50.0 - rsi_vals).clip(0, 30) / 30.0, 0.0, 1.0))
    sig_sma  = np.where(is_call_mom, (price_vs_sma20 > 0).astype(float), (price_vs_sma20 < 0).astype(float))
    sig_vwap = np.where(is_call_mom, (price_vs_vwap > 0).astype(float), (price_vs_vwap < 0).astype(float))

    # Weighted confluence: configurable via momentum_weights
    mw = config.get("momentum_weights", {})
    w_rsi  = mw.get("rsi", 0.35)
    w_ret5 = mw.get("ret5d", 0.30)
    w_sma  = mw.get("sma", 0.20)
    w_vwap = mw.get("vwap", 0.15)
    momentum_score = pd.Series(
        w_ret5 * sig_ret5 + w_rsi * sig_rsi + w_sma * sig_sma + w_vwap * sig_vwap,
        index=df.index
    ).clip(0, 1)
    df["momentum_score"] = momentum_score
    df["momentum_confluence"] = momentum_score
    
    # Blend 30-day and 90-day IV percentile for a more stable IV rank signal.
    # 90-day context prevents over-reacting to short-term vol spikes.
    iv_pct_30 = pd.to_numeric(df.get("iv_percentile_30", df.get("iv_percentile", pd.Series(np.nan, index=df.index))), errors="coerce")
    iv_pct_90 = pd.to_numeric(df.get("iv_percentile_90", pd.Series(np.nan, index=df.index)), errors="coerce")
    # Where 90-day is available, blend 60/40 (30d/90d); otherwise fall back to 30d alone
    iv_pct_series = iv_pct_30.where(iv_pct_90.isna(), 0.6 * iv_pct_30 + 0.4 * iv_pct_90)
    iv_rank_score = iv_pct_series.clip(0, 1).fillna(0.5) if mode == "Premium Selling" else (1.0 - iv_pct_series.clip(0, 1)).fillna(0.5)
    # Dampen IV rank score by history confidence (Low = 0.5, Medium = 0.8, High = 1.0)
    iv_conf = df.get("iv_confidence", pd.Series("High", index=df.index))
    conf_mult = iv_conf.map({"High": 1.0, "Medium": 0.8, "Low": 0.5}).fillna(0.7)
    iv_rank_score = iv_rank_score * conf_mult
    catalyst_score = pd.Series(0.3, index=df.index).mask(df["event_flag"] == "EARNINGS_NEARBY", 0.8)
    dte_norm = ((df["T_years"] * 365.0 - min_dte) / max(1, (max_dte - min_dte))).clip(0, 1)
    trader_pref_score = (0.6 * liquidity + 0.4 * spread_score) if trader_profile.lower().startswith("day") else (0.5 * delta_quality + 0.5 * dte_norm)

    # IV Edge Score: sigmoid mapping preserves tail information (replaces ±20% linear clip)
    iv_vs_hv = df.get("iv_vs_hv", pd.Series(0.0, index=df.index)).fillna(0.0)
    if mode == "Premium Selling":
        iv_edge_score = pd.Series(_sigmoid_scale(iv_vs_hv.values, center=0.0, scale=8.0), index=df.index)
    else:
        iv_edge_score = pd.Series(_sigmoid_scale(-iv_vs_hv.values, center=0.0, scale=8.0), index=df.index)

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

    # Initialise components only computed in the non-Premium-Selling branch.
    # Stays as NaN in Premium Selling mode → calibrator dropna() handles it cleanly.
    pcr_score = pd.Series(float("nan"), index=df.index)
    gex_score = pd.Series(float("nan"), index=df.index)
    oi_change_score = pd.Series(float("nan"), index=df.index)
    sentiment_score_component = pd.Series(float("nan"), index=df.index)

    # Weight Application
    if mode == "Premium Selling":
        weights = config.get("premium_selling_weights", {})
        ror_score = rank_norm(df["return_on_risk"].fillna(df["return_on_risk"].median()))
        w = {k: weights.get(k, 0.0) for k in ["pop", "return_on_risk", "iv_rank", "liquidity", "theta", "ev", "trader_pref"]}
        w["em_realism"] = weights.get("em_realism", 0.05)
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (w["pop"]*pop_score + w["return_on_risk"]*ror_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity + w["theta"]*theta_score + w["ev"]*ev_score + w["trader_pref"]*trader_pref_score + w["em_realism"]*em_realism_score) / w_sum
        try:
            _cdf = pd.DataFrame({"PoP": w["pop"]*pop_score, "RoR": w["return_on_risk"]*ror_score,
                                  "IV rank": w["iv_rank"]*iv_rank_score, "Liq": w["liquidity"]*liquidity,
                                  "Theta": w["theta"]*theta_score, "EV": w["ev"]*ev_score,
                                  "EM real": w["em_realism"]*em_realism_score}, index=df.index)
            df["score_drivers"] = _cdf.apply(lambda r: " · ".join(r.nlargest(3).index.tolist()), axis=1)
        except Exception:
            df["score_drivers"] = ""
    else:
        is_call_series = df["type"].str.lower() == "call"

        # Build weight dict early so zero-weight guards can reference it
        dw = {
            # Rebalanced: profitability (pop+rr+ev=30%), direction (momentum=10%),
            # vol signals (iv_rank+iv_edge+vrp=21%), execution (liq+spread+theta=15%)
            "pop": 0.13, "iv_mispricing": 0.05, "rr": 0.10, "momentum": 0.10,
            "iv_rank": 0.08, "liquidity": 0.08, "catalyst": 0.00, "theta": 0.06,
            "ev": 0.07, "trader_pref": 0.00, "iv_edge": 0.08, "skew_align": 0.02,
            "gamma_theta": 0.00, "pcr": 0.00, "gex": 0.01, "oi_change": 0.00,
            "sentiment": 0.00, "option_rvol": 0.00, "vrp": 0.05, "gamma_pin": 0.00,
            "max_pain": 0.01, "iv_velocity": 0.05, "em_realism": 0.00,
            "gamma_magnitude": 0.03, "vega_risk": 0.03, "term_structure": 0.04,
            "spread": 0.01,
        }
        cw = load_ic_adjusted_weights(config)
        w = {k: cw.get(k, dw[k]) for k in dw}

        # PCR score (skip if zero weight)
        if w.get("pcr", 0) > 0:
            pcr_vals = pd.to_numeric(df.get("pcr", pd.Series(np.nan, index=df.index)), errors="coerce")
            pcr_score_call = (1 - np.clip(pcr_vals / 2.0, 0, 1)).fillna(0.5)
            pcr_score_put = np.clip(pcr_vals / 2.0, 0, 1).fillna(0.5)
            pcr_score = pd.Series(np.where(is_call_series, pcr_score_call, pcr_score_put), index=df.index)
        else:
            pcr_score = pd.Series(0.5, index=df.index)

        # GEX score: distance-based sigmoid (replaces binary 0.3/0.7)
        gex_flip = pd.to_numeric(df.get("gex_flip_price", pd.Series(np.nan, index=df.index)), errors="coerce")
        underlying_s = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce")
        gex_flip_valid = gex_flip.notna() & (underlying_s > 0)
        gex_dist_pct = ((underlying_s - gex_flip) / underlying_s).fillna(0.0)
        # Calls: above GEX flip (positive gamma) is favorable; puts: below is favorable
        gex_score_call = pd.Series(_sigmoid_scale(gex_dist_pct.values, center=0.0, scale=30.0), index=df.index)
        gex_score_put = pd.Series(_sigmoid_scale(-gex_dist_pct.values, center=0.0, scale=30.0), index=df.index)
        gex_score = pd.Series(np.where(is_call_series, gex_score_call, gex_score_put), index=df.index)
        gex_score = gex_score.where(gex_flip_valid, 0.5)  # neutral when no flip data

        # OI change score (skip if zero weight)
        if w.get("oi_change", 0) > 0:
            oi_chg = pd.to_numeric(df.get("oi_change", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
            oi_change_score = rank_norm(oi_chg)
        else:
            oi_change_score = pd.Series(0.5, index=df.index)

        # Sentiment score (skip if zero weight)
        if w.get("sentiment", 0) > 0:
            raw_sent = pd.to_numeric(
                df.get("sentiment_score", pd.Series(0.0, index=df.index)), errors="coerce"
            ).fillna(0.0).clip(-1.0, 1.0)
            sent_call = ((raw_sent + 0.5) / 1.0).clip(0, 1)
            sent_put = ((0.5 - raw_sent) / 1.0).clip(0, 1)
            sentiment_score_component = pd.Series(
                np.where(is_call_series, sent_call, sent_put), index=df.index
            )
        else:
            sentiment_score_component = pd.Series(0.5, index=df.index)

        # Option RVOL score (skip if zero weight)
        if w.get("option_rvol", 0) > 0:
            option_rvol_score = rank_norm(
                pd.to_numeric(df.get("option_rvol", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
            ).clip(0, 1)
        else:
            option_rvol_score = pd.Series(0.5, index=df.index)
        df["option_rvol_score"] = option_rvol_score

        # Skew combined score: blend directional alignment with percentile rank
        iv_skew_rank_vals = pd.to_numeric(
            df.get("iv_skew_rank", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        skew_rank_score = pd.Series(
            np.where(is_call_series, 1.0 - iv_skew_rank_vals, iv_skew_rank_vals),
            index=df.index,
        ).clip(0, 1)
        skew_combined_score = (0.5 * skew_align_score + 0.5 * skew_rank_score).clip(0, 1)
        df["skew_align_score"] = skew_combined_score

        # VRP score: sigmoid scaling preserves information at extremes (vs old linear clip)
        vrp_vals = pd.to_numeric(
            df.get("vrp_mean", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        if mode == "Premium Selling":
            vrp_score = _sigmoid_scale(vrp_vals, center=0.025, scale=12.0)
        else:
            vrp_score = _sigmoid_scale(-vrp_vals, center=0.025, scale=12.0)
        df["vrp_score"] = vrp_score

        # Vega risk score: penalizes high vega exposure when IV rank is already elevated
        # High vega + high IV rank = mean-reversion risk (IV likely to compress)
        vega_vals = pd.to_numeric(df.get("vega", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).abs()
        vega_dollar = vega_vals * 100  # per contract
        iv_rank_for_vega = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
        vega_risk_raw = vega_dollar.rank(pct=True) * iv_rank_for_vega
        vega_risk_score = (1.0 - vega_risk_raw).clip(0, 1)
        df["vega_risk_score"] = vega_risk_score

        # Gamma pin score (skip if zero weight)
        if w.get("gamma_pin", 0) > 0:
            gamma_pin_dist = pd.to_numeric(
                df.get("gamma_pin_dist_pct", pd.Series(100.0, index=df.index)), errors="coerce"
            ).fillna(100.0)
            dte_arr = df["T_years"].values * 365.0
            gamma_pin_score = pd.Series(
                np.where(
                    dte_arr <= 14,
                    np.clip(1.0 - (gamma_pin_dist.values / 10.0), 0.0, 1.0),
                    0.5,
                ),
                index=df.index,
            )
        else:
            gamma_pin_score = pd.Series(0.5, index=df.index)
        df["gamma_pin_score"] = gamma_pin_score

        # Gamma magnitude score: high gamma near expiry = outsized convexity opportunity
        gamma_vals_mag = pd.to_numeric(df.get("gamma", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        underlying_vals = pd.to_numeric(df.get("underlying", pd.Series(100.0, index=df.index)), errors="coerce").fillna(100.0)
        gamma_dollar = (gamma_vals_mag * underlying_vals).abs()
        dte_vals = df["T_years"].values * 365.0
        expiry_boost = np.where(dte_vals < 14, 1.5, np.where(dte_vals < 21, 1.2, 1.0))
        gamma_mag_raw = gamma_dollar * expiry_boost
        gamma_magnitude_score = gamma_mag_raw.rank(pct=True).fillna(0.5)
        df["gamma_magnitude_score"] = gamma_magnitude_score

        # Max pain score: near max pain is favorable for the market structure (sellers get paid)
        # Score high if within 3%, drop off linearly to 0.3 at 10%+
        max_pain_dist = pd.to_numeric(df.get("max_pain_dist_pct", pd.Series(pd.NA, index=df.index)), errors="coerce").fillna(10.0)
        max_pain_score = pd.Series(np.clip(1.0 - (max_pain_dist.values / 8.0), 0.3, 1.0), index=df.index)
        df["max_pain_score"] = max_pain_score

        # IV Velocity score: rewards sellers when IV is expanding, buyers when contracting
        iv_trend_vals = df.get("iv_trend", pd.Series("stable", index=df.index))
        if not isinstance(iv_trend_vals, pd.Series):
            iv_trend_vals = pd.Series("stable", index=df.index)
        # Sellers want IV to contract (option cheapens); buyers want IV to expand (vega profit)
        _is_seller_mode = (mode == "Premium Selling")
        iv_velocity_raw = np.where(
            iv_trend_vals == "expanding",
            0.0 if _is_seller_mode else 1.0,
            np.where(
                iv_trend_vals == "contracting",
                1.0 if _is_seller_mode else 0.0,
                0.5,
            ),
        )
        iv_velocity_score = pd.Series(iv_velocity_raw, index=df.index)
        df["iv_velocity_score"] = iv_velocity_score

        # Term structure score: contango/backwardation impacts trade viability
        ts_spread = pd.to_numeric(
            df.get("term_structure_spread", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        if mode == "Premium Selling":
            # Sellers benefit from contango (positive spread = front IV < back IV)
            term_structure_score = _sigmoid_scale(ts_spread, center=0.0, scale=8.0)
        else:
            # Buyers benefit from backwardation (negative spread = front IV > back IV)
            term_structure_score = _sigmoid_scale(-ts_spread, center=0.0, scale=8.0)
        df["term_structure_score"] = term_structure_score

        # IV surface mispricing score: rewards buying cheap vs surface, selling rich vs surface
        resid = df.get("iv_surface_residual", pd.Series(0.0, index=df.index))
        resid = pd.to_numeric(resid, errors="coerce").fillna(0.0)
        # For buyers (calls): negative residual = cheap = good → score = clip(-resid * 5, 0, 1)
        # For sellers (puts in premium selling): positive residual = rich = good
        is_sell = (mode == "Premium Selling")
        if is_sell:
            iv_mispricing_score = pd.Series(np.clip(resid * 5, 0, 1), index=df.index)
        else:
            iv_mispricing_score = pd.Series(np.clip(-resid * 5, 0, 1), index=df.index)
        # Dampen mispricing score by surface fit confidence
        surf_conf = pd.to_numeric(df.get("iv_surface_confidence", 1.0), errors="coerce").fillna(0.0)
        iv_mispricing_score = iv_mispricing_score * surf_conf.clip(0, 1)
        # Bug fix: when no SVI surface fit exists (surf_conf ≈ 0), the multiplication
        # above yields 0.0 — not the neutral 0.5.  Restore neutrality for unfit contracts.
        iv_mispricing_score = iv_mispricing_score.where(surf_conf > 0.05, 0.5)
        df["iv_mispricing_score"] = iv_mispricing_score

        # Apply VIX regime multipliers — adjust weights based on current vol environment
        regime_mults = config.get("vix_regime_multipliers", {}).get(
            vix_regime_weights.get("regime", "normal") if isinstance(vix_regime_weights, dict) else "normal",
            {}
        )
        for k, mult in regime_mults.items():
            if k in w:
                w[k] *= mult
        # Re-normalize so weights sum to 1.0
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (
            w["pop"]*pop_score + w["em_realism"]*em_realism_score + w["rr"]*rr_score
            + w["momentum"]*momentum_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity
            + w["catalyst"]*catalyst_score + w["theta"]*theta_score + w["ev"]*ev_score
            + w["trader_pref"]*trader_pref_score + w["iv_edge"]*iv_edge_score
            + w["skew_align"]*skew_combined_score + w["gamma_theta"]*gamma_theta_score
            + w["pcr"]*pcr_score + w["gex"]*gex_score + w["oi_change"]*oi_change_score
            + w["sentiment"]*sentiment_score_component + w["option_rvol"]*option_rvol_score
            + w["vrp"]*vrp_score + w["gamma_pin"]*gamma_pin_score
            + w["max_pain"]*max_pain_score + w["iv_velocity"]*iv_velocity_score
            + w["iv_mispricing"]*iv_mispricing_score
            + w["gamma_magnitude"]*gamma_magnitude_score
            + w["vega_risk"]*vega_risk_score
            + w["term_structure"]*term_structure_score
            + w.get("spread", 0)*spread_score
        ) / w_sum
        try:
            _cdf = pd.DataFrame({
                "PoP": w["pop"]*pop_score, "EV": w["ev"]*ev_score,
                "RR": w["rr"]*rr_score, "IV edge": w["iv_edge"]*iv_edge_score,
                "Liq": w["liquidity"]*liquidity, "Theta": w["theta"]*theta_score,
                "Mom": w["momentum"]*momentum_score, "Skew": w["skew_align"]*skew_combined_score,
                "Sent": w["sentiment"]*sentiment_score_component,
                "VRP": w["vrp"]*vrp_score, "IV vel": w["iv_velocity"]*iv_velocity_score,
                "SVI": w["iv_mispricing"]*iv_mispricing_score,
                "Gam": w["gamma_magnitude"]*gamma_magnitude_score,
                "Vega": w["vega_risk"]*vega_risk_score,
                "TSpr": w["term_structure"]*term_structure_score,
            }, index=df.index)
            _neg_cdf = pd.DataFrame({
                "spread": pd.Series(0.0, index=df.index),
                "earnings": pd.Series(0.0, index=df.index),
            }, index=df.index)
            def _fmt_drivers(row, neg_row):
                top3 = row.nlargest(3)
                pos_parts = [f"+{k}({v:.2f})" for k, v in top3.items() if v > 0]
                neg_parts = [f"-{k}({v:.2f})" for k, v in neg_row.items() if v < 0]
                parts = pos_parts
                if neg_parts:
                    parts = pos_parts + ["|"] + neg_parts[:2]
                return " ".join(parts)
            df["score_drivers"] = [
                _fmt_drivers(_cdf.iloc[i], _neg_cdf.iloc[i]) for i in range(len(_cdf))
            ]
        except Exception:
            df["score_drivers"] = ""

    # Adjustments — earnings penalty scaled by historical IV crush magnitude
    _earn_nearby_mask = df["event_flag"] == "EARNINGS_NEARBY"
    if _earn_nearby_mask.any():
        is_seller = (mode == "Premium Selling")
        try:
            from .data_fetching import get_historical_iv_crush
            _ticker = df["symbol"].iloc[0] if "symbol" in df.columns else ""
            crush_data = get_historical_iv_crush(_ticker) if _ticker else {}
        except Exception:
            crush_data = {}
        avg_crush = crush_data.get("avg_crush", 0.25)
        # Continuous scaling: penalty proportional to crush magnitude
        # Buyers: high crush = bigger penalty. Sellers: high crush = opportunity
        if is_seller:
            _earn_penalty = max(0.01, min(0.08, avg_crush * 0.15))
        else:
            _earn_penalty = max(-0.15, min(-0.02, -0.05 * (avg_crush / 0.20)))
        df.loc[_earn_nearby_mask, "quality_score"] += _earn_penalty
        # Store crush info for thesis display
        df.loc[_earn_nearby_mask, "avg_iv_crush"] = avg_crush
        if "score_drivers" in df.columns:
            crush_str = f" earnings({_earn_penalty:+.2f},crush={avg_crush:.0%})"
            for _idx in df.index[_earn_nearby_mask]:
                df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + crush_str
    elif "score_drivers" in df.columns:
        pass  # no earnings nearby — no adjustment needed
    # Reward earnings plays where IV is actually underpriced vs realized vol
    if "Earnings Play" in df.columns and "is_underpriced" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["is_underpriced"]), "quality_score"] += 0.08
    df.loc[df["Trend_Aligned"], "quality_score"] += 0.05
    df.loc[df["decay_warning"], "quality_score"] -= 0.20
    # Gamma ramp: near-expiry gamma explosion is a structural risk — penalise hard
    if "gamma_ramp" in df.columns:
        df.loc[df["gamma_ramp"], "quality_score"] -= 0.15
    df.loc[df["sr_warning"] != "", "quality_score"] -= 0.10
    if "seasonal_win_rate" in df.columns:
        df.loc[df["seasonal_win_rate"] >= 0.8, "quality_score"] += 0.10
        df.loc[df["seasonal_win_rate"] <= 0.2, "quality_score"] -= 0.10
    df.loc[df["oi_wall_warning"] != "", "quality_score"] -= 0.10
    df["squeeze_play"] = (df.get("is_squeezing", pd.Series(False, index=df.index))) & (df.get("Unusual_Whale", pd.Series(False, index=df.index)))
    # Confirmed squeeze (trend-aligned) gets larger bonus; unconfirmed gets small nudge
    _squeeze_confirmed = df["squeeze_play"] & (df.get("Trend_Aligned", pd.Series(False, index=df.index)))
    df.loc[_squeeze_confirmed, "quality_score"] += 0.10
    df.loc[df["squeeze_play"] & ~_squeeze_confirmed, "quality_score"] += 0.04
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

    # Macro event penalty (sector-aware)
    try:
        from .macro_analyzer import get_macro_penalty
        _macro_pen, _macro_active, _macro_desc = get_macro_penalty(config, sector_etf=sector_etf)
        if _macro_active and _macro_pen != 0.0:
            df["quality_score"] += _macro_pen
            df["macro_event_flag"] = _macro_desc
    except Exception:
        pass

    # Tiered bid-ask spread penalty (reduced — spread_score now weighted in composite)
    _spread_pct = pd.to_numeric(df.get("spread_pct", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    _spread_penalty = pd.Series(0.0, index=df.index)
    _spread_penalty = _spread_penalty.where(_spread_pct <= 0.10, -0.04)
    _spread_penalty = _spread_penalty.where(_spread_pct <= 0.15, -0.08)
    df["quality_score"] += _spread_penalty
    # Append spread penalty to score_drivers
    if "score_drivers" in df.columns:
        _has_spread_penalty = _spread_penalty < 0
        for _idx in df.index[_has_spread_penalty]:
            _pen_val = float(_spread_penalty.loc[_idx])
            df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + f" -spread({_pen_val:.2f})"

    df["quality_score"] = df["quality_score"].fillna(0.0).clip(0, 1)

    # PoP soft floor for buyer modes: dampen very low probability trades
    _buyer_modes = {"Single-stock", "Scan", "Watchlist"}
    if mode in _buyer_modes or mode not in ("Premium Selling", "Credit Spreads", "Iron Condor"):
        _low_pop = df["prob_profit"] < 0.25
        if _low_pop.any():
            df.loc[_low_pop, "quality_score"] *= 0.6
            if "score_drivers" in df.columns:
                for _idx in df.index[_low_pop]:
                    df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + " -low_pop(-40%)"

    # Residual earnings crush penalty (reduced — primary crush adjustment now in PoP/EV)
    if "predicted_iv_crush" in df.columns and "Earnings Play" in df.columns:
        _crush_vals = pd.to_numeric(df["predicted_iv_crush"], errors="coerce").fillna(0.0)
        _earn_mask = df["Earnings Play"] == "YES"
        crush_penalty = (_crush_vals * 0.3).clip(0, 0.06)
        df.loc[_earn_mask, "quality_score"] -= crush_penalty[_earn_mask]

    # Catastrophic risk gate: if 3+ structural risks overlap, hard-cap quality_score
    _risk_flags = pd.DataFrame({
        "gamma_ramp":       df.get("gamma_ramp", pd.Series(False, index=df.index)).astype(bool),
        "decay_warning":    df.get("decay_warning", pd.Series(False, index=df.index)).astype(bool),
        "earnings_nearby":  (df.get("event_flag", pd.Series("", index=df.index)) == "EARNINGS_NEARBY"),
        "macro_risk":       df.get("macro_warning", pd.Series("", index=df.index)).str.contains("MACRO RISK", na=False),
        "sr_warning":       df.get("sr_warning", pd.Series("", index=df.index)) != "",
    }).astype(int)
    _risk_count = _risk_flags.sum(axis=1)
    # Graduated scaling instead of hard cap: 3→×0.85, 4→×0.70, 5+→cap 0.40
    _risk_mult = pd.Series(1.0, index=df.index)
    _risk_mult[_risk_count == 3] = 0.85
    _risk_mult[_risk_count == 4] = 0.70
    _risk_mult[_risk_count >= 5] = 0.50
    df["quality_score"] = df["quality_score"] * _risk_mult
    df.loc[_risk_count >= 5, "quality_score"] = df.loc[_risk_count >= 5, "quality_score"].clip(upper=0.40)
    df["risk_flag_count"] = _risk_count

    # Save components
    df["ev_score"] = ev_score
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"], df["delta_quality"], df["iv_quality"] = liquidity, delta_quality, iv_quality
    df["spread_score"], df["theta_score"], df["momentum_score"] = spread_score, theta_score, momentum_score
    df["iv_rank_score"], df["catalyst_score"] = iv_rank_score, catalyst_score
    df["iv_advantage_score"] = iv_edge_score  # mode-aware: buyers rewarded for IV < HV
    df["pop_score"] = pop_score   # stored for backtester IC analysis
    df["rr_score"]  = rr_score    # stored for backtester IC analysis
    df["trader_pref_score"] = trader_pref_score
    df["gamma_theta_score"] = gamma_theta_score
    df["pcr_score"]         = pcr_score
    df["gex_score"]         = gex_score
    df["oi_change_score"]   = oi_change_score
    # Save normalised sentiment under a distinct name so the raw `sentiment_score`
    # input column (used by display) is preserved.
    df["sentiment_score_norm"] = sentiment_score_component

    # Long Gamma mode: compute dedicated score with inverted IV rank
    # Done after all component scores are saved so we can reference them by column name
    if mode == "Long Gamma":
        lg_w = config.get("long_gamma_weights", {
            "iv_cheap": 0.30, "squeeze": 0.25, "rvol": 0.20, "momentum": 0.15, "liquidity": 0.10
        })
        # Invert IV rank: low IV = good for buying options
        _iv_cheap = (1.0 - pd.to_numeric(
            df.get("iv_rank_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)).clip(0, 1)
        # Invert BB width: tight bands = compressed vol = primed to expand
        _bb_raw = pd.to_numeric(
            df.get("bb_width_pct", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _squeeze = (1.0 - _bb_raw).clip(0, 1)
        # Relative volume: rank-normalised (high rvol = catalyst brewing)
        _rvol_raw = pd.to_numeric(
            df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce"
        ).fillna(1.0)
        _n = len(_rvol_raw)
        _rvol_n = ((_rvol_raw.rank(method="average", na_option="keep") - 1.0) / max(_n - 1, 1)).clip(0, 1) if _n > 1 else pd.Series(0.5, index=df.index)
        _mom = pd.to_numeric(
            df.get("momentum_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _liq = pd.to_numeric(
            df.get("liquidity_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _lg_sum = sum(lg_w.values()) or 1.0
        df["long_gamma_score"] = (
            lg_w.get("iv_cheap", 0.30) * _iv_cheap
            + lg_w.get("squeeze", 0.25) * _squeeze
            + lg_w.get("rvol", 0.20) * _rvol_n
            + lg_w.get("momentum", 0.15) * _mom
            + lg_w.get("liquidity", 0.10) * _liq
        ) / _lg_sum
        # Override quality_score so existing sort/display logic works unchanged
        df["quality_score"] = df["long_gamma_score"]
    else:
        df["long_gamma_score"] = pd.NA

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
    vrp_data: dict = None,
    news_data=None,
    dividend_yield: float = 0.0,
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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
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
        if df.empty:
            return df
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

    if df.empty:
        return df

    # IV Smile Outlier Filter: remove bad-print IV rows before enrichment
    df = filter_iv_smile_outliers(
        df,
        iv_threshold=config.get("iv_outlier_threshold", 0.30),
        min_volume=config.get("iv_outlier_min_volume", 10),
    )
    if df.empty:
        return df

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors='coerce')
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    ov_iv_m = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(ov_iv_m if pd.notna(ov_iv_m) else 0.25)

    # Attach EWMA vol column if provided
    if hv_ewma is not None and "hv_ewma" not in df.columns:
        df["hv_ewma"] = hv_ewma

    # Attach VRP data
    if vrp_data:
        df["vrp_mean"] = vrp_data.get("vrp_mean", 0.0)
        df["vrp_regime"] = vrp_data.get("vrp_regime", "UNKNOWN")
    else:
        df["vrp_mean"] = 0.0
        df["vrp_regime"] = "UNKNOWN"

    # Attach term structure spread
    if term_structure_spread is not None:
        df["term_structure_spread"] = term_structure_spread

    # 1. Call Helper: Metrics
    _div_yield = float(df["dividend_yield"].iloc[0]) if "dividend_yield" in df.columns and not df.empty else dividend_yield
    df = calculate_metrics(
        df, risk_free_rate, earnings_date, config, iv_rank, iv_percentile,
        sentiment_score, macro_risk_active, sector_perf, tnx_change_pct,
        short_interest=short_interest, next_ex_div=next_ex_div,
        earnings_move_data=earnings_move_data, mode=mode,
        dividend_yield=_div_yield,
    )

    # 2. Call Helper: Scores
    _sector_etf = sector_perf.get("sector_etf") if sector_perf else None
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte, sector_etf=_sector_etf)

    # Strategy recommendation for Long Gamma mode
    if mode == "Long Gamma":
        _is_sq = df.get("is_squeezing", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        _adx = pd.to_numeric(df.get("adx_14", pd.Series(20.0, index=df.index)), errors="coerce").fillna(20.0)
        _rsi = pd.to_numeric(df.get("rsi_14", pd.Series(50.0, index=df.index)), errors="coerce").fillna(50.0)
        _iv_pct = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.3, index=df.index)), errors="coerce").fillna(0.3)
        _rvol = pd.to_numeric(df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
        _underlying = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _sma50 = pd.to_numeric(df.get("sma_50", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _neutral_rsi = (_rsi >= 45) & (_rsi <= 55)
        _conditions = [
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct < 0.15),
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct >= 0.15),
            _is_sq & (_adx >= 20) & (_rsi > 55) & (_underlying > _sma50),
            _is_sq & (_adx >= 20) & (_rsi < 45) & (_underlying < _sma50),
            (~_is_sq) & (_rvol > 1.5) & (_adx > 25),
        ]
        _choices = ["Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Directional Debit Spread"]
        df["recommended_strategy"] = np.select(_conditions, _choices, default="Monitor")
    else:
        df["recommended_strategy"] = ""

    # Final Filters
    if mode == "Long Gamma":
        # No delta target for straddles/strangles — they span the full delta range by design
        pass
    elif mode == "Premium Selling":
        d_min = fc.get("premium_selling_delta_min", 0.15)
        d_max = fc.get("premium_selling_delta_max", 0.40)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    else:
        d_min = fc.get("delta_min", 0.15)
        d_max = fc.get("delta_max", 0.35)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    if mode not in ("Premium Selling", "Long Gamma"):
        df = df[df["rr_ratio"] >= 0.25].copy()

    if df.empty:
        return df

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


def normalize_spreads_for_ranking(spreads_df: pd.DataFrame, mode: str = "Credit Spreads") -> pd.DataFrame:
    """
    Convert credit spread candidates into a picks-compatible DataFrame row format
    so they can be scored alongside single-leg options.

    Maps spread fields to the closest equivalent single-leg fields:
    - premium -> net_credit (the credit received is the "premium" analog)
    - prob_profit -> estimated from net_credit / spread_width (P(expire worthless))
    - delta -> short_strike delta (already computed in the source data)
    - rr_ratio -> max_profit / max_loss
    - quality_score -> existing quality_score from find_credit_spreads
    """
    if spreads_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in spreads_df.head(5).iterrows():
        net_credit = float(row.get("net_credit", 0) or 0)
        max_profit = float(row.get("max_profit", 0) or 0)
        max_loss = float(row.get("max_loss", 1) or 1)
        spread_width = abs(float(row.get("short_strike", 0)) - float(row.get("long_strike", 0)))

        # Probability of max profit: net_credit / spread_width (breakeven-based PoP proxy)
        pop_proxy = (net_credit / spread_width) if spread_width > 0 else 0.5
        pop_proxy = min(max(pop_proxy, 0.3), 0.9)

        rr = (max_profit / 100) / net_credit if net_credit > 0 else 0.0

        normalized = {
            "symbol": row.get("symbol", ""),
            "type": f"{row.get('type', 'spread').upper()} SPREAD",
            "strike": row.get("short_strike", 0),
            "expiration": row.get("expiration", ""),
            "premium": net_credit,
            "underlying": row.get("underlying", 0) if "underlying" in row else 0,
            "prob_profit": pop_proxy,
            "rr_ratio": rr,
            "quality_score": float(row.get("quality_score", 0.5)),
            "impliedVolatility": 0.0,  # not meaningful for spreads
            "delta": 0.0,
            "volume": 0,
            "openInterest": 0,
            "ev_per_contract": max_profit / 100 * pop_proxy - net_credit * (1 - pop_proxy),
            "spread_pct": 0.05,  # spreads have defined risk, treat as tight
            "_is_spread": True,
            "_spread_type": row.get("type", ""),
            "_max_profit": max_profit,
            "_max_loss": max_loss,
        }
        rows.append(normalized)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Fill required columns with safe defaults so enrich_and_score doesn't error
    for col in ["T_years", "abs_delta", "iv_rank", "iv_percentile_30", "hv_30d",
                "iv_skew_rank", "option_rvol", "gamma_pin_dist_pct", "vrp_mean"]:
        if col not in result.columns:
            result[col] = 0.0
    return result


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
                'entry_price', 'entry_underlying', 'delta', 'iv', 'hv', 'iv_rank',
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
                    'entry_price': row.get('premium', ''),
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
    # Non-TTY: return default immediately to avoid hanging in pipes/CI
    if not sys.stdin.isatty() and default is not None:
        return default
    if HAS_ENHANCED_CLI:
        colored_prompt = fmt.colorize(prompt, fmt.Colors.BRIGHT_CYAN)
        if default is not None:
            colored_default = fmt.colorize(f"[{default}]", fmt.Colors.DIM)
            sys.stdout.write(f"{colored_prompt} {colored_default}: {fmt.Colors.RESET}")
        else:
            sys.stdout.write(f"{colored_prompt}: {fmt.Colors.RESET}")
        sys.stdout.flush()
        val = sys.stdin.readline().strip()
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
                print("  ⚠️  No price data available")
                continue
            
            # Find closest date to expiration
            hist_dates = hist.index.date
            closest_date = min(hist_dates, key=lambda d: abs((d - exp_date).days))
            filtered = hist[hist.index.date == closest_date]
            if filtered.empty:
                print("  ⚠️  No matching price for expiry date")
                continue
            exit_price = float(filtered['Close'].iloc[0])
            
            # Calculate intrinsic value at expiration
            strike = float(trade['strike'])
            option_type = trade['type'].lower()
            
            if option_type == 'call':
                intrinsic_value = max(0, exit_price - strike)
            else:  # put
                intrinsic_value = max(0, strike - exit_price)
            
            entry_price = float(trade['entry_price'])
            exit_premium = intrinsic_value
            
            # P/L per share
            pnl_per_share = exit_premium - entry_price
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
        vrp_data = context.get("vrp_data", {})

        context_log = []
        if hv:
            context_log.append(f"HV (30d): {hv:.2%}")
        if iv_rank:
            context_log.append(f"IV Rank: {iv_rank:.2f}")
        if earnings_date:
            context_log.append(f"Earnings: {earnings_date.strftime('%Y-%m-%d')}")
        if context.get("rvol"):
            context_log.append(f"RVOL: {context['rvol']:.2f}x")
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
            vrp_data=vrp_data,
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
        elif mode == "Long Gamma":
            from .filters import filter_long_gamma
            lg_filtered = filter_long_gamma(df_scored)
            if not lg_filtered.empty:
                result["picks"].append(lg_filtered)
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


def run_scan(mode: str, tickers: List[str], budget: Optional[float], max_expiries: int, min_dte: int, max_dte: int, trader_profile: str, logger: logging.Logger, market_trend: str, volatility_regime: str, macro_risk_active: bool = False, tnx_change_pct: float = 0.0, verbose: bool = True, custom_weights: Optional[Dict] = None, show_surface: bool = False, surface_mode: str = "braille", surface_type: str = "pnl", show_contours: bool = True, compact: bool = False):
    # Determine mode booleans for internal logic

    # === LOAD CONFIGURATION ===
    if verbose:
        print("\nLoading configuration...")
    config = load_config("config.json")

    # Merge custom weights if provided (from UI)
    if custom_weights:
        config['composite_weights'].update(custom_weights)

    if verbose:
        print("✓ Configuration loaded")

    # === SECTOR RELATIVE STRENGTH ===
    sector_ctx = None
    if config.get("sector_analysis", {}).get("enabled", True):
        try:
            from .sector_analyzer import SectorAnalyzer
            sector_ctx = SectorAnalyzer().get_sector_context()
            if verbose and sector_ctx.top_sectors:
                print(f"  Sector leaders: {', '.join(sector_ctx.top_sectors)}")
                if sector_ctx.mean_reversion_setups:
                    print(f"  Mean-reversion setups: {', '.join(sector_ctx.mean_reversion_setups)}")
        except Exception as _sa_exc:
            logger.warning("SectorAnalyzer failed: %s", _sa_exc)

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

    # Discovery scan: sort tickers in top-3 RS sectors to the front of the queue
    if mode == "Discovery scan" and sector_ctx and sector_ctx.top_sectors:
        from .data_fetching import SECTOR_MAP as _SM
        _top_set = set(sector_ctx.top_sectors)
        tickers = sorted(tickers, key=lambda s: 0 if _SM.get(s.upper()) in _top_set else 1)

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

    results_buffer: Dict[str, Any] = {}

    # Phase 1 — Pre-fetch all chains in parallel with a progress bar
    def _fetch_one(sym: str):
        try:
            return sym, fetch_options_yfinance(sym, max_expiries)
        except Exception as exc:
            return sym, {"error": str(exc)}

    raw_results: Dict[str, Any] = {}
    with _suppress_scan_noise():
        with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
            _future_map = {executor.submit(_fetch_one, sym): sym for sym in tickers}
            if HAS_ENHANCED_CLI and verbose:
                bar_fmt = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                _pbar = tqdm(
                    total=len(tickers), desc="  Fetching", unit="",
                    leave=False, dynamic_ncols=False, bar_format=bar_fmt, file=sys.stdout,
                )
            else:
                _pbar = None
            for _fut in as_completed(_future_map):
                _sym, _res = _fut.result()
                raw_results[_sym] = _res
                if _pbar is not None:
                    _pbar.update(1)
            if _pbar is not None:
                _pbar.close()
                print(flush=True)  # clean line after bar clears

    # Phase 2 — Score each fetched result
    for symbol in tickers:
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
            # Cross-sectional normalization across the FULL combined batch.
            # Must run here (not per-ticker) so scores reflect quality relative
            # to every contract scanned, not just the ~5-15 from one ticker's chain.
            # Risk-flag caps applied per-ticker earlier are preserved as the raw input.
            picks = _cross_section_normalize(picks)
    
    if all_credit_spreads:
        credit_spreads_df = pd.concat(all_credit_spreads, ignore_index=True)
    
    if all_iron_condors:
        iron_condors_df = pd.concat(all_iron_condors, ignore_index=True)

    # Inject credit spreads into picks pool for unified AI ranking
    if not credit_spreads_df.empty and mode not in ("Credit Spreads", "Iron Condor"):
        try:
            spread_picks = normalize_spreads_for_ranking(credit_spreads_df, mode)
            if not spread_picks.empty:
                picks = pd.concat([picks, spread_picks], ignore_index=True)
        except Exception:
            pass

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
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, budget=budget, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact)
        elif verbose:
            print("\nNo options found within budget.")

    elif mode == "Discovery scan":
        if not picks.empty:
            final_df = picks.sort_values("quality_score", ascending=False)
            final_df = categorize_by_premium(final_df, budget=None)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact)
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
                print_report(final_df.head(10), underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact)
        elif verbose:
            print("\nNo premium selling candidates found.")

    else:
        # Single stock mode
        if not picks.empty:
            final_df = picks.copy()
            final_df = categorize_by_premium(final_df, budget=None)
            if verbose:
                print_report(final_df, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact)
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

    # Post-run warning summary
    n_scored = len(picks) if not picks.empty else 0
    if _SCAN_WARNINGS[0] > 0:
        _warn_msg = f"Scan complete: {n_scored} contracts scored, {_SCAN_WARNINGS[0]} warnings logged (see DEBUG log for details)"
    else:
        _warn_msg = f"Scan complete: {n_scored} contracts scored"
    if verbose:
        print(f"\n  {_warn_msg}")
    _SCAN_WARNINGS[0] = 0  # reset for next scan

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
            'tnx_change_pct': tnx_change_pct,
            'sector_ctx': sector_ctx,
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
        type_ = str(row.get('type', 'N/A')).upper()
        
        # Determine strike display (Single vs Spread vs Condor)
        if 'short_strike' in row and 'long_strike' in row:
            # Credit Spread
            strike_val = f"{row['short_strike']:.0f}/{row['long_strike']:.0f}"
        elif 'short_put_strike' in row:
            # Iron Condor
            strike_val = f"{row['short_put_strike']:.0f}/{row['short_call_strike']:.0f}"
        else:
            # Single option
            strike_val = f"{row.get('strike', 0.0):.1f}"
            
        exp = row.get('expiration', 'N/A')
        if isinstance(exp, str):
            exp = exp.split("T")[0]
        
        # Determine premium display
        premium = row.get('premium') or row.get('net_credit') or row.get('total_credit') or 0.0
        quality = row.get('quality_score', 0.0)
        
        print(f"  [{i+1}] {symbol:<5} {type_:<12} {strike_val:>12} {exp} | Prem: ${premium:>6.2f} | Qual: {quality:.2f}")

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
    date_str = now_et.strftime("%Y-%m-%d")

    # US market holidays (NYSE/CBOE) — update annually
    _US_MARKET_HOLIDAYS_2026 = {
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day (observed)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    }
    if date_str in _US_MARKET_HOLIDAYS_2026:
        return False, f"Markets closed — US market holiday ({time_str}). Data is stale."

    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets closed — it's {day_name} ({time_str}). Data is stale."

    if hhmm < 930:
        return False, f"Pre-market ({time_str}). Options open at 09:30 ET — bid/ask spreads not reliable yet."

    if hhmm >= 1600:
        return False, f"After-hours ({time_str}). Options closed at 16:00 ET — quotes are stale."

    return True, f"Market open ({time_str})"


def _run_ai_pipeline(picks: "pd.DataFrame", volatility_regime: str, verbose: bool = True, sector_ctx=None) -> "Optional[pd.DataFrame]":
    """Thin wrapper: delegates to ai_rank pipeline so CLI and ai_rank.py share one code path."""
    try:
        from ai_rank import score_and_rank
        from src.ranking import print_ranked_table
        from src.config_ai import AI_CONFIG
        from src.data_fetching import _CHAIN_CACHE
    except Exception as exc:
        msg = f"AI scoring unavailable — import failed: {type(exc).__name__}: {exc}"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        import traceback as _tb
        _tb.print_exc()
        return None

    # Ensure at least one OpenRouter / Anthropic key is set before trying any network call
    _key_env_vars = [
        "OPENROUTER_API_KEY", "OPENROUTER_ARCEE_KEY",
        "OPENROUTER_STEPFUN_KEY", "OPENROUTER_NVIDIA_KEY",
        "OPENROUTER_HUNTER_KEY", "ANTHROPIC_API_KEY",
    ]
    if not any(os.environ.get(k) for k in _key_env_vars):
        msg = "AI scoring skipped — no API key set. Add OPENROUTER_API_KEY (or similar) to .env"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        return None

    try:
        vix_map = {"Low": "low", "Normal": "normal", "High": "high"}
        vix_regime = vix_map.get(str(volatility_regime), "normal")

        candidates = picks.sort_values("quality_score", ascending=False).head(20).copy()

        ticker_contexts = {}
        if AI_CONFIG.get("two_pass_enabled", True):
            for sym in candidates["symbol"].unique():
                if sym in _CHAIN_CACHE:
                    ticker_contexts[sym] = _CHAIN_CACHE[sym].get("context", {})

        if verbose:
            _tp = "two-pass " if ticker_contexts else ""
            print(f"\n  Running AI {_tp}analysis on top {len(candidates)} picks...")

        ranked = score_and_rank(candidates, ticker_contexts, vix_regime, sector_ctx=sector_ctx)
        if verbose:
            print_ranked_table(ranked, top_n=10)
        return ranked
    except Exception as exc:
        msg = f"AI scoring failed: {type(exc).__name__}: {exc}"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        import traceback as _tb
        _tb.print_exc()
        return None


def run_top_scan(
    tickers: List[str],
    top_n: int = 10,
    mode: str = "Discovery scan",
    export_csv: bool = False,
    min_dte: int = 7,
    max_dte: int = 45,
    max_expiries: int = 4,
) -> Optional[pd.DataFrame]:
    """Fetch and score contracts across all tickers, return top_n sorted by quality_score.

    Groups results into DTE buckets: Short (7-14), Standard (15-30), Swing (31-45).
    Prints a ranked table and optionally saves a CSV.
    """
    from .cli_display import print_top_n_table

    _logger = setup_logging()
    config = load_config("config.json")
    rfr = get_risk_free_rate()
    vix_level = get_vix_level()
    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()

    all_rows = []
    for sym in tickers:
        try:
            data = fetch_options_yfinance(sym, max_expiries)
            result = _score_fetched_data(
                sym, data, mode, min_dte, max_dte,
                rfr, config, vix_weights, "swing",
                None, macro_risk_active, tnx_change_pct,
            )
            if result.get("success"):
                for picks_df in result.get("picks", []):
                    if not picks_df.empty:
                        all_rows.append(picks_df)
        except Exception:
            continue

    if not all_rows:
        print("No results from top scan.")
        return None

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("quality_score", ascending=False).reset_index(drop=True)
    top = combined.head(top_n).copy()

    print_top_n_table(top, top_n)

    if export_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"scan_results_{ts}.csv"
        export_cols = [
            "symbol", "type", "strike", "expiration", "T_years",
            "bid", "ask", "premium", "delta", "impliedVolatility",
            "iv_rank_30", "prob_profit", "ev_per_contract",
            "quality_score", "score_drivers",
        ]
        export_df = top[[c for c in export_cols if c in top.columns]].copy()
        if "T_years" in export_df.columns:
            export_df["DTE"] = (export_df["T_years"] * 365.0).round(0).astype(int)
            export_df.drop(columns=["T_years"], inplace=True, errors="ignore")
        export_df.to_csv(fname, index=False)
        print(f"\nExported {len(export_df)} rows to {fname}")

    return top


def main():
    # ── CLI argument parsing (Phase 7) ───────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--help", "-h", action="store_true", help="Show help and exit")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--close-trades", action="store_true", help="Update trade log with closing P/L")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI analysis after scan")
    parser.add_argument("--top", type=int, default=None, metavar="N", help="Run top-N cross-ticker scan and exit")
    parser.add_argument("--export", type=str, default=None, metavar="FORMAT", help="Export results to file (csv)")
    parser.add_argument("--watchlist", type=str, default=None, metavar="NAME", help="Use named watchlist from config as ticker input")
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching (requests, AI scores, IV history)")
    parser.add_argument("--surface", action="store_true", help="Show 3D P&L risk surface for top pick")
    parser.add_argument("--surface-mode", choices=["ascii", "braille"], default="braille", help="Surface render mode (default: braille)")
    parser.add_argument("--surface-greek", choices=["delta", "gamma", "vega", "theta"], default=None, help="Show greek sensitivity surface (implies --surface)")
    parser.add_argument("--no-contours", action="store_true", help="Disable contour lines on surface")
    parser.add_argument("--viz", action="store_true", help="Launch interactive 3D visualizer in browser after scan")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompts, use config defaults")
    parser.add_argument("--compact", action="store_true", help="Compact per-pick output (3 lines per pick)")
    parser.add_argument("--mode", type=str, default=None, choices=["ticker", "all", "discover", "sell", "spreads", "iron", "portfolio", "mylist"], help="Scan mode (skip mode menu)")
    parser.add_argument("--ticker", type=str, default=None, metavar="SYM", help="Ticker symbol (implies --mode ticker)")
    args, _ = parser.parse_known_args()

    if args.no_cache:
        try:
            import requests_cache as _rc
            _rc.uninstall_cache()
        except Exception:
            pass
        # Disable IV history DB reads (bypass_cache flag on data_fetching)
        try:
            from . import data_fetching as _df_mod
            _df_mod._NO_CACHE = True
        except Exception:
            pass
        logging.getLogger(__name__).info("All caching disabled via --no-cache")

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
                ("--top N",        "Cross-ticker top-N scan (default 10), grouped by DTE bucket"),
                ("--export csv",   "Export top scan results to scan_results_YYYYMMDD_HHMM.csv"),
                ("--watchlist N",  "Use named watchlist from config (liquid_large_cap, sector_etfs, high_iv, income)"),
                ("--no-cache",    "Disable all caching (requests, AI scores, IV history)"),
                ("--surface",      "Show 3D P&L risk surface for top pick (braille hi-res by default)"),
                ("--surface-mode", "Surface render: ascii or braille (default: braille)"),
                ("--surface-greek","Show greek surface: delta, gamma, vega, theta"),
                ("--no-contours",  "Disable contour lines on surface"),
                ("--viz",          "Open interactive 3D visualizer (Plotly, opens in browser)"),
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

    # ── Config validation ─────────────────────────────────────────────────────
    _cw = config.get("composite_weights", {})
    _cw_sum = sum(_cw.values()) if _cw else 0
    if _cw and (_cw_sum > 1.1 or _cw_sum < 0.9):
        _warn = f"composite_weights sum to {_cw_sum:.2f} (expected ~1.0)"
        print(fmt.colorize(f"  \u26a0 Config: {_warn}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0 Config: {_warn}")

    # ── Calibration banner ────────────────────────────────────────────────────
    try:
        from .backtester import get_calibration_status
        _cal_n, _cal_status = get_calibration_status()
        _cal_line = f"  Paper-trade calibration: {_cal_n} closed — {_cal_status}"
        if HAS_ENHANCED_CLI:
            _color = fmt.Colors.BRIGHT_GREEN if "available" in _cal_status else fmt.Colors.DIM
            print(fmt.colorize(_cal_line, _color))
        else:
            print(_cal_line)
    except Exception:
        pass

    if not getattr(args, 'no_ai', False):
        # Ensure .env is loaded before checking for keys
        _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.isfile(_env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=_env_path, override=False)
            except ImportError:
                # Manual fallback: parse KEY=VALUE lines
                with open(_env_path) as _ef:
                    for _line in _ef:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _, _v = _line.partition("=")
                            _k, _v = _k.strip(), _v.strip().strip("'\"")
                            if _k and _v:
                                os.environ.setdefault(_k, _v)
        _has_any_key = any(
            os.environ.get(k) for k in [
                "OPENROUTER_API_KEY", "OPENROUTER_ARCEE_KEY",
                "OPENROUTER_STEPFUN_KEY", "OPENROUTER_NVIDIA_KEY",
                "ANTHROPIC_API_KEY",
            ]
        )
        if not _has_any_key:
            _warn = "No AI API keys found in environment. AI scoring will be skipped. Set keys in .env file."
            print(fmt.colorize(f"  \u26a0 {_warn}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0 {_warn}")

    # ── --watchlist: resolve named watchlist tickers from config ─────────────
    _watchlist_tickers = None
    if args.watchlist:
        _wl_name = args.watchlist.lower().replace("-", "_")
        _wls = config.get("watchlists", {})
        if _wl_name in _wls:
            _watchlist_tickers = _wls[_wl_name]
            print(f"Using watchlist '{_wl_name}': {len(_watchlist_tickers)} tickers")
        else:
            _available = list(_wls.keys())
            print(f"Unknown watchlist '{_wl_name}'. Available: {', '.join(_available)}")
            sys.exit(1)

    # ── --top N: run cross-ticker top-N scan and exit ─────────────────────────
    if args.top is not None:
        _top_n = max(1, args.top)
        _top_tickers = _watchlist_tickers or config.get("watchlists", {}).get("liquid_large_cap", [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
            "JPM", "BAC", "GS", "V", "MA", "AMD", "XOM", "CVX",
        ])
        _do_export = (args.export or "").lower() == "csv"
        print(f"\nRunning top-{_top_n} scan across {len(_top_tickers)} tickers...")
        run_top_scan(_top_tickers, top_n=_top_n, export_csv=_do_export)
        sys.exit(0)

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

    # ── Config Validation ──────────────────────────────────────────────────
    try:
        from .config_validator import validate_core_config
        _cfg_warnings = validate_core_config(config)
        if _cfg_warnings:
            for _cw in _cfg_warnings:
                print(fmt.format_warning(f"Config: {_cw}") if HAS_ENHANCED_CLI else f"  WARNING: Config: {_cw}")
    except Exception:
        pass

    # ── Regime Dashboard ─────────────────────────────────────────────────────
    try:
        from .regime_dashboard import print_regime_dashboard
        import threading as _t
        import io as _io
        import sys as _sys
        print("  Loading market data...", end="", flush=True)
        _result = [None]
        def _fetch():
            buf = _io.StringIO()
            _old = _sys.stdout
            _sys.stdout = buf
            try:
                print_regime_dashboard(WIDTH)
            finally:
                _sys.stdout = _old
            _result[0] = buf.getvalue()
        _th = _t.Thread(target=_fetch, daemon=True)
        _th.start()
        _th.join(timeout=8)  # hard 8-second cap on regime fetch
        print("\r" + " " * 30 + "\r", end="")
        if _result[0]:
            print(_result[0], end="")
    except Exception:
        print()  # newline after "Loading..." if dashboard fails

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
    try:
        import threading
        _update_done = threading.Event()
        _update_err = [None]
        def _bg_update():
            try:
                pm.update_positions()
            except Exception as e:
                _update_err[0] = e
            finally:
                _update_done.set()
        _t = threading.Thread(target=_bg_update, daemon=True)
        _t.start()
        if not _update_done.wait(timeout=15):
            print(fmt.format_warning("Portfolio update timed out — skipping live price refresh") if HAS_ENHANCED_CLI else "⚠ Portfolio update timed out")
        elif _update_err[0]:
            pass  # silently ignore update errors
    except Exception:
        pass

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

    # ── --mode / --ticker CLI bypass ──────────────────────────────────────────
    _mode_map_cli = {
        "ticker": "TICKER", "all": "ALL", "discover": "DISCOVER",
        "sell": "SELL", "spreads": "SPREADS", "iron": "IRON",
        "portfolio": "PORTFOLIO", "mylist": "MY LIST",
    }
    if args.ticker:
        symbol_input = args.ticker.upper()
    elif args.mode:
        symbol_input = _mode_map_cli.get(args.mode.lower(), "DISCOVER")
    else:
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

    if is_my_list_mode:
        mode = "Discovery scan"
    elif is_discovery_mode:
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

    if _watchlist_tickers and not is_my_list_mode:
        tickers = _watchlist_tickers
        print(f"  Using --watchlist tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_my_list_mode:
        tickers = _wl_tickers
        print(f"  Scanning your watchlist: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_discovery_mode or is_premium_selling_mode or is_credit_spread_mode or is_iron_condor_mode:
        tickers = prompt_for_tickers()
        print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    elif is_budget_mode:
        try:
            budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
            if budget <= 0:
                print("Budget must be greater than 0.")
                sys.exit(1)
        except Exception:
            print("Invalid budget amount.")
            sys.exit(1)
        scan_type = prompt_input("Enter 1 for TARGETED or 2 for DISCOVERY", "1")
        if scan_type == "2":
            tickers = prompt_for_tickers()
        else:
            default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
            tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    elif is_ticker_mode:
        ticker_sym = prompt_input("Enter stock ticker symbol", "AAPL").upper()
        if not ticker_sym.isalnum():
            print("Please enter a valid alphanumeric ticker.")
            sys.exit(1)
        tickers = [ticker_sym]
    else:
        if not symbol_input.isalnum():
            print("Please enter a valid alphanumeric ticker.")
            sys.exit(1)
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

    f_config = config.get("filters", {})
    if is_iron_condor_mode:
        default_min_dte = str(f_config.get("min_days_to_expiration_iron", 30))
        default_max_dte = str(f_config.get("max_days_to_expiration_iron", 60))
    else:
        default_min_dte = str(f_config.get("min_days_to_expiration", 7))
        default_max_dte = str(f_config.get("max_days_to_expiration", 45))

    if args.auto:
        max_expiries = config.get("max_expirations", 4)
        min_dte = int(default_min_dte)
        max_dte = int(default_max_dte)
        trader_profile = "swing"
    else:
        try:
            max_expiries = int(prompt_input("How many nearest expirations to scan", "4"))
        except Exception:
            print("Invalid number for expirations.")
            sys.exit(1)
        try:
            min_dte = int(prompt_input("Minimum days to expiration (DTE)", default_min_dte))
            max_dte = int(prompt_input("Maximum days to expiration (DTE)", default_max_dte))
        except Exception:
            print("Invalid DTE inputs.")
            sys.exit(1)
        profile_choice = prompt_input("Enter 1 for Swing or 2 for Day trader", "1").strip()
        trader_profile = "day" if profile_choice == "2" else "swing"

    # Account size for position sizing in order tickets
    if not args.auto:
        _acct_input = prompt_input("Account size in USD for position sizing (Enter to skip)", "").strip()
        if _acct_input:
            try:
                config["_account_size"] = float(_acct_input)
            except ValueError:
                pass

    _is_single_stock = (mode == "Single-stock")
    _repeat_count = 0

    try:
        while True:
            show_surface = getattr(args, 'surface', False) or getattr(args, 'surface_greek', None) is not None
            surface_mode = getattr(args, 'surface_mode', 'braille')
            surface_greek = getattr(args, 'surface_greek', None)
            surface_type = surface_greek if surface_greek else 'pnl'
            show_contours = not getattr(args, 'no_contours', False)
            scan_results = run_scan(mode=mode, tickers=tickers, budget=budget, max_expiries=max_expiries, min_dte=min_dte, max_dte=max_dte, trader_profile=trader_profile, logger=logger, market_trend=market_trend, volatility_regime=volatility_regime, macro_risk_active=macro_risk_active, tnx_change_pct=tnx_change_pct, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=getattr(args, 'compact', False))
            if scan_results is None:
                sys.exit(0)

            picks = scan_results.picks

            # ── AI Analysis ────────────────────────────────────────────────
            _ai_ranked = None
            if not picks.empty and not getattr(args, 'no_ai', False):
                _ai_ranked = _run_ai_pipeline(picks, volatility_regime, verbose=True,
                                               sector_ctx=scan_results.market_context.get("sector_ctx"))

            # Pull spread/condor results for the save menu
            _credit_spreads = scan_results.credit_spreads
            _iron_condors   = scan_results.iron_condors
            _has_results = (
                not picks.empty
                or (isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty)
                or (isinstance(_iron_condors,   pd.DataFrame) and not _iron_condors.empty)
            )

            # ── Auto-export if --export csv was passed ──────────────────────
            if getattr(args, "export", None) and str(args.export).lower() == "csv" and not picks.empty:
                _ts = datetime.now().strftime("%Y%m%d_%H%M")
                _auto_fname = f"scan_results_{_ts}.csv"
                _export_cols = [
                    "symbol", "type", "strike", "expiration",
                    "bid", "ask", "premium", "delta", "impliedVolatility",
                    "iv_rank_30", "prob_profit", "ev_per_contract",
                    "quality_score", "score_drivers",
                ]
                _auto_df = picks[[c for c in _export_cols if c in picks.columns]].copy()
                _auto_df.to_csv(_auto_fname, index=False)
                _msg = f"Auto-exported {len(_auto_df)} rows to {_auto_fname}"
                print(fmt.format_success(_msg) if HAS_ENHANCED_CLI else f"  \u2713 {_msg}")

            # ── Auto-launch 3D visualizer if --viz was passed ─────────────
            if getattr(args, 'viz', False) and _has_results and not picks.empty:
                try:
                    from .visualizer_3d import OptionsVisualizer
                    _viz_cfg = load_config("config.json")
                    _viz = OptionsVisualizer(scan_results, config=_viz_cfg)
                    _viz.show()
                    msg = "3D Visualizer opened in browser"
                    print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                except ImportError:
                    print(fmt.format_warning("Plotly required for --viz: pip install plotly") if HAS_ENHANCED_CLI else "  plotly required for --viz")
                except Exception as _viz_exc:
                    logger.warning("Visualizer failed: %s", _viz_exc)

            # ── Collapsed post-scan prompt (loops so V → P → L all work in one sitting) ──
            if _has_results:
              while True:
                if HAS_ENHANCED_CLI:
                    save_label = fmt.colorize("Save/Export?", fmt.Colors.BRIGHT_CYAN)
                    p_opt = fmt.colorize("[P]", fmt.Colors.BRIGHT_YELLOW) + " Paper trade top pick"
                    c_opt = fmt.colorize("[C]", fmt.Colors.BRIGHT_YELLOW) + " CSV"
                    l_opt = fmt.colorize("[L]", fmt.Colors.BRIGHT_YELLOW) + " Log trades"
                    v_opt = fmt.colorize("[V]", fmt.Colors.BRIGHT_YELLOW) + " 3D Visualizer"
                    skip_opt = fmt.colorize("[Enter]", fmt.Colors.DIM) + " Skip"
                    print(f"\n  {save_label}  {p_opt}  \u00b7  {c_opt}  \u00b7  {l_opt}  \u00b7  {v_opt}  \u00b7  {skip_opt}")
                else:
                    print("\n  Save/Export?  [P] Paper trade top pick  [C] CSV  [L] Log trades  [V] 3D Visualizer  [Enter] Skip")
                save_choice = prompt_input("Choice", "").strip().upper()

                if save_choice == "":
                    break  # Enter → done with save menu

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
                            "strategy_name": f"Short {str(top_pick_row['type']).capitalize()}",
                            "entry_iv": top_pick_row.get("impliedVolatility"),
                            "entry_delta": top_pick_row.get("delta"),
                            "entry_gamma": top_pick_row.get("gamma"),
                            "entry_vega": top_pick_row.get("vega"),
                            "entry_theta": top_pick_row.get("theta"),
                            "dividend_yield": top_pick_row.get("dividend_yield"),
                            # Component scores — enable backtester IC analysis once 30+ trades close
                            "pop_score": top_pick_row.get("pop_score"),
                            "ev_score": top_pick_row.get("ev_score"),
                            "rr_score": top_pick_row.get("rr_score"),
                            "liquidity_score": top_pick_row.get("liquidity_score"),
                            "momentum_score": top_pick_row.get("momentum_score"),
                            "iv_rank_score": top_pick_row.get("iv_rank_score"),
                            "theta_score": top_pick_row.get("theta_score"),
                            "iv_edge_score": top_pick_row.get("iv_advantage_score"),
                            "vrp_score": top_pick_row.get("vrp_score"),
                            "iv_mispricing_score": top_pick_row.get("iv_mispricing_score"),
                            "skew_align_score": top_pick_row.get("skew_align_score"),
                            "vega_risk_score": top_pick_row.get("vega_risk_score"),
                            "term_structure_score": top_pick_row.get("term_structure_score"),
                            # v7: remaining 14 components — full IC coverage
                            "catalyst_score": top_pick_row.get("catalyst_score"),
                            "em_realism_score": top_pick_row.get("em_realism_score"),
                            "gamma_theta_score": top_pick_row.get("gamma_theta_score"),
                            "gex_score": top_pick_row.get("gex_score"),
                            "gamma_magnitude_score": top_pick_row.get("gamma_magnitude_score"),
                            "gamma_pin_score": top_pick_row.get("gamma_pin_score"),
                            "iv_velocity_score": top_pick_row.get("iv_velocity_score"),
                            "max_pain_score": top_pick_row.get("max_pain_score"),
                            "oi_change_score": top_pick_row.get("oi_change_score"),
                            "option_rvol_score": top_pick_row.get("option_rvol_score"),
                            "pcr_score": top_pick_row.get("pcr_score"),
                            "sentiment_score_norm": top_pick_row.get("sentiment_score_norm"),
                            "spread_score": top_pick_row.get("spread_score"),
                            "trader_pref_score": top_pick_row.get("trader_pref_score"),
                        }
                        pm.log_trade(trade_dict)
                        msg = f"Paper trade logged: {top_pick_row['symbol']} {str(top_pick_row['type']).upper()} ${top_pick_row['strike']:.0f}"
                        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                        # Offer inline portfolio view
                        _view = prompt_input("View portfolio? (y/n)", "n").strip().lower()
                        if _view in ("y", "yes"):
                            try:
                                from .check_pnl import view_portfolio
                                view_portfolio()
                            except Exception as _pnl_exc:
                                print(f"  Could not load portfolio: {_pnl_exc}")

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

                elif save_choice == "V":
                    try:
                        from .visualizer_3d import OptionsVisualizer
                        _viz_cfg = load_config("config.json")
                        _viz = OptionsVisualizer(scan_results, config=_viz_cfg)
                        _viz.show()
                        msg = "3D Visualizer opened in browser"
                        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                    except ImportError:
                        msg = "Plotly required for visualizer: pip install plotly"
                        print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  {msg}")
                    except Exception as _viz_exc:
                        msg = f"Visualizer error: {_viz_exc}"
                        print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  {msg}")

                elif save_choice == "L":
                    log_src = picks if not picks.empty else (
                        _credit_spreads if isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty
                        else _iron_condors
                    )
                    if isinstance(log_src, pd.DataFrame) and not log_src.empty:
                        picks_to_log = select_trades_to_log(log_src)
                        if not picks_to_log.empty:
                            log_trade_entry(picks_to_log, mode)
                            
                            # Also log to PaperManager for portfolio visibility
                            today_str = datetime.now().strftime("%Y-%m-%d")
                            for _, row in picks_to_log.iterrows():
                                try:
                                    if "short_strike" in row or "net_credit" in row:
                                        # It's a Credit Spread
                                        pm.log_spread({
                                            "date": today_str,
                                            "ticker": row["symbol"],
                                            "expiration": row["expiration"],
                                            "short_strike": row["short_strike"],
                                            "long_strike": row["long_strike"],
                                            "type": row["type"],
                                            "net_credit": row["net_credit"],
                                            "max_profit": row.get("max_profit", 0),
                                            "max_loss": row.get("max_loss", 0),
                                            "quality_score": row.get("quality_score", 0.5)
                                        })
                                    elif "total_credit" in row:
                                        # It's an Iron Condor
                                        pm.log_spread({
                                            "date": today_str,
                                            "ticker": row["symbol"],
                                            "expiration": row["expiration"],
                                            "short_strike": row["short_put_strike"],
                                            "long_strike": row["long_put_strike"],
                                            "type": "Iron Condor",
                                            "net_credit": row["total_credit"],
                                            "max_profit": row.get("max_profit", 0), # total_credit * 100 handled inside log_spread? No.
                                            "max_loss": row.get("max_risk", 0),
                                            "quality_score": row.get("quality_score", 0.5)
                                        })
                                    else:
                                        # It's a single option
                                        trade_dict = {
                                            "date": today_str,
                                            "ticker": row["symbol"],
                                            "expiration": row["expiration"],
                                            "strike": row["strike"],
                                            "type": str(row["type"]).capitalize(),
                                            "entry_price": (
                                                safe_float(row.get("ask") or None)
                                                or safe_float(row.get("lastPrice"))
                                                or safe_float(row.get("premium"), 0.0)
                                            ),
                                            "quality_score": row.get("quality_score", 0.5),
                                            "strategy_name": f"Short {str(row['type']).capitalize()}",
                                            "entry_iv": row.get("impliedVolatility"),
                                            "entry_delta": row.get("delta"),
                                            "entry_gamma": row.get("gamma"),
                                            "entry_vega": row.get("vega"),
                                            "entry_theta": row.get("theta"),
                                            "pop_score": row.get("pop_score"),
                                            "ev_score": row.get("ev_score"),
                                            "rr_score": row.get("rr_score"),
                                            "liquidity_score": row.get("liquidity_score"),
                                            "momentum_score": row.get("momentum_score"),
                                            "iv_rank_score": row.get("iv_rank_score"),
                                            "theta_score": row.get("theta_score"),
                                            "iv_edge_score": row.get("iv_advantage_score"),
                                            "vrp_score": row.get("vrp_score"),
                                            "iv_mispricing_score": row.get("iv_mispricing_score"),
                                            "skew_align_score": row.get("skew_align_score"),
                                            "vega_risk_score": row.get("vega_risk_score"),
                                            "term_structure_score": row.get("term_structure_score"),
                                            "catalyst_score": row.get("catalyst_score"),
                                            "em_realism_score": row.get("em_realism_score"),
                                            "gamma_theta_score": row.get("gamma_theta_score"),
                                            "gex_score": row.get("gex_score"),
                                            "gamma_magnitude_score": row.get("gamma_magnitude_score"),
                                            "gamma_pin_score": row.get("gamma_pin_score"),
                                            "iv_velocity_score": row.get("iv_velocity_score"),
                                            "max_pain_score": row.get("max_pain_score"),
                                            "oi_change_score": row.get("oi_change_score"),
                                            "option_rvol_score": row.get("option_rvol_score"),
                                            "pcr_score": row.get("pcr_score"),
                                            "sentiment_score_norm": row.get("sentiment_score_norm"),
                                            "spread_score": row.get("spread_score"),
                                            "trader_pref_score": row.get("trader_pref_score"),
                                        }
                                        pm.log_trade(trade_dict)
                                except Exception as _log_exc:
                                    print(f"  Error logging to DB: {_log_exc}")

                            msg = f"Logged {len(picks_to_log)} trades."
                            print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2705 {msg}")

                else:
                    _msg = "Unknown choice — press P / C / L / V or Enter to skip"
                    print(fmt.format_warning(_msg) if HAS_ENHANCED_CLI else f"  {_msg}")
                # Loop back and re-prompt so V → P → L all work in one sitting

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

    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
