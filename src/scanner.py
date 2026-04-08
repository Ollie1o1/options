#!/usr/bin/env python3
import sys
import os
import json
import logging
import uuid
import time
import threading as _threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import contextlib

import pandas as pd
import numpy as np
import yfinance as yf

from .types import ScanResult
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
from .oi_snapshot import load_oi_snapshot
from .cli_display import (
    get_display_width, print_executive_summary,
    print_report, print_news_panel,
    print_credit_spreads_report, print_iron_condor_report,
)

# Optional imports
try:
    from .simulation import monte_carlo_pop, batch_monte_carlo_pop
    HAS_SIMULATION = True
except ImportError:
    HAS_SIMULATION = False

try:
    from .vol_analytics import compute_vol_cone, compute_iv_surface, classify_vol_regime
    HAS_VOL_ANALYTICS = True
except ImportError:
    HAS_VOL_ANALYTICS = False

# Enhanced CLI modules
try:
    from . import formatting as fmt
    HAS_ENHANCED_CLI = True
except ImportError:
    HAS_ENHANCED_CLI = False

_SCAN_WARNINGS = [0]

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
    _root = logging.getLogger()
    _saved_root = _root.level
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
        "composite_weights": {
            "pop": 0.18, "em_realism": 0.12, "rr": 0.15, "momentum": 0.10,
            "iv_rank": 0.10, "liquidity": 0.15, "catalyst": 0.05,
            "theta": 0.10, "ev": 0.05, "trader_pref": 0.10
        },
        "moneyness_band": 0.15,
        "target_delta": 0.40,
        "earnings_buffer_days": 5,
        "monte_carlo_simulations": 10000,
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25}
    }
    try:
        if not os.path.exists(config_path): return default_config
        with open(config_path, 'r') as f:
            config = json.load(f)
            for key in default_config:
                if key not in config: config[key] = default_config[key]
            return config
    except Exception: return default_config

_IC_WEIGHTS_CACHE: dict | None = None
_IC_RECALIB_RUNNING: bool = False
_IC_RECALIB_LOCK = _threading.Lock()

def _maybe_trigger_recalib(cache_path: str) -> None:
    global _IC_RECALIB_RUNNING
    with _IC_RECALIB_LOCK:
        if _IC_RECALIB_RUNNING: return
    try:
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < (7 * 86400): return
    except OSError: pass

    def _run():
        global _IC_RECALIB_RUNNING, _IC_WEIGHTS_CACHE
        try:
            import sqlite3 as _sqlite3
            db_path = "paper_trades.db"
            if not os.path.exists(db_path): return
            with _sqlite3.connect(db_path) as _conn:
                n = _conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL").fetchone()[0]
            if n < 30: return
            from .backtester import run_paper_trade_ic
            ic_data = run_paper_trade_ic()
            if ic_data.get("n_trades", 0) >= 30:
                with open(cache_path, "w") as _f: json.dump(ic_data, _f, indent=2)
                with _IC_RECALIB_LOCK: _IC_WEIGHTS_CACHE = None
        except Exception: pass
        finally:
            with _IC_RECALIB_LOCK: _IC_RECALIB_RUNNING = False

    with _IC_RECALIB_LOCK: _IC_RECALIB_RUNNING = True
    _threading.Thread(target=_run, daemon=True).start()

def load_ic_adjusted_weights(config: Dict, cache_path: str = "ic_weights_cache.json") -> Dict:
    global _IC_WEIGHTS_CACHE
    with _IC_RECALIB_LOCK:
        if _IC_WEIGHTS_CACHE is not None: return _IC_WEIGHTS_CACHE
    _maybe_trigger_recalib(cache_path)
    base_weights = config.get("composite_weights", {}) or {}
    try:
        if not os.path.exists(cache_path):
            with _IC_RECALIB_LOCK: _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        with open(cache_path, "r") as f: cache = json.load(f)
        component_ic = cache.get("component_ic", {})
        key_map = {"pop_score": "pop", "ev_score": "ev", "rr_score": "rr", "liquidity_score": "liquidity", "momentum_score": "momentum", "iv_rank_score": "iv_rank", "theta_score": "theta"}
        ic_vals = {key_map[k]: max(0.0, float(v)) for k, v in component_ic.items() if k in key_map and cache.get("component_pvalues", {}).get(k, 1.0) < 0.10}
        if not ic_vals:
            with _IC_RECALIB_LOCK: _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        ic_total = sum(ic_vals.values()) or 1.0
        blended = {k: 0.7*float(v) + 0.3*(ic_vals.get(k, 0)/ic_total) if k in ic_vals else v for k, v in base_weights.items()}
        with _IC_RECALIB_LOCK: _IC_WEIGHTS_CACHE = blended
        return _IC_WEIGHTS_CACHE
    except Exception:
        with _IC_RECALIB_LOCK: _IC_WEIGHTS_CACHE = base_weights
        return _IC_WEIGHTS_CACHE

def _invalidate_ic_weights_cache() -> None:
    global _IC_WEIGHTS_CACHE
    _IC_WEIGHTS_CACHE = None

def setup_logging() -> logging.Logger:
    """Configures and returns the scanner logger."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("options_screener")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f"logs/scan_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger

def calculate_probability_of_profit(option_type, S, K, T, sigma, premium, r=0.0, q=0.0):
    try:
        S, K, T, sigma, premium = map(np.asanyarray, [S, K, T, sigma, premium])
        T = np.maximum(T, 1.0 / (365.0 * 24.0))
        is_call = (option_type.lower() == "call") if isinstance(option_type, str) else (np.char.lower(np.asanyarray(option_type).astype(str)) == "call")
        breakeven = np.where(is_call, K + premium, K - premium)
        F = S * np.exp((r - q) * T)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (np.log(F / breakeven) - (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pop = np.where(is_call, norm_cdf(d), 1.0 - norm_cdf(d))
        return float(pop) if np.isscalar(S) else pop
    except Exception: return None

def calculate_expected_move(S, sigma, T):
    try:
        move = np.asanyarray(S) * np.asanyarray(sigma) * np.sqrt(np.asanyarray(T))
        return float(move) if move.ndim == 0 else move
    except Exception: return None

def calculate_probability_of_touch(option_type, S, K, T, sigma):
    try:
        S, K, T, sigma = map(np.asanyarray, [S, K, T, sigma])
        T = np.maximum(T, 1.0 / (365.0 * 24.0))
        is_call = (option_type.lower() == "call") if isinstance(option_type, str) else (np.char.lower(np.asanyarray(option_type).astype(str)) == "call")
        with np.errstate(divide='ignore', invalid='ignore'):
            d2 = (np.log(S / K) - (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pot = np.where(is_call, np.where(K > S, 2 * norm_cdf(d2), 1.0), np.where(K < S, 2 * (1.0 - norm_cdf(d2)), 1.0))
        return float(pot) if np.isscalar(S) else pot
    except Exception: return None

def calculate_risk_reward(option_type, premium, S, K, expected_move=None, em_factor=0.68):
    try:
        premium, S, K = map(np.asanyarray, [premium, S, K])
        is_call = (option_type.lower() == "call") if isinstance(option_type, str) else (np.char.lower(np.asanyarray(option_type).astype(str)) == "call")
        max_loss = premium * 100
        breakeven = np.where(is_call, K + premium, K - premium)
        target = np.where(is_call, S + em_factor * (expected_move or S*0.5), S - em_factor * (expected_move or S*0.5))
        payoff = np.where(is_call, np.maximum(0.0, target - K), np.maximum(0.0, K - target))
        rr = np.where(premium > 0, np.maximum(0.0, payoff - premium) / premium, 0.0)
        return (float(max_loss), float(breakeven), float(rr)) if premium.ndim == 0 else (max_loss, breakeven, rr)
    except Exception: return None, None, None

def calculate_metrics(df, risk_free_rate, earnings_date, config, iv_rank, iv_percentile, sentiment_score, macro_risk_active, sector_perf, tnx_change_pct, **kwargs):
    df["Vol_OI_Ratio"] = (df["volume"] / df["openInterest"].replace(0, np.nan))
    df["Unusual_Whale"] = (df["Vol_OI_Ratio"] > 1.5) & (df["volume"] > 500)
    df["Earnings Play"] = "NO"
    if earnings_date and earnings_date > datetime.now(timezone.utc):
        df.loc[(df["exp_dt"] > earnings_date), "Earnings Play"] = "YES"
    
    S, K, T, IV, types = df["underlying"].values, df["strike"].values, df["T_years"].values, np.maximum(1e-9, df["impliedVolatility"].values), df["type"].values
    q = float(kwargs.get("dividend_yield", 0))
    df["delta"] = bs_delta(types, S, K, T, risk_free_rate, IV, q)
    df["abs_delta"] = np.abs(df["delta"])
    df["gamma"] = bs_gamma(S, K, T, risk_free_rate, IV, q)
    df["vega"] = bs_vega(S, K, T, risk_free_rate, IV, q)
    df["theta"] = bs_theta(types, S, K, T, risk_free_rate, IV, q)
    
    df["expected_move"] = calculate_expected_move(S, IV, T)
    df["prob_profit"] = calculate_probability_of_profit(types, S, K, T, IV, df["premium"].values, r=risk_free_rate, q=q)
    df["rr_ratio"] = calculate_risk_reward(types, df["premium"].values, S, K, df["expected_move"].values)[2]
    
    from .risk_engine import run_risk_checks
    df = run_risk_checks(df, current_price=float(S[0]) if len(S) > 0 else 0.0, config=config)
    return df

def _cross_section_normalize(df):
    if len(df) <= 1: return df
    raw = df["quality_score"].copy()
    df["quality_score"] = (((raw - 0.28) / 0.54).clip(0, 1) ** 0.65).round(4)
    return df

def calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte, **kwargs):
    def rank_norm(s): return (s.rank() - 1) / (len(s) - 1) if len(s) > 1 else pd.Series([0.5]*len(s))
    liquidity = rank_norm(df["volume"] + df["openInterest"])
    pop_score = df["prob_profit"].fillna(0.5).clip(0, 1)
    rr_score = pd.to_numeric(df["rr_ratio"], errors='coerce').fillna(0).clip(0, 1)
    
    cw = load_ic_adjusted_weights(config)
    w = {k: cw.get(k, 0.1) for k in ["pop", "rr", "liquidity", "theta", "ev"]}
    w_sum = sum(w.values()) or 1.0
    df["quality_score"] = (w.get("pop",0)*pop_score + w.get("rr",0)*rr_score + w.get("liquidity",0)*liquidity) / w_sum
    return df

def enrich_and_score(df, min_dte, max_dte, risk_free_rate, config, vix_regime_weights, mode="Single-stock", **kwargs):
    if df is None or df.empty:
        return pd.DataFrame()
    df["exp_dt"] = pd.to_datetime(df["expiration"], utc=True)
    df["T_years"] = (df["exp_dt"] - datetime.now(timezone.utc)).dt.total_seconds() / (365*24*3600)
    df = df[(df["T_years"] >= min_dte/365.0) & (df["T_years"] <= max_dte/365.0)].copy()
    df["premium"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["premium"]
    earnings_date = kwargs.pop("earnings_date", None)
    df = calculate_metrics(df, risk_free_rate, earnings_date, config, **kwargs)
    df = calculate_scores(df, config, vix_regime_weights, kwargs.get("trader_profile", "swing"), mode, min_dte, max_dte)
    return df.sort_values("quality_score", ascending=False).reset_index(drop=True)

def _score_fetched_data(symbol, data_result, mode, min_dte, max_dte, rfr, config, vix_weights, trader_profile, **kwargs):
    res = {"symbol": symbol, "picks": [], "success": False, "error": None}
    try:
        df = data_result.get("df")
        if df is None or df.empty: return res
        df = enrich_and_score(df, min_dte, max_dte, rfr, config, vix_weights, mode=mode, trader_profile=trader_profile, **data_result.get("context", {}), **kwargs)
        if not df.empty:
            res["picks"].append(df)
            res["success"] = True
        return res
    except Exception as e:
        res["error"] = str(e)
        return res

def run_scan(mode, tickers, budget, max_expiries, min_dte, max_dte, trader_profile, logger, market_trend, volatility_regime, **kwargs):
    config = load_config()
    vix_level = get_vix_level()
    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    rfr = get_risk_free_rate()
    
    results = {}
    with _suppress_scan_noise():
        with ThreadPoolExecutor(max_workers=8) as executor:
            futs = {executor.submit(fetch_options_yfinance, s, max_expiries): s for s in tickers}
            for f in as_completed(futs):
                s = futs[f]
                results[s] = _score_fetched_data(s, f.result(), mode, min_dte, max_dte, rfr, config, vix_weights, trader_profile, **kwargs)
    
    all_p = [r["picks"][0] for r in results.values() if r["success"] and r["picks"]]
    picks = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()
    if not picks.empty: picks = _cross_section_normalize(picks)
    
    return ScanResult(picks=picks, spreads=pd.DataFrame(), credit_spreads=pd.DataFrame(), iron_condors=pd.DataFrame(), top_pick=picks.sort_values("quality_score", ascending=False).iloc[0] if not picks.empty else None, underlying_price=picks.iloc[0]["underlying"] if not picks.empty else 0.0, rfr=rfr, chain_iv_median=picks["impliedVolatility"].median() if not picks.empty else 0.0, timestamp=datetime.now().isoformat(), ticker_contexts={}, market_context={})
