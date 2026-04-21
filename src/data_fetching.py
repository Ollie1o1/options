#!/usr/bin/env python3
from __future__ import annotations
"""
Data fetching utilities for the options screener.
Handles all yfinance interactions with a Single-Fetch architecture for performance.
"""

import time
import math
import logging
import random
import functools
import threading
import warnings
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from contextlib import closing
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError

# Lazy import for slow modules
requests_cache = None

logger = logging.getLogger(__name__)

# Lazy imports for slow modules (yfinance, finvizfinance)
yf = None  # yfinance Ticker imported on first use
Performance = None  # finvizfinance imported on first use

try:
    from .news_fetcher import fetch_news_and_events
    _HAS_NEWS_FETCHER = True
except Exception:
    _HAS_NEWS_FETCHER = False

# Suppress noisy third-party library output at startup
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# Lazy cache initialization (deferred to first use to avoid startup slowdown)
_CACHE_INITIALIZED = False
_YF_IMPORTED = False
_FINVIZ_IMPORTED = False

def _init_request_cache():
    """Initialize request cache on first use (lazy-loaded to avoid startup slowdown).

    CRITICAL: allowable_codes=[200] prevents caching 429 rate-limit responses. Without this
    filter, once Yahoo 429s any request it stays cached for 15 min and every retry hits the
    cached error — poisoning the scan for an hour.
    """
    global _CACHE_INITIALIZED, requests_cache
    if _CACHE_INITIALIZED:
        return
    _CACHE_INITIALIZED = True
    try:
        import requests_cache as _rc
        requests_cache = _rc
        requests_cache.install_cache(
            'finance_cache', backend='sqlite', expire_after=900,
            allowable_codes=[200],
            backend_options={'pragmas': {'journal_mode': 'wal'}},
        )
    except Exception:
        try:
            if requests_cache is None:
                import requests_cache as _rc
                requests_cache = _rc
            # Older requests_cache versions don't support backend_options
            requests_cache.install_cache('finance_cache', backend='sqlite', expire_after=900, allowable_codes=[200])
        except Exception:
            try:
                # Fallback: install without allowable_codes filter (older versions)
                requests_cache.install_cache('finance_cache', backend='sqlite', expire_after=900)
            except Exception:
                pass  # Cache unavailable; requests proceed uncached

def _init_yfinance():
    """Import yfinance on first use (lazy-loaded to avoid startup slowdown)."""
    global yf, _YF_IMPORTED
    if _YF_IMPORTED:
        return
    _YF_IMPORTED = True
    import yfinance as _yf
    yf = _yf

def _init_finviz():
    """Import finvizfinance on first use (lazy-loaded to avoid startup slowdown)."""
    global Performance, _FINVIZ_IMPORTED
    if _FINVIZ_IMPORTED:
        return
    _FINVIZ_IMPORTED = True
    from finvizfinance.screener.performance import Performance as _Performance
    Performance = _Performance

# Shared curl_cffi session with timeout to prevent yfinance hangs on rate-limited connections (lazy)
_yf_session = None  # Initialized on first use

def _init_yf_session():
    """Initialize curl_cffi session on first use (lazy-loaded to avoid startup slowdown)."""
    global _yf_session
    if _yf_session is not None:
        return
    try:
        from curl_cffi import requests as _cffi_requests
        _yf_session = _cffi_requests.Session(impersonate="chrome")
        _yf_session.timeout = 20  # 20s default for all requests
    except ImportError:
        _yf_session = None  # Let yfinance create its own session

# Global cache bypass flag (set by --no-cache CLI flag)
_NO_CACHE = False

# In-memory caches
_HV_CACHE: Dict[str, float] = {}
_MOMENTUM_CACHE: Dict[str, Tuple] = {}
_IV_RANK_CACHE: Dict[str, Tuple] = {}
_SENTIMENT_CACHE: Dict[str, float] = {}
_SEASONALITY_CACHE: Dict[str, float] = {}
_CHAIN_CACHE: dict = {}
_NEWS_CACHE: Dict[str, list] = {}
_INFO_CACHE: Dict[str, dict] = {}
_FETCH_TIMESTAMPS: Dict[str, datetime] = {}  # symbol → last fetch time

# Module-level sector map — shared by sector_analyzer, ranking, and options_screener
# ~180 major optionable names mapped to their SPDR sector ETF
SECTOR_MAP: Dict[str, str] = {
    # ── XLK — Technology ─────────────────────────────────────────────────────
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
    "QCOM": "XLK", "AMAT": "XLK", "MU": "XLK", "AVGO": "XLK", "TXN": "XLK",
    "LRCX": "XLK", "KLAC": "XLK", "MRVL": "XLK", "ON": "XLK", "CSCO": "XLK",
    "NOW": "XLK", "CRM": "XLK", "ADBE": "XLK", "ORCL": "XLK", "IBM": "XLK",
    "ACN": "XLK", "INTU": "XLK", "PANW": "XLK", "CRWD": "XLK", "SNOW": "XLK",
    "DDOG": "XLK", "PLTR": "XLK", "ZS": "XLK", "FTNT": "XLK", "HPQ": "XLK",
    "HPE": "XLK", "DELL": "XLK", "STX": "XLK", "WDC": "XLK", "CTSH": "XLK",
    "XLK": "XLK",  # ETF itself maps to itself
    "QQQ": "XLK",  # Nasdaq-100 is ~50% tech

    # ── XLF — Financials ─────────────────────────────────────────────────────
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "BLK": "XLF", "SCHW": "XLF", "AXP": "XLF", "COF": "XLF",
    "USB": "XLF", "PNC": "XLF", "TFC": "XLF", "ICE": "XLF", "CME": "XLF",
    "SPGI": "XLF", "MCO": "XLF", "V": "XLF", "MA": "XLF", "PYPL": "XLF",
    "FIS": "XLF", "FISV": "XLF", "XYZ": "XLF", "ALLY": "XLF", "DFS": "XLF",
    "MTB": "XLF", "RF": "XLF", "HBAN": "XLF", "KEY": "XLF", "CFG": "XLF",
    "XLF": "XLF",

    # ── XLE — Energy ──────────────────────────────────────────────────────────
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    "OXY": "XLE", "PSX": "XLE", "VLO": "XLE", "MPC": "XLE", "HES": "XLE",
    "DVN": "XLE", "FANG": "XLE", "APA": "XLE", "BKR": "XLE", "HAL": "XLE",
    "MRO": "XLE", "PXD": "XLE",
    "XLE": "XLE",

    # ── XLY — Consumer Discretionary ─────────────────────────────────────────
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "TGT": "XLY", "LOW": "XLY", "GM": "XLY", "F": "XLY",
    "BKNG": "XLY", "ABNB": "XLY", "UBER": "XLY", "DRI": "XLY", "YUM": "XLY",
    "RCL": "XLY", "CCL": "XLY", "MGM": "XLY", "WYNN": "XLY", "LVS": "XLY",
    "DKNG": "XLY", "ETSY": "XLY", "ROST": "XLY", "TJX": "XLY", "BBY": "XLY",
    "EXPE": "XLY", "LYFT": "XLY", "DASH": "XLY",
    "XLY": "XLY",

    # ── XLP — Consumer Staples ────────────────────────────────────────────────
    "WMT": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "KHC": "XLP", "MDLZ": "XLP",
    "GIS": "XLP", "K": "XLP", "SYY": "XLP", "STZ": "XLP", "TSN": "XLP",
    "CAG": "XLP", "CPB": "XLP", "HRL": "XLP",
    "XLP": "XLP",

    # ── XLV — Health Care ────────────────────────────────────────────────────
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "LLY": "XLV", "TMO": "XLV", "ABT": "XLV", "BMY": "XLV", "AMGN": "XLV",
    "GILD": "XLV", "CVS": "XLV", "CI": "XLV", "HUM": "XLV", "ELV": "XLV",
    "ISRG": "XLV", "DXCM": "XLV", "IDXX": "XLV", "REGN": "XLV", "VRTX": "XLV",
    "SYK": "XLV", "BSX": "XLV", "MDT": "XLV", "EW": "XLV", "MRNA": "XLV",
    "BIIB": "XLV", "ILMN": "XLV", "ZBH": "XLV", "BAX": "XLV", "BDX": "XLV",
    "XLV": "XLV",

    # ── XLI — Industrials ────────────────────────────────────────────────────
    "BA": "XLI", "CAT": "XLI", "GE": "XLI", "HON": "XLI", "MMM": "XLI",
    "RTX": "XLI", "UPS": "XLI", "FDX": "XLI", "DE": "XLI", "ETN": "XLI",
    "LMT": "XLI", "NOC": "XLI", "GD": "XLI", "URI": "XLI", "EMR": "XLI",
    "PH": "XLI", "IR": "XLI", "CTAS": "XLI", "VRSK": "XLI", "FAST": "XLI",
    "WM": "XLI", "RSG": "XLI", "CSX": "XLI", "NSC": "XLI", "UNP": "XLI",
    "DAL": "XLI", "UAL": "XLI", "AAL": "XLI", "LUV": "XLI", "UBER": "XLI",
    "XLI": "XLI",

    # ── XLB — Materials ──────────────────────────────────────────────────────
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "ECL": "XLB", "FCX": "XLB",
    "NEM": "XLB", "NUE": "XLB", "CF": "XLB", "MOS": "XLB", "DOW": "XLB",
    "DD": "XLB", "PPG": "XLB", "ALB": "XLB", "CTVA": "XLB", "CE": "XLB",
    "IFF": "XLB", "PKG": "XLB", "IP": "XLB",
    "GDX": "XLB",  # gold miners ETF ≈ materials
    "XLB": "XLB",

    # ── XLU — Utilities ──────────────────────────────────────────────────────
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "EXC": "XLU",
    "AEP": "XLU", "XEL": "XLU", "ES": "XLU", "ETR": "XLU", "PPL": "XLU",
    "FE": "XLU", "PCG": "XLU", "AWK": "XLU", "CMS": "XLU", "DTE": "XLU",
    "WEC": "XLU", "NI": "XLU",
    "XLU": "XLU",

    # ── XLRE — Real Estate ───────────────────────────────────────────────────
    "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE", "EQIX": "XLRE", "SPG": "XLRE",
    "PSA": "XLRE", "WELL": "XLRE", "O": "XLRE", "ARE": "XLRE", "AVB": "XLRE",
    "EQR": "XLRE", "MAA": "XLRE", "UDR": "XLRE", "DLR": "XLRE", "VTR": "XLRE",
    "NLY": "XLRE", "AGNC": "XLRE", "IRM": "XLRE",
    "XLRE": "XLRE",

    # ── XLC — Communication Services ─────────────────────────────────────────
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "VZ": "XLC", "T": "XLC", "TMUS": "XLC", "CHTR": "XLC",
    "EA": "XLC", "TTWO": "XLC", "SNAP": "XLC", "PINS": "XLC", "RBLX": "XLC",
    "MTCH": "XLC", "ZM": "XLC", "PARA": "XLC", "WBD": "XLC", "FOXA": "XLC",
    "XLC": "XLC",
}


def clear_chain_cache() -> None:
    """Clear all in-session caches between scans."""
    _CHAIN_CACHE.clear()
    _FETCH_TIMESTAMPS.clear()
    _NEWS_CACHE.clear()
    _INFO_CACHE.clear()
    _SENTIMENT_CACHE.clear()
    _SEASONALITY_CACHE.clear()


def get_data_age_seconds() -> Optional[float]:
    """Return age in seconds of the oldest fetch in this session, or None if no fetches."""
    if not _FETCH_TIMESTAMPS:
        return None
    oldest = min(_FETCH_TIMESTAMPS.values())
    return (datetime.now() - oldest).total_seconds()


# --- Abstract Data Provider ---

class BaseDataProvider(ABC):
    """Abstract interface for options/market data sources."""
    @abstractmethod
    def fetch_chain(self, symbol: str) -> Dict[str, Any]: ...
    @abstractmethod
    def fetch_spot(self, symbol: str) -> float: ...


class YFinanceProvider(BaseDataProvider):
    """Concrete provider backed by yfinance (default)."""
    def fetch_chain(self, symbol: str) -> Dict[str, Any]:
        return fetch_options_yfinance(symbol, max_expiries=4)

    def fetch_spot(self, symbol: str) -> float:
        _init_yfinance()
        _init_yf_session()
        return float(yf.Ticker(symbol, session=_yf_session).fast_info["lastPrice"])

    async def fetch_chain_async(self, symbol: str) -> Dict[str, Any]:
        """Non-blocking wrapper around fetch_chain using asyncio.to_thread."""
        import asyncio
        return await asyncio.to_thread(self.fetch_chain, symbol)

    async def fetch_spot_async(self, symbol: str) -> float:
        """Non-blocking wrapper around fetch_spot using asyncio.to_thread."""
        import asyncio
        return await asyncio.to_thread(self.fetch_spot, symbol)


# ---------------------------------------------------------------------------
# yahooquery-backed provider
# ---------------------------------------------------------------------------

def fetch_options_yahooquery(symbol: str, max_expiries: int) -> Dict[str, Any]:
    """
    Fetch options data via yahooquery and return the same dict structure as
    fetch_options_yfinance so the two providers are interchangeable.

    Reuses all existing math helpers (HV, RSI, IV rank, etc.) — only the I/O
    layer (HTTP calls) differs.  Fields that require yfinance-specific APIs
    (news, sentiment, short interest) are set to safe defaults.
    """
    try:
        from yahooquery import Ticker as YQTicker
    except ImportError:
        raise RuntimeError("yahooquery not installed — run: pip install yahooquery")

    tkr_yq = YQTicker(symbol, timeout=15)

    # 1. Price history (1y daily) ------------------------------------------------
    raw_hist = tkr_yq.history(period="1y", interval="1d")
    if isinstance(raw_hist, str) or raw_hist is None:
        raise RuntimeError(f"yahooquery: no price history for {symbol}")

    # yahooquery returns MultiIndex (symbol, date); extract single-ticker slice
    if isinstance(raw_hist.index, pd.MultiIndex) and "symbol" in raw_hist.index.names:
        try:
            hist = raw_hist.xs(symbol.upper(), level="symbol")
        except KeyError:
            hist = raw_hist.xs(symbol.lower(), level="symbol")
    else:
        hist = raw_hist.copy()

    # Normalize column names to Title-case (match yfinance: Close, High, etc.)
    hist.columns = [c.title() for c in hist.columns]
    if "Adj Close" not in hist.columns and "Adjclose" in hist.columns:
        hist.rename(columns={"Adjclose": "Adj Close"}, inplace=True)
    hist.index = pd.DatetimeIndex(hist.index)

    if hist.empty or "Close" not in hist.columns:
        raise RuntimeError(f"yahooquery: unusable history for {symbol}")

    # 2. Derived metrics (same helpers as yfinance path) -------------------------
    underlying = safe_float(hist["Close"].iloc[-1])
    if not underlying or underlying <= 0:
        price_info = tkr_yq.price
        underlying = safe_float(
            (price_info or {}).get(symbol.upper(), {}).get("regularMarketPrice")
        )

    hv_30d_rolling = calculate_historical_volatility(hist, period=30)
    hv_ewma = calculate_ewma_volatility(hist, span=20)
    hv_parkinson = calculate_parkinson_volatility(hist, period=30)
    if hv_30d_rolling and hv_ewma and hv_parkinson:
        hv_30d = 0.34 * hv_30d_rolling + 0.33 * hv_ewma + 0.33 * hv_parkinson
    elif hv_30d_rolling and hv_ewma:
        hv_30d = 0.5 * hv_30d_rolling + 0.5 * hv_ewma
    else:
        hv_30d = hv_30d_rolling or hv_ewma

    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct = \
        calculate_momentum_indicators(hist)
    rvol = calculate_rvol(hist)
    vwap, fib_50, fib_618 = calculate_technical_levels(hist)

    # 3. Secondary context (yahooquery where possible; defaults otherwise) -------
    earnings_date = None
    try:
        ed = tkr_yq.earnings_dates
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            future_mask = ed.index > pd.Timestamp.now()
            if future_mask.any():
                earnings_date = ed.index[future_mask][0].to_pydatetime()
    except Exception as exc:
        logger.debug("Earnings date fetch failed for %s: %s", symbol, exc)

    sector_perf = get_sector_performance(symbol)
    # yahooquery doesn't expose sentiment/news/short-interest without extra APIs
    sentiment_score = 0.0
    news_headlines: List[str] = []
    news_data = None
    seasonal_win_rate = None
    short_interest = None
    next_ex_div = None

    # 4. Option chain ------------------------------------------------------------
    try:
        chain_full = tkr_yq.option_chain
    except Exception as exc:
        raise RuntimeError(f"yahooquery: option_chain failed for {symbol}: {exc}")

    if isinstance(chain_full, str) or chain_full is None or (
        hasattr(chain_full, "empty") and chain_full.empty
    ):
        raise RuntimeError(f"yahooquery: no option chain for {symbol}")

    # Determine the index structure (varies: 2- or 3-level MultiIndex)
    idx_names = list(chain_full.index.names)
    has_symbol_level = "symbol" in idx_names

    # Get sorted expiration Timestamps
    expirations_raw = sorted(chain_full.index.get_level_values("expiration").unique())
    num_exp = max(max_expiries, 2) if max_expiries == 1 else max_expiries
    expirations_to_use = expirations_raw[:num_exp]

    frames: List[pd.DataFrame] = []
    for exp_ts in expirations_to_use:
        exp_str = exp_ts.strftime("%Y-%m-%d") if hasattr(exp_ts, "strftime") else str(exp_ts)[:10]
        for yq_type, mapped_type in [("calls", "call"), ("puts", "put")]:
            try:
                if has_symbol_level:
                    sub = chain_full.xs(
                        (symbol.upper(), exp_ts, yq_type),
                        level=("symbol", "expiration", "optionType"),
                    )
                else:
                    sub = chain_full.xs(
                        (exp_ts, yq_type), level=("expiration", "optionType")
                    )
            except KeyError:
                continue

            if sub is None or sub.empty:
                continue

            sub = sub.copy()
            sub["type"] = mapped_type
            sub["expiration"] = exp_str
            sub["symbol"] = symbol.upper()
            for col in ["strike", "lastPrice", "bid", "ask", "volume",
                        "openInterest", "impliedVolatility"]:
                if col not in sub.columns:
                    sub[col] = pd.NA
            frames.append(sub)

    if not frames:
        raise RuntimeError(f"yahooquery: no usable option frames for {symbol}")

    df = pd.concat(frames, ignore_index=True)

    # Max pain
    max_pain_val = calculate_max_pain(df)
    df["max_pain"] = max_pain_val

    # 5. Enrich DataFrame (identical to fetch_options_yfinance lines 922+) -------
    df["underlying"] = underlying
    df["hv_30d"] = hv_30d
    df["ret_5d"] = ret_5d
    df["rsi_14"] = rsi_14
    df["adx_14"] = adx_14
    df["atr_trend"] = atr_trend
    df["sma_20"] = sma_20
    df["sma_50"] = sma_50
    df["high_20"] = high_20
    df["low_20"] = low_20
    df["sentiment_score"] = sentiment_score
    df["seasonal_win_rate"] = seasonal_win_rate
    df["is_squeezing"] = is_squeezing
    df["bb_width_pct"] = bb_width_pct
    df["rvol"] = rvol
    df["short_interest"] = short_interest
    df["vwap"] = vwap
    df["fib_50"] = fib_50
    df["fib_618"] = fib_618

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)

    iv_rank_30 = iv_pct_30 = iv_rank_90 = iv_pct_90 = None
    iv_confidence = "Low"
    if pd.notna(median_iv) and median_iv > 0:
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, iv_confidence = \
            get_iv_rank_percentile_from_history(hist, median_iv, ticker=symbol)

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90
    df["iv_confidence"] = iv_confidence

    # IV velocity and trend
    _yq_iv_trend = "stable"
    _yq_iv_velocity = 0.0
    if iv_pct_30 is not None and iv_pct_90 is not None:
        _diff = float(iv_pct_30) - float(iv_pct_90)
        if _diff > 0.05:
            _yq_iv_trend = "expanding"
            _yq_iv_velocity = _diff
        elif _diff < -0.05:
            _yq_iv_trend = "contracting"
            _yq_iv_velocity = _diff
    df["iv_trend"] = _yq_iv_trend
    df["iv_velocity"] = _yq_iv_velocity
    df["dividend_yield"] = 0.0

    # Term structure spread
    term_structure_spread = None
    if len(expirations_to_use) >= 2 and "expiration" in df.columns:
        try:
            df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
            sorted_exps = sorted(df["exp_dt"].dropna().unique())
            if len(sorted_exps) >= 2:
                def _atm_iv(d: pd.DataFrame):
                    d = d.copy()
                    d["dist"] = (d["strike"] - underlying).abs()
                    strike = d.loc[d["dist"].idxmin(), "strike"]
                    ivs = d[d["strike"] == strike]["impliedVolatility"].dropna()
                    return ivs.mean() if not ivs.empty else None

                f_iv = _atm_iv(df[df["exp_dt"] == sorted_exps[0]])
                b_iv = _atm_iv(df[df["exp_dt"] == sorted_exps[1]])
                if f_iv and b_iv:
                    term_structure_spread = b_iv - f_iv
        except Exception as exc:
            logger.debug("Term structure spread computation failed: %s", exc)

    return {
        "df": df,
        "history_df": hist,
        "context": {
            "hv": hv_30d,
            "bb_width_pct": bb_width_pct,
            "hv_ewma": hv_ewma,
            "hv_parkinson": hv_parkinson,
            "iv_rank": iv_rank_30,
            "iv_percentile": iv_pct_30,
            "earnings_date": earnings_date,
            "earnings_move_data": None,   # requires yfinance historical earnings
            "sentiment_score": sentiment_score,
            "news_headlines": news_headlines,
            "seasonal_win_rate": seasonal_win_rate,
            "term_structure_spread": term_structure_spread,
            "sector_perf": sector_perf,
            "rvol": rvol,
            "short_interest": short_interest,
            "next_ex_div": next_ex_div,
            "vwap": vwap,
            "fib_50": fib_50,
            "fib_618": fib_618,
            "news_data": news_data,
            "iv_confidence": iv_confidence,
            "iv_trend": _yq_iv_trend,
            "iv_velocity": _yq_iv_velocity,
            "dividend_yield": 0.0,
        },
    }


class YahooQueryProvider(BaseDataProvider):
    """
    Option data provider backed by yahooquery.

    Uses a different HTTP client and endpoint structure than yfinance so the
    two providers hit independent rate limits and can be raced in parallel.
    yahooquery also supports batching multiple symbols in a single HTTP round-
    trip via fetch_chain_batch().
    """

    def fetch_chain(self, symbol: str) -> Dict[str, Any]:
        return fetch_options_yahooquery(symbol, max_expiries=4)

    def fetch_spot(self, symbol: str) -> float:
        from yahooquery import Ticker as YQTicker
        price = YQTicker(symbol, timeout=10).price
        val = (price or {}).get(symbol.upper(), {}).get("regularMarketPrice")
        if val is None:
            raise RuntimeError(f"yahooquery: no spot price for {symbol}")
        return float(val)

    def fetch_chain_batch(
        self, symbols: List[str], max_expiries: int = 4
    ) -> Dict[str, Any]:
        """
        Fetch option chains for ALL symbols in a single yahooquery batch call.

        Returns a dict {symbol: chain_dict_or_error} with the same structure as
        fetch_options_yfinance so results are drop-in replacements.
        """
        from yahooquery import Ticker as YQTicker

        results: Dict[str, Any] = {}
        # One ticker object covers all symbols; yahooquery batches internally
        batch = YQTicker(symbols, asynchronous=True, timeout=30)

        for symbol in symbols:
            try:
                results[symbol] = fetch_options_yahooquery.__wrapped__(
                    symbol, max_expiries, _batch_ticker=batch
                ) if hasattr(fetch_options_yahooquery, "__wrapped__") \
                    else fetch_options_yahooquery(symbol, max_expiries)
            except Exception as exc:
                results[symbol] = {"error": str(exc)}

        return results


class FanOutProvider(BaseDataProvider):
    """
    Meta-provider that races multiple providers in parallel and returns the
    first successful result.  Provides redundancy: if yfinance is rate-limited,
    yahooquery picks up the slack, and vice versa.
    """

    def __init__(self, providers: List[BaseDataProvider]) -> None:
        self.providers = providers

    def fetch_chain(self, symbol: str) -> Dict[str, Any]:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.providers)
        ) as executor:
            future_to_provider = {
                executor.submit(p.fetch_chain, symbol): p
                for p in self.providers
            }
            errors = []
            for fut in concurrent.futures.as_completed(future_to_provider):
                try:
                    return fut.result()
                except Exception as exc:
                    prov = future_to_provider[fut]
                    errors.append((type(prov).__name__, str(exc)))

        raise RuntimeError(
            f"All providers failed for {symbol}: " +
            "; ".join(f"{name}: {err}" for name, err in errors)
        )

    def fetch_spot(self, symbol: str) -> float:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.providers)
        ) as executor:
            future_to_provider = {
                executor.submit(p.fetch_spot, symbol): p
                for p in self.providers
            }
            errors = []
            for fut in concurrent.futures.as_completed(future_to_provider):
                try:
                    return fut.result()
                except Exception as exc:
                    prov = future_to_provider[fut]
                    errors.append((type(prov).__name__, str(exc)))

        raise RuntimeError(
            f"All providers failed for spot({symbol}): " +
            "; ".join(f"{name}: {err}" for name, err in errors)
        )


# Module-level singleton — swap out for testing or alternative sources (lazy-loaded)
_DATA_PROVIDER = None  # Lazy-initialized on first use

def _make_default_provider() -> BaseDataProvider:
    """Build the default provider, using FanOut if yahooquery is available."""
    _init_yfinance()  # Ensure yfinance is available
    try:
        import yahooquery  # noqa: F401
        return FanOutProvider([YFinanceProvider(), YahooQueryProvider()])
    except ImportError:
        return YFinanceProvider()

def _get_data_provider() -> BaseDataProvider:
    """Get the default data provider, initializing it on first use."""
    global _DATA_PROVIDER
    if _DATA_PROVIDER is None:
        _DATA_PROVIDER = _make_default_provider()
    return _DATA_PROVIDER

# --- IV History Persistence ---

def _get_iv_db_path() -> str:
    try:
        import json
        with open("config.json") as f:
            cfg = json.load(f)
        return cfg.get("iv_history_db_path", "iv_cache.db")
    except Exception:
        return "iv_cache.db"


def _init_iv_db(db_path: str) -> None:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iv_history (
                    ticker TEXT,
                    date   TEXT,
                    iv_value REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.commit()
    except Exception as exc:
        logger.debug("IV DB init failed: %s", exc)


def _upsert_iv(ticker: str, date_str: str, iv: float, db_path: str) -> None:
    try:
        _init_iv_db(db_path)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO iv_history (ticker, date, iv_value) VALUES (?, ?, ?)",
                (ticker, date_str, float(iv)),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("IV upsert failed for %s: %s", ticker, exc)


def _init_skew_db(db_path: str) -> None:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skew_history (
                    ticker TEXT,
                    date   TEXT,
                    skew_value REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.commit()
    except Exception as exc:
        logger.debug("Skew DB init failed: %s", exc)


def _upsert_skew(ticker: str, date_str: str, skew: float, db_path: str) -> None:
    try:
        _init_skew_db(db_path)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skew_history (ticker, date, skew_value) VALUES (?, ?, ?)",
                (ticker, date_str, float(skew)),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Skew upsert failed for %s: %s", ticker, exc)


def _load_skew_history(ticker: str, db_path: str) -> List[float]:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT skew_value FROM skew_history WHERE ticker=? ORDER BY date ASC",
                (ticker,),
            ).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    except Exception as exc:
        logger.debug("Skew history load failed for %s: %s", ticker, exc)
        return []


def get_skew_percentile(ticker: str, current_skew: float) -> float:
    """Upsert today's skew and return the historical percentile rank (0.5 if <10 rows)."""
    try:
        db_path = _get_iv_db_path()
        date_str = datetime.now().strftime("%Y-%m-%d")
        _upsert_skew(ticker, date_str, current_skew, db_path)
        history = _load_skew_history(ticker, db_path)
        if len(history) < 10:
            return 0.5
        arr = np.array(history, dtype=float)
        return float(np.clip((arr < current_skew).sum() / len(arr), 0.0, 1.0))
    except Exception:
        return 0.5


def calculate_vrp(
    hist: pd.DataFrame,
    current_iv: float,
    lookback_periods: int = 12,
) -> Dict[str, Any]:
    """Compute Volatility Risk Premium (IV - Realized Vol) statistics.

    Returns a dict with vrp_mean, vrp_std, vrp_percentile, vrp_regime.
    Falls back to a safe default dict on error or <60 rows of history.
    """
    default: Dict[str, Any] = {
        "vrp_mean": 0.0, "vrp_std": 0.0, "vrp_percentile": 0.5, "vrp_regime": "UNKNOWN",
    }
    try:
        if hist.empty or len(hist) < 60:
            return default
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(returns) < 30:
            return default
        rolling_rv = returns.rolling(window=30).std().dropna() * np.sqrt(252)
        if len(rolling_rv) < lookback_periods:
            return default
        current_rv = float(rolling_rv.iloc[-1])
        if current_rv <= 0:
            return default
        iv_hv_ratio = np.clip(current_iv / current_rv, 0.5, 3.0)
        hist_ivs = rolling_rv * iv_hv_ratio
        vrp_series = hist_ivs - rolling_rv
        recent_vrp = vrp_series.iloc[-lookback_periods:]
        vrp_mean = float(recent_vrp.mean())
        vrp_std = float(recent_vrp.std()) if len(recent_vrp) > 1 else 0.0
        vrp_current = current_iv - current_rv
        vrp_pctile = float(np.clip((vrp_series < vrp_current).sum() / len(vrp_series), 0.0, 1.0))
        if vrp_mean >= 0.05:
            regime = "HIGH_PREMIUM"
        elif vrp_mean >= 0.01:
            regime = "NORMAL"
        elif vrp_mean >= -0.02:
            regime = "FAIR"
        else:
            regime = "CHEAP"
        return {"vrp_mean": vrp_mean, "vrp_std": vrp_std, "vrp_percentile": vrp_pctile, "vrp_regime": regime}
    except Exception as exc:
        logger.debug("VRP computation failed: %s", exc)
        return default


def _load_iv_history(ticker: str, db_path: str) -> List[float]:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT iv_value FROM iv_history WHERE ticker=? ORDER BY date ASC",
                (ticker,),
            ).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    except Exception as exc:
        logger.debug("IV history load failed for %s: %s", ticker, exc)
        return []


def iv_history_coverage(ticker: str) -> Dict[str, Any]:
    """Return coverage stats for ticker's IV history in iv_cache.db."""
    try:
        db_path = _get_iv_db_path()
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM iv_history WHERE ticker=?", (ticker,)
            ).fetchone()
        days = int(row[0]) if row else 0
        if days >= 252:
            confidence = "High"
        elif days >= 30:
            confidence = "Medium"
        else:
            confidence = "Low"
        return {"ticker": ticker, "days": days, "confidence": confidence}
    except Exception as e:
        logger.warning("iv_history_coverage failed for %s: %s", ticker, e)
        return {"ticker": ticker, "days": 0, "confidence": "Low"}


# --- Retry Decorator ---
# Global token-bucket throttle for Yahoo requests.
# All threads share one gate so aggregate request rate stays under Yahoo's ~60-100 req/min ceiling.
_YF_THROTTLE_LOCK = threading.Lock()
_YF_LAST_CALL_TS = [0.0]
_YF_MIN_INTERVAL = 0.25  # seconds between any two Yahoo calls (4 req/sec aggregate — fast mode; 429 cooldown will back off if hit)

def _yf_throttle():
    with _YF_THROTTLE_LOCK:
        now = time.monotonic()
        wait = _YF_LAST_CALL_TS[0] + _YF_MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _YF_LAST_CALL_TS[0] = now

def _yf_cooldown(seconds: float):
    """Push the next-allowed call forward by `seconds` across all threads (use after 429)."""
    with _YF_THROTTLE_LOCK:
        _YF_LAST_CALL_TS[0] = max(_YF_LAST_CALL_TS[0], time.monotonic()) + seconds


def retry_with_backoff(retries=3, backoff_in_seconds=1, error_types=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    msg = str(e).lower()
                    # yfinance has a bug (history.py:224) where a rate-limited/empty response
                    # causes `data['chart']` to raise TypeError('NoneType' object is not subscriptable).
                    # Treat that as a rate-limit signal so the global cooldown kicks in.
                    is_rate_limit = (
                        ("too many requests" in msg)
                        or ("rate limited" in msg)
                        or ("429" in msg)
                        or ("nonetype" in msg and "subscriptable" in msg)
                    )
                    if is_rate_limit:
                        # Global cooldown: all other threads wait too, so we don't hammer Yahoo after a 429.
                        _yf_cooldown(15.0)
                    if x == retries:
                        logging.warning(f"Function {func.__name__} failed after {retries} retries: {e}")
                        # When the failure is a Python-level bug (NoneType, etc.) rather than a
                        # network/rate-limit issue, dump the traceback so we can pinpoint it.
                        if not is_rate_limit and "no options" not in msg and "price history" not in msg:
                            try:
                                import os, traceback
                                _log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scan_errors.log")
                                sym_hint = args[0] if args else kwargs.get("symbol", "?")
                                with open(_log, "a") as _f:
                                    _f.write(f"\n=== {func.__name__}({sym_hint}) ===\n{traceback.format_exc()}\n")
                            except Exception:
                                pass
                        raise e
                    base = backoff_in_seconds * (2 ** x)
                    if is_rate_limit:
                        base = max(base, 6.0 * (x + 1))  # 6s, 12s, 18s on rate limits
                    sleep = base + random.uniform(0, 1)
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

# --- Core Data Fetching Functions ---

@retry_with_backoff(retries=3, backoff_in_seconds=2, error_types=(RuntimeError, URLError, ConnectionError, OSError))
def get_dynamic_tickers(scan_type: str, max_tickers: int = 50) -> List[str]:
    """
    Fetches a list of tickers from Finviz based on a given scan type.
    Falls back to a hardcoded list if fetching fails.
    """
    BACKUP_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL"]
    
    filters_dict = {
        'Option/Short': 'Optionable',
        'Average Volume': 'Over 500K',
        'Country': 'USA',
    }
    order = 'Change'  # Default for gainers
    if scan_type == 'high_iv':
        order = 'Volatility (Month)'

    try:
        _init_finviz()
        fperformance = Performance()
        fperformance.set_filter(filters_dict=filters_dict)
        df = fperformance.screener_view(order=order, limit=max_tickers, verbose=0)

        if df.empty:
            logging.warning("Finviz returned empty dataframe. Using backup tickers.")
            return BACKUP_TICKERS[:max_tickers]

        return df['Ticker'].tolist()
    except Exception as e:
        logging.warning(f"Could not fetch '{scan_type}' from Finviz: {e}. Using backup tickers.")
        return BACKUP_TICKERS[:max_tickers]

def get_underlying_price(ticker: Any) -> Optional[float]:
    # Try fast_info, then info, then last close
    try:
        fi = getattr(ticker, "fast_info", None)
        if fi:
            lp = safe_float(getattr(fi, "last_price", None))
            if lp:
                return lp
    except Exception:
        pass
    try:
        info = _get_info_cached(ticker.ticker, ticker)
        lp = safe_float(info.get("regularMarketPrice"))
        if lp:
            return lp
    except Exception:
        pass
    try:
        hist = ticker.history(period="5d", interval="1d")
        if not hist.empty:
            return safe_float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _get_news_cached(symbol: str, ticker: "yf.Ticker") -> list:
    """Fetch ticker.news once per session, cache result."""
    if symbol in _NEWS_CACHE:
        return _NEWS_CACHE[symbol]
    try:
        news = ticker.news if ticker else []
    except Exception:
        news = []
    _NEWS_CACHE[symbol] = news or []
    return _NEWS_CACHE[symbol]


def _get_info_cached(symbol: str, ticker: "yf.Ticker") -> dict:
    """Fetch ticker.info once per session, cache result."""
    if symbol in _INFO_CACHE:
        return _INFO_CACHE[symbol]
    try:
        info = (ticker.info if ticker else None) or {}
    except Exception:
        info = {}
    _INFO_CACHE[symbol] = info
    return _INFO_CACHE[symbol]


@retry_with_backoff(retries=2, backoff_in_seconds=1)
def get_sentiment(ticker: yf.Ticker) -> Optional[float]:
    try:
        from textblob import TextBlob
    except ImportError:
        return None

    key = f"{ticker.ticker}:sentiment"
    if key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[key]

    try:
        news = _get_news_cached(ticker.ticker, ticker)
        if not news:
            return None
        headlines = ". ".join([item.get("title", "") for item in news])
        if not headlines.strip():
            return None
        blob = TextBlob(headlines)
        score = blob.sentiment.polarity
        _SENTIMENT_CACHE[key] = score
        return score
    except Exception as exc:
        logger.debug("Sentiment analysis failed: %s", exc)
        return None

def get_news_headlines(ticker: yf.Ticker, max_headlines: int = 3) -> list:
    """Return up to max_headlines recent news titles for a ticker."""
    try:
        news = _get_news_cached(ticker.ticker, ticker)
        if not news:
            return []
        titles = []
        for item in news[:max_headlines]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                titles.append(title.strip())
        return titles[:max_headlines]
    except Exception as exc:
        logger.debug("News headlines fetch failed: %s", exc)
        return []

def _read_seasonality_cache(symbol: str, month: int, db_path: str = "iv_cache.db") -> Optional[float]:
    """Read cached seasonality win rate. Returns None if stale (>7 days) or missing."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS seasonality_cache (
                symbol TEXT, month INTEGER, win_rate REAL, updated TEXT,
                PRIMARY KEY (symbol, month))""")
            row = conn.execute(
                "SELECT win_rate, updated FROM seasonality_cache WHERE symbol=? AND month=?",
                (symbol, month)
            ).fetchone()
        if row is None:
            return None
        updated = datetime.fromisoformat(row[1])
        if datetime.now() - updated > timedelta(days=7):
            return None
        return row[0]
    except Exception:
        return None


def _write_seasonality_cache(symbol: str, month: int, win_rate: float, db_path: str = "iv_cache.db") -> None:
    """Write seasonality win rate to SQLite cache."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS seasonality_cache (
                symbol TEXT, month INTEGER, win_rate REAL, updated TEXT,
                PRIMARY KEY (symbol, month))""")
            conn.execute(
                "INSERT OR REPLACE INTO seasonality_cache (symbol, month, win_rate, updated) VALUES (?, ?, ?, ?)",
                (symbol, month, win_rate, datetime.now().isoformat())
            )
            conn.commit()
    except Exception:
        pass


@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_seasonality(ticker: yf.Ticker) -> Optional[float]:
    key = f"{ticker.ticker}:seasonality"
    if key in _SEASONALITY_CACHE:
        return _SEASONALITY_CACHE[key]
    current_month = datetime.now().month
    cached = _read_seasonality_cache(ticker.ticker, current_month)
    if cached is not None:
        _SEASONALITY_CACHE[key] = cached
        return cached
    try:
        hist = ticker.history(period="5y", interval="1mo")
        if hist.empty:
            return None
        monthly_data = hist[hist.index.month == current_month]
        if monthly_data.empty:
            return None
        wins = (monthly_data['Close'] > monthly_data['Open']).sum()
        total = len(monthly_data)
        win_rate = wins / total if total > 0 else 0.0
        _SEASONALITY_CACHE[key] = win_rate
        _write_seasonality_cache(ticker.ticker, current_month, win_rate)
        return win_rate
    except Exception as exc:
        logger.debug("Seasonality check failed for %s: %s", ticker.ticker, exc)
        return None

def get_next_earnings_date(ticker: yf.Ticker) -> Optional[datetime]:
    try:
        try:
            ed = ticker.get_earnings_dates(limit=1)
            if ed is not None and not ed.empty:
                dt = ed.index[0]
                if not isinstance(dt, datetime):
                    dt = datetime.fromtimestamp(dt.timestamp())
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except Exception:
            pass
        try:
            cal = getattr(ticker, "calendar", None)
            if cal is not None and not getattr(cal, "empty", False):
                if hasattr(cal, "index") and len(cal.index) > 0:
                    dt = cal.index[0]
                    if not isinstance(dt, datetime):
                        # date objects have no .timestamp(); use combine instead
                        import datetime as _dt_mod
                        if isinstance(dt, _dt_mod.date):
                            dt = datetime.combine(dt, datetime.min.time())
                        else:
                            dt = datetime.fromtimestamp(float(dt))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
        except Exception:
            pass
    except Exception:
        return None
    return None

_rfr_cache: dict = {"value": None, "ts": 0.0}

def get_risk_free_rate() -> float:
    """Fetch current risk-free rate from ^IRX (13-week T-bill), cached for 15 min."""
    _init_yfinance()  # Lazy init yfinance on first use
    _init_yf_session()  # Lazy init curl_cffi session
    import time as _time
    now = _time.time()
    if _rfr_cache["value"] is not None and now - _rfr_cache["ts"] < 900:
        return _rfr_cache["value"]
    default_rate = 0.045
    rate_result = default_rate
    try:
        tbill = yf.Ticker("^IRX", session=_yf_session)
        try:
            fi = getattr(tbill, "fast_info", None)
            if fi:
                rate = safe_float(getattr(fi, "last_price", None))
                if rate and rate > 0:
                    rate_result = rate / 100.0
        except Exception:
            pass
        if rate_result == default_rate:
            hist = tbill.history(period="5d", interval="1d")
            if not hist.empty:
                rate = safe_float(hist["Close"].iloc[-1])
                if rate and rate > 0:
                    rate_result = rate / 100.0
    except Exception:
        pass
    _rfr_cache["value"] = rate_result
    _rfr_cache["ts"] = now
    return rate_result

def get_vix_level() -> Optional[float]:
    _init_yfinance()
    _init_yf_session()
    try:
        vix = yf.Ticker("^VIX", session=_yf_session)
        try:
            fi = getattr(vix, "fast_info", None)
            if fi:
                level = safe_float(getattr(fi, "last_price", None))
                if level and level > 0:
                    return level
        except Exception:
            pass
        hist = vix.history(period="5d", interval="1d")
        if not hist.empty:
            level = safe_float(hist["Close"].iloc[-1])
            if level and level > 0:
                return level
    except Exception:
        pass
    return None

def determine_vix_regime(vix_level: Optional[float], config: Dict) -> Tuple[str, Dict]:
    if vix_level is None:
        w = dict(config.get("composite_weights", {}))
        w["regime"] = "normal"
        return "normal", w
    vix_regimes = config.get("vix_regimes", {})
    if vix_level < vix_regimes.get("low", {}).get("threshold", 15):
        regime = "low"
    elif vix_level > vix_regimes.get("high", {}).get("threshold", 25):
        regime = "high"
    else:
        regime = "normal"
    w = dict(vix_regimes.get(regime, {}).get("weights", config.get("composite_weights", {})))
    w["regime"] = regime
    return regime, w

# --- New / Refactored Calculation Functions (Using Cached History) ---

def calculate_historical_volatility(hist: pd.DataFrame, period: int = 30) -> Optional[float]:
    """Calculate annualized volatility from history DataFrame."""
    try:
        if hist.empty or len(hist) < period:
            return None
        # Use last 'period' days
        subset = hist.iloc[-(period+1):].copy()
        returns = np.log(subset['Close'] / subset['Close'].shift(1)).dropna()
        if len(returns) < 2:
            return None
        daily_vol = returns.std()
        return daily_vol * math.sqrt(252)
    except Exception as exc:
        logger.debug("HV calculation failed: %s", exc)
        return None

def calculate_ewma_volatility(hist: pd.DataFrame, span: int = 20) -> Optional[float]:
    """
    EWMA annualized volatility (exponentially weighted).
    More responsive to recent price moves than a simple rolling std,
    giving a better near-term realized vol estimate for EV calculations.
    """
    try:
        if hist.empty or len(hist) < 10:
            return None
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        if len(returns) < 5:
            return None
        ewm_var = (returns ** 2).ewm(span=span, adjust=False).mean()
        if ewm_var.empty:
            return None
        return float(np.sqrt(ewm_var.iloc[-1] * 252))
    except Exception as exc:
        logger.debug("EWMA vol calculation failed: %s", exc)
        return None

def calculate_parkinson_volatility(hist: pd.DataFrame, period: int = 30) -> Optional[float]:
    """Parkinson (1980) high-low range estimator of realized vol.
    More efficient than close-to-close: uses intraday range to capture
    moves missed by close-to-close (e.g., intraday spikes that close flat).
    Factor: 1 / (4 * ln(2)) ≈ 0.3607
    """
    try:
        if hist.empty or "High" not in hist.columns or "Low" not in hist.columns:
            return None
        subset = hist.iloc[-period:].copy()
        log_hl = np.log(subset["High"] / subset["Low"])
        if (log_hl < 0).any() or len(log_hl) < 5:
            return None
        parkinson_var = (log_hl ** 2).mean() / (4 * math.log(2))
        return float(math.sqrt(parkinson_var * 252))
    except Exception:
        return None


def calculate_max_pain(df_chain: pd.DataFrame, underlying: float = 0.0) -> Optional[float]:
    """
    Max pain strike: the strike at which total option holder loss is maximized
    (i.e., market makers/sellers profit most). Near expiry, price tends to pin here.

    For each candidate strike K:
      call_pain = sum over all call strikes k < K of: (K - k) * call_OI(k)
      put_pain  = sum over all put strikes k > K of: (k - K) * put_OI(k)
      total_pain(K) = call_pain + put_pain
    Max pain = argmin(total_pain)
    """
    try:
        if df_chain.empty or "strike" not in df_chain.columns:
            return None
        strikes = sorted(df_chain["strike"].unique())
        calls = df_chain[df_chain["type"] == "call"].set_index("strike")["openInterest"].fillna(0)
        puts  = df_chain[df_chain["type"] == "put"].set_index("strike")["openInterest"].fillna(0)

        min_pain = float("inf")
        max_pain_strike = None
        for k in strikes:
            call_pain = sum((k - s) * calls.get(s, 0) for s in strikes if s < k)
            put_pain  = sum((s - k) * puts.get(s, 0)  for s in strikes if s > k)
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = float(k)
        return max_pain_strike
    except Exception:
        return None


def calculate_momentum_indicators(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool, Optional[float], Optional[float], Optional[float]]:
    """Calculate RSI, ATR, ADX, SMA-20, SMA-50, etc. from history DataFrame."""
    try:
        if hist.empty or len(hist) < 21:
            return None, None, None, None, None, None, False, None, None, None

        close = hist["Close"].astype(float)
        high = hist.get("High", close).astype(float)
        low = hist.get("Low", close).astype(float)

        # 5-day return
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else None

        # RSI 14
        delta = close.diff().dropna()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None

        # ATR Trend
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        atr_trend = None
        if len(atr.dropna()) >= 20:
            recent_atr = atr.iloc[-1]
            atr_ma = atr.rolling(window=20).mean().iloc[-1]
            if atr_ma and atr_ma > 0:
                atr_trend = float(recent_atr / atr_ma - 1.0)

        # ADX-14: trend strength (>25 = trending, <20 = ranging)
        adx_14 = None
        try:
            if len(close) >= 28:
                plus_dm = (high - high.shift(1)).clip(lower=0.0)
                minus_dm = (low.shift(1) - low).clip(lower=0.0)
                # Zero out when the other is larger
                plus_dm[plus_dm < minus_dm] = 0.0
                minus_dm[minus_dm < plus_dm] = 0.0
                _atr_smooth = atr.bfill()
                plus_di = 100.0 * (plus_dm.rolling(14).mean() / _atr_smooth.replace(0, np.nan))
                minus_di = 100.0 * (minus_dm.rolling(14).mean() / _atr_smooth.replace(0, np.nan))
                dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
                adx_series = dx.rolling(14).mean()
                if adx_series.notna().any():
                    adx_14 = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else None
        except Exception:
            pass

        # 20-day stats
        sma_20 = float(close.rolling(window=20).mean().iloc[-1])
        high_20 = float(high.rolling(window=20).max().iloc[-1])
        low_20 = float(low.rolling(window=20).min().iloc[-1])

        # 50-day SMA (confirms medium-term trend; requires at least 50 bars)
        sma_50 = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else None

        # Squeeze
        bb_std = close.rolling(window=20).std()
        bb_upper = sma_20 + (bb_std.iloc[-1] * 2)
        bb_lower = sma_20 - (bb_std.iloc[-1] * 2)
        kc_atr = atr.iloc[-1]
        kc_upper = sma_20 + (kc_atr * 1.5)
        kc_lower = sma_20 - (kc_atr * 1.5)
        is_squeezing = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        # BB Width percentile: (BB upper - BB lower) / SMA-20, ranked vs history
        # Low value = volatility compressed = primed to expand
        bb_width_pct = None
        try:
            bb_width_series = (close.rolling(window=20).std() * 4.0) / close.rolling(window=20).mean().replace(0, np.nan)
            bb_width_series = bb_width_series.dropna()
            if len(bb_width_series) >= 20:
                current_width = bb_width_series.iloc[-1]
                bb_width_pct = float((bb_width_series < current_width).mean())
        except Exception:
            pass

        return ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct
    except Exception:
        return None, None, None, None, None, None, False, None, None, None

def calculate_rvol(hist: pd.DataFrame) -> Optional[float]:
    """Calculate Relative Volume (Current Vol / 30-day Avg Vol)."""
    try:
        if hist.empty or len(hist) < 30:
            return None
        
        # Ensure Volume is numeric
        vol = hist["Volume"].astype(float)
        
        # Current volume (last row)
        current_vol = vol.iloc[-1]
        
        # Average of previous 30 days (excluding current if it's a partial day, 
        # but usually we just take the rolling mean. For robustness, let's use last 30 days inc current)
        avg_vol = vol.rolling(window=30).mean().iloc[-1]
        
        if avg_vol and avg_vol > 0:
            return float(current_vol / avg_vol)
        return None
    except Exception:
        return None

def calculate_technical_levels(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate VWAP (approx) and Fibonacci levels (0.5, 0.618)."""
    try:
        if hist.empty:
            return None, None, None
        
        # VWAP Approximation (Session VWAP if intraday data, else just typical price)
        # Since we usually fetch 1y daily, we can't do true session VWAP. 
        # We'll use the last day's Typical Price as a proxy for "Session Pivot".
        last = hist.iloc[-1]
        vwap = (last['High'] + last['Low'] + last['Close']) / 3.0
        
        # Fibonacci Levels (based on 3-month High/Low)
        # 3 months ~ 63 trading days
        subset = hist.iloc[-63:] if len(hist) > 63 else hist
        period_high = subset['High'].max()
        period_low = subset['Low'].min()
        
        diff = period_high - period_low
        fib_50 = period_high - (0.5 * diff)
        fib_618 = period_high - (0.618 * diff)
        
        return vwap, fib_50, fib_618
    except Exception:
        return None, None, None

def get_iv_rank_percentile_from_history(
    hist: pd.DataFrame,
    current_iv: float,
    ticker: str = "",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """Calculate IV Rank/Percentile, persisting daily IV for improving confidence over time.

    Returns a 5-tuple: (iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, confidence).
    confidence is "High" (>=252 stored days), "Medium" (>=30), or "Low" (HV proxy).
    """
    confidence = "Low"

    # --- Persist today's IV and try to use stored history ---
    if not _NO_CACHE:
        try:
            db_path = _get_iv_db_path()
            if ticker:
                date_str = datetime.now().strftime("%Y-%m-%d")
                _upsert_iv(ticker, date_str, current_iv, db_path)
                iv_history = _load_iv_history(ticker, db_path)
            else:
                iv_history = []

            if len(iv_history) >= 30:
                iv_arr = np.array(iv_history, dtype=float)
                iv_arr = iv_arr[np.isfinite(iv_arr)]
                if len(iv_arr) < 30:
                    pass  # fall through to HV-proxy
                else:
                    iv_min = iv_arr.min()
                    iv_max = iv_arr.max()
                    if iv_max - iv_min < 1e-8:
                        iv_rank_30, iv_pct_30 = 0.5, 0.5
                    else:
                        iv_rank_30 = float(np.clip((current_iv - iv_min) / (iv_max - iv_min), 0, 1))
                        iv_pct_30 = float(np.clip((iv_arr < current_iv).sum() / len(iv_arr), 0, 1))

                    if len(iv_arr) >= 90:
                        iv_arr_90 = iv_arr[-90:]
                        iv_min_90, iv_max_90 = iv_arr_90.min(), iv_arr_90.max()
                        if iv_max_90 - iv_min_90 < 1e-8:
                            iv_rank_90, iv_pct_90 = 0.5, 0.5
                        else:
                            iv_rank_90 = float(np.clip((current_iv - iv_min_90) / (iv_max_90 - iv_min_90), 0, 1))
                            iv_pct_90 = float(np.clip((iv_arr_90 < current_iv).sum() / len(iv_arr_90), 0, 1))
                    else:
                        iv_rank_90, iv_pct_90 = iv_rank_30, iv_pct_30

                    confidence = "High" if len(iv_history) >= 252 else "Medium"
                    return iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, confidence
        except Exception:
            pass

    # --- Fall back to HV-proxy computation ---
    try:
        if hist.empty or len(hist) < 30:
            return None, None, None, None, confidence

        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

        def _iv_stats(window):
            if len(returns) < window:
                return None, None
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            if len(rolling_vol) < 2:
                return None, None
            iv_min = rolling_vol.min()
            iv_max = rolling_vol.max()
            if iv_max - iv_min <= 0:
                return 0.5, 0.5
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
            iv_percentile = (rolling_vol < current_iv).sum() / len(rolling_vol)
            return float(iv_rank), float(iv_percentile)

        iv_rank_30, iv_pct_30 = _iv_stats(30)
        iv_rank_90, iv_pct_90 = _iv_stats(90)
        return iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, confidence
    except Exception:
        return None, None, None, None, confidence

def get_short_interest(ticker: yf.Ticker) -> Optional[float]:
    """Fetch short interest (shortPercentOfFloat) from ticker info."""
    try:
        info = _get_info_cached(ticker.ticker, ticker)
        si = info.get("shortPercentOfFloat", None)
        if si is None:
            return None
        si = float(si)
        if not (0 <= si <= 200):
            return None
        if si > 1:
            si = si / 100.0
        return si
    except Exception:
        return None

def get_sector_performance(ticker_symbol: str, hist: "pd.DataFrame | None" = None) -> Dict:
    _init_yfinance()
    _init_yf_session()
    sector_etf = SECTOR_MAP.get(ticker_symbol, "SPY")
    try:
        # Derive ticker 5d return from existing history if available
        if hist is not None and len(hist) >= 6:
            tkr_ret = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1)
        else:
            tkr = yf.Ticker(ticker_symbol, session=_yf_session)
            tkr_hist = tkr.history(period="5d")
            if len(tkr_hist) < 2:
                return {}
            tkr_ret = float(tkr_hist['Close'].iloc[-1] / tkr_hist['Close'].iloc[0] - 1)

        etf = yf.Ticker(sector_etf, session=_yf_session)
        etf_hist = etf.history(period="5d")
        if len(etf_hist) < 2:
            return {}
        etf_ret = float(etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[0] - 1)

        return {
            "sector_etf": sector_etf,
            "ticker_return": tkr_ret,
            "sector_return": etf_ret
        }
    except Exception:
        return {}

@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_macro_risk() -> bool:
    _init_yfinance()
    _init_yf_session()
    try:
        eurusd = yf.Ticker("EURUSD=X", session=_yf_session)
        _yf_throttle()
        try:
            fx_hist = eurusd.history(period="1mo")
        except TypeError:
            return False
        if not fx_hist.empty and len(fx_hist) > 5:
            daily_range = (fx_hist['High'] - fx_hist['Low']) / fx_hist['Open']
            current_range = daily_range.iloc[-1]
            avg_range = daily_range.mean()
            std_range = daily_range.std()
            if current_range > (avg_range + 2 * std_range):
                return True
        return False
    except Exception:
        return False

@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_yield_spike() -> Tuple[bool, float]:
    _init_yfinance()
    _init_yf_session()
    try:
        tnx = yf.Ticker("^TNX", session=_yf_session)
        _yf_throttle()
        try:
            tnx_hist = tnx.history(period="5d")
        except TypeError:
            return False, 0.0
        if not tnx_hist.empty and len(tnx_hist) >= 2:
            tnx_close = tnx_hist['Close']
            tnx_change_pct = (tnx_close.iloc[-1] - tnx_close.iloc[-2]) / tnx_close.iloc[-2]
            return (tnx_change_pct > 0.025), tnx_change_pct
        return False, 0.0
    except Exception:
        return False, 0.0

# In-memory cache for market context (15-min TTL) — avoids re-fetching SPY/VIX/EURUSD/TNX
# on every scan within the same session
_market_context_cache: dict = {"value": None, "ts": 0.0}
_MARKET_CONTEXT_TTL = 900  # 15 minutes

def get_market_context() -> Tuple[str, str, bool, float]:
    import time as _time
    now = _time.time()
    if _market_context_cache["value"] is not None and now - _market_context_cache["ts"] < _MARKET_CONTEXT_TTL:
        return _market_context_cache["value"]

    _init_yfinance()  # Lazy init yfinance on first use
    _init_yf_session()  # Lazy init curl_cffi session
    _init_request_cache()  # Lazy init: first network call triggers cache setup
    market_trend = "Unknown"
    volatility_regime = "Unknown"
    macro_risk_active = False
    tnx_change_pct = 0.0

    # Fetch SPY, VIX, EURUSD, TNX in parallel to cut wall-clock time ~4x.
    # Each inner call is throttled via _yf_throttle() and guards yfinance's
    # NoneType-subscript bug on rate-limited responses. Each .result() has a
    # hard wall-clock timeout so startup can never hang indefinitely.
    def _fetch_spy():
        try:
            spy = yf.Ticker("SPY", session=_yf_session)
            _yf_throttle()
            try:
                return spy.history(period="3mo")
            except TypeError:
                return None
        except Exception:
            return None

    def _fetch_vix():
        try:
            vix = yf.Ticker("^VIX", session=_yf_session)
            _yf_throttle()
            try:
                return vix.history(period="5d")
            except TypeError:
                return None
        except Exception:
            return None

    def _safe_result(fut, timeout, default):
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return default

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            spy_fut = pool.submit(_fetch_spy)
            vix_fut = pool.submit(_fetch_vix)
            macro_fut = pool.submit(check_macro_risk)
            yield_fut = pool.submit(check_yield_spike)

            spy_hist = _safe_result(spy_fut, 20, None)
            if spy_hist is not None and not spy_hist.empty:
                spy_sma50 = spy_hist['Close'].rolling(window=50).mean().iloc[-1]
                spy_current = spy_hist['Close'].iloc[-1]
                market_trend = "Bull" if spy_current > spy_sma50 else "Bear"

            vix_hist = _safe_result(vix_fut, 15, None)
            if vix_hist is not None and not vix_hist.empty:
                vix_current = vix_hist['Close'].iloc[-1]
                volatility_regime = "High" if vix_current > 20 else "Low"

            macro_risk_active = _safe_result(macro_fut, 15, False)
            _, tnx_change_pct = _safe_result(yield_fut, 15, (False, 0.0))

        result = (market_trend, volatility_regime, macro_risk_active, tnx_change_pct)
        _market_context_cache["value"] = result
        _market_context_cache["ts"] = now
        return result
    except Exception:
        return "Unknown", "Unknown", False, 0.0


def _process_option_chain(tkr: yf.Ticker, symbol: str, exp: str) -> List[pd.DataFrame]:
    try:
        # Wrap ticker.option_chain in try/except to handle invalid/empty expirations gracefully
        _yf_throttle()
        oc = tkr.option_chain(exp)
    except Exception as e:
        logger.debug("Could not fetch options for %s on %s: %s", symbol, exp, e)
        return []

    sub_frames = []
    for opt_type, df in [("call", oc.calls), ("put", oc.puts)]:
        if df is None or df.empty:
            continue
        sub = df.copy()
        sub["type"] = opt_type
        sub["expiration"] = exp
        sub["symbol"] = symbol.upper()
        for col in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
            if col not in sub.columns:
                sub[col] = pd.NA
        sub_frames.append(sub)

    if not sub_frames:
        return []

    # Max pain requires both calls AND puts — compute on the combined chain
    combined = pd.concat(sub_frames, ignore_index=True)
    max_pain_val = calculate_max_pain(combined)
    for sub in sub_frames:
        sub["max_pain"] = max_pain_val

    return sub_frames

def calculate_implied_earnings_move(tkr: yf.Ticker, earnings_date: Optional[datetime], df_chain: pd.DataFrame, underlying: float) -> Optional[dict]:
    """
    Compare the market-implied earnings move (ATM straddle / underlying)
    with the stock's historical earnings actual moves (last 8 quarters).
    Returns dict with implied_move_pct, hist_avg_move, hist_beat_rate, is_cheap.
    """
    if earnings_date is None or df_chain.empty or underlying <= 0:
        return None
    try:
        df_chain = df_chain.copy()
        df_chain["exp_dt"] = pd.to_datetime(df_chain["expiration"], errors="coerce", utc=True)
        try:
            earnings_dt = pd.Timestamp(earnings_date).tz_localize("UTC") if earnings_date.tzinfo is None else pd.Timestamp(earnings_date).tz_convert("UTC")
        except Exception:
            earnings_dt = pd.Timestamp(earnings_date, tz="UTC")
        post_earnings_exps = df_chain[df_chain["exp_dt"] > earnings_dt]["expiration"].unique()
        if len(post_earnings_exps) == 0:
            return None
        nearest_exp = sorted(post_earnings_exps)[0]

        exp_df = df_chain[df_chain["expiration"] == nearest_exp].copy()
        exp_df["strike_dist"] = (exp_df["strike"] - underlying).abs()
        atm_strike = exp_df.loc[exp_df["strike_dist"].idxmin(), "strike"]

        atm_call = exp_df[(exp_df["strike"] == atm_strike) & (exp_df["type"] == "call")]
        atm_put = exp_df[(exp_df["strike"] == atm_strike) & (exp_df["type"] == "put")]

        if atm_call.empty or atm_put.empty:
            return None

        call_mid = (atm_call.iloc[0]["bid"] + atm_call.iloc[0]["ask"]) / 2
        put_mid = (atm_put.iloc[0]["bid"] + atm_put.iloc[0]["ask"]) / 2

        if call_mid <= 0 or put_mid <= 0:
            return None

        straddle = call_mid + put_mid

        # OTM strangle blending for a more accurate implied move estimate
        strangle = None
        try:
            sorted_strikes = sorted(exp_df["strike"].unique())
            atm_idx = sorted_strikes.index(atm_strike) if atm_strike in sorted_strikes else None
            if atm_idx is not None:
                otm_call_strike = sorted_strikes[atm_idx + 1] if atm_idx + 1 < len(sorted_strikes) else None
                otm_put_strike = sorted_strikes[atm_idx - 1] if atm_idx - 1 >= 0 else None
                if otm_call_strike and otm_put_strike:
                    otm_call = exp_df[(exp_df["strike"] == otm_call_strike) & (exp_df["type"] == "call")]
                    otm_put = exp_df[(exp_df["strike"] == otm_put_strike) & (exp_df["type"] == "put")]
                    if not otm_call.empty and not otm_put.empty:
                        oc_mid = (otm_call.iloc[0]["bid"] + otm_call.iloc[0]["ask"]) / 2
                        op_mid = (otm_put.iloc[0]["bid"] + otm_put.iloc[0]["ask"]) / 2
                        if oc_mid > 0 and op_mid > 0:
                            strangle = oc_mid + op_mid
        except Exception:
            strangle = None

        if strangle is not None:
            implied_move_pct = ((straddle + strangle) / 2) / underlying
        else:
            implied_move_pct = straddle / underlying

        # Get ATM IV for crush prediction
        atm_iv_val = None
        try:
            call_iv = float(atm_call.iloc[0].get("impliedVolatility", 0) or 0)
            put_iv = float(atm_put.iloc[0].get("impliedVolatility", 0) or 0)
            if call_iv > 0 and put_iv > 0:
                atm_iv_val = (call_iv + put_iv) / 2
            elif call_iv > 0:
                atm_iv_val = call_iv
            elif put_iv > 0:
                atm_iv_val = put_iv
        except Exception:
            pass

        hist_moves = []
        predicted_iv_crush = None
        crush_confidence = ""
        try:
            earn_dates = tkr.get_earnings_dates(limit=16)
            if earn_dates is not None and not earn_dates.empty:
                earn_dates = earn_dates.dropna(subset=["EPS Actual"])
                hist = tkr.history(period="3y", interval="1d")
                if not hist.empty:
                    for earn_dt in earn_dates.index[:8]:
                        earn_date_loc = hist.index.searchsorted(earn_dt)
                        if earn_date_loc > 0 and earn_date_loc < len(hist):
                            pre_price = float(hist["Close"].iloc[earn_date_loc - 1])
                            post_price = float(hist["Close"].iloc[min(earn_date_loc + 1, len(hist) - 1)])
                            if pre_price > 0:
                                hist_moves.append(abs(post_price / pre_price - 1.0))
                    # IV crush prediction using 3y history
                    if atm_iv_val and atm_iv_val > 0 and len(hist) >= 60:
                        crush_confidence = "high" if len(hist) >= 252 else "medium"
                        post_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                        if len(post_returns) >= 30:
                            post_event_hv = float(post_returns.iloc[-30:].std() * np.sqrt(252))
                            predicted_iv_crush = max(0.0, atm_iv_val - post_event_hv)
        except Exception:
            pass

        if not hist_moves:
            return {
                "implied_move_pct": implied_move_pct,
                "hist_avg_move": None,
                "hist_beat_rate": None,
                "is_cheap": None,
                "predicted_iv_crush": predicted_iv_crush,
                "crush_confidence": crush_confidence,
            }

        hist_avg = sum(hist_moves) / len(hist_moves)
        beat_rate = sum(1 for m in hist_moves if m > implied_move_pct) / len(hist_moves)
        is_cheap = hist_avg > implied_move_pct

        return {
            "implied_move_pct": implied_move_pct,
            "hist_avg_move": hist_avg,
            "hist_beat_rate": beat_rate,
            "is_cheap": is_cheap,
            "predicted_iv_crush": predicted_iv_crush,
            "crush_confidence": crush_confidence,
        }
    except Exception:
        return None


@retry_with_backoff(retries=3, backoff_in_seconds=3, error_types=(Exception,))
def fetch_options_yfinance(symbol: str, max_expiries: int) -> Dict:
    """
    Fetch options data and all related context using a Single-Fetch architecture.
    Returns a dictionary with 'df' (options chain) and 'context' (all derived metrics).
    """
    _init_yfinance()  # Lazy init yfinance on first use
    _init_yf_session()  # Lazy init curl_cffi session
    _init_request_cache()  # Lazy init: first fetch triggers cache setup, not import

    if symbol in _CHAIN_CACHE:
        return _CHAIN_CACHE[symbol]

    try:
        tkr = yf.Ticker(symbol, session=_yf_session)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to initialize ticker {symbol}: {e}")

    # 1. Fetch History ONCE (1 year daily)
    try:
        _yf_throttle()
        hist = tkr.history(period="1y", interval="1d")
    except TypeError as e:
        # yfinance bug: on rate-limited/empty responses it hits `data['chart']` where data is None.
        # Convert to an explicit rate-limit error so the retry decorator's cooldown kicks in.
        if "subscriptable" in str(e).lower():
            raise RuntimeError(f"Rate limited while fetching history for {symbol}") from e
        raise
    except (ValueError, KeyError, URLError) as e:
        logging.warning(f"Failed to fetch history for {symbol}: {e}")
        hist = pd.DataFrame()

    if hist is None or hist.empty:
        raise RuntimeError(f"Could not fetch price history for {symbol}")

    # 2. Derive Metrics from History
    underlying = safe_float(hist["Close"].iloc[-1])
    hv_30d_rolling = calculate_historical_volatility(hist, period=30)
    hv_ewma = calculate_ewma_volatility(hist, span=20)
    hv_parkinson = calculate_parkinson_volatility(hist, period=30)
    # Blend rolling, EWMA, and Parkinson vol: rolling gives stability, EWMA gives recency, Parkinson captures intraday range
    if hv_30d_rolling and hv_ewma and hv_parkinson:
        hv_30d = 0.34 * hv_30d_rolling + 0.33 * hv_ewma + 0.33 * hv_parkinson
    elif hv_30d_rolling and hv_ewma:
        hv_30d = 0.5 * hv_30d_rolling + 0.5 * hv_ewma
    else:
        hv_30d = hv_30d_rolling or hv_ewma
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50, adx_14, bb_width_pct = calculate_momentum_indicators(hist)
    rvol = calculate_rvol(hist)
    vwap, fib_50, fib_618 = calculate_technical_levels(hist)

    # 3. Fetch Other Data (Earnings, Sentiment, Seasonality) — PARALLEL
    _aux_results: Dict[str, Any] = {}
    def _aux(name, fn, *args, **kwargs):
        try:
            return name, fn(*args, **kwargs)
        except Exception:
            return name, None

    # aux_pool=3 fast mode: global throttle still caps aggregate rate, parallelism just hides latency.
    with ThreadPoolExecutor(max_workers=3) as aux_pool:
        aux_futures = [
            aux_pool.submit(_aux, "earnings", get_next_earnings_date, tkr),
            aux_pool.submit(_aux, "sentiment", get_sentiment, tkr),
            aux_pool.submit(_aux, "news", get_news_headlines, tkr),
            aux_pool.submit(_aux, "seasonality", check_seasonality, tkr),
            aux_pool.submit(_aux, "sector", get_sector_performance, symbol, hist),
            aux_pool.submit(_aux, "short_interest", get_short_interest, tkr),
        ]
        for fut in as_completed(aux_futures):
            name, val = fut.result()
            _aux_results[name] = val

    earnings_date = _aux_results.get("earnings")
    sentiment_score = _aux_results.get("sentiment")
    news_headlines = _aux_results.get("news") or []
    seasonal_win_rate = _aux_results.get("seasonality")
    sector_perf = _aux_results.get("sector") or {}
    short_interest = _aux_results.get("short_interest")

    # Rich news + analyst data (uses cached news from _get_news_cached)
    news_data = None
    if _HAS_NEWS_FETCHER:
        try:
            news_data = fetch_news_and_events(symbol, ticker_obj=tkr, max_age_hours=72, max_headlines=5)
            if news_data and news_data.top_headlines:
                news_headlines = news_data.top_headlines
        except Exception:
            pass

    # Next ex-dividend date
    next_ex_div = None
    try:
        divs = tkr.dividends
        if divs is not None and not divs.empty:
            future_divs = divs[divs.index > pd.Timestamp.now(tz='UTC')]
            next_ex_div = future_divs.index[0].date() if not future_divs.empty else None
        else:
            next_ex_div = None
    except Exception:
        next_ex_div = None

    # 4. Fetch Options Chains
    try:
        _yf_throttle()
        expirations = tkr.options
    except TypeError as e:
        if "subscriptable" in str(e).lower():
            raise RuntimeError(f"Rate limited while fetching expirations for {symbol}") from e
        raise
    except (ValueError, KeyError, AttributeError) as e:
        raise RuntimeError(f"Failed to fetch options expirations for {symbol}: {e}")

    if not expirations:
        raise RuntimeError(f"No options expirations available for {symbol}.")

    num_expiries_to_fetch = max_expiries
    if max_expiries == 1 and len(expirations) > 1:
        num_expiries_to_fetch = 2
    expirations_to_scan = expirations[:num_expiries_to_fetch]

    # Serial expiration fetch: relies on global _yf_throttle inside _process_option_chain.
    # Was ThreadPoolExecutor(max_workers=4) — combined with outer workers=3 that gave 12-way
    # concurrency per symbol and triggered 429 storms.
    frames = []
    for _exp in expirations_to_scan:
        try:
            frames.extend(_process_option_chain(tkr, symbol, _exp))
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"No options data frames fetched for {symbol}.")

    df = pd.concat(frames, ignore_index=True)

    # Option RVOL: contract-level volume relative to OI-normalized baseline
    _oi_vals = pd.to_numeric(df["openInterest"], errors="coerce").fillna(1)
    _vol_vals = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    _baseline = (_oi_vals / 30.0).clip(lower=1.0)
    df["option_rvol"] = (_vol_vals / _baseline).clip(upper=50.0)

    # Max pain calculation
    max_pain_strike = calculate_max_pain(df, underlying)

    # Compute earnings move analysis now that chain is available
    earnings_move_data = calculate_implied_earnings_move(tkr, earnings_date, df, underlying)

    # 5. Enrich DataFrame with Context
    df["underlying"] = underlying
    df["hv_30d"] = hv_30d
    df["max_pain_strike"] = max_pain_strike
    df["ret_5d"] = ret_5d
    df["rsi_14"] = rsi_14
    df["adx_14"] = adx_14
    df["atr_trend"] = atr_trend
    df["sma_20"] = sma_20
    df["sma_50"] = sma_50
    df["high_20"] = high_20
    df["low_20"] = low_20
    df["sentiment_score"] = sentiment_score
    df["seasonal_win_rate"] = seasonal_win_rate
    df["is_squeezing"] = is_squeezing
    df["bb_width_pct"] = bb_width_pct
    df["rvol"] = rvol
    df["short_interest"] = short_interest
    df["vwap"] = vwap
    df["fib_50"] = fib_50
    df["fib_618"] = fib_618

    # IV Rank Calculation
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)

    # VRP: Volatility Risk Premium computation
    _vrp_iv = float(median_iv) if pd.notna(median_iv) and median_iv > 0 else (hv_30d or 0.25)
    vrp_data = calculate_vrp(hist, _vrp_iv)

    iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = None, None, None, None
    iv_confidence = "Low"
    if pd.notna(median_iv) and median_iv > 0:
        try:
            _iv_db = _get_iv_db_path()
            _iv_date = datetime.now().strftime("%Y-%m-%d")
            _upsert_iv(symbol, _iv_date, float(median_iv), _iv_db)
        except Exception as _uiv_exc:
            logger.warning("IV upsert failed for %s: %s", symbol, _uiv_exc)
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, iv_confidence = get_iv_rank_percentile_from_history(
            hist, median_iv, ticker=symbol
        )

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90
    df["iv_confidence"] = iv_confidence

    # IV velocity and trend: compare 30-day vs 90-day IV percentile
    iv_trend = "stable"
    iv_velocity = 0.0
    if iv_pct_30 is not None and iv_pct_90 is not None:
        diff = float(iv_pct_30) - float(iv_pct_90)
        if diff > 0.05:
            iv_trend = "expanding"
            iv_velocity = diff
        elif diff < -0.05:
            iv_trend = "contracting"
            iv_velocity = diff
    df["iv_trend"] = iv_trend
    df["iv_velocity"] = iv_velocity

    # Dividend yield from ticker.info
    dividend_yield = 0.0
    try:
        _info = _get_info_cached(symbol, tkr)
        _dy = _info.get("dividendYield", 0)
        dividend_yield = float(_dy) if _dy else 0.0
        # yfinance returns dividendYield as percentage (e.g. 3.13 for 3.13%)
        # BS formulas expect decimal (0.0313). Convert if > 0.20 (no stock has 20%+ yield).
        if dividend_yield > 0.20:
            dividend_yield /= 100.0
        dividend_yield = min(dividend_yield, 0.15)  # hard clamp for data anomalies
    except Exception:
        dividend_yield = 0.0
    df["dividend_yield"] = dividend_yield

    # Term Structure
    term_structure_spread = None
    if len(expirations_to_scan) >= 2 and 'expiration' in df.columns:
        try:
            df['exp_dt'] = pd.to_datetime(df['expiration'], errors='coerce', utc=True)
            sorted_expiries = sorted(df['exp_dt'].dropna().unique())
            if len(sorted_expiries) >= 2:
                front = df[df['exp_dt'] == sorted_expiries[0]]
                back = df[df['exp_dt'] == sorted_expiries[1]]
                
                def get_atm_iv(d):
                    d['dist'] = (d['strike'] - underlying).abs()
                    strike = d.loc[d['dist'].idxmin()]['strike']
                    ivs = d[d['strike'] == strike]['impliedVolatility'].dropna()
                    return ivs.mean() if not ivs.empty else None
                
                f_iv = get_atm_iv(front.copy())
                b_iv = get_atm_iv(back.copy())
                
                if f_iv and b_iv:
                    term_structure_spread = b_iv - f_iv
        except Exception:
            pass

    # IV skew percentile rank
    try:
        if "iv_skew" in df.columns:
            avg_skew = float(df["iv_skew"].replace(0.0, np.nan).median(skipna=True))
            if not math.isnan(avg_skew):
                skew_pctile = get_skew_percentile(symbol, avg_skew)
                df["iv_skew_rank"] = skew_pctile
            else:
                df["iv_skew_rank"] = 0.5
        else:
            df["iv_skew_rank"] = 0.5
    except Exception:
        df["iv_skew_rank"] = 0.5

    # ── Polygon.io enrichment (opt-in, silent no-op when key absent) ────────────
    unusual_options_flow = None
    company_description = None
    market_cap = None
    try:
        from src.polygon_client import PolygonClient
        _poly = PolygonClient()
        if _poly._enabled:
            logger.debug("Polygon enrichment: fetching data for %s", symbol)

            # VWAP override from Polygon prev-close (snapshot requires higher plan tier)
            prev = _poly.get_prev_close(symbol)
            if prev:
                try:
                    poly_vwap = prev.get("vw")
                    if poly_vwap and float(poly_vwap) > 0:
                        vwap = float(poly_vwap)
                        logger.debug("Polygon enrichment: VWAP overridden to %.4f for %s", vwap, symbol)
                except Exception:
                    pass

            # Unusual options flow (requires options plan — silent no-op if 403)
            import time as _time
            _time.sleep(0.2)
            poly_cfg = {}
            try:
                from src.config_ai import AI_CONFIG
                poly_cfg = AI_CONFIG.get("polygon", {})
            except Exception:
                pass
            min_prem = poly_cfg.get("unusual_flow_min_premium", 10_000)
            unusual_options_flow = _poly.get_unusual_options_flow(symbol, min_premium=min_prem)

            # Company description and market cap
            _time.sleep(0.2)
            details = _poly.get_ticker_details(symbol)
            if details:
                company_description = details.get("description", "")
                raw_mc = details.get("market_cap")
                if raw_mc:
                    try:
                        market_cap = float(raw_mc)
                    except Exception:
                        pass
    except Exception as _poly_exc:
        logger.debug("Polygon enrichment failed for %s: %s", symbol, _poly_exc)

    # Return structured result
    result = {
        "df": df,
        "history_df": hist,
        "context": {
            "hv": hv_30d,
            "bb_width_pct": bb_width_pct,
            "hv_ewma": hv_ewma,
            "hv_parkinson": hv_parkinson,
            "iv_rank": iv_rank_30,
            "iv_percentile": iv_pct_30,
            "earnings_date": earnings_date,
            "earnings_move_data": earnings_move_data,
            "sentiment_score": sentiment_score,
            "news_headlines": news_headlines,
            "seasonal_win_rate": seasonal_win_rate,
            "term_structure_spread": term_structure_spread,
            "sector_perf": sector_perf,
            "rvol": rvol,
            "short_interest": short_interest,
            "next_ex_div": next_ex_div,
            "vwap": vwap,
            "fib_50": fib_50,
            "fib_618": fib_618,
            "news_data": news_data,
            "iv_confidence": iv_confidence,
            "unusual_options_flow": unusual_options_flow,
            "company_description": company_description,
            "market_cap": market_cap,
            "iv_skew_rank": float(df["iv_skew_rank"].iloc[0]) if "iv_skew_rank" in df.columns and not df.empty else 0.5,
            "vrp_data": vrp_data,
            "max_pain_strike": max_pain_strike,
            "iv_trend": iv_trend,
            "iv_velocity": iv_velocity,
            "dividend_yield": dividend_yield,
        }
    }
    _CHAIN_CACHE[symbol] = result
    _FETCH_TIMESTAMPS[symbol] = datetime.now()
    return result


# ---------------------------------------------------------------------------
# Historical IV crush estimation
# ---------------------------------------------------------------------------

def get_historical_iv_crush(ticker: str, n_quarters: int = 8, iv_db_path: str = "iv_cache.db") -> Dict[str, Any]:
    """
    Estimate historical IV crush around earnings dates.

    Uses yfinance earnings dates and attempts to compute IV change pre vs post earnings.
    Falls back to historical realized move as a proxy if IV data not available.

    Returns dict with keys:
        avg_crush: float (mean IV crush as fraction, e.g. 0.35 = 35% crush)
        std_crush: float (std dev of crush)
        n_events: int (number of earnings events analyzed)
        crushes: list of individual crush values
    Returns empty dict if insufficient data.
    """
    try:
        # ------------------------------------------------------------------
        # Check cache first
        # ------------------------------------------------------------------
        try:
            with closing(sqlite3.connect(iv_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS iv_crush_cache "
                    "(ticker TEXT PRIMARY KEY, avg_crush REAL, std_crush REAL, "
                    "n_events INT, updated TEXT)"
                )
                row = conn.execute(
                    "SELECT avg_crush, std_crush, n_events, updated "
                    "FROM iv_crush_cache WHERE ticker = ?",
                    (ticker.upper(),),
                ).fetchone()
            if row is not None:
                updated = datetime.fromisoformat(row[3])
                if (datetime.now() - updated).days < 7:
                    return {
                        "avg_crush": row[0],
                        "std_crush": row[1],
                        "n_events": row[2],
                        "crushes": [],  # individual values not cached
                    }
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Fetch earnings dates
        # ------------------------------------------------------------------
        _init_yfinance()
        _init_yf_session()
        tkr = yf.Ticker(ticker, session=_yf_session)
        earnings_dates = None
        try:
            ed = tkr.earnings_dates
            if ed is not None and not ed.empty:
                earnings_dates = ed.index.tolist()
        except Exception:
            pass

        if earnings_dates is None:
            try:
                cal = tkr.calendar
                if cal is not None:
                    if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                        earnings_dates = cal["Earnings Date"].tolist()
                    elif isinstance(cal, dict) and "Earnings Date" in cal:
                        val = cal["Earnings Date"]
                        earnings_dates = val if isinstance(val, list) else [val]
            except Exception:
                pass

        if not earnings_dates:
            return {}

        # Keep only past dates, limit to n_quarters
        now = datetime.now(timezone.utc)
        past_dates = []
        for d in earnings_dates:
            try:
                dt = pd.Timestamp(d)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                if dt < now:
                    past_dates.append(dt)
            except Exception:
                continue

        past_dates = sorted(past_dates, reverse=True)[:n_quarters]

        if len(past_dates) < 3:
            return {}

        # ------------------------------------------------------------------
        # Compute realized move around each earnings date as crush proxy
        # ------------------------------------------------------------------
        hist = tkr.history(period="3y", auto_adjust=True)
        if hist is None or hist.empty:
            return {}

        if hist.index.tzinfo is None:
            hist.index = hist.index.tz_localize("UTC")

        crushes: list = []
        for ed in past_dates:
            try:
                # Find the nearest trading day on or after the earnings date
                mask = hist.index >= (ed - timedelta(days=1))
                nearby = hist.loc[mask].head(5)
                if len(nearby) < 2:
                    continue

                # 1-day return around the event
                close_before = nearby.iloc[0]["Close"]
                close_after = nearby.iloc[1]["Close"]
                if close_before <= 0:
                    continue

                realized_move = abs((close_after - close_before) / close_before)
                # IV crush estimate: IV typically drops ~1.5x the realized move for ATM
                crush = realized_move * 1.5
                crushes.append(crush)
            except Exception:
                continue

        if len(crushes) < 3:
            return {}

        arr = np.array(crushes)
        avg_crush = float(np.mean(arr))
        std_crush = float(np.std(arr, ddof=1)) if len(crushes) > 1 else 0.0

        # ------------------------------------------------------------------
        # Cache the result
        # ------------------------------------------------------------------
        try:
            with closing(sqlite3.connect(iv_db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS iv_crush_cache "
                    "(ticker TEXT PRIMARY KEY, avg_crush REAL, std_crush REAL, "
                    "n_events INT, updated TEXT)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO iv_crush_cache "
                    "(ticker, avg_crush, std_crush, n_events, updated) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ticker.upper(), avg_crush, std_crush, len(crushes),
                     datetime.now().isoformat()),
                )
                conn.commit()
        except Exception:
            pass

        return {
            "avg_crush": avg_crush,
            "std_crush": std_crush,
            "n_events": len(crushes),
            "crushes": crushes,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Async batch pipeline
# ---------------------------------------------------------------------------

async def batch_fetch_async(
    symbols: List[str],
    max_concurrent: int = 20,
    provider: Optional[BaseDataProvider] = None,
) -> Dict[str, Any]:
    """
    Concurrently fetch option chains for up to max_concurrent symbols at once.

    Uses asyncio.to_thread to avoid blocking the event loop while yfinance
    makes synchronous HTTP calls.  Each symbol result is a dict returned by
    the provider's fetch_chain method, or {"error": str} on failure.
    """
    import asyncio
    _provider = provider or _get_data_provider()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(sym: str):
        async with semaphore:
            try:
                return sym, await asyncio.to_thread(_provider.fetch_chain, sym)
            except Exception as exc:
                return sym, {"error": str(exc)}

    tasks = [_fetch_one(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return dict(results)


def batch_fetch(
    symbols: List[str],
    max_concurrent: int = 20,
    provider: Optional[BaseDataProvider] = None,
) -> Dict[str, Any]:
    """
    Synchronous entry point for the async batch fetcher.

    Safe to call from both synchronous code and from within a running event
    loop (e.g. Streamlit / Jupyter).
    """
    import asyncio

    async def _run():
        return await batch_fetch_async(symbols, max_concurrent, provider)

    try:
        # get_running_loop() raises RuntimeError if no loop is running (Python 3.7+)
        asyncio.get_running_loop()
        # A loop IS running (Streamlit, Jupyter, etc.) — run in a worker thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        # No running loop — safe to create one
        return asyncio.run(_run())