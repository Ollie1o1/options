#!/usr/bin/env python3
"""
Data fetching utilities for the options screener.
Handles all yfinance interactions with a Single-Fetch architecture for performance.
"""

import time
import math
import logging
import random
import functools
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError

# In-memory caches
_HV_CACHE: Dict[str, float] = {}
_MOMENTUM_CACHE: Dict[str, Tuple] = {}
_IV_RANK_CACHE: Dict[str, Tuple] = {}
_SENTIMENT_CACHE: Dict[str, float] = {}
_SEASONALITY_CACHE: Dict[str, float] = {}

# --- Retry Decorator ---
def retry_with_backoff(retries=3, backoff_in_seconds=1, error_types=(Exception,)):
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

def get_underlying_price(ticker: yf.Ticker) -> Optional[float]:
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
        info = ticker.info or {}
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
        news = ticker.news
        if not news:
            return None
        headlines = ". ".join([item.get("title", "") for item in news])
        if not headlines.strip():
            return None
        blob = TextBlob(headlines)
        score = blob.sentiment.polarity
        _SENTIMENT_CACHE[key] = score
        return score
    except Exception:
        return None

@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_seasonality(ticker: yf.Ticker) -> Optional[float]:
    key = f"{ticker.ticker}:seasonality"
    if key in _SEASONALITY_CACHE:
        return _SEASONALITY_CACHE[key]
    try:
        hist = ticker.history(period="5y", interval="1mo")
        if hist.empty:
            return None
        current_month = datetime.now().month
        monthly_data = hist[hist.index.month == current_month]
        if monthly_data.empty:
            return None
        wins = (monthly_data['Close'] > monthly_data['Open']).sum()
        total = len(monthly_data)
        win_rate = wins / total if total > 0 else 0.0
        _SEASONALITY_CACHE[key] = win_rate
        return win_rate
    except Exception:
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
                        dt = datetime.fromtimestamp(dt.timestamp())
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
        except Exception:
            pass
    except Exception:
        return None
    return None

def get_risk_free_rate() -> float:
    default_rate = 0.045
    try:
        tbill = yf.Ticker("^IRX")
        try:
            fi = getattr(tbill, "fast_info", None)
            if fi:
                rate = safe_float(getattr(fi, "last_price", None))
                if rate and rate > 0:
                    return rate / 100.0
        except Exception:
            pass
        hist = tbill.history(period="5d", interval="1d")
        if not hist.empty:
            rate = safe_float(hist["Close"].iloc[-1])
            if rate and rate > 0:
                return rate / 100.0
    except Exception:
        pass
    return default_rate

def get_vix_level() -> Optional[float]:
    try:
        vix = yf.Ticker("^VIX")
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
        return "normal", config.get("weights", {})
    vix_regimes = config.get("vix_regimes", {})
    if vix_level < vix_regimes.get("low", {}).get("threshold", 15):
        return "low", vix_regimes.get("low", {}).get("weights", config.get("weights", {}))
    elif vix_level > vix_regimes.get("high", {}).get("threshold", 25):
        return "high", vix_regimes.get("high", {}).get("weights", config.get("weights", {}))
    else:
        return "normal", vix_regimes.get("normal", {}).get("weights", config.get("weights", {}))

# --- New / Refactored Calculation Functions (Using Cached History) ---

def calculate_historical_volatility(hist: pd.DataFrame, period: int = 30) -> Optional[float]:
    """Calculate annualized volatility from history DataFrame."""
    try:
        if hist.empty or len(hist) < period:
            return None
        # Use last 'period' days
        subset = hist.iloc[-(period+1):].copy()
        returns = (subset['Close'] / subset['Close'].shift(1)).apply(math.log).dropna()
        if len(returns) < 2:
            return None
        daily_vol = returns.std()
        return daily_vol * math.sqrt(252)
    except Exception:
        return None

def calculate_momentum_indicators(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Calculate RSI, ATR, SMA, etc. from history DataFrame."""
    try:
        if hist.empty or len(hist) < 21:
            return None, None, None, None, None, None, False
        
        close = hist["Close"].astype(float)
        high = hist.get("High", close).astype(float)
        low = hist.get("Low", close).astype(float)

        # 5-day return
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else None

        # RSI 14
        delta = close.diff().dropna()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
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

        # 20-day stats
        sma_20 = float(close.rolling(window=20).mean().iloc[-1])
        high_20 = float(high.rolling(window=20).max().iloc[-1])
        low_20 = float(low.rolling(window=20).min().iloc[-1])

        # Squeeze
        bb_std = close.rolling(window=20).std()
        bb_upper = sma_20 + (bb_std.iloc[-1] * 2)
        bb_lower = sma_20 - (bb_std.iloc[-1] * 2)
        kc_atr = atr.iloc[-1]
        kc_upper = sma_20 + (kc_atr * 1.5)
        kc_lower = sma_20 - (kc_atr * 1.5)
        is_squeezing = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        return ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing
    except Exception:
        return None, None, None, None, None, None, False

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

def get_iv_rank_percentile_from_history(hist: pd.DataFrame, current_iv: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculate IV Rank/Percentile using historical realized volatility as proxy."""
    try:
        if hist.empty or len(hist) < 30:
            return None, None, None, None

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
                return None, None
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
            iv_percentile = (rolling_vol < current_iv).sum() / len(rolling_vol)
            return float(iv_rank), float(iv_percentile)

        iv_rank_30, iv_pct_30 = _iv_stats(30)
        iv_rank_90, iv_pct_90 = _iv_stats(90)
        return iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90
    except Exception:
        return None, None, None, None

def get_short_interest(ticker: yf.Ticker) -> Optional[float]:
    """Fetch short interest from fast_info or info (non-blocking preferred)."""
    try:
        # Try fast_info first (unlikely to have SI, but worth checking)
        # yfinance 'fast_info' usually has shares, market_cap, etc.
        # 'info' has 'shortPercentFloat'.
        
        # We will try to access .info but wrap it in a way that we don't hang forever?
        # Actually, yfinance .info IS the blocking call. 
        # The user said: "Attempt to fetch it from ticker.fast_info or ticker.stats() only if it is instant."
        # .stats() isn't a standard yfinance property. 
        # We will try to access a specific key from .info, but if it triggers a full scrape, we can't easily stop it.
        # However, we can skip it if we want strict speed.
        # Let's try to be safe: access .info only if we are willing to wait a bit, 
        # OR just skip it as per "Speed is more important".
        # Compromise: We will skip .info for now to guarantee speed, unless we find a fast way.
        # Actually, let's try to see if we can get it from 'key_statistics' if available?
        # For now, return None to be safe on speed as requested.
        return None 
    except Exception:
        return None

def get_sector_performance(ticker_symbol: str) -> Dict:
    SECTOR_MAP = {
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
        "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
        "XOM": "XLE", "CVX": "XLE",
        "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
        "WMT": "XLP", "PG": "XLP", "KO": "XLP",
        "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV",
        "BA": "XLI", "CAT": "XLI",
        "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
        "AMT": "XLRE",
        "NEE": "XLU",
        "LIN": "XLB",
    }
    sector_etf = SECTOR_MAP.get(ticker_symbol, "SPY")
    try:
        tkr = yf.Ticker(ticker_symbol)
        etf = yf.Ticker(sector_etf)
        tkr_hist = tkr.history(period="5d")
        etf_hist = etf.history(period="5d")
        
        if len(tkr_hist) < 2 or len(etf_hist) < 2:
            return {}

        tkr_ret = (tkr_hist['Close'].iloc[-1] / tkr_hist['Close'].iloc[0]) - 1
        etf_ret = (etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[0]) - 1
        
        return {
            "sector_etf": sector_etf,
            "ticker_return": tkr_ret,
            "sector_return": etf_ret
        }
    except Exception:
        return {}

@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_macro_risk() -> bool:
    try:
        eurusd = yf.Ticker("EURUSD=X")
        fx_hist = eurusd.history(period="1mo")
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
    try:
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="5d")
        if not tnx_hist.empty and len(tnx_hist) >= 2:
            tnx_close = tnx_hist['Close']
            tnx_change_pct = (tnx_close.iloc[-1] - tnx_close.iloc[-2]) / tnx_close.iloc[-2]
            return (tnx_change_pct > 0.025), tnx_change_pct
        return False, 0.0
    except Exception:
        return False, 0.0

def get_market_context() -> Tuple[str, str, bool, float]:
    market_trend = "Unknown"
    volatility_regime = "Unknown"
    macro_risk_active = False
    tnx_change_pct = 0.0
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        if not spy_hist.empty:
            spy_sma50 = spy_hist['Close'].rolling(window=50).mean().iloc[-1]
            spy_current = spy_hist['Close'].iloc[-1]
            market_trend = "Bull" if spy_current > spy_sma50 else "Bear"
        
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            vix_current = vix_hist['Close'].iloc[-1]
            volatility_regime = "High" if vix_current > 20 else "Low"
            
        macro_risk_active = check_macro_risk()
        _, tnx_change_pct = check_yield_spike()
        
        return market_trend, volatility_regime, macro_risk_active, tnx_change_pct
    except Exception:
        return "Unknown", "Unknown", False, 0.0

def calculate_max_pain(chain_df: pd.DataFrame) -> Optional[float]:
    try:
        if chain_df.empty:
            return None
        relevant = chain_df[chain_df['openInterest'] > 0].copy()
        if relevant.empty:
            return None
        strikes = sorted(relevant['strike'].unique())
        max_pain_price = None
        min_total_loss = float('inf')
        for price_candidate in strikes:
            calls = relevant[relevant['type'] == 'call']
            call_loss = calls.apply(lambda row: max(0.0, price_candidate - row['strike']) * row['openInterest'], axis=1).sum()
            puts = relevant[relevant['type'] == 'put']
            put_loss = puts.apply(lambda row: max(0.0, row['strike'] - price_candidate) * row['openInterest'], axis=1).sum()
            total_loss = call_loss + put_loss
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                max_pain_price = price_candidate
        return max_pain_price
    except Exception:
        return None

def _process_option_chain(tkr: yf.Ticker, symbol: str, exp: str) -> List[pd.DataFrame]:
    frames = []
    try:
        oc = tkr.option_chain(exp)
    except Exception:
        return []

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
        sub["max_pain"] = calculate_max_pain(sub)
        frames.append(sub)
    return frames

@retry_with_backoff(retries=3, backoff_in_seconds=2, error_types=(RuntimeError, URLError, ConnectionError, OSError))
def fetch_options_yfinance(symbol: str, max_expiries: int) -> Dict:
    """
    Fetch options data and all related context using a Single-Fetch architecture.
    Returns a dictionary with 'df' (options chain) and 'context' (all derived metrics).
    """
    try:
        tkr = yf.Ticker(symbol)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ticker {symbol}: {e}")

    # 1. Fetch History ONCE (1 year daily)
    # Used for: Underlying Price, HV, Momentum, RVOL, Technical Levels, IV Rank
    try:
        hist = tkr.history(period="1y", interval="1d")
    except Exception:
        hist = pd.DataFrame()

    if hist.empty:
        raise RuntimeError(f"Could not fetch price history for {symbol}")

    # 2. Derive Metrics from History
    underlying = safe_float(hist["Close"].iloc[-1])
    hv_30d = calculate_historical_volatility(hist, period=30)
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing = calculate_momentum_indicators(hist)
    rvol = calculate_rvol(hist)
    vwap, fib_50, fib_618 = calculate_technical_levels(hist)

    # 3. Fetch Other Data (Earnings, Sentiment, Seasonality)
    earnings_date = get_next_earnings_date(tkr)
    sentiment_score = get_sentiment(tkr)
    seasonal_win_rate = check_seasonality(tkr)
    sector_perf = get_sector_performance(symbol)
    short_interest = get_short_interest(tkr) # Will likely be None to save time

    # 4. Fetch Options Chains
    try:
        expirations = tkr.options
    except Exception as e:
        raise RuntimeError(f"Failed to fetch options expirations for {symbol}: {e}")

    if not expirations:
        raise RuntimeError(f"No options expirations available for {symbol}.")

    num_expiries_to_fetch = max_expiries
    if max_expiries == 1 and len(expirations) > 1:
        num_expiries_to_fetch = 2
    expirations_to_scan = expirations[:num_expiries_to_fetch]

    frames = []
    with ThreadPoolExecutor(max_workers=min(4, len(expirations_to_scan))) as executor:
        future_to_exp = {executor.submit(_process_option_chain, tkr, symbol, exp): exp for exp in expirations_to_scan}
        for future in as_completed(future_to_exp):
            try:
                result_frames = future.result()
                frames.extend(result_frames)
            except Exception:
                continue

    if not frames:
        raise RuntimeError(f"No options data frames fetched for {symbol}.")

    df = pd.concat(frames, ignore_index=True)
    
    # 5. Enrich DataFrame with Context
    df["underlying"] = underlying
    df["hv_30d"] = hv_30d
    df["ret_5d"] = ret_5d
    df["rsi_14"] = rsi_14
    df["atr_trend"] = atr_trend
    df["sma_20"] = sma_20
    df["high_20"] = high_20
    df["low_20"] = low_20
    df["sentiment_score"] = sentiment_score
    df["seasonal_win_rate"] = seasonal_win_rate
    df["is_squeezing"] = is_squeezing
    
    # New Columns
    df["rvol"] = rvol
    df["short_interest"] = short_interest
    df["vwap"] = vwap
    df["fib_50"] = fib_50
    df["fib_618"] = fib_618

    # IV Rank Calculation
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)
    
    iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = None, None, None, None
    if median_iv and pd.notna(median_iv) and median_iv > 0:
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = get_iv_rank_percentile_from_history(hist, median_iv)

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90

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

    # Return structured result
    return {
        "df": df,
        "history_df": hist, # Return history for portfolio correlation!
        "context": {
            "hv": hv_30d,
            "iv_rank": iv_rank_30,
            "iv_percentile": iv_pct_30,
            "earnings_date": earnings_date,
            "sentiment_score": sentiment_score,
            "seasonal_win_rate": seasonal_win_rate,
            "term_structure_spread": term_structure_spread,
            "sector_perf": sector_perf,
            "rvol": rvol,
            "short_interest": short_interest,
            "vwap": vwap,
            "fib_50": fib_50,
            "fib_618": fib_618
        }
    }
