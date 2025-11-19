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


def get_dynamic_tickers(scan_type: str, max_tickers: int = 50) -> List[str]:
    """
    Fetches a list of tickers from Finviz based on a given scan type.
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
        df = fperformance.screener_view(order=order, limit=max_tickers)

        if df.empty:
            raise RuntimeError("No tickers found matching the criteria.")

        tickers = df['Ticker'].tolist()
        return tickers
    except Exception as e:
        raise RuntimeError(f"Could not fetch '{scan_type}' from Finviz: {e}")


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

# Simple in-process caches to avoid refetching HV/IV/momentum for the same ticker
_HV_CACHE: Dict[str, float] = {}
_MOMENTUM_CACHE: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
_IV_RANK_CACHE: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}

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
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Cache for sentiment scores to avoid re-fetching
_SENTIMENT_CACHE: Dict[str, float] = {}


def get_sentiment(ticker: yf.Ticker) -> Optional[float]:
    """
    Fetches news headlines and calculates a sentiment polarity score using TextBlob.
    """
    if not HAS_TEXTBLOB:
        return None

    key = f"{ticker.ticker}:sentiment"
    if key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[key]

    try:
        # yfinance .news returns a list of dicts with 'title' and 'publisher'
        news = ticker.news
        if not news:
            return None

        # Combine all headlines into a single text block for analysis
        headlines = ". ".join([item.get("title", "") for item in news])
        if not headlines.strip():
            return None

        # Create a TextBlob and get the polarity score
        blob = TextBlob(headlines)
        score = blob.sentiment.polarity

        _SENTIMENT_CACHE[key] = score
        return score
    except Exception:
        # Fail gracefully if news fetch or sentiment analysis fails
        return None


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


def get_vix_level() -> Optional[float]:
    """Fetch current VIX level from yfinance."""
    try:
        vix = yf.Ticker("^VIX")
        # Try fast_info first
        try:
            fi = getattr(vix, "fast_info", None)
            if fi:
                level = safe_float(getattr(fi, "last_price", None))
                if level and level > 0:
                    return level
        except Exception:
            pass
        
        # Try info dict
        try:
            info = vix.info or {}
            level = safe_float(info.get("regularMarketPrice"))
            if level and level > 0:
                return level
        except Exception:
            pass
        
        # Try recent history
        hist = vix.history(period="5d", interval="1d")
        if not hist.empty:
            level = safe_float(hist["Close"].iloc[-1])
            if level and level > 0:
                return level
    except Exception:
        pass
    
    return None


def determine_vix_regime(vix_level: Optional[float], config: Dict) -> Tuple[str, Dict]:
    """Determine volatility regime and return appropriate weights."""
    if vix_level is None:
        # Default to normal regime
        return "normal", config.get("weights", {})
    
    vix_regimes = config.get("vix_regimes", {})
    
    if vix_level < vix_regimes.get("low", {}).get("threshold", 15):
        return "low", vix_regimes.get("low", {}).get("weights", config.get("weights", {}))
    elif vix_level > vix_regimes.get("high", {}).get("threshold", 25):
        return "high", vix_regimes.get("high", {}).get("weights", config.get("weights", {}))
    else:
        return "normal", vix_regimes.get("normal", {}).get("weights", config.get("weights", {}))


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


def get_historical_volatility(ticker: yf.Ticker, period: int = 30) -> Optional[float]:
    """Fetch historical volatility (annualized) from recent price data.

    Uses an in-memory cache keyed by ticker symbol and period to avoid
    repeated downloads within a single run.
    """
    try:
        key = f"{ticker.ticker}:{period}"  # type: ignore[attr-defined]
        if key in _HV_CACHE:
            return _HV_CACHE[key]

        hist = ticker.history(period=f"{period+10}d", interval="1d")
        if hist.empty or len(hist) < period:
            return None
        
        # Calculate log returns
        returns = (hist['Close'] / hist['Close'].shift(1)).apply(math.log).dropna()
        
        if len(returns) < 2:
            return None
        
        # Annualized standard deviation
        daily_vol = returns.std()
        annual_vol = daily_vol * math.sqrt(252)  # Trading days
        
        _HV_CACHE[key] = annual_vol
        return annual_vol
    except Exception:
        return None


def get_momentum_indicators(ticker: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute basic price momentum indicators.

    Returns a tuple of (5-day return, 14-day RSI, ATR trend).
    All values are floats or None if insufficient data.
    """
    try:
        key = f"{ticker.ticker}:momentum"  # type: ignore[attr-defined]
        if key in _MOMENTUM_CACHE:
            return _MOMENTUM_CACHE[key]

        # Use ~90 trading days of history for stable indicators
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 20:
            return None, None, None

        close = hist["Close"].astype(float)
        high = hist.get("High", close)
        low = hist.get("Low", close)

        # 5-day simple return
        if len(close) >= 6:
            ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        else:
            ret_5d = None

        # RSI (14-day) implementation without external TA libraries
        delta = close.diff().dropna()
        if len(delta) >= 14:
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            rsi_14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None
        else:
            rsi_14 = None

        # ATR trend (14-day ATR vs its own 20-day average)
        high = high.astype(float)
        low = low.astype(float)
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        if len(atr.dropna()) < 20:
            atr_trend = None
        else:
            recent_atr = atr.iloc[-1]
            atr_ma = atr.rolling(window=20).mean().iloc[-1]
            if atr_ma and atr_ma > 0:
                atr_trend = float(recent_atr / atr_ma - 1.0)
            else:
                atr_trend = None

        result = (ret_5d, rsi_14, atr_trend)
        _MOMENTUM_CACHE[key] = result
        return result
    except Exception:
        return None, None, None


def get_iv_rank_percentile(
    ticker: yf.Ticker,
    current_iv: float,
    period_days: int = 252,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculate 30-day and 90-day IV Rank and IV Percentile.

    Uses rolling realized volatility as a proxy for historical IV.
    Returns (iv_rank_30, iv_percentile_30, iv_rank_90, iv_percentile_90).
    """
    try:
        key = f"{ticker.ticker}:iv_rank:{current_iv:.6f}"  # type: ignore[attr-defined]
        if key in _IV_RANK_CACHE:
            return _IV_RANK_CACHE[key]

        # Fetch ~1 year of daily data
        hist = ticker.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 30:
            return None, None, None, None

        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        if len(returns) < 30:
            return None, None, None, None

        def _iv_stats(window: int) -> Tuple[Optional[float], Optional[float]]:
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

        result = (iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90)
        _IV_RANK_CACHE[key] = result
        return result
    except Exception:
        return None, None, None, None


def get_next_earnings_date(ticker: yf.Ticker) -> Optional[datetime]:
    """Fetch next earnings date from yfinance.

    Tries multiple yfinance endpoints but always fails gracefully if data is missing.
    Returns a timezone-aware UTC datetime or None.
    """
    try:
        # Preferred: get_earnings_dates (newer yfinance API)
        try:
            ed = ticker.get_earnings_dates(limit=1)
            if ed is not None and not ed.empty:
                # Index is the earnings date
                dt = ed.index[0]
                if not isinstance(dt, datetime):
                    dt = datetime.fromtimestamp(dt.timestamp())  # type: ignore[attr-defined]
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except Exception:
            pass

        # Fallback: older calendar interface
        try:
            cal = getattr(ticker, "calendar", None)
            if cal is not None and not getattr(cal, "empty", False):
                # yfinance often uses the index as the event date
                if hasattr(cal, "index") and len(cal.index) > 0:
                    dt = cal.index[0]
                    if not isinstance(dt, datetime):
                        dt = datetime.fromtimestamp(dt.timestamp())  # type: ignore[attr-defined]
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
        except Exception:
            pass
    except Exception:
        # Any failure just returns None so the rest of the pipeline can continue
        return None

    return None


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


def get_underlying_price(ticker: yf.Ticker) -> Optional[float]:
    # Try fast_info, then info, then last close
    try:
        fi = getattr(ticker, "fast_info", None)
        if fi:
            lp = safe_float(getattr(fi, "last_price", None))
            if lp:
                return lp
            # Sometimes as dict
            lp = safe_float(getattr(fi, "last_price", None) or fi.get("last_price") if isinstance(fi, dict) else None)
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


def get_risk_free_rate() -> float:
    """
    Fetch the current risk-free rate from yfinance using 13-week Treasury bill (^IRX).
    Returns annualized rate as decimal (e.g., 0.045 for 4.5%).
    Falls back to 4.5% if unable to fetch.
    """
    default_rate = 0.045
    try:
        # ^IRX is the 13-week Treasury bill yield (quoted as annual %)
        tbill = yf.Ticker("^IRX")
        # Try fast_info first
        try:
            fi = getattr(tbill, "fast_info", None)
            if fi:
                rate = safe_float(getattr(fi, "last_price", None))
                if rate and rate > 0:
                    return rate / 100.0  # Convert from percentage to decimal
        except Exception:
            pass
        
        # Try info dict
        try:
            info = tbill.info or {}
            rate = safe_float(info.get("regularMarketPrice"))
            if rate and rate > 0:
                return rate / 100.0
        except Exception:
            pass
        
        # Try recent history
        hist = tbill.history(period="5d", interval="1d")
        if not hist.empty:
            rate = safe_float(hist["Close"].iloc[-1])
            if rate and rate > 0:
                return rate / 100.0
    except Exception:
        pass
    
    # Fallback to default
    return default_rate


def fetch_options_yfinance(symbol: str, max_expiries: int) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[datetime], Optional[float]]:
    """Fetch options data and enriched context for a symbol.

    Returns a DataFrame of options with underlying-level context (HV, momentum,
    IV rank/percentiles, earnings date) attached as columns, plus summary
    values for HV and IV metrics.
    """
    try:
        tkr = yf.Ticker(symbol)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ticker {symbol}: {e}")

    underlying = get_underlying_price(tkr)
    if underlying is None:
        raise RuntimeError(f"Could not determine underlying price for ticker {symbol}.")

    # Historical volatility and momentum indicators
    hv = get_historical_volatility(tkr, period=30)
    ret_5d, rsi_14, atr_trend = get_momentum_indicators(tkr)

    # Upcoming earnings
    earnings_date = get_next_earnings_date(tkr)

    # Sentiment analysis
    sentiment_score = get_sentiment(tkr)

    try:
        expirations = tkr.options
    except (URLError, ConnectionError, OSError) as e:  # type: ignore[name-defined]
        raise RuntimeError(f"Network error while fetching expirations for {symbol}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch options expirations for {symbol}: {e}")

    if not expirations:
        raise RuntimeError(f"No options expirations available for {symbol}.")

    expirations = expirations[:max_expiries]
    frames = []
    median_iv = None

    for exp in expirations:
        try:
            oc = tkr.option_chain(exp)
        except (URLError, ConnectionError, OSError):  # type: ignore[name-defined]
            # Network issue for this expiration; skip but continue others
            continue
        except Exception:
            # Skip this expiration if it fails
            continue
        for opt_type, df in [("call", oc.calls), ("put", oc.puts)]:
            if df is None or df.empty:
                continue
            sub = df.copy()
            sub["type"] = opt_type
            sub["expiration"] = exp
            sub["symbol"] = symbol.upper()

            # Normalize column names we rely on
            # yfinance has: 'strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility','lastTradeDate','inTheMoney','contractSymbol'
            for col in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
                if col not in sub.columns:
                    sub[col] = pd.NA

            frames.append(sub)

    if not frames:
        raise RuntimeError(f"No options data frames fetched from yfinance for {symbol}.")

    df = pd.concat(frames, ignore_index=True)
    df["underlying"] = underlying
    df["hv_30d"] = hv  # Add historical volatility column
    df["ret_5d"] = ret_5d
    df["rsi_14"] = rsi_14
    df["atr_trend"] = atr_trend
    df["sentiment_score"] = sentiment_score

    # Calculate median IV for IV Rank/Percentile calculation
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)

    # Get IV Rank and Percentile (30 and 90 day)
    iv_rank, iv_percentile = None, None
    iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = None, None, None, None
    if median_iv and pd.notna(median_iv) and median_iv > 0:
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = get_iv_rank_percentile(tkr, median_iv)
        # Backwards-compatible aggregate values (use 30-day by default)
        iv_rank = iv_rank_30
        iv_percentile = iv_pct_30

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90

    return df, hv, iv_rank, iv_percentile, earnings_date, sentiment_score


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
    rr_raw = df["rr_ratio"].fillna(0.0)
    rr_score = pd.Series(0.2, index=df.index)
    rr_score = rr_score.mask(rr_raw >= 2.0, 0.5)
    rr_score = rr_score.mask(rr_raw >= 3.0, 0.8)
    rr_score = rr_score.mask(rr_raw >= 4.0, 1.0)

    # EV score (rank-normalized expected value per contract)
    ev_score = rank_norm(df["ev_per_contract"].fillna(df["ev_per_contract"].median()))

    # EM realism score (already 0-1, fallback to neutral 0.5)
    em_realism_score = df["em_realism_score"].fillna(0.5).clip(lower=0.0, upper=1.0)

    # Theta decay pressure => score where lower pressure is better
    theta_raw = df["theta_decay_pressure"].replace([pd.NA, pd.NaT], np.nan)
    theta_rank = rank_norm(theta_raw.fillna(theta_raw.median()))
    theta_score = (1.0 - theta_rank).clip(lower=0.0, upper=1.0)
    # For very short-dated options, weight theta risk more heavily by slightly
    # compressing high scores (i.e., making high TDP hurt more)
    short_dte_mask = (df["T_years"] * 365.0) <= 7
    theta_score = theta_score.where(~short_dte_mask, theta_score * 0.7)

    # Momentum score combining 5d return, RSI distance from 50, and ATR trend
    ret_score = rank_norm(df.get("ret_5d", pd.Series(0.0, index=df.index)).fillna(0.0))
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


def print_report(df_picks: pd.DataFrame, underlying_price: float, rfr: float, num_expiries: int, min_dte: int, max_dte: int, mode: str = "Single-stock", budget: Optional[float] = None):
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
        unique_tickers = df_picks["symbol"].nunique()
        print(f"  OPTIONS SCREENER REPORT - DISCOVERY MODE ({unique_tickers} Tickers)")
    elif mode == "Premium Selling":
        unique_tickers = df_picks["symbol"].nunique()
        print(f"  OPTIONS SCREENER REPORT - PREMIUM SELLING ({unique_tickers} Tickers)")
    else:
        print(f"  OPTIONS SCREENER REPORT - {df_picks.iloc[0]['symbol']}")
    print("="*80)
    
    if mode == "Single-stock":
        print(f"  Stock Price: ${underlying_price:.2f}")
    elif mode == "Budget scan":
        print(f"  Budget Constraint: ${budget:.2f} per contract (premium × 100)")
        print(f"  Categories: LOW ($0-${budget*0.33:.2f}) | MEDIUM (${budget*0.33:.2f}-${budget*0.66:.2f}) | HIGH (${budget*0.66:.2f}-${budget:.2f})")
    elif mode in ["Discovery scan", "Premium Selling"]:
        print(f"  Scan Type: Top opportunities across all price ranges (no budget limit)")
        print(f"  Categories: LOW (bottom 33%) | MEDIUM (middle 33%) | HIGH (top 33%) by premium")
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
        if mode in ["Budget scan", "Discovery scan"]:
            print(f"  {'Tkr':<5} {'Whale':<3} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<7} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*81)
        else:
            print(f"  {'Whale':<3} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<8} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*79)
        
        for _, r in sub.iterrows():
            exp = pd.to_datetime(r["expiration"]).date()
            moneyness = determine_moneyness(r)
            dte = int(r["T_years"] * 365)
            whale_emoji = "🐋" if r.get("Unusual_Whale", False) else ""
            
            # Main line with aligned columns
            if mode in ["Budget scan", "Discovery scan"]:
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
            print(f"    ↳ Analysis:  {analysis_line}\n")


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
            "quality_score", "liquidity_flag", "spread_flag", "event_flag", "price_bucket"
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


def run_scan(mode: str, tickers: List[str], budget: Optional[float], max_expiries: int, min_dte: int, max_dte: int, trader_profile: str, logger: logging.Logger):
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

    # Collect data from all tickers (parallel for speed)
    all_frames = []
    ticker_metadata = {}  # Store IV Rank, earnings dates per ticker
    errors = []
    def _fetch(tkr: str):
        try:
            df_raw, hv, iv_rank, iv_percentile, earnings_date, sentiment_score = fetch_options_yfinance(tkr, max_expiries=max_expiries)
            return {
                "ticker": tkr,
                "df": df_raw,
                "hv": hv,
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile,
                "earnings_date": earnings_date,
                "sentiment_score": sentiment_score,
                "error": None
            }
        except RuntimeError as e:
            return {
                "ticker": tkr,
                "df": None,
                "hv": None,
                "iv_rank": None,
                "iv_percentile": None,
                "earnings_date": None,
                "sentiment_score": None,
                "error": str(e),
            }
        except Exception as e:
            return {
                "ticker": tkr,
                "df": None,
                "hv": None,
                "iv_rank": None,
                "iv_percentile": None,
                "earnings_date": None,
                "sentiment_score": None,
                "error": f"Unexpected error: {e}",
            }

    # Parallelize across tickers; cap workers to avoid rate limits
    max_workers = min(10, len(tickers)) if len(tickers) > 1 else 1
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_fetch, t): t for t in tickers}
            for fut in as_completed(futures):
                time.sleep(0.5) # Rate limit
                res = fut.result()
                t = res["ticker"]
                if res["df"] is not None and not res["df"].empty:
                    all_frames.append(res["df"])
                    # Store metadata per ticker
                    ticker_metadata[t] = {
                        "hv": res["hv"],
                        "iv_rank": res["iv_rank"],
                        "iv_percentile": res["iv_percentile"],
                        "earnings_date": res["earnings_date"],
                        "sentiment_score": res["sentiment_score"]
                    }
                    hv = res["hv"]
                    hv_str = f" HV:{hv*100:.1f}%" if hv else ""
                    iv_rank_str = f" IVR:{res['iv_rank']:.2f}" if res["iv_rank"] else ""
                    print(f"  Fetched {t} ✓{hv_str}{iv_rank_str}")
                else:
                    print(f"  Fetched {t} (no data)")
                if res["error"]:
                    errors.append({"ticker": t, "error": res["error"]})
    else:
        res = _fetch(tickers[0])
        if res["df"] is not None and not res["df"].empty:
            all_frames.append(res["df"])
            ticker_metadata[tickers[0]] = {
                "hv": res["hv"],
                "iv_rank": res["iv_rank"],
                "iv_percentile": res["iv_percentile"],
                "earnings_date": res["earnings_date"],
                "sentiment_score": res["sentiment_score"]
            }
            hv = res["hv"]
            hv_str = f" HV:{hv*100:.1f}%" if hv else ""
            iv_rank_str = f" IVR:{res['iv_rank']:.2f}" if res["iv_rank"] else ""
            print(f"  Fetched {tickers[0]} ✓{hv_str}{iv_rank_str}")
        elif res["error"]:
            errors.append({"ticker": tickers[0], "error": res["error"]})

    if errors:
        for err in errors:
            tkr = err.get("ticker", "?")
            msg = err.get("error", "")
            if "Network error" in msg:
                print(f"⚠️  Network error for {tkr}: {msg}. Check your connection and retry.")
            elif "No options data frames" in msg or "No options expirations" in msg:
                print(f"⚠️  No options data for {tkr}; try a more liquid symbol such as SPY or AAPL.")
            else:
                print(f"⚠️  Skipping {tkr}: {msg}")

    if not all_frames:
        print("\nNo options data retrieved from any ticker.")
        return None

    # Combine all data
    df_combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nProcessing {len(df_combined)} total options contracts...")

    # Score and filter - process per ticker to get correct metadata
    scored_frames = []
    for ticker_symbol in df_combined["symbol"].unique():
        ticker_df = df_combined[df_combined["symbol"] == ticker_symbol].copy()
        metadata = ticker_metadata.get(ticker_symbol, {})

        df_ticker_scored = enrich_and_score(
            ticker_df,
            min_dte=min_dte,
            max_dte=max_dte,
            risk_free_rate=rfr,
            config=config,
            vix_regime_weights=vix_weights,
            trader_profile=trader_profile,
            mode=mode,
            iv_rank=metadata.get("iv_rank"),
            iv_percentile=metadata.get("iv_percentile"),
            earnings_date=metadata.get("earnings_date"),
            sentiment_score=metadata.get("sentiment_score"),
        )

        if not df_ticker_scored.empty:
            scored_frames.append(df_ticker_scored)

    if not scored_frames:
        print("No contracts passed filters (check DTE bounds or liquidity).")
        return None

    df_scored = pd.concat(scored_frames, ignore_index=True)
    print(f"Scored {len(df_scored)} contracts after filtering.")

    if df_scored.empty:
        print("No contracts passed filters (check DTE bounds or liquidity).")
        return None

    # Apply budget filter if in budget mode
    if is_budget_mode:
        df_scored["contract_cost"] = df_scored["premium"] * 100
        df_scored = df_scored[df_scored["contract_cost"] <= budget].copy()
        if df_scored.empty:
            print(f"No contracts found within budget of ${budget:.2f}.")
            return None
        print(f"Found {len(df_scored)} contracts within budget.")

        # Show budget distribution for user clarity
        print(f"\nBudget Categories:")
        print(f"  LOW:    $0 - ${budget*0.33:.2f} (0-33% of budget)")
        print(f"  MEDIUM: ${budget*0.33:.2f} - ${budget*0.66:.2f} (33-66% of budget)")
        print(f"  HIGH:   ${budget*0.66:.2f} - ${budget:.2f} (66-100% of budget)")
    elif is_discovery_mode:
        # Discovery mode: no budget filter, use quantiles for categorization
        print(f"\nDiscovery Mode: Analyzing {len(df_scored)} quality options across all price ranges...")

    # Categorize and pick
    # Use budget categorization for budget mode, quantile for single-stock and discovery
    df_bucketed = categorize_by_premium(df_scored, budget=budget if is_budget_mode else None)

    # Diversify tickers in budget and discovery modes
    picks = pick_top_per_bucket(df_bucketed, per_bucket=5, diversify_tickers=(is_budget_mode or is_discovery_mode))
    if picks.empty:
        print("Could not produce picks in the requested buckets.")
        return None

    # Find vertical spreads
    spreads = find_vertical_spreads(df_scored)

    # Get underlying price for report (first ticker in single mode, or 0 in budget mode)
    underlying_price = df_scored.iloc[0]["underlying"] if not df_scored.empty and not is_budget_mode else 0.0

    # Structured JSONL logging of the picks and context
    log_picks_json(
        logger,
        picks,
        context={
            "mode": mode,
            "budget": budget,
            "max_expiries": max_expiries,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "risk_free_rate": rfr,
        },
    )
    # Enhanced scoring for top pick: balance quality with practical factors
    picks_copy = picks.copy()
    chain_iv_median = picks["impliedVolatility"].median(skipna=True)

    # Weighted overall score
    # Favor: high quality, good liquidity, moderate IV, tight spread, balanced delta
    picks_copy["overall_score"] = (
        0.40 * picks_copy["quality_score"] +
        0.20 * picks_copy["liquidity_score"] +
        0.15 * picks_copy["spread_score"] +
        0.15 * picks_copy["delta_quality"] +
        0.10 * picks_copy["iv_quality"]
    )

    top_pick = picks_copy.sort_values("overall_score", ascending=False).iloc[0]

    return {
        'picks': picks,
        'top_pick': top_pick,
        'spreads': spreads,
        'rfr': rfr,
        'chain_iv_median': chain_iv_median,
        'underlying_price': underlying_price,
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
    print("  4. Enter 'SELL' for Premium Selling analysis (short puts)\n")

    symbol_input = prompt_input("Enter stock ticker, 'ALL', 'DISCOVER', or 'SELL'", "").upper()

    # Determine mode
    is_budget_mode = (symbol_input == "ALL")
    is_discovery_mode = (symbol_input == "DISCOVER" or symbol_input == "")
    is_premium_selling_mode = (symbol_input == "SELL")

    if is_discovery_mode:
        mode = "Discovery scan"
    elif is_budget_mode:
        mode = "Budget scan"
    elif is_premium_selling_mode:
        mode = "Premium Selling"
    else:
        mode = "Single-stock"

    budget = None
    tickers = []

    if is_discovery_mode or is_premium_selling_mode:
        # Discovery mode: scan top 100 most-traded options tickers
        if is_premium_selling_mode:
            print("\n=== PREMIUM SELLING MODE ===")
            print("Scanning top 100 most-traded options tickers for short put opportunities...")
        else:
            print("\n=== DISCOVERY MODE ===")
            print("Scanning top 100 most-traded options tickers for best opportunities...")

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
    try:
        max_expiries = int(prompt_input("How many nearest expirations to scan", "4"))
        if max_expiries <= 0 or max_expiries > 12:
            print("Please choose between 1 and 12 expirations.")
            sys.exit(1)
    except Exception:
        print("Invalid number for expirations.")
        sys.exit(1)

    try:
        min_dte = int(prompt_input("Minimum days to expiration (DTE)", "7"))
        max_dte = int(prompt_input("Maximum days to expiration (DTE)", "120"))
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
        )
        if scan_results is None:
            sys.exit(0)

        # Unpack results
        picks = scan_results['picks']
        top_pick = scan_results['top_pick']
        spreads = scan_results['spreads']
        rfr = scan_results['rfr']
        chain_iv_median = scan_results['chain_iv_median']
        underlying_price = scan_results['underlying_price']
        
        # Print main report
        print_report(picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode, budget)
        
        # Print spreads report
        print_spreads_report(spreads)

        # Compute and display top overall pick
        print("\n" + "="*80)
        print("  ⭐ TOP OVERALL PICK")
        print("="*80)
        
        exp = pd.to_datetime(top_pick["expiration"]).date()
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
        
        # Liquidity assessment
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
            unique_tickers = picks["symbol"].nunique()
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
