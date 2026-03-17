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
import warnings
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import requests_cache
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
from finvizfinance.screener.performance import Performance

logger = logging.getLogger(__name__)

try:
    from .news_fetcher import fetch_news_and_events, NewsData
    _HAS_NEWS_FETCHER = True
except Exception:
    _HAS_NEWS_FETCHER = False

# Suppress noisy third-party library output at startup
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# Install intelligent request caching (15-minute expiry)
try:
    requests_cache.install_cache('finance_cache', backend='sqlite', expire_after=900)
except Exception:
    pass  # Cache unavailable; requests proceed uncached

# In-memory caches
_HV_CACHE: Dict[str, float] = {}
_MOMENTUM_CACHE: Dict[str, Tuple] = {}
_IV_RANK_CACHE: Dict[str, Tuple] = {}
_SENTIMENT_CACHE: Dict[str, float] = {}
_SEASONALITY_CACHE: Dict[str, float] = {}

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
        return float(yf.Ticker(symbol).fast_info["lastPrice"])

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
    hv_30d = (0.5 * hv_30d_rolling + 0.5 * hv_ewma) if (hv_30d_rolling and hv_ewma) \
        else (hv_30d_rolling or hv_ewma)

    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50 = \
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
    except Exception:
        pass

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
    df["atr_trend"] = atr_trend
    df["sma_20"] = sma_20
    df["sma_50"] = sma_50
    df["high_20"] = high_20
    df["low_20"] = low_20
    df["sentiment_score"] = sentiment_score
    df["seasonal_win_rate"] = seasonal_win_rate
    df["is_squeezing"] = is_squeezing
    df["rvol"] = rvol
    df["short_interest"] = short_interest
    df["vwap"] = vwap
    df["fib_50"] = fib_50
    df["fib_618"] = fib_618

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)

    iv_rank_30 = iv_pct_30 = iv_rank_90 = iv_pct_90 = None
    iv_confidence = "Low"
    if median_iv and pd.notna(median_iv) and median_iv > 0:
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, iv_confidence = \
            get_iv_rank_percentile_from_history(hist, median_iv, ticker=symbol)

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90
    df["iv_confidence"] = iv_confidence

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
        except Exception:
            pass

    return {
        "df": df,
        "history_df": hist,
        "context": {
            "hv": hv_30d,
            "hv_ewma": hv_ewma,
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
                    errors.append(str(exc))

        raise RuntimeError(
            f"All providers failed for {symbol}: {'; '.join(errors)}"
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
                    errors.append(str(exc))

        raise RuntimeError(
            f"All providers failed for spot({symbol}): {'; '.join(errors)}"
        )


# Module-level singleton — swap out for testing or alternative sources
def _make_default_provider() -> BaseDataProvider:
    """Build the default provider, using FanOut if yahooquery is available."""
    try:
        import yahooquery  # noqa: F401
        return FanOutProvider([YFinanceProvider(), YahooQueryProvider()])
    except ImportError:
        return YFinanceProvider()


_DATA_PROVIDER: BaseDataProvider = _make_default_provider()

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
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_history (
                ticker TEXT,
                date   TEXT,
                iv_value REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _upsert_iv(ticker: str, date_str: str, iv: float, db_path: str) -> None:
    try:
        _init_iv_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO iv_history (ticker, date, iv_value) VALUES (?, ?, ?)",
            (ticker, date_str, float(iv)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_iv_history(ticker: str, db_path: str) -> List[float]:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT iv_value FROM iv_history WHERE ticker=? ORDER BY date ASC",
            (ticker,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0] is not None]
    except Exception:
        return []

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

def get_news_headlines(ticker: yf.Ticker, max_headlines: int = 3) -> list:
    """Return up to max_headlines recent news titles for a ticker."""
    try:
        news = ticker.news
        if not news:
            return []
        titles = []
        for item in news[:max_headlines]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                titles.append(title.strip())
        return titles[:max_headlines]
    except Exception:
        return []

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
        returns = np.log(subset['Close'] / subset['Close'].shift(1)).dropna()
        if len(returns) < 2:
            return None
        daily_vol = returns.std()
        return daily_vol * math.sqrt(252)
    except Exception:
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
        return float(np.sqrt(ewm_var.iloc[-1] * 252))
    except Exception:
        return None

def calculate_momentum_indicators(hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool, Optional[float]]:
    """Calculate RSI, ATR, SMA-20, SMA-50, etc. from history DataFrame."""
    try:
        if hist.empty or len(hist) < 21:
            return None, None, None, None, None, None, False, None
        
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

        return ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50
    except Exception:
        return None, None, None, None, None, None, False, None

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
            iv_min = iv_arr.min()
            iv_max = iv_arr.max()
            if iv_max - iv_min <= 0:
                iv_rank_30, iv_pct_30 = 0.5, 0.5
            else:
                iv_rank_30 = float(np.clip((current_iv - iv_min) / (iv_max - iv_min), 0, 1))
                iv_pct_30 = float(np.clip((iv_arr < current_iv).sum() / len(iv_arr), 0, 1))

            if len(iv_history) >= 90:
                iv_arr_90 = iv_arr[-90:]
                iv_min_90, iv_max_90 = iv_arr_90.min(), iv_arr_90.max()
                if iv_max_90 - iv_min_90 <= 0:
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
        info = ticker.info or {}
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
            call_loss = (np.maximum(0.0, price_candidate - calls['strike'].values) * calls['openInterest'].values).sum()
            puts = relevant[relevant['type'] == 'put']
            put_loss = (np.maximum(0.0, puts['strike'].values - price_candidate) * puts['openInterest'].values).sum()
            total_loss = call_loss + put_loss
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                max_pain_price = price_candidate
        return max_pain_price
    except Exception:
        return None

def _process_option_chain(tkr: yf.Ticker, symbol: str, exp: str) -> List[pd.DataFrame]:
    try:
        # Wrap ticker.option_chain in try/except to handle invalid/empty expirations gracefully
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

        implied_move_pct = (call_mid + put_mid) / underlying

        hist_moves = []
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
        except Exception:
            pass

        if not hist_moves:
            return {"implied_move_pct": implied_move_pct, "hist_avg_move": None, "hist_beat_rate": None, "is_cheap": None}

        hist_avg = sum(hist_moves) / len(hist_moves)
        beat_rate = sum(1 for m in hist_moves if m > implied_move_pct) / len(hist_moves)
        is_cheap = hist_avg > implied_move_pct

        return {
            "implied_move_pct": implied_move_pct,
            "hist_avg_move": hist_avg,
            "hist_beat_rate": beat_rate,
            "is_cheap": is_cheap
        }
    except Exception:
        return None


@retry_with_backoff(retries=3, backoff_in_seconds=2, error_types=(RuntimeError, URLError, ConnectionError, OSError))
def fetch_options_yfinance(symbol: str, max_expiries: int) -> Dict:
    """
    Fetch options data and all related context using a Single-Fetch architecture.
    Returns a dictionary with 'df' (options chain) and 'context' (all derived metrics).
    """
    try:
        tkr = yf.Ticker(symbol)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to initialize ticker {symbol}: {e}")

    # 1. Fetch History ONCE (1 year daily)
    try:
        hist = tkr.history(period="1y", interval="1d")
    except (ValueError, KeyError, URLError) as e:
        logging.warning(f"Failed to fetch history for {symbol}: {e}")
        hist = pd.DataFrame()

    if hist.empty:
        raise RuntimeError(f"Could not fetch price history for {symbol}")

    # 2. Derive Metrics from History
    underlying = safe_float(hist["Close"].iloc[-1])
    hv_30d_rolling = calculate_historical_volatility(hist, period=30)
    hv_ewma = calculate_ewma_volatility(hist, span=20)
    # Blend rolling and EWMA vol: rolling gives stability, EWMA gives recency
    if hv_30d_rolling and hv_ewma:
        hv_30d = 0.5 * hv_30d_rolling + 0.5 * hv_ewma
    else:
        hv_30d = hv_30d_rolling or hv_ewma
    ret_5d, rsi_14, atr_trend, sma_20, high_20, low_20, is_squeezing, sma_50 = calculate_momentum_indicators(hist)
    rvol = calculate_rvol(hist)
    vwap, fib_50, fib_618 = calculate_technical_levels(hist)

    # 3. Fetch Other Data (Earnings, Sentiment, Seasonality)
    earnings_date = get_next_earnings_date(tkr)
    sentiment_score = get_sentiment(tkr)
    news_headlines = get_news_headlines(tkr)
    seasonal_win_rate = check_seasonality(tkr)
    sector_perf = get_sector_performance(symbol)

    # Rich news + analyst data (multi-source aggregation)
    news_data = None
    if _HAS_NEWS_FETCHER:
        try:
            news_data = fetch_news_and_events(symbol, ticker_obj=tkr, max_age_hours=72, max_headlines=5)
            # Keep news_headlines in sync so AI context is consistent
            if news_data and news_data.top_headlines:
                news_headlines = news_data.top_headlines
        except Exception:
            pass
    short_interest = get_short_interest(tkr)

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
        expirations = tkr.options
    except (ValueError, KeyError, AttributeError) as e:
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

    # Compute earnings move analysis now that chain is available
    earnings_move_data = calculate_implied_earnings_move(tkr, earnings_date, df, underlying)

    # 5. Enrich DataFrame with Context
    df["underlying"] = underlying
    df["hv_30d"] = hv_30d
    df["ret_5d"] = ret_5d
    df["rsi_14"] = rsi_14
    df["atr_trend"] = atr_trend
    df["sma_20"] = sma_20
    df["sma_50"] = sma_50
    df["high_20"] = high_20
    df["low_20"] = low_20
    df["sentiment_score"] = sentiment_score
    df["seasonal_win_rate"] = seasonal_win_rate
    df["is_squeezing"] = is_squeezing
    df["rvol"] = rvol
    df["short_interest"] = short_interest
    df["vwap"] = vwap
    df["fib_50"] = fib_50
    df["fib_618"] = fib_618

    # IV Rank Calculation
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    median_iv = df["impliedVolatility"].median(skipna=True)
    
    iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90 = None, None, None, None
    iv_confidence = "Low"
    if median_iv and pd.notna(median_iv) and median_iv > 0:
        iv_rank_30, iv_pct_30, iv_rank_90, iv_pct_90, iv_confidence = get_iv_rank_percentile_from_history(
            hist, median_iv, ticker=symbol
        )

    df["iv_rank_30"] = iv_rank_30
    df["iv_percentile_30"] = iv_pct_30
    df["iv_rank_90"] = iv_rank_90
    df["iv_percentile_90"] = iv_pct_90
    df["iv_confidence"] = iv_confidence

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

    # ── Polygon.io enrichment (opt-in, silent no-op when key absent) ────────────
    unusual_options_flow = None
    company_description = None
    market_cap = None
    try:
        from src.polygon_client import PolygonClient
        _poly = PolygonClient()
        if _poly._enabled:
            logger.debug("Polygon enrichment: fetching data for %s", symbol)

            # VWAP override from Polygon snapshot (more accurate than computed)
            snap = _poly.get_snapshot(symbol)
            if snap:
                try:
                    poly_vwap = snap["ticker"]["day"]["vw"]
                    if poly_vwap and float(poly_vwap) > 0:
                        vwap = float(poly_vwap)
                        logger.debug("Polygon enrichment: VWAP overridden to %.4f for %s", vwap, symbol)
                except Exception:
                    pass

            # Unusual options flow
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
    return {
        "df": df,
        "history_df": hist,
        "context": {
            "hv": hv_30d,
            "hv_ewma": hv_ewma,
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
        }
    }


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
    _provider = provider or _DATA_PROVIDER
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