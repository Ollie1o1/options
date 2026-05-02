"""Crypto data layer: Deribit options + Binance funding + yfinance spot history.

All public endpoints — no API key required for the read-only data we need.
Times are UTC throughout. Caching via src.crypto.cache (SQLite WAL).
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import cache as _cache

_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
_BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
_HTTP_TIMEOUT = 10
_USER_AGENT = "options-screener-crypto/1.0"


def _http_get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """Single GET with timeout + user-agent. Returns parsed JSON or None on failure."""
    try:
        r = requests.get(
            url,
            params=params or {},
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        if r.status_code != 200:
            return None
        return r.json()
    except (requests.RequestException, ValueError):
        return None


# ── Deribit ──────────────────────────────────────────────────────────────

def get_index_price(currency: str = "BTC") -> Optional[float]:
    """Spot index price for currency (BTC or ETH) in USD."""
    cache_key = f"index_{currency.upper()}"
    cached = _cache.get("deribit_index", cache_key)
    if cached is not None:
        return float(cached.get("price"))
    data = _http_get(
        f"{_DERIBIT_BASE}/get_index_price",
        {"index_name": f"{currency.lower()}_usd"},
    )
    if not data or "result" not in data:
        return None
    price = data["result"].get("index_price")
    if price is None:
        return None
    _cache.put("deribit_index", cache_key, {"price": float(price)})
    return float(price)


def get_options_chain(currency: str = "BTC") -> pd.DataFrame:
    """Full Deribit options chain (every active expiry, every strike, both sides).

    Returns a DataFrame with columns: instrument_name, underlying_price, strike,
    type ('call' / 'put'), expiration (UTC date), dte, mark_iv, bid_iv, ask_iv,
    bid_price, ask_price, mark_price, mid_price, volume, open_interest,
    underlying_index, expiration_timestamp_ms.

    Empty DataFrame on fetch failure.
    """
    cache_key = f"chain_{currency.upper()}"
    cached = _cache.get("deribit_chain", cache_key)
    if cached is not None:
        df = pd.DataFrame(cached)
        if not df.empty and "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        return df

    data = _http_get(
        f"{_DERIBIT_BASE}/get_book_summary_by_currency",
        {"currency": currency.upper(), "kind": "option"},
    )
    if not data or "result" not in data:
        return pd.DataFrame()
    rows: List[dict] = []
    now_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    for item in data["result"]:
        name = item.get("instrument_name", "")
        # BTC-2MAY26-67000-C  →  base/expiry/strike/option-type
        try:
            _, exp_str, strike_str, kind = name.split("-")
            exp_date = _dt.datetime.strptime(exp_str, "%d%b%y").replace(
                tzinfo=_dt.timezone.utc
            )
            strike = float(strike_str)
            opt_type = "call" if kind.upper() == "C" else "put"
        except (ValueError, AttributeError):
            continue
        underlying = item.get("underlying_price") or item.get("estimated_delivery_price")
        if not underlying or underlying <= 0:
            continue
        bid_p = item.get("bid_price") or 0.0
        ask_p = item.get("ask_price") or 0.0
        mark_p = item.get("mark_price") or 0.0
        # Deribit prices are in BTC/ETH terms; convert to USD per contract for clarity
        bid_usd = float(bid_p) * float(underlying)
        ask_usd = float(ask_p) * float(underlying)
        mark_usd = float(mark_p) * float(underlying)
        mid_usd = (bid_usd + ask_usd) / 2 if (bid_usd > 0 and ask_usd > 0) else mark_usd
        exp_ms = item.get("creation_timestamp", now_ms)
        # The chain summary doesn't include expiration_timestamp directly, derive it:
        exp_ts_ms = int(exp_date.timestamp() * 1000)
        dte_days = max(0.0, (exp_ts_ms - now_ms) / (1000 * 86400))
        rows.append({
            "instrument_name": name,
            "underlying_price": float(underlying),
            "strike": strike,
            "type": opt_type,
            "expiration": exp_date.date().isoformat(),
            "dte": dte_days,
            "mark_iv": float(item.get("mark_iv") or 0.0) / 100.0,  # Deribit reports IV as percent
            "bid_iv": float(item.get("bid_iv") or 0.0) / 100.0,
            "ask_iv": float(item.get("ask_iv") or 0.0) / 100.0,
            "bid_price": bid_usd,
            "ask_price": ask_usd,
            "mark_price": mark_usd,
            "mid_price": mid_usd,
            "volume": float(item.get("volume") or 0.0),
            "open_interest": float(item.get("open_interest") or 0.0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["spread_pct"] = (df["ask_price"] - df["bid_price"]) / df["mark_price"].replace(0, float("nan"))
    _cache.put("deribit_chain", cache_key, df.to_dict("records"))
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    return df


# ── Binance funding (perp) ───────────────────────────────────────────────

def get_funding_rate(symbol: str = "BTCUSDT") -> Optional[Dict[str, float]]:
    """Latest funding rate + mark/index price from Binance USDT-M perp.

    Returns dict {funding_rate, mark_price, index_price, basis_pct} or None.
    funding_rate is the *next* funding (8h interval typically); annualize by × 3 × 365.
    """
    cache_key = f"funding_{symbol.upper()}"
    cached = _cache.get("binance_funding", cache_key)
    if cached is not None:
        return cached
    data = _http_get(f"{_BINANCE_FAPI}/premiumIndex", {"symbol": symbol.upper()})
    if not data or "lastFundingRate" not in data:
        return None
    try:
        out = {
            "funding_rate": float(data["lastFundingRate"]),
            "mark_price": float(data["markPrice"]),
            "index_price": float(data["indexPrice"]),
        }
        out["basis_pct"] = (out["mark_price"] - out["index_price"]) / out["index_price"]
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    _cache.put("binance_funding", cache_key, out)
    return out


def get_funding_history(symbol: str = "BTCUSDT", limit: int = 100) -> pd.DataFrame:
    """Last `limit` funding rates (8h bars) for z-score computation."""
    cache_key = f"funding_history_{symbol.upper()}_{limit}"
    cached = _cache.get("binance_funding", cache_key, ttl=900)  # 15 min
    if cached is not None:
        return pd.DataFrame(cached)
    data = _http_get(
        f"{_BINANCE_FAPI}/fundingRate",
        {"symbol": symbol.upper(), "limit": int(limit)},
    )
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for item in data:
        try:
            rows.append({
                "funding_time": int(item["fundingTime"]),
                "funding_rate": float(item["fundingRate"]),
            })
        except (KeyError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        _cache.put("binance_funding", cache_key, df.to_dict("records"))
    return df


# ── yfinance spot history (for realized vol + 200d MA) ───────────────────

def get_spot_history(currency: str = "BTC", days: int = 365) -> pd.DataFrame:
    """Daily OHLC history for the spot via yfinance. Used for realized vol + MA."""
    cache_key = f"yf_history_{currency.upper()}_{days}"
    cached = _cache.get("yf_history", cache_key)
    if cached is not None:
        df = pd.DataFrame(cached)
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    sym = f"{currency.upper()}-USD"
    try:
        t = yf.Ticker(sym)
        h = t.history(period=f"{int(days)}d", interval="1d")
    except Exception:
        return pd.DataFrame()
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.reset_index()
    if "Date" in h.columns:
        h["Date"] = h["Date"].astype(str)
    _cache.put("yf_history", cache_key, h.to_dict("records"))
    h["Date"] = pd.to_datetime(h["Date"])
    return h


def realized_vol(returns: pd.Series, window: int = 30, annualize: bool = True) -> float:
    """Standard realized vol from log returns. Annualized by sqrt(365) (24/7 market)."""
    if returns is None or returns.empty:
        return float("nan")
    r = returns.dropna().tail(window)
    if r.empty:
        return float("nan")
    sd = float(r.std())
    if not math.isfinite(sd):
        return float("nan")
    return sd * math.sqrt(365) if annualize else sd
