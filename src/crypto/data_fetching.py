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
_BYBIT_BASE   = "https://api.bybit.com/v5/market"
_OKX_BASE     = "https://www.okx.com/api/v5/public"
_DYDX_BASE    = "https://indexer.dydx.trade/v4"
_HTTP_TIMEOUT = 10
_USER_AGENT = "options-screener-crypto/1.0"

# Funding cycle hours per exchange. Used to normalize all rates to "per 8h"
# so they're directly comparable in the dashboard / divergence math.
_FUNDING_CYCLE_HOURS = {
    "binance": 8,
    "bybit":   8,
    "okx":     8,
    "dydx":    1,   # dYdX v4 funds hourly
}


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


# ── Cross-exchange funding ───────────────────────────────────────────────

def _normalize_to_8h(rate: float, cycle_hours: int) -> float:
    """Scale a funding rate to per-8h-equivalent for cross-venue comparison."""
    if cycle_hours <= 0:
        return rate
    return float(rate) * (8.0 / float(cycle_hours))


def _funding_binance(symbol: str = "BTCUSDT") -> Optional[Dict[str, float]]:
    data = _http_get(f"{_BINANCE_FAPI}/premiumIndex", {"symbol": symbol.upper()})
    if not data or "lastFundingRate" not in data:
        return None
    try:
        rate_native = float(data["lastFundingRate"])
        return {
            "exchange": "binance",
            "cycle_hours": _FUNDING_CYCLE_HOURS["binance"],
            "rate_native": rate_native,
            "rate_8h": _normalize_to_8h(rate_native, _FUNDING_CYCLE_HOURS["binance"]),
            "mark_price": float(data["markPrice"]),
            "index_price": float(data["indexPrice"]),
        }
    except (KeyError, ValueError):
        return None


def _funding_bybit(symbol: str = "BTCUSDT") -> Optional[Dict[str, float]]:
    data = _http_get(
        f"{_BYBIT_BASE}/tickers",
        {"category": "linear", "symbol": symbol.upper()},
    )
    if not data or data.get("retCode") != 0:
        return None
    try:
        items = data.get("result", {}).get("list", [])
        if not items:
            return None
        item = items[0]
        rate_native = float(item.get("fundingRate") or 0)
        return {
            "exchange": "bybit",
            "cycle_hours": _FUNDING_CYCLE_HOURS["bybit"],
            "rate_native": rate_native,
            "rate_8h": _normalize_to_8h(rate_native, _FUNDING_CYCLE_HOURS["bybit"]),
            "mark_price": float(item.get("markPrice") or 0),
            "index_price": float(item.get("indexPrice") or 0),
        }
    except (KeyError, ValueError, TypeError):
        return None


def _funding_okx(currency: str = "BTC") -> Optional[Dict[str, float]]:
    inst = f"{currency.upper()}-USDT-SWAP"
    data = _http_get(f"{_OKX_BASE}/funding-rate", {"instId": inst})
    if not data or data.get("code") != "0":
        return None
    try:
        items = data.get("data", [])
        if not items:
            return None
        item = items[0]
        rate_native = float(item.get("fundingRate") or 0)
        # OKX funding-rate endpoint is funding-only — fetch mark/index separately.
        ticker = _http_get(f"https://www.okx.com/api/v5/market/ticker", {"instId": inst})
        mark = idx = 0.0
        if ticker and ticker.get("code") == "0":
            try:
                row = ticker.get("data", [])[0]
                mark = float(row.get("last") or row.get("markPx") or 0)
                idx  = float(row.get("idxPx") or mark)
            except (KeyError, ValueError, IndexError):
                pass
        return {
            "exchange": "okx",
            "cycle_hours": _FUNDING_CYCLE_HOURS["okx"],
            "rate_native": rate_native,
            "rate_8h": _normalize_to_8h(rate_native, _FUNDING_CYCLE_HOURS["okx"]),
            "mark_price": mark,
            "index_price": idx,
        }
    except (KeyError, ValueError, TypeError):
        return None


def _funding_dydx(currency: str = "BTC") -> Optional[Dict[str, float]]:
    ticker = f"{currency.upper()}-USD"
    data = _http_get(f"{_DYDX_BASE}/perpetualMarkets", {"ticker": ticker})
    if not data:
        return None
    try:
        market = data.get("markets", {}).get(ticker)
        if not market:
            return None
        # dYdX `nextFundingRate` is per-hour. Mark price not directly exposed
        # via this endpoint; oraclePrice is the closest proxy.
        rate_native = float(market.get("nextFundingRate") or 0)
        oracle = float(market.get("oraclePrice") or 0)
        return {
            "exchange": "dydx",
            "cycle_hours": _FUNDING_CYCLE_HOURS["dydx"],
            "rate_native": rate_native,
            "rate_8h": _normalize_to_8h(rate_native, _FUNDING_CYCLE_HOURS["dydx"]),
            "mark_price": oracle,
            "index_price": oracle,
        }
    except (KeyError, ValueError, TypeError):
        return None


def get_aggregated_funding(currency: str = "BTC") -> Dict[str, Any]:
    """Funding rate from all 4 venues + cross-venue divergence stats.

    Returns a dict with structure::

        {
          'exchanges': {
            'binance': { rate_native, rate_8h, mark_price, index_price, cycle_hours, ... },
            'bybit':   {...},
            'okx':     {...},
            'dydx':    {...},
          },
          'divergence': {
            'max_8h':   highest rate observed across venues
            'min_8h':   lowest rate observed across venues
            'spread_bps': (max_8h - min_8h) in basis points
            'mean_8h':  cross-venue mean
            'std_8h':   cross-venue std
            'venue_count': number of venues that responded
          }
        }

    All rates normalized to per-8h equivalent. Divergence omitted if fewer
    than 2 venues respond.
    """
    cache_key = f"agg_funding_{currency.upper()}"
    cached = _cache.get("binance_funding", cache_key, ttl=60)  # 1-min TTL across all venues
    if cached is not None:
        return cached

    sym_usdt = f"{currency.upper()}USDT"
    results: Dict[str, Optional[Dict[str, float]]] = {
        "binance": _funding_binance(sym_usdt),
        "bybit":   _funding_bybit(sym_usdt),
        "okx":     _funding_okx(currency),
        "dydx":    _funding_dydx(currency),
    }
    exchanges = {k: v for k, v in results.items() if v is not None}
    out: Dict[str, Any] = {"exchanges": exchanges, "divergence": {}}
    if len(exchanges) >= 2:
        rates = [e["rate_8h"] for e in exchanges.values()]
        max_r = max(rates)
        min_r = min(rates)
        mean_r = sum(rates) / len(rates)
        var_r = sum((r - mean_r) ** 2 for r in rates) / len(rates)
        std_r = var_r ** 0.5
        out["divergence"] = {
            "max_8h": max_r,
            "min_8h": min_r,
            "spread_bps": (max_r - min_r) * 10000,
            "mean_8h": mean_r,
            "std_8h": std_r,
            "venue_count": len(exchanges),
        }
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
