"""Cached, network-isolated data adapters: Deribit DVOL + option trades,
Binance funding, yfinance spot. Parsers are pure; fetchers hit public endpoints.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

_DERIBIT = "https://www.deribit.com/api/v2/public"
_BINANCE = "https://fapi.binance.com/fapi/v1"


def parse_dvol(payload: dict) -> pd.DataFrame:
    rows = payload["result"]["data"]
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "close"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df[["Date", "close"]].rename(columns={"close": "dvol"})


def parse_funding(payload: list) -> pd.DataFrame:
    df = pd.DataFrame(payload)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["fundingTime", "funding_rate"]]


def _http_get(url: str, params: dict, timeout: int = 30) -> object:
    return requests.get(url, params=params, timeout=timeout).json()


def load_dvol(currency: str = "BTC", days: int = 760) -> pd.DataFrame:
    now = int(time.time() * 1000)
    start = now - days * 24 * 3600 * 1000
    payload = _http_get(f"{_DERIBIT}/get_volatility_index_data",
                        dict(currency=currency, start_timestamp=start,
                             end_timestamp=now, resolution=86400))
    return parse_dvol(payload)


def load_funding(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    payload = _http_get(f"{_BINANCE}/fundingRate", dict(symbol=symbol, limit=limit))
    return parse_funding(payload)


def load_option_trades(currency: str = "BTC", days: int = 7,
                       count: int = 1000) -> pd.DataFrame:
    now = int(time.time() * 1000)
    start = now - days * 24 * 3600 * 1000
    payload = _http_get(f"{_DERIBIT}/get_last_trades_by_currency_and_time",
                        dict(currency=currency, kind="option", start_timestamp=start,
                             end_timestamp=now, count=count, include_old="true"))
    return pd.DataFrame(payload["result"]["trades"])


def load_spot(currency: str = "BTC", days: int = 900) -> pd.DataFrame:
    from src.crypto.data_fetching import get_spot_history
    df = get_spot_history(currency, days=days)[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df
