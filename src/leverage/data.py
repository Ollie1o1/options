"""Binance USDⓈ-M futures (fapi) klines + funding, parquet-cached, with a Bybit
fallback. The only module in src/leverage that touches the network."""
from __future__ import annotations
import os
import time
from typing import Callable, Optional, Tuple
import pandas as pd
import requests

_FAPI = "https://fapi.binance.com"
_BYBIT = "https://api.bybit.com"
_KLINE_LIMIT = 1500
_DEFAULT_CACHE = "data/leverage_ohlcv"
_INTERVAL_MIN = {"5m": 5, "15m": 15}
_BYBIT_INT = {"5m": "5", "15m": "15"}


def _klines_to_frame(raw: list) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "_ct",
            "_qv", "_n", "_tb", "_tq", "_ig"]
    df = pd.DataFrame(raw, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.set_index("open_time")


def fetch_klines_binance(symbol: str, interval: str, start_ms: int,
                         end_ms: int) -> pd.DataFrame:
    frames, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(f"{_FAPI}/fapi/v1/klines", params={
            "symbol": symbol, "interval": interval, "startTime": cur,
            "endTime": end_ms, "limit": _KLINE_LIMIT}, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(_klines_to_frame(rows))
        cur = rows[-1][0] + _INTERVAL_MIN[interval] * 60_000
        if len(rows) < _KLINE_LIMIT:
            break
        time.sleep(0.25)
    if not frames:
        return _klines_to_frame([])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")]


def _bybit_to_frame(rows: list) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "_turn"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms",
                                     utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.set_index("open_time").sort_index()


def fetch_klines_bybit(symbol: str, interval: str, start_ms: int,
                       end_ms: int) -> pd.DataFrame:
    frames, cur = [], start_ms
    step = _INTERVAL_MIN[interval] * 60_000 * 1000  # 1000 bars per page
    while cur < end_ms:
        r = requests.get(f"{_BYBIT}/v5/market/kline", params={
            "category": "linear", "symbol": symbol,
            "interval": _BYBIT_INT[interval], "start": cur,
            "end": min(cur + step, end_ms), "limit": 1000}, timeout=15)
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            break
        frames.append(_bybit_to_frame(rows))
        cur += step
        time.sleep(0.25)
    if not frames:
        return _klines_to_frame([])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def _fetch_with_fallback(symbol: str, interval: str, start_ms: int,
                         end_ms: int) -> pd.DataFrame:
    """Binance USDⓈ-M first; on a network/HTTP failure (including the US
    geo-block, HTTP 451) fall back to Bybit. An empty result (no new bars) is
    NOT a failure — it returns as-is, so routine cache top-ups never trigger a
    needless second exchange call."""
    try:
        return fetch_klines_binance(symbol, interval, start_ms, end_ms)
    except requests.RequestException:
        return fetch_klines_bybit(symbol, interval, start_ms, end_ms)


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    frames, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(f"{_FAPI}/fapi/v1/fundingRate", params={
            "symbol": symbol, "startTime": cur, "endTime": end_ms,
            "limit": 1000}, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        df = pd.DataFrame(rows)
        df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["rate"] = df["fundingRate"].astype(float)
        frames.append(df[["funding_time", "rate"]].set_index("funding_time"))
        cur = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < 1000:
            break
        time.sleep(0.25)
    return pd.concat(frames) if frames else pd.DataFrame(columns=["rate"])


def _write_cache(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _resample(df5: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = interval.replace("m", "min")
    out = df5.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return out


def load_history(symbol: str, cache_dir: str = _DEFAULT_CACHE,
                 fetcher: Optional[Callable] = None,
                 days: int = 550) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned (df5, df15). Cache-first; tops up missing recent bars.
    `fetcher(symbol, interval, start_ms, end_ms)` is injectable for tests."""
    fetch = fetcher or _fetch_with_fallback
    path5 = os.path.join(cache_dir, symbol, "5m.parquet")
    cached = _read_cache(path5)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    if cached is not None and len(cached):
        top_start = int(cached.index[-1].timestamp() * 1000) + 1
        try:
            fresh = fetch(symbol, "5m", top_start, end_ms)
        except Exception:
            # Cache is the resilience layer — a failed top-up (both exchanges
            # down/blocked) must not throw away 550 days of valid history.
            fresh = _klines_to_frame([])
        df5 = pd.concat([cached, fresh])
        df5 = df5[~df5.index.duplicated(keep="last")].sort_index()
    else:
        # Cold start: no cache to fall back on, so a fetch failure propagates.
        df5 = fetch(symbol, "5m", start_ms, end_ms).sort_index()
    _write_cache(path5, df5)
    df15 = _resample(df5, "15m")
    df5.attrs["symbol"] = symbol
    df15.attrs["symbol"] = symbol
    return df5, df15
