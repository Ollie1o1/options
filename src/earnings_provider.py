"""Finnhub free-tier provider for earnings dates and dividend yields.

Yahoo's earnings dates are unreliable, and the screener's IV-crush penalty and
earnings-move logic depend on them. Finnhub's free tier exposes a clean earnings
calendar and a stock-metrics endpoint (indicated annual dividend yield), no
payment required — just a free API key.

This module is a thin, isolated provider:

  * It activates only when an API key is available (``FINNHUB_API_KEY`` env var or
    config ``data_providers.finnhub_api_key``). Without a key every entry point
    returns None, so callers transparently fall back to the existing yfinance path.
  * HTTP is injected (``fetcher`` maps (url, params) -> parsed JSON) so the logic
    is unit-testable without network.
  * Nothing here raises; failures degrade to None.
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Callable, Optional

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT_S = 6

Fetcher = Callable[[str, dict], Any]


def resolve_api_key(config: Optional[dict] = None) -> Optional[str]:
    """Free Finnhub key from env or config; None if unset."""
    key = os.environ.get("FINNHUB_API_KEY")
    if key:
        return key.strip() or None
    try:
        if config is not None:
            k = (config.get("data_providers") or {}).get("finnhub_api_key")
            if k:
                return str(k).strip() or None
    except Exception:
        pass
    return None


def _http_json(url: str, params: dict) -> Any:
    import requests
    resp = requests.get(url, params=params, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def next_earnings_date(symbol: str,
                       api_key: Optional[str] = None,
                       fetcher: Fetcher = _http_json,
                       today: Optional[_dt.date] = None,
                       horizon_days: int = 200) -> Optional[_dt.datetime]:
    """Next confirmed/estimated earnings date at/after today, or None.

    Returns a tz-naive datetime at midnight UTC for the earliest future entry in
    Finnhub's earnings calendar. None when no key, no future entry, or any error.
    """
    try:
        if not api_key or not symbol:
            return None
        today = today or _dt.date.today()
        end = today + _dt.timedelta(days=horizon_days)
        js = fetcher(_BASE + "/calendar/earnings", {
            "from": today.isoformat(),
            "to": end.isoformat(),
            "symbol": symbol.upper(),
            "token": api_key,
        })
        rows = (js or {}).get("earningsCalendar") or []
        future: list[_dt.date] = []
        for r in rows:
            ds = r.get("date")
            if not ds:
                continue
            try:
                d = _dt.date.fromisoformat(str(ds)[:10])
            except ValueError:
                continue
            if d >= today:
                future.append(d)
        if not future:
            return None
        nxt = min(future)
        return _dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def dividend_yield(symbol: str,
                   api_key: Optional[str] = None,
                   fetcher: Fetcher = _http_json) -> Optional[float]:
    """Indicated annual dividend yield as a fraction (0.0045 == 0.45%), or None.

    None when no key, a non-payer (null yield), or any error.
    """
    try:
        if not api_key or not symbol:
            return None
        js = fetcher(_BASE + "/stock/metric", {
            "symbol": symbol.upper(),
            "metric": "all",
            "token": api_key,
        })
        val = ((js or {}).get("metric") or {}).get("dividendYieldIndicatedAnnual")
        if val is None:
            return None
        return round(float(val) / 100.0, 6)
    except Exception:
        return None
