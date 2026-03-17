"""Polygon.io API client — opt-in enrichment layer.

Silent no-op when ``POLYGON_API_KEY`` is absent from the environment.
All methods return ``None`` on any error; they never raise.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"


@dataclass
class PolygonNewsItem:
    headline: str
    publisher: str
    published_utc: datetime
    sentiment: float          # 0.0 when not provided by API
    url: str
    tickers: list[str] = field(default_factory=list)


class PolygonClient:
    """Thin wrapper around the Polygon.io REST API.

    Parameters
    ----------
    api_key:
        Polygon API key.  When omitted the constructor reads
        ``os.environ.get("POLYGON_API_KEY")``.  If no key is found the
        client is disabled and all methods immediately return ``None``.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("POLYGON_API_KEY", "")
        self._enabled: bool = bool(key)
        self._headers: dict[str, str] = {"Authorization": f"Bearer {key}"} if key else {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a GET request and return the parsed JSON body, or ``None``."""
        if not self._enabled:
            return None
        url = f"{BASE_URL}{path}"
        try:
            resp = requests.get(url, headers=self._headers, params=params or {}, timeout=5)
            if resp.status_code == 429:
                logger.debug("Polygon rate limit (429) for %s", path)
                return None
            if not resp.ok:
                logger.debug("Polygon HTTP %s for %s", resp.status_code, path)
                return None
            return resp.json()
        except requests.exceptions.Timeout:
            logger.debug("Polygon request timed out: %s", path)
            return None
        except Exception as exc:
            logger.debug("Polygon request failed for %s: %s", path, exc)
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_news(
        self,
        ticker: str,
        limit: int = 10,
        max_age_hours: int = 72,
    ) -> Optional[list[PolygonNewsItem]]:
        """Fetch recent ticker-filtered news articles.

        Returns
        -------
        list[PolygonNewsItem] or None
            ``None`` when the client is disabled or the request fails.
            An empty list means the request succeeded but no articles matched.
        """
        if not self._enabled:
            return None
        data = self._get(
            "/v2/reference/news",
            params={
                "ticker": ticker,
                "limit": limit,
                "order": "desc",
                "sort": "published_utc",
            },
        )
        if data is None:
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items: list[PolygonNewsItem] = []
        for article in data.get("results", []):
            pub_str = article.get("published_utc", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except Exception:
                pub_dt = datetime.now(timezone.utc)

            if pub_dt < cutoff:
                continue

            # Polygon may or may not include sentiment per article
            sentiment = 0.0
            insights = article.get("insights", [])
            for ins in insights:
                if ins.get("ticker", "").upper() == ticker.upper():
                    raw = ins.get("sentiment", "")
                    if raw == "positive":
                        sentiment = 0.5
                    elif raw == "negative":
                        sentiment = -0.5
                    break

            items.append(PolygonNewsItem(
                headline=article.get("title", "").strip(),
                publisher=article.get("publisher", {}).get("name", "Polygon") if isinstance(article.get("publisher"), dict) else str(article.get("publisher", "Polygon")),
                published_utc=pub_dt,
                sentiment=sentiment,
                url=article.get("article_url", "").strip(),
                tickers=article.get("tickers", []),
            ))
        return items

    def get_snapshot(self, ticker: str) -> Optional[dict]:
        """Fetch the current day snapshot for *ticker* (includes VWAP).

        Returns the raw Polygon response dict (``ticker`` key contains the data).
        """
        if not self._enabled:
            return None
        time.sleep(0.2)
        return self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")

    def get_options_snapshot(
        self,
        ticker: str,
        limit: int = 250,
    ) -> Optional[list[dict]]:
        """Fetch the options chain snapshot for *ticker*.

        Returns the ``results`` list from Polygon's v3 snapshot endpoint.
        """
        if not self._enabled:
            return None
        time.sleep(0.2)
        data = self._get(
            f"/v3/snapshot/options/{ticker}",
            params={"limit": limit},
        )
        if data is None:
            return None
        return data.get("results", [])

    def get_ticker_details(self, ticker: str) -> Optional[dict]:
        """Fetch reference details for *ticker* (description, market cap, etc.)."""
        if not self._enabled:
            return None
        time.sleep(0.2)
        data = self._get(f"/v3/reference/tickers/{ticker}")
        if data is None:
            return None
        return data.get("results")

    def get_unusual_options_flow(
        self,
        ticker: str,
        min_premium: int = 10_000,
    ) -> Optional[list[dict]]:
        """Return options contracts with unusual volume/OI and large dollar premium.

        Filters the full options snapshot for contracts where:
        - ``day.volume / open_interest > 2.0``
        - ``day.volume * details.strike_price * 100 >= min_premium``

        Returns results sorted by dollar volume descending.
        """
        if not self._enabled:
            return None
        snapshot = self.get_options_snapshot(ticker)
        if snapshot is None:
            return None

        unusual: list[dict] = []
        for contract in snapshot:
            day = contract.get("day") or {}
            details = contract.get("details") or {}
            oi = contract.get("open_interest") or 0
            vol = day.get("volume") or 0
            strike = details.get("strike_price") or 0

            if oi <= 0 or vol <= 0 or strike <= 0:
                continue

            vol_oi_ratio = vol / oi
            dollar_vol = vol * strike * 100

            if vol_oi_ratio > 2.0 and dollar_vol >= min_premium:
                contract["_dollar_volume"] = dollar_vol
                contract["_vol_oi_ratio"] = vol_oi_ratio
                unusual.append(contract)

        unusual.sort(key=lambda c: c.get("_dollar_volume", 0), reverse=True)
        return unusual
