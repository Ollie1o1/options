"""I/O: fetch + parse the free sources. Every fetch is failure-safe ([]/None)
and short-timeout — the world pulse must never delay or break the screener.

Validated live 2026-06-11: Google News RSS (true publisher in each item's
<source url=...>), CNBC/MarketWatch RSS, StockTwits symbol streams (tagged
Bullish/Bearish). CNN Fear & Greed rejects bots — deliberately not used.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

TIMEOUT_S = 8

GOOGLE_NEWS_TOPICS = ["federal reserve", "inflation", "stock market",
                      "tariffs trade", "geopolitics conflict"]
RSS_FEEDS = {
    "cnbc.com": ("https://search.cnbc.com/rs/search/combinedcms/view.xml"
                 "?partnerId=wrss01&id=100003114"),
    "marketwatch.com": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
}
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except Exception:
        return ""


def parse_rss(xml_text: str, default_source: str = "") -> List[Dict[str, Any]]:
    """RSS 2.0 → [{title, source, published, url}]. Pure; never raises.
    Google News items carry the true publisher in <source url=...>."""
    if not xml_text or "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        published: Optional[datetime] = None
        pub = item.findtext("pubDate")
        if pub:
            try:
                published = parsedate_to_datetime(pub)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                published = None
        src_el = item.find("source")
        source = default_source
        if src_el is not None:
            source = _domain(src_el.get("url") or "") or default_source
        out.append({"title": title, "source": source, "published": published,
                    "url": (item.findtext("link") or "").strip()})
    return out


def parse_stocktwits(payload: Dict[str, Any]) -> Dict[str, Any]:
    """StockTwits stream → {tagged, bull_ratio}. Pure."""
    bull = bear = 0
    for m in (payload or {}).get("messages") or []:
        tag = (((m.get("entities") or {}).get("sentiment") or {}) or {}).get("basic")
        if tag == "Bullish":
            bull += 1
        elif tag == "Bearish":
            bear += 1
    tagged = bull + bear
    return {"tagged": tagged,
            "bull_ratio": (bull / tagged) if tagged else None}


# ── live fetchers (failure-safe) ─────────────────────────────────────────────

def _get(url: str):
    import requests
    return requests.get(url, timeout=TIMEOUT_S,
                        headers={"User-Agent": "options-screener/1.0"})


def fetch_google_news(topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for topic in topics or GOOGLE_NEWS_TOPICS:
        try:
            q = topic.replace(" ", "+")
            resp = _get(f"https://news.google.com/rss/search?q={q}"
                        f"&hl=en-US&gl=US&ceid=US:en")
            items.extend(parse_rss(resp.text, default_source="news.google.com"))
        except Exception:
            continue
    return items


def fetch_direct_feeds() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for domain, url in RSS_FEEDS.items():
        try:
            items.extend(parse_rss(_get(url).text, default_source=domain))
        except Exception:
            continue
    return items


def fetch_crowd(symbols=("SPY", "QQQ")) -> Dict[str, Any]:
    """StockTwits crowd gauge across index proxies."""
    bull = tagged = 0
    for sym in symbols:
        try:
            r = parse_stocktwits(_get(STOCKTWITS_URL.format(symbol=sym)).json())
            if r["bull_ratio"] is not None:
                bull += round(r["bull_ratio"] * r["tagged"])
                tagged += r["tagged"]
        except Exception:
            continue
    return {"tagged": tagged, "bull_ratio": (bull / tagged) if tagged else None}


def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        key = (it.get("title") or "").lower()[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def fetch_all(topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    return dedupe(fetch_google_news(topics) + fetch_direct_feeds())
