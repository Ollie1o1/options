"""Multi-source news and event aggregator for the options screener.

Sources (all free, no paid API keys required):
  1. Yahoo Finance RSS  — per-ticker feed, real-time headlines
  2. Finviz ticker news — curated financial news with source attribution
  3. yfinance analyst recommendations — upgrade / downgrade history
  4. Alpha Vantage NEWS_SENTIMENT — optional; set ALPHA_VANTAGE_API_KEY in .env

Design principles:
  - Every source is independently try/except-wrapped — one failure never kills the rest
  - Results are cached in-process for the duration of a scan run
  - Returns a NewsData object that is always safe to consume (never None)
  - Zero new pip dependencies beyond what is already in requirements.txt
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from email.utils import parsedate_to_datetime

_NEWS_CACHE: dict = {}


def _news_cache_key(symbol: str) -> str:
    hour = datetime.utcnow().strftime("%Y%m%d%H")
    return f"{symbol.upper()}:{hour}"

import requests

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False

try:
    from finvizfinance.quote import finvizquote
    _HAS_FINVIZ = True
except Exception:
    _HAS_FINVIZ = False

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    headline: str
    source: str          # e.g. "Yahoo Finance", "Reuters", "Seeking Alpha"
    published: datetime
    sentiment: float     # TextBlob polarity: -1.0 (very negative) … +1.0 (very positive)
    url: str = ""
    relevance: float = 1.0   # 0–1; boosted when ticker is in headline


@dataclass
class AnalystChange:
    firm: str
    action: str          # "upgrade", "downgrade", "initiate", "reiterate", "maintain"
    from_grade: str
    to_grade: str
    date: datetime
    price_target: Optional[float] = None


@dataclass
class NewsData:
    symbol: str
    items: List[NewsItem] = field(default_factory=list)
    analyst_changes: List[AnalystChange] = field(default_factory=list)
    # Aggregate sentiment: weighted mean of individual item sentiments (-1 … +1)
    aggregate_sentiment: float = 0.0
    # Top 3 headline strings for compact display and AI context
    top_headlines: List[str] = field(default_factory=list)
    # Flags derived from headlines / sentiment
    has_negative_catalyst: bool = False  # strong sell-off risk keywords detected
    has_positive_catalyst: bool = False  # strong gap-up risk keywords detected
    # Unusual news volume: True when >5 headlines in last 24h (can spike IV)
    unusual_news_volume: bool = False
    fetched_at: Optional[datetime] = None


# ── Sentiment helpers ──────────────────────────────────────────────────────────

_NEGATIVE_KEYWORDS = {
    "miss", "misses", "disappoints", "fraud", "recall", "lawsuit", "downgrade",
    "guidance cut", "lays off", "layoff", "bankruptcy", "investigation", "subpoena",
    "warning", "revenue miss", "earnings miss", "profit warning", "fine", "penalty",
    "accounting", "restatement", "delisted", "sec probe", "class action", "short",
}

_POSITIVE_KEYWORDS = {
    "beat", "beats", "upgrade", "raised", "record", "buyback", "acquisition",
    "merger", "dividend increase", "guidance raised", "raised guidance",
    "blowout", "smashes", "exceeds", "strong demand", "fda approval", "fda approves",
    "partnership", "deal", "contract", "milestone", "breakout",
}


def _sanitize(text: str) -> str:
    """Strip non-ASCII and common mojibake from RSS feed text."""
    return text.encode("ascii", errors="replace").decode("ascii").replace("?", " ").strip()


def _score_headline_sentiment(headline: str) -> float:
    """Compute sentiment for one headline. TextBlob if available, keyword fallback."""
    if not headline:
        return 0.0
    hl_lower = headline.lower()

    if _HAS_TEXTBLOB:
        try:
            tb_score = TextBlob(headline).sentiment.polarity
        except Exception:
            tb_score = 0.0
    else:
        tb_score = 0.0

    # Keyword boost — nudge scores toward meaningful extremes
    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in hl_lower)
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in hl_lower)
    kw_boost = (pos_hits - neg_hits) * 0.15

    return max(-1.0, min(1.0, tb_score + kw_boost))


def _boost_relevance(headline: str, symbol: str) -> float:
    """Return 1.0 if the ticker appears in the headline, else 0.6."""
    if symbol.lower() in headline.lower():
        return 1.0
    return 0.6


# ── Source 1: Yahoo Finance RSS ────────────────────────────────────────────────

_YF_RSS_URL = "https://finance.yahoo.com/rss/headline?s={symbol}"
_RSS_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _fetch_yf_rss(symbol: str, max_age_hours: int = 72) -> List[NewsItem]:
    """Fetch and parse Yahoo Finance per-ticker RSS feed."""
    items: List[NewsItem] = []
    try:
        url = _YF_RSS_URL.format(symbol=symbol)
        resp = requests.get(url, timeout=10, headers={"User-Agent": "options-screener/1.0"})
        if resp.status_code != 200:
            return items
        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return items

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        for entry in channel.findall("item"):
            title_el = entry.find("title")
            link_el = entry.find("link")
            pub_el = entry.find("pubDate")
            source_el = entry.find("source")

            headline = _sanitize(title_el.text.strip()) if title_el is not None and title_el.text else ""
            if not headline:
                continue

            url_str = link_el.text.strip() if link_el is not None and link_el.text else ""
            source_name = _sanitize(source_el.text.strip()) if source_el is not None and source_el.text else "Yahoo Finance"

            pub_dt = datetime.now(timezone.utc)
            if pub_el is not None and pub_el.text:
                try:
                    pub_dt = parsedate_to_datetime(pub_el.text.strip())
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            if pub_dt < cutoff:
                continue

            sentiment = _score_headline_sentiment(headline)
            relevance = _boost_relevance(headline, symbol)
            items.append(NewsItem(
                headline=headline,
                source=source_name,
                published=pub_dt,
                sentiment=sentiment,
                url=url_str,
                relevance=relevance,
            ))

    except Exception as exc:
        logger.debug("YF RSS fetch failed for %s: %s", symbol, exc)
    return items


# ── Source 2: Finviz ticker news ───────────────────────────────────────────────

def _fetch_finviz_news(symbol: str, max_age_hours: int = 72) -> List[NewsItem]:
    """Fetch news from Finviz via finvizfinance library."""
    items: List[NewsItem] = []
    if not _HAS_FINVIZ:
        return items
    try:
        import pandas as pd
        fq = finvizquote(symbol)
        news_df = fq.ticker_news()
        if news_df is None or (hasattr(news_df, "empty") and news_df.empty):
            return items

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        for _, row in news_df.iterrows():
            headline = _sanitize(str(row.get("Title", "") or "").strip())
            if not headline:
                continue
            source = _sanitize(str(row.get("Source", "Finviz") or "Finviz").strip())
            url_str = str(row.get("Link", "") or "").strip()

            raw_date = row.get("Date", None)
            pub_dt = datetime.now(timezone.utc)
            if raw_date is not None:
                try:
                    if hasattr(raw_date, "tzinfo"):
                        pub_dt = raw_date if raw_date.tzinfo else raw_date.replace(tzinfo=timezone.utc)
                    else:
                        pub_dt = pd.to_datetime(raw_date, utc=True).to_pydatetime()
                except Exception:
                    pass

            if pub_dt < cutoff:
                continue

            sentiment = _score_headline_sentiment(headline)
            relevance = _boost_relevance(headline, symbol)
            items.append(NewsItem(
                headline=headline,
                source=source,
                published=pub_dt,
                sentiment=sentiment,
                url=url_str,
                relevance=relevance,
            ))
    except Exception as exc:
        logger.debug("Finviz news fetch failed for %s: %s", symbol, exc)
    return items


# ── Source 3: Analyst upgrades / downgrades via yfinance ──────────────────────

_ACTION_MAP = {
    "up": "upgrade", "down": "downgrade", "init": "initiate",
    "reit": "reiterate", "main": "maintain", "strong buy": "strong buy",
}


def _normalise_action(raw: str) -> str:
    rl = raw.lower()
    for key, val in _ACTION_MAP.items():
        if key in rl:
            return val
    return raw.strip()


def _fetch_analyst_changes(ticker_obj, days: int = 30) -> List[AnalystChange]:
    """Pull analyst recommendation changes from yfinance."""
    changes: List[AnalystChange] = []
    try:
        import pandas as pd
        recs = ticker_obj.recommendations
        if recs is None or (hasattr(recs, "empty") and recs.empty):
            return changes

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for idx, row in recs.iterrows():
            try:
                if hasattr(idx, "tzinfo"):
                    rec_dt = idx if idx.tzinfo else idx.replace(tzinfo=timezone.utc)
                else:
                    rec_dt = pd.Timestamp(idx).tz_localize("UTC")

                if rec_dt < cutoff:
                    continue

                firm = str(row.get("Firm", "") or "").strip()
                to_grade = str(row.get("To Grade", "") or "").strip()
                from_grade = str(row.get("From Grade", "") or "").strip()
                action = _normalise_action(str(row.get("Action", "maintain") or "maintain"))
                pt = None
                for pt_col in ["Price Target", "Target", "price_target"]:
                    if pt_col in row and row[pt_col]:
                        try:
                            pt = float(row[pt_col])
                        except Exception:
                            pass
                        break

                if firm or to_grade:
                    changes.append(AnalystChange(
                        firm=firm,
                        action=action,
                        from_grade=from_grade,
                        to_grade=to_grade,
                        date=rec_dt,
                        price_target=pt,
                    ))
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Analyst changes fetch failed: %s", exc)
    return changes


# ── Source 4: Alpha Vantage (optional) ────────────────────────────────────────

_AV_URL = "https://www.alphavantage.co/query"


def _fetch_alpha_vantage(symbol: str, api_key: str, max_age_hours: int = 72) -> List[NewsItem]:
    """Fetch news sentiment from Alpha Vantage NEWS_SENTIMENT endpoint (free tier)."""
    items: List[NewsItem] = []
    try:
        from datetime import timezone as tz
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": api_key,
            "limit": 20,
            "sort": "LATEST",
        }
        resp = requests.get(_AV_URL, params=params, timeout=12)
        if resp.status_code != 200:
            return items
        data = resp.json()
        feed = data.get("feed", [])
        if not feed:
            return items

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        for article in feed:
            headline = _sanitize(article.get("title", "").strip())
            if not headline:
                continue
            source = _sanitize(article.get("source", "Alpha Vantage").strip())
            url_str = article.get("url", "").strip()

            pub_str = article.get("time_published", "")
            pub_dt = datetime.now(timezone.utc)
            if pub_str:
                try:
                    pub_dt = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            if pub_dt < cutoff:
                continue

            # Alpha Vantage provides ticker-level sentiment scores
            av_sentiment = 0.0
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    try:
                        av_sentiment = float(ts.get("ticker_sentiment_score", 0.0))
                    except Exception:
                        pass
                    break

            relevance = 1.0
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    try:
                        relevance = float(ts.get("relevance_score", 1.0))
                    except Exception:
                        pass
                    break

            items.append(NewsItem(
                headline=headline,
                source=source,
                published=pub_dt,
                sentiment=av_sentiment,
                url=url_str,
                relevance=relevance,
            ))
    except Exception as exc:
        logger.debug("Alpha Vantage news fetch failed for %s: %s", symbol, exc)
    return items


# ── Polygon news helper ────────────────────────────────────────────────────────

def _fetch_polygon_news_items(symbol: str, client, max_age_hours: int = 72) -> List[NewsItem]:
    """Fetch and convert Polygon.io news items for *symbol*."""
    items: List[NewsItem] = []
    try:
        raw_poly = client.get_news(symbol, limit=10, max_age_hours=max_age_hours)
        if raw_poly:
            for pi in raw_poly:
                headline = _sanitize(pi.headline)
                if not headline:
                    continue
                items.append(NewsItem(
                    headline=headline,
                    source=f"Polygon/{pi.publisher}",
                    published=pi.published_utc,
                    sentiment=pi.sentiment,
                    url=pi.url,
                    relevance=1.2,   # slightly higher priority in sort
                ))
    except Exception as exc:
        logger.debug("Polygon news fetch failed for %s: %s", symbol, exc)
    return items


# ── Aggregation ────────────────────────────────────────────────────────────────

def _deduplicate(items: List[NewsItem]) -> List[NewsItem]:
    """Remove near-duplicate headlines (same first 60 chars)."""
    seen: set[str] = set()
    out: List[NewsItem] = []
    for item in items:
        key = item.headline[:60].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _compute_aggregate_sentiment(items: List[NewsItem]) -> float:
    """Weighted mean sentiment, with higher-relevance items weighted more."""
    if not items:
        return 0.0
    total_w, total_s = 0.0, 0.0
    for it in items:
        w = it.relevance
        total_w += w
        total_s += it.sentiment * w
    return total_s / total_w if total_w > 0 else 0.0


def fetch_news_and_events(
    symbol: str,
    ticker_obj=None,
    max_age_hours: int = 72,
    max_headlines: int = 5,
) -> NewsData:
    """
    Aggregate news and analyst events for *symbol* from all configured sources.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. "AAPL").
    ticker_obj:
        A yfinance Ticker object (already instantiated by the caller).
        Used for analyst recommendation fetching.
    max_age_hours:
        Only include items published within this window.
    max_headlines:
        Maximum number of items to include in NewsData.items.

    Returns
    -------
    NewsData — always a valid object, never raises.
    """
    key = _news_cache_key(symbol)
    if key in _NEWS_CACHE:
        return _NEWS_CACHE[key]

    result = NewsData(
        symbol=symbol,
        fetched_at=datetime.now(timezone.utc),
    )

    # Build concurrent fetch list
    fetch_fns = [
        ("yf_rss", lambda: _fetch_yf_rss(symbol, max_age_hours=max_age_hours)),
        ("finviz", lambda: _fetch_finviz_news(symbol, max_age_hours=max_age_hours)),
    ]
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if av_key:
        fetch_fns.append(("alpha_vantage", lambda: _fetch_alpha_vantage(symbol, av_key, max_age_hours=max_age_hours)))
    # Polygon concurrent fetch
    try:
        from src.polygon_client import PolygonClient as _PC
        _poly = _PC()
        if _poly._enabled:
            fetch_fns.append(("polygon", lambda: _fetch_polygon_news_items(symbol, _poly, max_age_hours)))
    except Exception:
        pass

    all_items: List[NewsItem] = []
    with ThreadPoolExecutor(max_workers=len(fetch_fns)) as executor:
        futures = {executor.submit(fn): name for name, fn in fetch_fns}
        for future in as_completed(futures, timeout=10):
            try:
                items = future.result(timeout=0)
                if items:
                    all_items.extend(items)
            except Exception:
                pass

    # Merge, deduplicate, sort newest-first (Polygon items get relevance boost)
    all_items = _deduplicate(all_items)
    all_items.sort(key=lambda x: (x.published, x.relevance), reverse=True)

    result.items = all_items[:max_headlines]

    # 4. Analyst changes (from yfinance object if available)
    if ticker_obj is not None:
        result.analyst_changes = _fetch_analyst_changes(ticker_obj, days=30)

    # Compute aggregate sentiment
    result.aggregate_sentiment = _compute_aggregate_sentiment(result.items)

    # Top headlines (plain text, for display + AI)
    result.top_headlines = [it.headline for it in result.items[:3]]

    # Catalyst flags — scan ALL fetched headlines for catalyst keywords
    all_headlines_lower = " ".join(it.headline.lower() for it in all_items)
    result.has_negative_catalyst = any(kw in all_headlines_lower for kw in _NEGATIVE_KEYWORDS)
    result.has_positive_catalyst = any(kw in all_headlines_lower for kw in _POSITIVE_KEYWORDS)

    # Unusual news volume: more than 5 articles in last 24h
    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_count = sum(1 for it in all_items if it.published >= recent_cutoff)
    result.unusual_news_volume = recent_count > 5

    _NEWS_CACHE[key] = result
    return result


# ── Display helpers ────────────────────────────────────────────────────────────

_SENTIMENT_BAR = {
    (0.3, 1.1):   ("(+) Positive", "\033[92m"),   # green
    (-0.3, 0.3):  ("(~) Neutral",  "\033[93m"),   # yellow
    (-1.1, -0.3): ("(-) Negative", "\033[91m"),   # red
}


def _sentiment_label(score: float) -> tuple[str, str]:
    """Return (label, ansi_color) for a sentiment score."""
    for (lo, hi), (label, color) in _SENTIMENT_BAR.items():
        if lo <= score < hi:
            return label, color
    return "● Neutral", "\033[93m"


def format_news_panel(
    news_data: NewsData,
    width: int = 100,
    use_color: bool = True,
) -> str:
    """
    Render a compact news panel for CLI display.

    Returns a multi-line string ready to be printed.
    """
    RESET = "\033[0m" if use_color else ""
    DIM   = "\033[2m"  if use_color else ""
    BOLD  = "\033[1m"  if use_color else ""
    CYAN  = "\033[96m" if use_color else ""
    RED   = "\033[91m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    YELLOW = "\033[93m" if use_color else ""
    MAGENTA = "\033[95m" if use_color else ""

    lines: list[str] = []

    sym = news_data.symbol
    now = datetime.now(timezone.utc)
    age_str = f"last {72}h"

    lines.append(f"\n{BOLD}{CYAN}  NEWS & EVENTS  --  {sym}  ({age_str}){RESET}")
    lines.append(f"  {'-' * min(width - 4, 76)}")

    if not news_data.items and not news_data.analyst_changes:
        lines.append(f"  {DIM}No recent news found.{RESET}")
        return "\n".join(lines)

    # Aggregate sentiment line
    s_label, s_color = _sentiment_label(news_data.aggregate_sentiment)
    s_color_esc = s_color if use_color else ""
    flags = []
    if news_data.has_negative_catalyst:
        flags.append(f"{RED}CATALYST-RISK{RESET}")
    if news_data.has_positive_catalyst:
        flags.append(f"{GREEN}CATALYST-UPSIDE{RESET}")
    if news_data.unusual_news_volume:
        flags.append(f"{YELLOW}UNUSUAL NEWS VOLUME{RESET}")
    flag_str = "  " + "  ".join(flags) if flags else ""
    lines.append(
        f"  Sentiment: {s_color_esc}{s_label}{RESET} "
        f"({news_data.aggregate_sentiment:+.2f}){flag_str}"
    )

    # Headlines
    if news_data.items:
        lines.append("")
        for i, item in enumerate(news_data.items[:5], 1):
            s_col = GREEN if item.sentiment > 0.1 else (RED if item.sentiment < -0.1 else DIM)
            age = now - item.published
            if age.total_seconds() < 3600:
                age_label = f"{int(age.total_seconds()/60)}m ago"
            elif age.total_seconds() < 86400:
                age_label = f"{int(age.total_seconds()/3600)}h ago"
            else:
                age_label = f"{int(age.total_seconds()/86400)}d ago"

            headline_trunc = item.headline[:width - 42] + "…" if len(item.headline) > width - 42 else item.headline
            lines.append(
                f"  {DIM}{i}.{RESET} {headline_trunc}\n"
                f"     {DIM}{item.source}  |  {age_label}  |  {s_col}{item.sentiment:+.2f}{RESET}"
            )

    # Analyst changes
    if news_data.analyst_changes:
        lines.append(f"\n  {BOLD}Analyst Activity (last 30d):{RESET}")
        for ch in news_data.analyst_changes[:4]:
            action = ch.action.upper()
            action_color = GREEN if ch.action in ("upgrade", "strong buy", "initiate") else (RED if ch.action == "downgrade" else DIM)
            grade_str = f"{ch.from_grade} → {ch.to_grade}" if ch.from_grade and ch.to_grade else (ch.to_grade or ch.from_grade)
            pt_str = f"  PT: ${ch.price_target:.0f}" if ch.price_target else ""
            date_str = ch.date.strftime("%b %d")
            lines.append(
                f"  {action_color}{action}{RESET}  {BOLD}{ch.firm}{RESET}  "
                f"{DIM}{grade_str}{pt_str}  |  {date_str}{RESET}"
            )

    lines.append(f"  {'-' * min(width - 4, 76)}")
    return "\n".join(lines)
