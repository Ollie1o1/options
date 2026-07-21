"""On-demand, single-ticker enrichment for the DISCOVER drill-down.

Plays the same role for one candidate that data.py plays for the whole
scan universe: an I/O layer, independently-degrading, never raising. Every
field here costs a network round-trip and is fetched only when the user
actually drills into a candidate — never for the whole scan (see
discover.py's module docstring for that cost discipline).

Descriptive context only, same as the rest of this package: short interest
and news are facts about the market's current positioning/coverage, never
a buy/sell recommendation.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from . import discover
from .discover import DeepRead

logger = logging.getLogger(__name__)


@dataclass
class DetailRead:
    ticker: str
    deep: DeepRead
    short_interest: Optional["object"] = None  # src.short_interest.ShortInterest
    news: Optional["object"] = None            # src.news_fetcher.NewsData


def fetch_detail(ticker: str, deep: Optional[DeepRead] = None) -> DetailRead:
    """Assemble a DetailRead for `ticker`. `deep` is reused as-is if the
    caller already has one (the top deep_limit ranked scan candidates);
    otherwise computed fresh here — this is what makes any ranked
    candidate drillable, not just the top few.

    short_interest and news are each independently try/except-wrapped: one
    dead source never blocks the other, matching deep_context()'s existing
    per-source isolation in discover.py.
    """
    if deep is None:
        deep = discover.deep_context(ticker)

    ticker_obj = None
    short_interest = None
    try:
        import yfinance as yf

        from src.short_interest import short_interest_detail

        ticker_obj = yf.Ticker(ticker)
        short_interest = short_interest_detail(ticker_obj.info)
    except Exception as exc:
        logger.debug("longterm detail: short interest fetch failed for %s: %s", ticker, exc)

    news = None
    try:
        from src.news_fetcher import fetch_news_and_events

        news = fetch_news_and_events(ticker, ticker_obj=ticker_obj)
    except Exception as exc:
        logger.debug("longterm detail: news fetch failed for %s: %s", ticker, exc)

    return DetailRead(ticker=ticker, deep=deep, short_interest=short_interest, news=news)
