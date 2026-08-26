"""Kalshi public market data — read-only, no key required.

Host matters: `api.elections.kalshi.com` serves market data unauthenticated
(verified 2026-08-26). The older `trading-api.kalshi.com` returns 401.

WHY THIS STORES BOTH SIDES AND NEVER A "PROBABILITY".

Measured on the live macro series 2026-08-26:

    KXFEDDECISION-28JAN-H0   yes 0.63 / 0.64      1-point spread
    KXFED-27APR-T3.75        yes 0.43 / 0.63     20-point spread
    KXCPI-26AUG-T0.8         yes 0.00 / 0.40     zero bid, one-sided

Every KXCPI contract had a zero bid. Collapsing 0.00/0.40 into a midpoint of
0.20 and calling it "the market's probability" would manufacture a precision
that does not exist — the same defect this codebase keeps finding, a number
describing something other than its label.

So the quote keeps bid, ask, last and open interest, and `mid()` REFUSES to
answer when the spread is too wide to mean anything. A caller that wants a
probability must say how much slop it will tolerate.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BASE = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 30
PAGE_LIMIT = 200

# Macro series worth archiving. Kalshi lists 13,486 series, overwhelmingly
# sports; archiving everything would be mostly noise. Add deliberately.
MACRO_SERIES = (
    "KXFED",            # fed funds upper bound thresholds
    "KXFEDDECISION",    # hike/cut size at a given meeting
    "KXCPI",            # monthly CPI change
    "KXECONSTATCPIYOY",  # year-over-year inflation
    "KXTERMINALRATE",   # highest fed funds rate
    "KXU3",             # unemployment rate
    "KXGDP",            # US GDP growth
    "KXRECSSNBER",      # recession call
)


@dataclass(frozen=True)
class Quote:
    series: str
    ticker: str
    title: Optional[str]
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    last: Optional[float]
    open_interest: Optional[float]
    close_time: Optional[str]

    @property
    def spread(self) -> Optional[float]:
        """Ask minus bid, or None if either side is absent."""
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def mid(self, max_spread: float = 0.05) -> Optional[float]:
        """Midpoint, but only when the market is tight enough to mean it.

        Returns None for a wide or one-sided book rather than inventing a
        number. A zero bid is treated as no bid: 0.00/0.40 is not a 20%
        probability, it is an absence of one.
        """
        if self.yes_bid is None or self.yes_ask is None:
            return None
        if self.yes_bid <= 0.0:
            return None
        spread = self.yes_ask - self.yes_bid
        if spread < 0 or spread > max_spread:
            return None
        return (self.yes_bid + self.yes_ask) / 2.0


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "options-screener/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _num(value: Any) -> Optional[float]:
    """float() or None. Never 0.0 as a stand-in for missing."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_markets(payload: Dict[str, Any], series: str) -> List[Quote]:
    """Quotes from one page. A market with no ticker is skipped."""
    out: List[Quote] = []
    for m in payload.get("markets") or []:
        ticker = m.get("ticker")
        if not ticker:
            continue
        out.append(Quote(
            series=series,
            ticker=str(ticker),
            title=m.get("title") or m.get("yes_sub_title"),
            yes_bid=_num(m.get("yes_bid_dollars")),
            yes_ask=_num(m.get("yes_ask_dollars")),
            last=_num(m.get("last_price_dollars")),
            open_interest=_num(m.get("open_interest_fp")),
            close_time=m.get("close_time"),
        ))
    return out


def fetch_series(series: str, max_pages: int = 10) -> List[Quote]:
    """Open markets for one series. [] on any failure — an archive job must
    never take down the scheduler."""
    quotes: List[Quote] = []
    cursor: Optional[str] = None
    try:
        for _ in range(max_pages):
            params = {"series_ticker": series, "limit": str(PAGE_LIMIT),
                      "status": "open"}
            if cursor:
                params["cursor"] = cursor
            payload = _get_json(f"{BASE}/markets?{urllib.parse.urlencode(params)}")
            quotes.extend(parse_markets(payload, series))
            cursor = payload.get("cursor")
            if not cursor:
                break
    except Exception:
        return []
    return quotes
