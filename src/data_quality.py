"""
Data-quality helpers: quote freshness classification and market-hours check.

Pure and network-free so they are unit-testable offline. This module is the
single source of truth for market-hours logic — ``options_screener._check_market_hours``
delegates here so both paths agree.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# US market holidays (NYSE/CBOE) — update annually.
_US_MARKET_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}


def check_market_hours() -> Tuple[bool, str]:
    """
    Returns (is_open: bool, message: str) in US Eastern time.

    Options are liquid 09:30-16:00 ET, Mon-Fri. Outside this window,
    yfinance data is stale and bid-ask spreads are unreliable.
    """
    try:
        from zoneinfo import ZoneInfo
        et_zone = ZoneInfo("America/New_York")
        now_et = datetime.now(et_zone)
    except Exception:
        # Fallback: rough UTC-4 (EDT) — acceptable for a warning-only check
        now_et = datetime.now(timezone(timedelta(hours=-4)))

    weekday = now_et.weekday()  # 0=Mon … 6=Sun
    hhmm = now_et.hour * 100 + now_et.minute
    time_str = now_et.strftime("%H:%M ET")
    date_str = now_et.strftime("%Y-%m-%d")

    if date_str in _US_MARKET_HOLIDAYS_2026:
        return False, f"Markets closed — US market holiday ({time_str}). Data is stale."

    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets closed — it's {day_name} ({time_str}). Data is stale."

    if hhmm < 930:
        return False, f"Pre-market ({time_str}). Options open at 09:30 ET — bid/ask spreads not reliable yet."

    if hhmm >= 1600:
        return False, f"After-hours ({time_str}). Options closed at 16:00 ET — quotes are stale."

    return True, f"Market open ({time_str})"


def classify_quote_freshness(quote_age_min, market_open: bool) -> str:
    """
    Classify how much to trust a quote given its age and market state.

    Returns one of:
      - "fresh"   — quote age <= 20 min during market hours
      - "delayed" — quote age <= 120 min during market hours
      - "stale"   — age > 120 min, OR any age while the market is closed
      - "unknown" — no usable timestamp (None / NaN)

    ``quote_age_min`` is minutes between the quote's strike time and the fetch.
    A small negative age (timestamp marginally in the future due to clock skew)
    is treated as fresh.
    """
    # Missing / unparseable timestamp -> we genuinely don't know.
    if quote_age_min is None:
        return "unknown"
    try:
        age = float(quote_age_min)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(age):
        return "unknown"

    # Market closed: any quote we have is stale by definition.
    if not market_open:
        return "stale"

    if age <= 20.0:
        return "fresh"
    if age <= 120.0:
        return "delayed"
    return "stale"
