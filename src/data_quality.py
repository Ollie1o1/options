"""
Data-quality helpers: quote freshness classification and market-hours check.

Pure and network-free so they are unit-testable offline. This module is the
single source of truth for market-hours logic — ``options_screener._check_market_hours``
delegates here so both paths agree.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

logger = logging.getLogger("data_quality")

# Verification tolerance: Yahoo's IV is "verified" when within this fraction of
# the IV implied by the contract's own mid price.
IV_VERIFY_TOLERANCE = 0.15

# US market holidays (NYSE/CBOE), keyed by year — add each new year's set.
# check_market_hours warns loudly when the current year is missing, so a
# stale calendar is visible instead of silently reporting "open" on holidays.
_US_MARKET_HOLIDAYS_BY_YEAR = {
    2026: {
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
    },
}


def check_market_hours(now_et: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Returns (is_open: bool, message: str) in US Eastern time.

    Options are liquid 09:30-16:00 ET, Mon-Fri. Outside this window,
    yfinance data is stale and bid-ask spreads are unreliable.

    ``now_et`` is injectable for tests; when omitted, the current US/Eastern
    time is used.
    """
    if now_et is None:
        try:
            from zoneinfo import ZoneInfo
            et_zone = ZoneInfo("America/New_York")
            now_et = datetime.now(et_zone)
        except Exception:
            # Fallback: rough UTC-4 (EDT; off by an hour under EST winter
            # time) — acceptable for a warning-only check
            now_et = datetime.now(timezone(timedelta(hours=-4)))

    weekday = now_et.weekday()  # 0=Mon … 6=Sun
    hhmm = now_et.hour * 100 + now_et.minute
    time_str = now_et.strftime("%H:%M ET")
    date_str = now_et.strftime("%Y-%m-%d")

    holidays = _US_MARKET_HOLIDAYS_BY_YEAR.get(now_et.year)
    if holidays is None:
        holidays = set()
        cal_warn = (f" [WARNING: holiday calendar not maintained for "
                    f"{now_et.year} — update _US_MARKET_HOLIDAYS_BY_YEAR in "
                    f"src/data_quality.py; holiday closures NOT detected]")
    else:
        cal_warn = ""

    if date_str in holidays:
        return False, f"Markets closed — US market holiday ({time_str}). Data is stale."

    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return False, f"Markets closed — it's {day_name} ({time_str}). Data is stale.{cal_warn}"

    if hhmm < 930:
        return False, f"Pre-market ({time_str}). Options open at 09:30 ET — bid/ask spreads not reliable yet.{cal_warn}"

    if hhmm >= 1600:
        return False, f"After-hours ({time_str}). Options closed at 16:00 ET — quotes are stale.{cal_warn}"

    return True, f"Market open ({time_str}){cal_warn}"


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


def implied_vol_from_price(option_type, S, K, T, r, market_price, q: float = 0.0) -> Optional[float]:
    """
    Solve implied volatility from a Black-Scholes price using Brent's method.

    Returns the sigma in [0.005, 5.0] that reprices the option to ``market_price``,
    or None when:
      - any input is invalid (non-positive S/K/T/price, or NaN), or
      - the price lies outside the model's range for that bracket (e.g. below
        intrinsic, or above the maximum achievable price), so no root exists.
    """
    from scipy.optimize import brentq
    from src.utils import bs_price

    try:
        S = float(S); K = float(K); T = float(T); r = float(r)
        mp = float(market_price); q = float(q)
    except (TypeError, ValueError):
        return None
    if any(math.isnan(x) for x in (S, K, T, r, mp, q)):
        return None
    if S <= 0 or K <= 0 or T <= 0 or mp <= 0:
        return None

    lo, hi = 0.005, 5.0

    def _f(sigma: float) -> float:
        return float(bs_price(option_type, S, K, T, r, sigma, q)) - mp

    try:
        f_lo = _f(lo)
        f_hi = _f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            return None
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        # No sign change -> price is outside the achievable range (below
        # intrinsic or above the model max): no implied vol exists.
        if f_lo * f_hi > 0:
            return None
        return float(brentq(_f, lo, hi, maxiter=100, xtol=1e-10))
    except (ValueError, RuntimeError):
        return None


def cross_validate_iv(df, r: float):
    """
    Verify Yahoo's reported impliedVolatility against the IV implied by each
    contract's own mid price.

    For every row with a valid ``mid`` and valid underlying/strike/T_years,
    solves IV from the mid and adds three columns:
      - ``iv_solved``       float — solved IV (NaN when unsolvable)
      - ``iv_residual_pct`` float — (yahoo_iv - iv_solved) / iv_solved (NaN if either missing)
      - ``iv_verified``     object — True if |residual| <= 15%, False if not,
                            None when no IV could be solved.

    Pure: does not mutate ``impliedVolatility`` (the caller decides whether to
    adopt ``iv_solved``). A per-contract dividend yield is read from a
    ``dividend_yield`` column when present, else 0.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    n = len(df)
    iv_solved = np.full(n, np.nan, dtype=float)
    iv_residual = np.full(n, np.nan, dtype=float)
    iv_verified: list = [None] * n

    if n == 0:
        df["iv_solved"] = iv_solved
        df["iv_residual_pct"] = iv_residual
        df["iv_verified"] = iv_verified
        return df

    has_div = "dividend_yield" in df.columns
    types = df.get("type")
    for pos, (idx, row) in enumerate(df.iterrows()):
        try:
            mid = row.get("mid", np.nan)
            S = row.get("underlying", np.nan)
            K = row.get("strike", np.nan)
            T = row.get("T_years", np.nan)
            opt = row.get("type", "call")
            q = float(row.get("dividend_yield", 0.0)) if has_div else 0.0
            if pd.isna(mid) or float(mid) <= 0:
                continue
            solved = implied_vol_from_price(opt, S, K, T, r, float(mid), q)
            if solved is None or solved <= 0:
                continue
            iv_solved[pos] = solved
            yahoo_iv = row.get("impliedVolatility", np.nan)
            if pd.notna(yahoo_iv) and float(yahoo_iv) > 0:
                resid = (float(yahoo_iv) - solved) / solved
                iv_residual[pos] = resid
                iv_verified[pos] = bool(abs(resid) <= IV_VERIFY_TOLERANCE)
            else:
                # Solved but nothing to compare against: treat as unverified-but-usable.
                iv_verified[pos] = None
        except Exception:
            continue

    df["iv_solved"] = iv_solved
    df["iv_residual_pct"] = iv_residual
    df["iv_verified"] = iv_verified
    return df
