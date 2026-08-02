"""P_live and F_live — what the market charges for the treated cohort's calls.

ATM rather than 25-delta, deliberately: strike granularity on $5-$10 names makes
a delta-targeted strike unmeasurable, and ATM is where the liquidity is. The
25-delta minus ATM gap is worth recording as a diagnostic — is the specific
upside wing marked up beyond the vol level? — but it is not the decision
statistic.

Calls only, because calls are the side being bought. The ATM put-minus-call IV
gap is the hard-to-borrow proxy: borrow cost depresses the synthetic forward,
and averaging the two sides would smear it into the vol measurement, which is
the one quantity this module exists to isolate.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from src.utils import bs_call

TENORS = (30, 60)          # calendar days: 21td and 42td horizons
MAX_REL_SPREAD = 0.60      # wider than this is unmeasurable, not zero
MIN_BID = 0.05
TENOR_BAND = (0.7, 1.5)    # a lone expiry is usable inside this multiple


def _atm_rows(chain: Sequence[dict], spot: float) -> dict:
    """Nearest-to-the-money usable call per expiry, keyed by DTE."""
    best: dict = {}
    for row in chain:
        if str(row.get("option_type", "call")).lower() != "call":
            continue
        bid, ask = row.get("bid"), row.get("ask")
        iv, strike, dte = row.get("iv"), row.get("strike"), row.get("dte")
        if None in (bid, ask, iv, strike, dte):
            continue
        if bid < MIN_BID or ask <= bid or iv <= 0:
            continue
        mid = (bid + ask) / 2.0
        if mid <= 0 or (ask - bid) / mid > MAX_REL_SPREAD:
            continue
        gap = abs(float(strike) - spot)
        cur = best.get(int(dte))
        if cur is None or gap < cur[0]:
            best[int(dte)] = (gap, float(iv), (ask - bid) / mid)
    return best


def atm_call_iv(chain: Sequence[dict], spot: float, tenor_days: int,
                risk_free: float = 0.04) -> Optional[float]:
    """ATM call IV at a fixed tenor, interpolated in total variance."""
    if spot <= 0:
        return None
    best = _atm_rows(chain, spot)
    if not best:
        return None

    exact = best.get(tenor_days)
    if exact is not None:
        return exact[1]

    below = [d for d in best if d < tenor_days]
    above = [d for d in best if d > tenor_days]
    if below and above:
        d0, d1 = max(below), min(above)
        v0, v1 = best[d0][1] ** 2 * d0, best[d1][1] ** 2 * d1
        w = (tenor_days - d0) / (d1 - d0)
        var = v0 + w * (v1 - v0)
        return math.sqrt(var / tenor_days) if var > 0 else None

    lo, hi = TENOR_BAND
    usable = [d for d in best if lo * tenor_days <= d <= hi * tenor_days]
    if not usable:
        return None
    nearest = min(usable, key=lambda d: abs(d - tenor_days))
    return best[nearest][1]


def relative_spread(chain: Sequence[dict], spot: float,
                    tenor_days: int) -> Optional[float]:
    """Quoted spread as a fraction of mid, at the nearest usable expiry."""
    best = _atm_rows(chain, spot)
    if not best:
        return None
    lo, hi = TENOR_BAND
    usable = [d for d in best if lo * tenor_days <= d <= hi * tenor_days] or list(best)
    nearest = min(usable, key=lambda d: abs(d - tenor_days))
    return best[nearest][2]


def premium_ratio(iv_treated: float, iv_control: float, spot: float,
                  tenor_days: int, risk_free: float = 0.04) -> Optional[float]:
    """Extra premium charged for the treated name, as a fraction of the control."""
    if iv_treated <= 0 or iv_control <= 0 or spot <= 0:
        return None
    t = tenor_days / 365.0
    c_t = float(bs_call(spot, spot, t, risk_free, iv_treated))
    c_c = float(bs_call(spot, spot, t, risk_free, iv_control))
    if not math.isfinite(c_t) or not math.isfinite(c_c) or c_c <= 0:
        return None
    return c_t / c_c - 1.0
