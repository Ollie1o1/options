"""Expiry settlement at intrinsic value.

At expiry an option is worth exactly its intrinsic value and there is no spread
to cross — in-the-money contracts are assigned, out-of-the-money ones expire.
That is precisely why holding to expiry is cheaper than managing to a target:
you pay the opening legs only.

Settling off quotes instead produced nonsense. An illiquid long leg showing a
zero bid could be "sold" for nothing while the short leg was bought back at a
full ask, giving a -6.20 exit on a $5-wide spread — a loss larger than the
structure can physically sustain.

The underlying price is recovered from put-call parity rather than fetched. At
expiry the discounting term vanishes, so for any strike quoted on both sides
S = K + C - P. Taking the median across every dual-quoted strike makes the
estimate robust to one bad quote.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from src.alloc.fills import Leg


def implied_spot(chain: Sequence[Dict[str, Any]],
                 expiration: str) -> Optional[float]:
    """Underlying price implied by put-call parity at one expiry.

    Returns None when no strike carries usable quotes on both sides.
    """
    exp = str(expiration)[:10]
    calls: Dict[float, float] = {}
    puts: Dict[float, float] = {}
    for c in chain:
        if str(c.get("expiration"))[:10] != exp:
            continue
        bid, ask = c.get("bid"), c.get("ask")
        if bid is None or ask is None or bid < 0 or ask < 0 or ask < bid:
            continue
        mid = (float(bid) + float(ask)) / 2.0
        strike = round(float(c["strike"]), 4)
        (calls if str(c["type"]).lower() == "call" else puts)[strike] = mid

    est = [k + calls[k] - puts[k] for k in calls.keys() & puts.keys()]
    if not est:
        return None
    return float(statistics.median(est))


def implied_spot_any(chain: Sequence[Dict[str, Any]],
                     preferred: Optional[str] = None) -> Optional[float]:
    """Underlying price from the preferred expiry, else from any expiry present.

    The underlying is the same whatever expiry you read it from, and on the
    expiry date itself the expiring contracts have usually already left the
    chain. Without this fallback a position could never be settled on the day
    it expired: it stayed open until the end of the sample and was then closed
    as `ticker_ended`, which discarded almost every mega-cap trade.
    """
    if preferred:
        spot = implied_spot(chain, preferred)
        if spot is not None:
            return spot
    for exp in sorted({str(c.get("expiration"))[:10] for c in chain}):
        spot = implied_spot(chain, exp)
        if spot is not None:
            return spot
    return None


def intrinsic(leg: Leg, spot: float) -> float:
    """What one contract is worth at expiry, per share. Never negative."""
    strike = float(leg.strike)
    if str(leg.type).lower() == "put":
        return max(0.0, strike - spot)
    return max(0.0, spot - strike)


def settle(legs: Sequence[Leg], spot: float) -> float:
    """Net proceeds per share of closing the position at expiry.

    Sign convention matches fills.fill_with_reason: positive is received,
    negative is paid. A short leg is bought back at intrinsic; a long leg is
    sold at intrinsic. No spread is crossed — settlement is not a trade.
    """
    net = 0.0
    for leg in legs:
        value = intrinsic(leg, spot)
        net += -value if leg.action == "sell" else value
    return round(net, 4)
