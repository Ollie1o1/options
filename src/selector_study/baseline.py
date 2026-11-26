"""Matched liquidity-first baseline for one generic bull-put selection.

Same ticker, session and expiration as the selector's pick. Comparable exposure: the
short put's |delta| within a tolerance of the selector's short delta, and the width within
a tolerance of the selector's width. Among those verticals, take the one with the lowest
round-trip bid-ask spread relative to its mid credit. Ties break on the smaller delta gap,
then the smaller width gap, then the higher short strike, then the higher long strike.

Pure: no I/O, no clock. The selector's own pick is eligible, so an identical choice is a
valid outcome and is reported as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.structure.chain_data import ChainContract, ChainSlice


@dataclass(frozen=True)
class Vertical:
    short: ChainContract
    long: ChainContract

    @property
    def width(self) -> float:
        return self.short.strike - self.long.strike

    @property
    def credit_mid(self) -> float:
        return self.short.mid - self.long.mid

    @property
    def credit_cross(self) -> float:
        return self.short.bid - self.long.ask

    @property
    def rel_spread(self) -> float:
        return ((self.short.ask - self.short.bid) + (self.long.ask - self.long.bid)) / self.credit_mid


def _two_sided(c: ChainContract, min_bid: float, min_oi: int) -> bool:
    return (c.bid >= min_bid and c.ask >= c.bid and c.mid > 0
            and int(c.open_interest or 0) >= min_oi)


def eligible_verticals(sl: ChainSlice, *, target_delta: float, target_width: float,
                       rules: Dict[str, Any], max_friction_pct: float) -> List[Vertical]:
    """Every bull-put vertical in the slice with exposure comparable to the target."""
    tol_d = float(rules["short_delta_tolerance"])
    tol_w = float(rules["width_tolerance_pct"])
    min_oi = int(rules.get("min_open_interest", 0))
    shorts = [c for c in sl.puts
              if _two_sided(c, float(rules["min_short_bid"]), min_oi)
              and c.delta is not None and math.isfinite(c.delta)
              and abs(abs(c.delta) - target_delta) <= tol_d + 1e-9]
    longs = [c for c in sl.puts if _two_sided(c, float(rules["min_long_bid"]), min_oi)]
    out: List[Vertical] = []
    for s in shorts:
        for lg in longs:
            v = Vertical(s, lg)
            if v.width <= 0 or abs(v.width - target_width) > tol_w * target_width + 1e-9:
                continue
            if not (v.credit_mid > float(rules["min_credit_mid"]) and v.credit_cross > 0
                    and v.credit_mid < v.width):
                continue
            if v.rel_spread > max_friction_pct:
                continue
            out.append(v)
    return out


def rank_key(v: Vertical, target_delta: float, target_width: float) -> Tuple[float, ...]:
    return (round(v.rel_spread, 10),
            round(abs(abs(v.short.delta or 0.0) - target_delta), 10),
            round(abs(v.width - target_width), 10),
            -v.short.strike, -v.long.strike)


def find_leg(sl: ChainSlice, strike: float) -> Optional[ChainContract]:
    for c in sl.puts:
        if abs(c.strike - strike) < 1e-6:
            return c
    return None


def match(sl: ChainSlice, selector_row: Dict[str, Any], *, rules: Dict[str, Any],
          max_friction_pct: float) -> Tuple[Optional[Vertical], str, int]:
    """(baseline vertical or None, status, eligible count) for one selector bull put.

    status: `identical` (baseline chose the selector's own legs), `matched`, or a
    `no_match:<reason>` string."""
    legs = selector_row.get("legs") or ()
    shorts = [lg for lg in legs if str(lg["side"]).upper() == "SELL"]
    longs = [lg for lg in legs if str(lg["side"]).upper() == "BUY"]
    if len(shorts) != 1 or len(longs) != 1:
        return None, "no_match:selector legs unreadable", 0
    s_leg = find_leg(sl, float(shorts[0]["strike"]))
    l_leg = find_leg(sl, float(longs[0]["strike"]))
    if s_leg is None or l_leg is None:
        return None, "no_match:selector legs absent from captured quotes", 0
    if s_leg.delta is None or not math.isfinite(s_leg.delta):
        return None, "no_match:selector short delta unavailable", 0
    target_delta = abs(s_leg.delta)
    target_width = s_leg.strike - l_leg.strike
    pool = eligible_verticals(sl, target_delta=target_delta, target_width=target_width,
                              rules=rules, max_friction_pct=max_friction_pct)
    if not pool:
        return None, "no_match:no eligible vertical with comparable exposure", 0
    best = min(pool, key=lambda v: rank_key(v, target_delta, target_width))
    same = abs(best.short.strike - s_leg.strike) < 1e-6 and abs(best.long.strike - l_leg.strike) < 1e-6
    return best, ("identical" if same else "matched"), len(pool)
