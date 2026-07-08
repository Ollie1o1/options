"""Perp position sizing: leverage is DERIVED from the stop, not chosen.

Lives in src/leverage (not src/core) because the existing src/core/sizing.py is
provenance-locked on this machine and cannot be edited; this is the leverage
package's own sizing surface. core/sizing.capped_quantity (the options ledger
cap) is unaffected and still used by src/crypto.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Sizing:
    risk_frac: float      # fraction of equity risked if stop hit
    risk_usd: float       # dollar risk if stop hit
    eff_leverage: float   # notional / equity
    notional: float       # position notional in USD
    qty: Optional[float]  # base-asset quantity (None if price not supplied)


def effective_leverage_size(equity: float, stop_distance_pct: float,
                            kelly_p: float = 0.42, kelly_b: float = 2.0,
                            risk_cap: float = 0.02,
                            min_leverage: float = 2.0,
                            lev_hard_cap: float = 5.0,
                            price: Optional[float] = None,
                            min_notional: float = 0.0) -> Optional[Sizing]:
    """Size a trade so leverage is DERIVED from the stop, not chosen.

    risk_frac = min(quarter-Kelly, risk_cap).
    eff_leverage = risk_frac / stop_distance_pct, clamped to lev_hard_cap
        (clamping reduces realized risk_frac so the cap is never breached).
    Returns None when the resulting eff_leverage is below `min_leverage` - a 1x
    setup is un-leveraged buy-and-hold (an explicit non-goal), so it is skipped
    rather than force-sized. The preferred 3-6x band is informational (surfaced
    in the ticket); 2.5x setups are acceptable, per the spec's own example.

    Returns None when the sized notional is below `min_notional` (the venue's
    minimum order size) — a small account that cannot meet the minimum should
    not fake a position.
    """
    if stop_distance_pct <= 0 or equity <= 0:
        return None
    kelly = (kelly_p * kelly_b - (1.0 - kelly_p)) / kelly_b
    risk_frac = min(0.25 * kelly, risk_cap)
    if risk_frac <= 0:
        return None
    eff_lev = risk_frac / stop_distance_pct
    if eff_lev > lev_hard_cap:
        eff_lev = lev_hard_cap
        risk_frac = eff_lev * stop_distance_pct  # back out the reduced risk
    if eff_lev < min_leverage:
        return None  # un-leveraged -> skip
    notional = equity * eff_lev
    if min_notional > 0.0 and notional < min_notional:
        return None  # can't meet the venue's minimum at this risk budget
    qty = (notional / price) if price else None
    return Sizing(risk_frac=risk_frac, risk_usd=risk_frac * equity,
                  eff_leverage=eff_lev, notional=notional, qty=qty)
