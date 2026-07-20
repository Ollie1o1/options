"""Pure zone-state math. Context only — never predictive scoring.

States: IN ZONE (spot at/below next unfilled level), NEAR (within 1 daily
sigma or 2% of the level, whichever is wider), WATCHING, FILLED (ladder
complete). Next tranche = highest unfilled level."""
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .plan import PlanName

IN_ZONE = "IN ZONE"
NEAR = "NEAR"
WATCHING = "WATCHING"
FILLED = "FILLED"

_LEVEL_TOL = 1e-6


@dataclass
class Snapshot:
    ticker: str
    spot: float
    high_52w: float
    low_52w: float
    ma200: Optional[float]
    daily_sigma: float          # fraction of spot, e.g. 0.041
    closes: List[float] = field(default_factory=list)


@dataclass
class ZoneRead:
    ticker: str
    state: str
    spot: float
    next_level: Optional[float]
    distance_pct: Optional[float]   # (spot-level)/level ×100; negative below
    sigma_dist: Optional[float]     # (spot-level)/(spot×sigma)
    drawdown_pct: float             # vs 52w high, negative below high
    above_ma200: Optional[bool]


def assess(name: PlanName, snap: Snapshot, filled: Set[float]) -> ZoneRead:
    drawdown = (snap.spot / snap.high_52w - 1.0) * 100 if snap.high_52w else 0.0
    above = (snap.spot >= snap.ma200) if snap.ma200 else None
    open_levels = [t.level for t in name.tranches
                   if not any(abs(t.level - f) < _LEVEL_TOL for f in filled)]
    if not open_levels:
        return ZoneRead(name.ticker, FILLED, snap.spot, None, None, None, drawdown, above)
    nxt = max(open_levels)
    dist_frac = (snap.spot - nxt) / nxt
    sigma_dist = ((snap.spot - nxt) / (snap.spot * snap.daily_sigma)
                  if snap.daily_sigma > 0 else None)
    if snap.spot <= nxt:
        state = IN_ZONE
    elif dist_frac <= max(snap.daily_sigma, 0.02):
        state = NEAR
    else:
        state = WATCHING
    return ZoneRead(name.ticker, state, snap.spot, nxt, dist_frac * 100,
                    sigma_dist, drawdown, above)
