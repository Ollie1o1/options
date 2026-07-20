"""Long-term discovery scan: find candidates worth watching in a sector.

Two-tier design (see docs/superpowers/specs/2026-07-20-longterm-discover-design.md):

  FAST tier (every candidate, zero extra network calls beyond the one batched
  price fetch already used by the buy-zone watcher): drawdown, distance below
  the 200-day moving average, 12-1 month momentum, real support/resistance
  levels, and this name's own empirical bounce-odds after drops this size.

  DEEP tier (bounded to the top ~6 ranked candidates, each a separate network
  round-trip): insider cluster-buy activity (EDGAR Form 4), days to next
  earnings, and lightweight fundamentals (P/E, margins, growth).

No claim of predictive edge anywhere in this module. Drawdown, support
levels, and bounce rates are historical facts about a stock's own past
behavior, not forecasts. Insider buying and fundamentals are descriptive
context, not a "buy signal" — a name can show every positive sign here and
still be a value trap.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .plan import Tranche
from .zones import Snapshot

# Mid-cap-or-larger, average volume > 200k shares/day, US-listed — screens out
# illiquid and penny-stock noise before anything else runs. Every sector
# filter below is built on this same gate.
_QUALITY_GATE = "cap_midover,sh_avgvol_o200,geo_usa"

# Finviz screener filter-code fragments (https://finviz.com/screener.ashx).
# ind_* = industry-level filter (narrower); sec_* = sector-level (broader).
# CONSUMER intentionally maps to Consumer Defensive (staples), not Cyclical —
# staples is the more common "value/long-term hold" hunting ground.
SECTOR_FILTERS: Dict[str, str] = {
    "SEMICONDUCTORS": f"ind_semiconductors,{_QUALITY_GATE}",
    "TECH": f"sec_technology,{_QUALITY_GATE}",
    "BANKS": f"ind_banks,{_QUALITY_GATE}",
    "HEALTHCARE": f"sec_healthcare,{_QUALITY_GATE}",
    "ENERGY": f"sec_energy,{_QUALITY_GATE}",
    "CONSUMER": f"sec_consumerdefensive,{_QUALITY_GATE}",
    "INDUSTRIALS": f"sec_industrials,{_QUALITY_GATE}",
    "UTILITIES": f"sec_utilities,{_QUALITY_GATE}",
    "REALESTATE": f"sec_realestate,{_QUALITY_GATE}",
    "MATERIALS": f"sec_basicmaterials,{_QUALITY_GATE}",
    "COMMUNICATIONS": f"sec_communicationservices,{_QUALITY_GATE}",
}


def universe(sector_keyword: str, limit: int = 30) -> List[str]:
    """Ticker list for a sector keyword, quality-filtered, most-liquid first.

    Raises ValueError (with the valid keyword list) for an unrecognized
    keyword — a typo should never silently scan the wrong thing or an empty
    universe. A recognized keyword whose live Finviz fetch fails returns an
    empty list rather than raising (finviz_tickers's own contract), so the
    caller can distinguish "bad input" from "the network is down right now."
    """
    from src.squeeze.universe import finviz_tickers

    keyword = sector_keyword.upper()
    f_params = SECTOR_FILTERS.get(keyword)
    if f_params is None:
        valid = ", ".join(sorted(SECTOR_FILTERS))
        raise ValueError(f"unknown sector {sector_keyword!r} — valid: {valid}")
    return finviz_tickers(f_params, order="-averagevolume", limit=limit)
