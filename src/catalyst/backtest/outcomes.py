"""Forward returns from each vintage, absolute and benchmark-relative.

An UNELAPSED horizon produces no row at all. It is never zero, and never
truncated to "whatever price exists today" — both would quietly mix a partial
window into an average labelled as a full one.

Measured against 2026-08-25 for the 12 quarter-starts 2023-01-01..2025-10-01:
3-month and 6-month horizons have all 12 vintages; the 12-month horizon has 11,
losing 2025-10-01. So each horizon may run on a different subset, and the
report prints the vintage count beside every figure.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

HORIZON_DAYS: Dict[int, int] = {3: 91, 6: 182, 12: 365}


@dataclass(frozen=True)
class Outcome:
    months: int
    absolute: Optional[float]
    relative: Optional[float]


def elapsed(vintage: str, months: int, today: str) -> bool:
    end = (dt.date.fromisoformat(vintage)
           + dt.timedelta(days=HORIZON_DAYS[months]))
    return end <= dt.date.fromisoformat(today)


MAX_GAP_DAYS = 7


def _nearest_on_or_after(prices: Dict[str, float], target: str,
                         max_gap_days: int = MAX_GAP_DAYS) -> Optional[float]:
    """First close on or after ``target``, but only within ``max_gap_days``.

    THE BOUND IS NOT OPTIONAL. Markets close on weekends and holidays, so an
    exact-date lookup would drop windows for no reason — but an UNBOUNDED
    search silently stretches the window instead. Measured while building
    this: a 91-day horizon from 2025-01-01 whose target date was missing
    snapped forward three months and returned a six-month return labelled as
    three-month. And a vintage before the series began snapped both ends onto
    the same first price, returning a confident 0.0% instead of None.

    Seven days covers any weekend or holiday run without letting a sparse or
    stale series masquerade as a dense one.
    """
    try:
        limit = (dt.date.fromisoformat(target)
                 + dt.timedelta(days=max_gap_days)).isoformat()
    except ValueError:
        return None
    for day in sorted(prices):
        if day >= target:
            return prices[day] if day <= limit else None
    return None


def forward_return(prices: Dict[str, float], start: str,
                   days: int) -> Optional[float]:
    """Simple return over ``days``, or None if either end is unavailable.

    None means "not measurable", never 0.0 — a zero return is a real outcome
    and must not be manufactured by a missing price.
    """
    if not prices:
        return None
    begin = _nearest_on_or_after(prices, start)
    if not begin or begin <= 0:
        return None
    target = (dt.date.fromisoformat(start) + dt.timedelta(days=days)).isoformat()
    end = _nearest_on_or_after(prices, target)
    if end is None:
        return None
    return (end / begin) - 1.0


def outcomes_for(ticker: str, vintage: str, today: str,
                 prices: Dict[str, float],
                 bench: Dict[str, float]) -> List[Outcome]:
    """One Outcome per ELAPSED horizon. Unelapsed horizons are omitted."""
    out: List[Outcome] = []
    for months, days in sorted(HORIZON_DAYS.items()):
        if not elapsed(vintage, months, today):
            continue
        absolute = forward_return(prices, vintage, days)
        if absolute is None:
            continue
        b = forward_return(bench, vintage, days) if bench else None
        out.append(Outcome(months=months, absolute=absolute,
                           relative=(absolute - b) if b is not None else None))
    return out
