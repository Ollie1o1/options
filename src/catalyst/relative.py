"""Board-relative superlatives — descriptive, never a ranking.

A bare "43 versions" or "±40%" tells a reader nothing, because they have no
reference for it. Naming the extremes of the CURRENT board supplies that
reference without supplying a judgement: "most-amended on this board" is a
fact about the forty rows on screen, not a claim that amendments matter. They
were measured on 2026-08-26 across ~2,100 observations and found to predict
nothing.

Three rules keep this honest:
  * None is skipped, never ranked — an unmeasured runway is not a short one;
  * fewer than MIN_N measurements claims nothing, because "the shortest of
    two" describes no spread;
  * a tie claims nothing, because naming one of two equals invents an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

#: Below this many measured values, an extreme is noise rather than a spread.
MIN_N = 3


@dataclass(frozen=True)
class Superlatives:
    """Which ticker holds each extreme, where one can honestly be named."""

    shortest_runway: Optional[str] = None
    longest_runway: Optional[str] = None
    widest_implied: Optional[str] = None
    most_amended: Optional[str] = None


def _extreme(pairs: Sequence[Tuple[str, float]],
             pick: Callable[[Any], float]) -> Optional[str]:
    """The ticker holding the min/max, or None if it cannot be named."""
    if len(pairs) < MIN_N:
        return None
    best = pick(value for _, value in pairs)
    winners = [ticker for ticker, value in pairs if value == best]
    return winners[0] if len(winners) == 1 else None


def compute(rows: Sequence[Any]) -> Superlatives:
    """Extremes of the rows actually shown.

    A cash-generative company is excluded from the runway comparison: it has
    no burn limit to measure, so it is not "the longest runway" — it is a
    different kind of object, and quarters is None for exactly that reason.
    """
    runway: List[Tuple[str, float]] = [
        (row.event.ticker, float(row.runway.quarters)) for row in rows
        if row.runway.quarters is not None and not row.runway.cash_generative]
    implied: List[Tuple[str, float]] = [
        (row.event.ticker, float(row.implied.move_pct)) for row in rows
        if row.implied.move_pct is not None]
    amended: List[Tuple[str, float]] = [
        (row.event.ticker, float(row.amendments.versions)) for row in rows
        if row.amendments.available and row.amendments.versions]

    return Superlatives(
        shortest_runway=_extreme(runway, min),
        longest_runway=_extreme(runway, max),
        widest_implied=_extreme(implied, max),
        most_amended=_extreme(amended, max),
    )


def note_for(ticker: str, field: str, sup: Superlatives) -> Optional[str]:
    """The annotation for one row's field, or None if it holds no extreme.

    Wording is deliberately distributional. Nothing here may imply that an
    extreme is good, bad, or worth acting on.
    """
    if field == "runway":
        if ticker == sup.shortest_runway:
            return "shortest runway shown"
        if ticker == sup.longest_runway:
            return "longest runway shown"
        return None
    if field == "implied":
        return "widest implied move shown" if ticker == sup.widest_implied else None
    if field == "amend":
        return "most-amended on this board" if ticker == sup.most_amended else None
    return None
