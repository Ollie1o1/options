"""Immutable carriers for a position's observed price path and a replay result.

`PricePath` is the single type every corpus loader must produce, so the replay
engine never learns which database a position came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PathPoint:
    """One day's quote for a position. `mid` of 0.0 is a price, not a gap."""
    date: str          # ISO YYYY-MM-DD
    bid: float
    ask: float
    mid: float
    spot: Optional[float]
    dte: int
    spread_imputed: bool = False
    """True when bid/ask were synthesized from mid because the source
    recorded no two-sided quote (the corpus stores bid/ask as 100% NULL for
    a large share of marks). False means bid/ask are a real observed quote."""


@dataclass(frozen=True)
class PricePath:
    """A position plus every quote observed for it, entry and exit included.

    `entry_price` is the structure's opening price as a positive magnitude;
    `is_credit` says which way it was traded, because a credit structure is
    closed by BUYING BACK (paying the ask) and a debit one by SELLING (taking
    the bid). Collapsing the two is the sign error that would make every
    result wrong in the same direction.
    """
    position_id: str
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    capital_at_risk: float
    is_credit: bool
    points: Tuple[PathPoint, ...]
    actual_exit_date: str
    actual_pnl_frac: float      # realised P&L as a fraction of capital at risk
    corpus: str                 # "A" or "B"

    @property
    def interior_points(self) -> Tuple[PathPoint, ...]:
        """Points strictly between entry and the actual exit.

        These are the only days on which an alternative exit rule could have
        acted. Entry is not a decision point, and the actual exit day is the
        baseline's own choice rather than a free one.
        """
        return tuple(p for p in self.points
                     if self.entry_date < p.date < self.actual_exit_date)


@dataclass(frozen=True)
class Outcome:
    """What a policy did on a path.

    `decision_points` is how many interior days the rule actually saw; a rule
    that "won" on a path offering it one look has not been tested.
    """
    position_id: str
    exit_date: str
    exit_reason: str
    pnl_frac_car: float
    decision_points: int
