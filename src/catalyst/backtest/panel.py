"""Vintage list and the feature panel — features only, no outcomes.

Outcomes live in outcomes.py so that building features can never accidentally
read a price from after the vintage.

Every feature is TRI-STATE. `funded_through=None` means we could not compute
it, which is different from False ("must raise"). Collapsing unknown into
False would silently move rows into the comparison group and bias H1.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from src.catalyst.models import Coverage

_QUARTER_STARTS = ((1, 1), (4, 1), (7, 1), (10, 1))


@dataclass(frozen=True)
class PanelRow:
    vintage: str
    ticker: str
    nct_id: str
    event_date: str
    phase: str
    funded_through: Optional[bool] = None
    amended: Optional[bool] = None
    enrollment: Optional[int] = None


def vintages(start: str, end: str) -> List[str]:
    """Quarter-start dates from start to end inclusive."""
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out: List[str] = []
    for year in range(lo.year, hi.year + 1):
        for month, day in _QUARTER_STARTS:
            d = dt.date(year, month, day)
            if lo <= d <= hi:
                out.append(d.isoformat())
    return out


def _board(as_of: str, nct_ids: Sequence[str],
           conn: sqlite3.Connection) -> Tuple[List[Any], Coverage]:
    from src.catalyst import pit
    return pit.board_as_of(as_of, nct_ids, conn)


def _runway(cik: int, as_of: str, event_date: str,
            conn: sqlite3.Connection) -> Any:
    from src.catalyst import pit
    return pit.runway_as_of(cik, as_of, event_date, conn)


def _amendments(nct_id: str, as_of: str, conn: sqlite3.Connection) -> Any:
    """Amendment history AS OF the vintage, from the cached version list.

    `design.amendments_for` fetches live and counts every change ever made, so
    an endpoint edited in 2025 marked a row "amended" at the 2023 vintage.
    Every other feature on this panel was already reconstructed point-in-time;
    this one was not, and H2 is the hypothesis about exactly this feature.
    """
    from src.catalyst import pit
    return pit.amendments_as_of(pit._versions(nct_id, conn, as_of=as_of), as_of)


def _cik(ticker: str) -> Optional[int]:
    from src.catalyst.runway import _cik as lookup
    return lookup(ticker)


def build(vintage: str, nct_ids: Sequence[str],
          conn: sqlite3.Connection) -> Tuple[List[PanelRow], Coverage]:
    """Feature rows for one vintage."""
    events, coverage = _board(vintage, nct_ids, conn)
    rows: List[PanelRow] = []
    for event in events:
        cik = _cik(event.ticker)
        funded: Optional[bool] = None
        if cik is not None:
            funded = _runway(cik, vintage, event.event_date, conn).funded_through
        amendments = _amendments(event.trial.nct_id, vintage, conn)
        amended: Optional[bool] = None
        if amendments.available:
            amended = amendments.outcomes_updated >= 2
        rows.append(PanelRow(
            vintage=vintage,
            ticker=event.ticker,
            nct_id=event.trial.nct_id,
            event_date=event.event_date,
            phase=event.trial.top_phase or event.phase,
            funded_through=funded,
            amended=amended,
            enrollment=event.trial.enrollment,
        ))
    return rows, coverage
