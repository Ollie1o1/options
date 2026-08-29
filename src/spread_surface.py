"""What crossing the spread actually costs, conditioned on the contract.

`execution_costs.py` charges one median half-spread per strategy name. A
strategy name is not a contract: measured over 606,986 two-sided quotes in
data/chain_archive.db, relative half-spread runs 0.062 deep OTM against 0.011
at the money, and 0.028 at open interest under 10 against 0.011 above 10k.
A single constant per structure is wrong in both directions at once.

This module models RELATIVE half-spread, not dollars. Dollar spread spans 200x
and is driven mostly by the price level of the contract; relative spans ~15x
and is the unit friction is actually paid in. Dollars fall out as rel * mid.

It takes contract characteristics and returns a cost. It knows nothing about
the ledger, the scanner or the gate.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

# Four edges => five buckets. Upper-exclusive: a value equal to an edge belongs
# to the bucket ABOVE it.
DELTA_EDGES: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.60)
DTE_EDGES: Tuple[float, ...] = (7.0, 21.0, 45.0, 90.0)
OI_EDGES: Tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)

# Below this many quotes a cell cannot set a cost constant. No cell in today's
# archive trips it; it exists for refits against a thinner archive or a finer
# bucketing. Mirrors MIN_OBSERVATIONS in execution_costs.py.
MIN_CELL_OBS = 30


def bucket_index(value: float, edges: Sequence[float]) -> int:
    """Index of the bucket `value` falls in. Upper-exclusive on each edge."""
    v = float(value)
    for i, edge in enumerate(edges):
        if v < edge:
            return i
    return len(edges)


def cell_key(abs_delta: Optional[float], dte: Optional[float],
             open_interest: Optional[float]) -> Tuple[int, int, int]:
    """Grid coordinates for a contract.

    A missing value is not zero cost. NULL open interest means "not recorded",
    and the conservative reading is the most illiquid bucket — assuming
    liquidity we did not observe would understate friction, which is the
    direction that flatters a book.
    """
    return (
        bucket_index(abs(float(abs_delta)) if abs_delta is not None else 0.0,
                     DELTA_EDGES),
        bucket_index(float(dte) if dte is not None else 0.0, DTE_EDGES),
        bucket_index(float(open_interest) if open_interest is not None else 0.0,
                     OI_EDGES),
    )


import sqlite3
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Dict, List, Set

DEFAULT_ARCHIVE = "data/chain_archive.db"
REFIT_COMMAND = ("PYTHONPATH=$PWD ~/.venvs/options/bin/python "
                 "-m src.spread_surface --fit")


@dataclass(frozen=True)
class Cell:
    n: int
    rel_half_spread: float
    median_depth: int


class SpreadSurface:
    def __init__(self, cells: Dict[Tuple[int, int, int], Cell],
                 stamp: dict):
        self.cells = cells
        self.stamp = stamp


_FIT_SQL = """
    SELECT (ask - bid) / 2.0                              AS half,
           (ask + bid) / 2.0                              AS mid,
           abs(delta)                                     AS ad,
           julianday(expiration) - julianday(snap_date)   AS dte,
           open_interest                                  AS oi,
           bid_size, ask_size, symbol, snap_date
    FROM chain_snapshots
    WHERE bid > 0 AND ask > bid AND delta IS NOT NULL
"""


def fit_surface(archive_db: str = DEFAULT_ARCHIVE) -> SpreadSurface:
    """Fit the surface from archived quotes.

    Only two-sided quotes count. A zero bid or a crossed book is missing data,
    and averaging it in would understate the real cost of crossing — the same
    rule execution_costs.measure_half_spreads already applies.
    """
    con = sqlite3.connect(archive_db)
    try:
        rows = con.execute(_FIT_SQL).fetchall()
    finally:
        con.close()

    rel: Dict[Tuple[int, int, int], List[float]] = {}
    depth: Dict[Tuple[int, int, int], List[float]] = {}
    symbols: Set[str] = set()
    dates: List[str] = []
    for half, mid, ad, dte, oi, bsz, asz, sym, snap in rows:
        if mid is None or mid <= 0 or half is None or half < 0:
            continue
        key = cell_key(ad, dte, oi)
        rel.setdefault(key, []).append(float(half) / float(mid))
        sides = [s for s in (bsz, asz) if s is not None]
        depth.setdefault(key, []).append(float(min(sides)) if sides else 0.0)
        symbols.add(sym)
        dates.append(snap)

    cells = {
        key: Cell(n=len(vals),
                  rel_half_spread=float(median(vals)),
                  median_depth=int(median(depth[key])))
        for key, vals in rel.items() if len(vals) >= MIN_CELL_OBS
    }
    stamp = {
        "fit_date": date.today().isoformat(),
        "rows": sum(len(v) for v in rel.values()),
        "symbols": sorted(symbols),
        "date_range": [min(dates), max(dates)] if dates else None,
        "refit_command": REFIT_COMMAND,
    }
    return SpreadSurface(cells, stamp)
