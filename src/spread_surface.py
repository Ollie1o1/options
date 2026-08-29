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
