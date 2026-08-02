"""The frozen exit ladder, simulated over a path of closes.

The study's outcome is a path MAXIMUM while the median endpoint is negative,
so a call held to expiry buys the bad half of the distribution. The exit
therefore triggers on the underlying's close rather than the option's mark
(marks in these names cannot be trusted to fire exits — see
MARK_TRUSTWORTHINESS_SPEC) and is sigma-scaled per name (a flat +20% fires on
noise in a 150-vol name and is a 2-sigma event in a 40-vol one).

There is no stop loss. The down-tail is explicitly NOT elevated — that is the
asymmetry finding — so there is no measured left-tail event to defend against,
and the horizon effect grows with time, so an early stop forfeits the maxima
this whole strategy exists to catch.
"""
from __future__ import annotations

from typing import List, NamedTuple, Sequence

TP1_MULT = 0.50    # sells half; ~the cohort's median path max
TP2_MULT = 1.25    # sells the rest; ~the +50% level, where lift is 4.5x base
HOLD_BARS = 42     # trading days HELD, not days to expiry


class Fill(NamedTuple):
    bar: int          # 1-based bar index after entry
    price: float      # underlying close at the fill
    fraction: float   # share of the position closed
    reason: str       # "tp1" | "tp2" | "time"


def simulate(path: Sequence[float], spot0: float, sigma_h: float,
             hold_bars: int = HOLD_BARS) -> List[Fill]:
    """Fills produced by the ladder over *path* (closes after entry).

    ``sigma_h`` is frozen at entry by the caller and never re-estimated: the
    spike itself would otherwise inflate the bar it is being measured against.
    """
    tp1 = spot0 * (1.0 + TP1_MULT * sigma_h)
    tp2 = spot0 * (1.0 + TP2_MULT * sigma_h)
    fills: List[Fill] = []
    remaining = 1.0
    last = min(len(path), hold_bars)

    for j in range(last):
        close = float(path[j])
        bar = j + 1
        if remaining > 0.5 + 1e-9 and close >= tp1:
            fills.append(Fill(bar=bar, price=close, fraction=0.5, reason="tp1"))
            remaining -= 0.5
        if remaining > 1e-9 and close >= tp2:
            fills.append(Fill(bar=bar, price=close, fraction=remaining, reason="tp2"))
            remaining = 0.0
            break

    if remaining > 1e-9 and last > 0:
        fills.append(Fill(bar=last, price=float(path[last - 1]),
                          fraction=remaining, reason="time"))
    return fills
