"""Stock splits, which this dataset does not adjust for.

The DoltHub chains are point-in-time and NOT split-adjusted. GOOG's strike level
goes 2245 -> 108 across 2022-07-18, AMZN 2434 -> 109, NVDA 1210 -> 122. Seventeen
such events affect fifteen universe symbols, six of them mega-caps.

Left unhandled this corrupts two things badly:

  SIGNALS   A trend signal comparing spot to its own average reads a 20:1 split
            as a 95% crash, and keeps reading a catastrophic downtrend for a
            year afterwards. Any "avoid downtrends" finding measured through
            that is partly measuring splits.

  P&L       A short put struck at 2200 becomes absurdly in-the-money when the
            data's spot drops to 108. In reality the CONTRACT splits too and the
            position is unharmed, but nothing in the chain records that, so the
            backtest books a catastrophic loss that never happened.

Contract adjustment cannot be reconstructed from this data, so positions that
span a split are closed at the last clean mark and flagged, never silently
carried. Excluding them is honest; pretending to price them is not.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set, Tuple

from src.dolt_options import READ_TIMEOUT_S

# A real underlying does not move by these factors in a day. Anything outside
# this band is a corporate action, not a price move.
LOW, HIGH = 0.6, 1.7

# ...but it very much can move that far over a year. The band is only meaningful
# between observations that are ADJACENT in market time. This dataset's cadence
# is roughly every other trading day and it has real holes, so 7 days covers a
# holiday week while still excluding month- and year-scale drift.
MAX_GAP_DAYS = 7


def detect_splits(db_path: str,
                  symbols: Optional[Sequence[str]] = None,
                  max_gap_days: int = MAX_GAP_DAYS,
                  ) -> Dict[str, Set[str]]:
    """symbol -> set of dates on which a split takes effect.

    Uses the mean listed strike as a scale proxy: strikes are re-listed around
    the new price, so the level moves with the split and is far more robust than
    any single contract's quote.

    Only compares observations within `max_gap_days` of each other. Comparing
    across a data gap reads ordinary drift as a corporate action: SPY's cache
    jumps from a mean strike of 228.9 on 2020-03-20 to 465.7 on 2022-01-03, and
    without this guard that 21-month doubling is reported as a split — closing
    every open position on a day when nothing happened. Eight of the fourteen
    tight-spread names tripped it at the 2022 window boundary alone.

    The tradeoff is explicit: a genuine split hidden inside a gap longer than
    `max_gap_days` is missed. Densifying the cache is what shrinks that risk.
    """
    conn = sqlite3.connect(db_path, timeout=READ_TIMEOUT_S)
    try:
        rows = conn.execute(
            "SELECT symbol, date, AVG(strike) FROM dolt_chain "
            "GROUP BY symbol, date ORDER BY symbol, date").fetchall()
    finally:
        conn.close()

    wanted = set(symbols) if symbols else None
    series: Dict[str, list] = defaultdict(list)
    for sym, date, level in rows:
        if wanted is None or sym in wanted:
            if level:
                series[sym].append((date, float(level)))

    out: Dict[str, Set[str]] = defaultdict(set)
    for sym, points in series.items():
        for i in range(1, len(points)):
            prev_date, prev = points[i - 1]
            cur_date, cur = points[i]
            if prev <= 0:
                continue
            try:
                gap = (_dt.date.fromisoformat(cur_date)
                       - _dt.date.fromisoformat(prev_date)).days
            except (TypeError, ValueError):
                continue
            if gap > max_gap_days:
                continue        # a hole in the data, not an overnight event
            ratio = cur / prev
            if ratio < LOW or ratio > HIGH:
                out[sym].add(cur_date)
    return dict(out)


def split_ratio(before: float, after: float) -> float:
    """Approximate split factor, for reporting only."""
    return before / after if after else 0.0
