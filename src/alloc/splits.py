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

import sqlite3
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set, Tuple

# A real underlying does not move by these factors in a day. Anything outside
# this band is a corporate action, not a price move.
LOW, HIGH = 0.6, 1.7


def detect_splits(db_path: str,
                  symbols: Optional[Sequence[str]] = None
                  ) -> Dict[str, Set[str]]:
    """symbol -> set of dates on which a split takes effect.

    Uses the mean listed strike as a scale proxy: strikes are re-listed around
    the new price, so the level moves with the split and is far more robust than
    any single contract's quote.
    """
    conn = sqlite3.connect(db_path)
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
            prev, cur = points[i - 1][1], points[i][1]
            if prev <= 0:
                continue
            ratio = cur / prev
            if ratio < LOW or ratio > HIGH:
                out[sym].add(points[i][0])
    return dict(out)


def split_ratio(before: float, after: float) -> float:
    """Approximate split factor, for reporting only."""
    return before / after if after else 0.0
