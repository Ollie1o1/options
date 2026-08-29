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

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# Four edges => five buckets. Upper-exclusive: a value equal to an edge belongs
# to the bucket ABOVE it.
DELTA_EDGES: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.60)
DTE_EDGES: Tuple[float, ...] = (7.0, 21.0, 45.0, 90.0)
OI_EDGES: Tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)

# Below this many quotes a cell cannot set a cost constant. No cell in today's
# archive trips it; it exists for refits against a thinner archive or a finer
# bucketing. Same floor-below-which-we-refuse pattern as MIN_OBSERVATIONS in
# execution_costs.py, not the same value (that one is 10).
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

    def _median_over(
        self, predicate: Callable[[Tuple[int, int, int]], bool]
    ) -> Optional[float]:
        vals = [c.rel_half_spread for k, c in self.cells.items()
                if predicate(k)]
        return float(median(vals)) if vals else None

    def relative(self, *, abs_delta: Optional[float], dte: Optional[float],
                 open_interest: Optional[float],
                 default: Optional[float] = None) -> Tuple[float, str]:
        """Relative half-spread for a contract, with its provenance.

        Always returns (value, provenance) rather than a bare float. A fallback
        that is indistinguishable from a measurement is how an invented number
        gets quoted as fact.
        """
        d, t, o = cell_key(abs_delta, dte, open_interest)

        cell = self.cells.get((d, t, o))
        if cell is not None:
            return cell.rel_half_spread, "cell"

        collapsed = self._median_over(lambda k: k[0] == d and k[1] == t)
        if collapsed is not None:
            return collapsed, "oi_collapsed"

        collapsed = self._median_over(lambda k: k[0] == d)
        if collapsed is not None:
            return collapsed, "dte_collapsed"

        overall = self._median_over(lambda k: True)
        if overall is not None:
            return overall, "global"

        if default is None:
            raise ValueError(
                "empty spread surface and no caller default; refusing to "
                "report a cost of zero")
        return float(default), "caller_default"

    def half_spread(self, mid: float, *, abs_delta: Optional[float],
                    dte: Optional[float], open_interest: Optional[float],
                    default: Optional[float] = None) -> float:
        """Dollars per share. A non-positive mid is not a free contract, it is
        a row the caller should skip."""
        m = float(mid)
        if m <= 0:
            raise ValueError(f"mid must be positive, got {m}")
        rel, _ = self.relative(abs_delta=abs_delta, dte=dte,
                               open_interest=open_interest, default=default)
        return rel * m

    def depth_ok(self, contracts: float, *, abs_delta: Optional[float],
                 dte: Optional[float],
                 open_interest: Optional[float]) -> bool:
        """Whether an order of this size sits inside displayed depth.

        An unmeasured cell returns False: no measurement is not permission.
        This is the measurable half of the market-impact question — 13.8% of
        archived quotes show under 5 contracts at the touch, and there is
        nothing in this repo to calibrate an impact coefficient against.
        """
        cell = self.cells.get(cell_key(abs_delta, dte, open_interest))
        if cell is None:
            return False
        return float(contracts) <= float(cell.median_depth)


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
        # A missing side is not evidence of depth (same convention as
        # cell_key above): only when BOTH sizes are recorded do we take the
        # tighter side. If either is missing we do not know how thin that
        # side is, so the observation resolves to 0 rather than silently
        # reusing the other side's size as a stand-in.
        depth.setdefault(key, []).append(
            float(min(bsz, asz)) if bsz is not None and asz is not None
            else 0.0)
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


DEFAULT_SURFACE_PATH = "data/spread_surface.json"


def save_surface(surface: SpreadSurface,
                 path: str = DEFAULT_SURFACE_PATH) -> None:
    blob = {
        "stamp": surface.stamp,
        "cells": [{"key": list(k), "n": c.n,
                   "rel_half_spread": c.rel_half_spread,
                   "median_depth": c.median_depth}
                  for k, c in sorted(surface.cells.items())],
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)


def load_surface(path: str = DEFAULT_SURFACE_PATH) -> SpreadSurface:
    """Load a fitted surface. A missing file yields an EMPTY surface, not a
    default-valued one: callers must supply their own default and see the
    `caller_default` provenance rather than silently inheriting a guess."""
    if not os.path.exists(path):
        return SpreadSurface({}, {})
    with open(path) as fh:
        blob = json.load(fh)
    cells = {tuple(c["key"]): Cell(n=int(c["n"]),
                                   rel_half_spread=float(c["rel_half_spread"]),
                                   median_depth=int(c["median_depth"]))
             for c in blob.get("cells", [])}
    return SpreadSurface(cells, blob.get("stamp", {}))


def _cli_main() -> None:
    p = argparse.ArgumentParser(
        description="Measured spread surface: fit, inspect, reprice")
    p.add_argument("--fit", action="store_true",
                   help=f"refit from the archive and write "
                        f"{DEFAULT_SURFACE_PATH}")
    p.add_argument("--archive", default=DEFAULT_ARCHIVE)
    p.add_argument("--out", default=DEFAULT_SURFACE_PATH)
    p.add_argument("--report", action="store_true",
                   help="reprice the closed book under the surface (binds "
                        "nothing)")
    p.add_argument("--ledger", default="paper_trades.db")
    p.add_argument("--surface", default=DEFAULT_SURFACE_PATH)
    args = p.parse_args()

    if args.fit:
        surface = fit_surface(args.archive)
        save_surface(surface, args.out)
        print(f"fitted {len(surface.cells)} cells from "
              f"{surface.stamp['rows']} quotes -> {args.out}")
        print(f"  symbols: {len(surface.stamp['symbols'])}  "
              f"range: {surface.stamp['date_range']}")
        return

    if args.report:
        from src.spread_surface_report import classify_tiers, render_report
        surface = load_surface(args.surface)
        tiers = classify_tiers(args.ledger, args.archive, surface)
        print(render_report(tiers, surface.stamp))
        return

    p.print_help()


if __name__ == "__main__":
    _cli_main()
