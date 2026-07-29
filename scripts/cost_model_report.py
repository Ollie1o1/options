#!/usr/bin/env python3
"""Re-price the closed multi-leg book under the flat vs the measured cost model.

Reporting only — writes nothing, changes no exit behaviour. The point is to see
whether the lines that carry the book survive their real cost of trading before
anyone changes how exits are priced.

Single-leg trades are excluded: their close path already charges 30% of the
live quoted spread, so the flat constant was never applied to them.

    python scripts/cost_model_report.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.execution_costs import CostModel, load_measured_model, reprice_pnl_pct  # noqa: E402

_MULTI_LEG = {"Bull Put", "Bear Call", "Iron Condor"}


def _n_legs(strategy: str) -> int:
    return 4 if strategy == "Iron Condor" else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--archive", default="data/chain_archive.db")
    ap.add_argument("--since", default="2026-05-27", help="cohort start date")
    args = ap.parse_args()

    measured = load_measured_model(args.archive, args.db)
    flat = CostModel(table={}, default_half_spread=0.05,
                     commission_per_contract=measured.commission_per_contract)

    print("Measured half-spreads (median $/share, from archived quotes of the")
    print("exact contracts logged, on their own entry dates):")
    for strategy in sorted(_MULTI_LEG):
        cell = measured.table.get(strategy, {})
        n = cell.get("n", 0)
        used = measured.half_spread(strategy)
        note = "" if n >= 10 else f"  (n={n} — too thin, using flat default)"
        print(f"  {strategy:12s} n={n:3d}  ${used:.3f}{note}")
    print()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT strategy_name, pnl_pct, net_credit, entry_price, capital_at_risk
        FROM trades
        WHERE status = 'CLOSED' AND pnl_pct IS NOT NULL AND date >= ?
          AND strategy_name IN ('Bull Put', 'Bear Call', 'Iron Condor')
        """,
        (args.since,),
    ).fetchall()
    conn.close()

    agg = defaultdict(lambda: {"n": 0, "old": 0.0, "new": 0.0, "risk": 0.0})
    for row in rows:
        strategy = row["strategy_name"]
        credit = row["net_credit"] or row["entry_price"] or 0
        if credit <= 0:
            continue
        legs = _n_legs(strategy)
        repriced = reprice_pnl_pct(row["pnl_pct"], strategy, credit, legs, flat, measured)
        cell = agg[strategy]
        cell["n"] += 1
        cell["old"] += row["pnl_pct"] * credit * 100
        cell["new"] += repriced * credit * 100
        cell["risk"] += row["capital_at_risk"] or 0

    print(f"{'strategy':12s} {'n':>4} {'as booked':>11} {'re-priced':>11} "
          f"{'change':>10} {'RoR before':>11} {'RoR after':>10}")
    tot_old = tot_new = tot_risk = 0.0
    for strategy in sorted(agg, key=lambda s: -agg[s]["new"]):
        c = agg[strategy]
        tot_old += c["old"]; tot_new += c["new"]; tot_risk += c["risk"]
        r_before = 100 * c["old"] / c["risk"] if c["risk"] else 0
        r_after = 100 * c["new"] / c["risk"] if c["risk"] else 0
        print(f"{strategy:12s} {c['n']:4d} {c['old']:+11.0f} {c['new']:+11.0f} "
              f"{c['new'] - c['old']:+10.0f} {r_before:+10.1f}% {r_after:+9.1f}%")
    if tot_risk:
        print(f"{'TOTAL':12s} {sum(c['n'] for c in agg.values()):4d} "
              f"{tot_old:+11.0f} {tot_new:+11.0f} {tot_new - tot_old:+10.0f} "
              f"{100 * tot_old / tot_risk:+10.1f}% {100 * tot_new / tot_risk:+9.1f}%")

    print("""
Read this as "the cost assumption was driving the comparison", not as "Bear Call
is the better trade". Four limits, all of which cut against over-reading it:

  - The half-spread is measured on the structure's anchor leg and applied to
    every leg. A Bull Put's long leg is further OTM and need not quote like its
    short leg.
  - Sample sizes are tens of contract-days, not thousands, drawn from an archive
    of 15 symbols over 19 snapshot dates in June-July 2026. One illiquid week
    moves a median at this n.
  - Medians hide the tail. The cost of getting out of a losing spread in a fast
    market is not the median quote of a calm one.
  - Nothing here re-examines entry or exit timing, only what the round trip was
    charged.

The defensible conclusion is narrow: a single flat constant cannot price both
of these structures, and the line that looked best was the one it flattered.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
