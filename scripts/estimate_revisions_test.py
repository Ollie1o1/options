#!/usr/bin/env python3
"""Date-gated estimator for Playbook 4 Arm B: Point-in-Time Analyst Estimate Revisions.

Frozen registration: docs/PREREG_PEAD_AND_REVISIONS_20261124.md
Refuses to compute forward outcomes before 2027-01-15 to prevent early peeking.
Use `--coverage-only` any time to inspect archive snapshot readiness without reading returns.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

UNLOCK_DATE = "2027-01-15"
MIN_SNAPSHOTS = 4
MIN_OBS_PER_SNAP = 40
ARCHIVE_DB = "data/estimates_archive.db"


def compute_rev_breadth_30d(db_path: str = ARCHIVE_DB) -> Dict[str, Dict[str, float]]:
    """Compute frozen `rev_breadth_30d` per (snap_date, symbol) strictly from eps_revisions."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT snap_date, symbol, period, field, value FROM estimate_snapshots "
            "WHERE dataset = 'eps_revisions' AND period IN ('0q', '+1y') "
            "AND field IN ('upLast30days', 'downLast30days') AND snap_date >= '2026-10-14'"
        ).fetchall()
    finally:
        con.close()

    acc: Dict[Tuple[str, str], Dict[str, float]] = {}
    for sdate, sym, _period, field, val in rows:
        if val is None or not np.isfinite(float(val)):
            continue
        bucket = acc.setdefault((str(sdate), str(sym).upper()), {"up": 0.0, "down": 0.0})
        if field == "upLast30days":
            bucket["up"] += float(val)
        elif field == "downLast30days":
            bucket["down"] += float(val)

    by_date: Dict[str, Dict[str, float]] = {}
    for (sdate, sym), counts in acc.items():
        tot = counts["up"] + counts["down"]
        if tot >= 2.0:
            by_date.setdefault(sdate, {})[sym] = (counts["up"] - counts["down"]) / tot
    return by_date


def select_non_overlapping_snapshots(dates: List[str], min_cal_gap_days: int = 21) -> List[str]:
    kept: List[str] = []
    for d in sorted(dates):
        if not kept:
            kept.append(d)
            continue
        gap = (dt.date.fromisoformat(d) - dt.date.fromisoformat(kept[-1])).days
        if gap >= min_cal_gap_days:
            kept.append(d)
    return kept


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Playbook 4 Arm B: Estimate Revisions (locked until 2027-01-15)")
    p.add_argument("--coverage-only", action="store_true", help="Print archive coverage only (zero outcomes)")
    p.add_argument("--db", default=ARCHIVE_DB)
    args = p.parse_args(argv)

    breadth = compute_rev_breadth_30d(args.db)
    snap_dates = sorted(breadth.keys())
    non_ov = select_non_overlapping_snapshots(snap_dates, min_cal_gap_days=21)

    if args.coverage_only:
        print(f"Estimate-revisions archive coverage ({args.db}):")
        print(f"  All snapshots (>=2026-10-14): {len(snap_dates)} -> {snap_dates}")
        print(f"  Non-overlapping (>=21d gap) : {len(non_ov)} / {MIN_SNAPSHOTS} min -> {non_ov}")
        for d in snap_dates:
            print(f"    {d}: {len(breadth[d])} symbols with valid rev_breadth_30d (min {MIN_OBS_PER_SNAP})")
        return 0

    today = dt.date.today().isoformat()
    if today < UNLOCK_DATE:
        print(
            f"REFUSED: Arm B is preregistered for readout on or after {UNLOCK_DATE} "
            f"(today is {today}). Run with --coverage-only to inspect archive growth.",
            file=sys.stderr,
        )
        return 2

    if len(non_ov) < MIN_SNAPSHOTS:
        print(f"INSUFFICIENT: only {len(non_ov)} non-overlapping snapshots (need {MIN_SNAPSHOTS}).")
        return 0

    # Full post-unlock evaluation (runs on/after 2027-01-15)
    from src.equity_playbooks.backtest import BenchmarkSeries, load_equity_ohlcv
    from src.outlook.backtest import spearman_ic

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db")
    if "SPY" not in ohlcv:
        print("ERROR: SPY missing from data/equity_ohlcv.db", file=sys.stderr)
        return 1
    spy = BenchmarkSeries(ohlcv["SPY"][0], ohlcv["SPY"][1])

    ics: List[float] = []
    for sd in non_ov:
        xs, ys = [], []
        for sym, score in breadth[sd].items():
            if sym not in ohlcv:
                continue
            ds, cs, _ = ohlcv[sym]
            import bisect
            i_in = bisect.bisect_right(ds, sd)
            if i_in + 21 >= len(ds):
                continue
            r_stk = float(cs[i_in + 21] / cs[i_in] - 1.0)
            r_spy = spy.window_return(ds[i_in], ds[i_in + 21])
            if r_spy is None:
                continue
            xs.append(score)
            ys.append(r_stk - r_spy)
        if len(xs) >= MIN_OBS_PER_SNAP:
            ic = spearman_ic(xs, ys)
            if ic is not None:
                ics.append(ic)
                print(f"  {sd}: n={len(xs)} rank_ic={ic:+.4f}")
    if len(ics) < MIN_SNAPSHOTS:
        print(f"INSUFFICIENT: only {len(ics)} snapshots met symbol floor ({MIN_OBS_PER_SNAP}).")
        return 0
    print(f"Mean cross-sectional rank IC across {len(ics)} non-overlapping snapshots: {float(np.mean(ics)):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
