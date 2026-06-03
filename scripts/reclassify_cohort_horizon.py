#!/usr/bin/env python
"""One-time: quarantine forward-cohort Long Calls that were logged below the
cohort DTE floor (the 3-day time-exit contamination described in status/STATUS.md).

Such trades entered at <floor DTE, so the time-exit force-closed them after the
min-hold floor before any swing could play out — their returns are ~3-day noise,
not the signal the gate is meant to measure. We flip them to paper_only=1: kept
for data, excluded from the validation-gate IC.

Scope is deliberately narrow:
  - strategy_name = 'Long Call'
  - paper_only = 0  (currently cohort-eligible)
  - date >= auto_log.phase1_start_date  (forward cohort only — the 100 historical
    Long Calls used for the out-of-sample read are left untouched)
  - DTE-at-entry (expiration - date) < cohort_min_dte

Dry-run by default. --apply backs up the DB first.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/reclassify_cohort_horizon.py
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/reclassify_cohort_horizon.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime


def _floor(config_path: str) -> int:
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except OSError:
        cfg = {}
    al = cfg.get("auto_log") or {}
    explicit = al.get("cohort_min_dte")
    if explicit is not None:
        return int(explicit)
    time_exit = (cfg.get("exit_rules") or {}).get("time_exit_dte", 21)
    return int(time_exit) + int(al.get("cohort_min_runway_days", 9))


def _phase1_start(config_path: str) -> str:
    with open(config_path) as f:
        cfg = json.load(f)
    start = (cfg.get("auto_log") or {}).get("phase1_start_date")
    if not start:
        raise SystemExit("config.json missing auto_log.phase1_start_date")
    return start


def find_candidates(db_path: str, floor: int, phase1_start: str):
    """Return list of (entry_id, ticker, date, expiration, dte, status, pnl_pct)."""
    sql = (
        "SELECT entry_id, ticker, date, expiration, status, pnl_pct, "
        "       CAST(julianday(expiration) - julianday(date) AS INTEGER) AS dte "
        "FROM trades "
        "WHERE strategy_name = 'Long Call' "
        "  AND COALESCE(paper_only, 0) = 0 "
        "  AND date >= ? "
        "  AND expiration IS NOT NULL AND date IS NOT NULL "
        "  AND (julianday(expiration) - julianday(date)) < ? "
        "ORDER BY date"
    )
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, (phase1_start, floor)).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--apply", action="store_true",
                    help="Actually update the DB (default is dry-run).")
    args = ap.parse_args()

    floor = _floor(args.config)
    phase1_start = _phase1_start(args.config)
    rows = find_candidates(args.db, floor, phase1_start)

    print(f"Cohort DTE floor: {floor}   forward-cohort start: {phase1_start}")
    print(f"Contaminated forward-cohort Long Calls (entry DTE < {floor}): {len(rows)}\n")
    if not rows:
        print("Nothing to reclassify.")
        return
    print(f"  {'entry_id':>8}  {'ticker':<6} {'entry':<11} {'exp':<11} {'dte':>3} "
          f"{'status':<7} {'pnl_pct':>8}")
    for eid, tkr, d, exp, status, pnl, dte in rows:
        pnl_s = f"{pnl:+.3f}" if pnl is not None else "  open"
        print(f"  {eid:>8}  {tkr:<6} {str(d)[:10]:<11} {str(exp)[:10]:<11} {dte:>3} "
              f"{str(status):<7} {pnl_s:>8}")

    if not args.apply:
        print(f"\nDRY-RUN: would set paper_only=1 on {len(rows)} trade(s). "
              f"Re-run with --apply to commit.")
        return

    backup = f"{args.db}.bak.{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(args.db, backup)
    ids = [r[0] for r in rows]
    with sqlite3.connect(args.db) as conn:
        conn.executemany("UPDATE trades SET paper_only = 1 WHERE entry_id = ?",
                         [(i,) for i in ids])
        conn.commit()
    print(f"\nBacked up to {backup}")
    print(f"Reclassified {len(ids)} forward-cohort Long Call(s) to paper_only=1.")


if __name__ == "__main__":
    main()
