#!/usr/bin/env python3
"""Apply the operator's ruling on reports/duplicate_trades_audit.md.

The audit flags CANDIDATES: same (ticker, strategy, strike, expiration) with the
same entry price to the cent, within three days. It deliberately rules on
nothing, because a contract legitimately re-picked a day later looks identical
at that resolution.

The ruling made on 2026-08-01 turned on one test the audit did not run: **did
the flagged day log anything else?** A catch-up replay — the documented failure
mode behind `auto_log.dedup_window_days` — re-logs the PREVIOUS day's set, so
the repeats would be most of what that day contains. They were not:

    2026-06-08:  33 logged |  5 flagged | 28 fresh
    2026-06-09:  16 logged |  5 flagged | 11 fresh
    2026-04-18:  15 logged |  6 flagged |  9 fresh
    2026-04-19:  10 logged |  6 flagged |  4 fresh

Every flagged day carries a full batch of unrelated fresh trades. These are
normal scans in which a deterministic screener re-picked a handful of the same
contracts on consecutive days — at quotes that had not moved, which is why the
mid, and therefore `entry_price` and `capital_at_risk`, are bit-identical while
`entry_iv` differs (one day less to expiry re-prices the vol, not the quote).

That leaves exactly one true double-log in 882 rows: two WFC Short Put rows
entered the SAME day with bit-identical `entry_iv`. The screener ran once, so
one snapshot cannot yield two independent decisions.

Marks, never deletes: `duplicate_of` points the excess row at the row it
duplicates. The record of what happened stays whole; the evidence stops
double-counting. Re-runnable and reversible (`--undo`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (duplicate_row, the_row_it_duplicates). Keep the FIRST logged row of a group:
# it is the original decision; the later row is the accidental echo.
RULED_DUPLICATES = [
    (91, 90),  # WFC Short Put $77.5 exp 2026-05-15 — same day, same snapshot
]


def _backup(db_path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join("backups", f"{os.path.basename(db_path)}.bak.{stamp}")
    os.makedirs("backups", exist_ok=True)
    shutil.copy2(db_path, dest)
    return dest


def apply(db_path: str, undo: bool = False, dry_run: bool = False) -> dict:
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if "duplicate_of" not in cols:
        conn.close()
        raise SystemExit(
            "ledger predates schema v17 — open it through PaperTradeManager "
            "once so the migration runs, then re-run this script")

    changed = []
    for dup_id, keep_id in RULED_DUPLICATES:
        row = conn.execute(
            "SELECT entry_id, ticker, strategy_name, pnl_usd, duplicate_of "
            "FROM trades WHERE entry_id = ?", (dup_id,)).fetchone()
        if row is None:
            continue
        target = None if undo else keep_id
        if row[4] == target:
            continue
        changed.append({"entry_id": row[0], "ticker": row[1],
                        "strategy": row[2], "pnl_usd": row[3],
                        "duplicate_of": target})
        if not dry_run:
            conn.execute("UPDATE trades SET duplicate_of = ? WHERE entry_id = ?",
                         (target, dup_id))
    if not dry_run:
        conn.commit()
    conn.close()
    return {"changed": changed, "dry_run": dry_run, "undo": undo}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--undo", action="store_true",
                    help="Clear the marks instead of setting them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        print(f"Backup: {_backup(args.db)}")
    res = apply(args.db, undo=args.undo, dry_run=args.dry_run)
    if not res["changed"]:
        print("Nothing to change — the ruling is already applied.")
        return
    verb = "Would clear" if args.dry_run and args.undo else \
           "Would mark" if args.dry_run else \
           "Cleared" if args.undo else "Marked"
    total = sum(float(c["pnl_usd"] or 0) for c in res["changed"])
    for c in res["changed"]:
        print(f"  {verb} #{c['entry_id']} {c['ticker']} {c['strategy']} "
              f"(${float(c['pnl_usd'] or 0):,.2f}) → duplicate_of="
              f"{c['duplicate_of']}")
    print(f"{verb} {len(res['changed'])} row(s), ${total:,.2f} of P&L "
          f"{'restored to' if args.undo else 'removed from'} the record.")


if __name__ == "__main__":
    main()
