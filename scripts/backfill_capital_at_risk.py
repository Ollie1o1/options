#!/usr/bin/env python3
"""Populate capital_at_risk on ledger rows that predate the column.

Deterministic — every input is already stored on the row, so this can be
re-run safely and never overwrites a value written at log time.

    python scripts/backfill_capital_at_risk.py --dry-run
    python scripts/backfill_capital_at_risk.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capital_risk import capital_at_risk  # noqa: E402

_FIELDS = ("entry_id", "ticker", "strike", "entry_price", "max_loss_usd",
           "quantity", "strategy_name")


def backfill(db_path: str, dry_run: bool = False) -> dict:
    """Fill NULL capital_at_risk values. Returns counts, writes nothing on dry_run.

    ``unbounded`` counts rows whose risk cannot be derived (naked calls, legacy
    rows missing fields). Those stay NULL on purpose: a 0 would read as free.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        if "capital_at_risk" not in columns:
            raise RuntimeError(
                f"{db_path} has no capital_at_risk column — it arrives with schema "
                f"v16. Open the ledger once (any PaperManager call, e.g. "
                f"`python -m src.check_pnl`) to run the migration, then re-run."
            )
        rows = conn.execute(
            f"SELECT {', '.join(_FIELDS)} FROM trades WHERE capital_at_risk IS NULL"
        ).fetchall()

        updates, unbounded = [], 0
        for row in rows:
            risk = capital_at_risk(
                strategy_name=row["strategy_name"] or "",
                entry_price=row["entry_price"],
                strike=row["strike"],
                max_loss_usd=row["max_loss_usd"],
                quantity=row["quantity"] if row["quantity"] is not None else 1.0,
                ticker=row["ticker"],
            )
            if risk is None:
                unbounded += 1
            else:
                updates.append((risk, row["entry_id"]))

        if updates and not dry_run:
            conn.executemany(
                "UPDATE trades SET capital_at_risk = ? WHERE entry_id = ?", updates
            )
            conn.commit()
    finally:
        conn.close()

    return {"scanned": len(rows), "updated": len(updates), "unbounded": unbounded}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    # Opening the ledger applies any pending migration, so the column exists
    # before the backfill looks for it.
    from src.paper_manager import PaperManager  # noqa: PLC0415 - keeps import cost off the test path

    PaperManager(db_path=args.db, config_path=args.config)

    result = backfill(args.db, dry_run=args.dry_run)
    verb = "would fill" if args.dry_run else "filled"
    print(
        f"{result['scanned']} rows missing capital_at_risk; "
        f"{verb} {result['updated']}; {result['unbounded']} left NULL (risk unbounded)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
