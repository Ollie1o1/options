#!/usr/bin/env python3
"""Re-read the ledger at its true fill price, using archived quotes.

Every logged entry was priced at the bid/ask MID (`options_screener.py:2160`
sets `premium = mid`) and charged slippage only on the way out, so entry
friction has always been zero in this book. `data/chain_archive.db` holds real
CBOE quotes for 15 symbols since 2026-06-10, which is enough to restate the
subset of trades it covers at what they would actually have filled for.

This writes ONLY the v18 columns. `entry_price`, `net_credit` and `pnl_usd` are
read-only here: the ledger records what happened, and a second reading of it
belongs beside the first, not on top of it. Every write is reversible with
--undo.

    python -m scripts.restate_execution --dry-run     # counts, writes nothing
    python -m scripts.restate_execution               # apply
    python -m scripts.restate_execution --undo        # clear the v18 columns
    python -m scripts.restate_execution --report      # honest P&L by strategy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import statistics
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import execution_restate as er  # noqa: E402
from src import execution_truth as et  # noqa: E402

DEFAULT_DB = "paper_trades.db"
DEFAULT_ARCHIVE = "data/chain_archive.db"

_V18_COLUMNS = ("entry_price_mid", "entry_price_fill", "entry_price_cross",
                "fill_policy", "fill_source")


def _ensure_v18(db_path: str) -> None:
    """Bring the ledger up to the schema this script writes.

    The live DB only migrates when a PaperManager is constructed, so a CLI
    backfill on a v17 file would otherwise die on `no such column`. Migrating
    here is additive — v18 only ADDs columns — and idempotent."""
    from src.paper_manager import PaperManager
    PaperManager(db_path=db_path)


def _backup(db_path: str) -> str:
    dest = f"{db_path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(db_path, dest)
    return dest


def _quote_lookup(archive: sqlite3.Connection, symbol: str, snap_date: str,
                  expiration: str):
    """(strike, type) -> (bid, ask) for one symbol/day/expiry, or None.

    Quotes are read once per trade rather than per leg: a spread must be priced
    from a single snapshot or the two legs come from different moments."""
    rows = archive.execute(
        "SELECT strike, type, bid, ask FROM chain_snapshots "
        "WHERE symbol=? AND snap_date=? AND expiration=?",
        (symbol, snap_date, expiration)).fetchall()
    table = {}
    for strike, opt_type, bid, ask in rows:
        if bid is None or ask is None:
            continue
        table[(round(float(strike), 4), str(opt_type).lower())] = (float(bid), float(ask))

    def lookup(strike: float, opt_type: str) -> Optional[tuple]:
        return table.get((round(float(strike), 4), str(opt_type).lower()))

    return lookup


def backfill(db_path: str = DEFAULT_DB, archive_path: str = DEFAULT_ARCHIVE,
             policy: str = "limit", k: Optional[float] = None,
             dry_run: bool = False, undo: bool = False,
             backup: bool = False) -> Dict[str, Any]:
    """Populate (or clear) the v18 execution columns. Returns counts."""
    if backup and not dry_run and os.path.exists(db_path):
        _backup(db_path)
    _ensure_v18(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if undo:
            n = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE fill_source IS NOT NULL").fetchone()[0]
            if not dry_run:
                conn.execute(
                    "UPDATE trades SET " +
                    ", ".join(f"{c}=NULL" for c in _V18_COLUMNS))
                conn.commit()
            return {"cleared": n, "scanned": n, "priced": 0, "unknown": 0}

        trades = [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE duplicate_of IS NULL ORDER BY entry_id")]

        archive = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
        counts = {"scanned": 0, "priced": 0, "unknown": 0}
        updates = []
        try:
            for row in trades:
                counts["scanned"] += 1
                date = (row.get("date") or "")[:10]
                lookup = _quote_lookup(archive, row.get("ticker") or "",
                                       date, row.get("expiration") or "")
                out = er.restate(row, lookup, policy=policy, k=k)
                if out["fill_source"] == "unknown":
                    counts["unknown"] += 1
                else:
                    counts["priced"] += 1
                updates.append((out, row["entry_id"]))
        finally:
            archive.close()

        if not dry_run:
            conn.executemany(
                "UPDATE trades SET entry_price_mid=?, entry_price_fill=?, "
                "entry_price_cross=?, fill_policy=?, fill_source=? WHERE entry_id=?",
                [(o["entry_price_mid"], o["entry_price_fill"], o["entry_price_cross"],
                  o["fill_policy"], o["fill_source"], eid) for o, eid in updates])
            conn.commit()
        return counts
    finally:
        conn.close()


def report(db_path: str = DEFAULT_DB) -> str:
    """P&L by strategy at each fill policy, split by how the fill was sourced.

    Measured and unpriceable rows are never pooled into one number."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE duplicate_of IS NULL AND status!='OPEN' "
            "AND fill_source='live_quote'")]
    finally:
        conn.close()

    if not rows:
        return ("No restated rows. Run `python -m scripts.restate_execution` first, "
                "or the archive covers none of the ledger.")

    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(r.get("strategy_name") or "?", []).append(r)

    out = [
        "Restated on archived quotes — rows the archive could price exactly.",
        "p* is the win rate the structure must beat to break even at that fill.",
        "",
        f"{'strategy':<14}{'n':>4}{'win%':>7}{'credit@mid':>12}{'@limit':>9}"
        f"{'@cross':>9}{'p*@mid':>9}{'p*@limit':>10}{'p*@cross':>10}",
    ]
    for name, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        wins = sum(1 for r in rs if (r.get("pnl_usd") or 0) > 0)
        priced = [r for r in rs if r.get("spread_width")]
        if not priced:
            continue

        def med(vals):
            vals = [v for v in vals if v is not None]
            return statistics.median(vals) if vals else float("nan")

        def med_breakeven(key):
            """Median of each trade's OWN p*, not p* of the median credit.

            The two diverge whenever widths are mixed — and they are, from $1
            to $29 in this book — by enough to flip a line from clearing its
            observed win rate to missing it."""
            return med([et.breakeven_win_rate(r[key], r["spread_width"])
                        for r in priced if r.get(key) is not None])

        m, l, c = (med([r["entry_price_mid"] for r in priced]),
                   med([r["entry_price_fill"] for r in priced]),
                   med([r["entry_price_cross"] for r in priced]))
        ps = [med_breakeven(k) for k in
              ("entry_price_mid", "entry_price_fill", "entry_price_cross")]
        fmt = lambda p: f"{100 * p:.1f}%" if p == p else "  n/a"
        out.append(
            f"{name:<14}{len(rs):>4}{100 * wins / len(rs):>6.1f}%"
            f"{m:>12.2f}{l:>9.2f}{c:>9.2f}"
            f"{fmt(ps[0]):>9}{fmt(ps[1]):>10}{fmt(ps[2]):>10}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--policy", default="limit", choices=list(et.POLICIES))
    ap.add_argument("--k", type=float, default=None,
                    help=f"limit aggressiveness, 0=mid 1=cross (default {et.DEFAULT_LIMIT_K})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--report", action="store_true", help="print honest P&L and exit")
    args = ap.parse_args()

    if args.report:
        print(report(args.db))
        return

    counts = backfill(args.db, args.archive, policy=args.policy, k=args.k,
                      dry_run=args.dry_run, undo=args.undo,
                      backup=not args.no_backup)
    if args.undo:
        print(f"cleared v18 columns on {counts['cleared']} rows"
              f"{' (dry run)' if args.dry_run else ''}")
        return
    print(f"scanned {counts['scanned']}  priced {counts['priced']}  "
          f"unpriceable {counts['unknown']}{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
