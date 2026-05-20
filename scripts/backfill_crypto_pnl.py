"""One-time backfill: derive quantity, recompute pnl_usd for
paper_trades_crypto.db. Fixes the x100 inflation AND applies the $1,000 cap.
Dry-run by default; --apply writes after taking a .bak snapshot."""
from __future__ import annotations
import argparse, os, shutil, sqlite3, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.sizing import capped_quantity

_CREDIT = ("Bear Call", "Iron Condor", "Bull Put")


def _unit_risk(strategy_name: str, entry_price, spread_width, net_credit):
    if any(strategy_name.startswith(s) for s in _CREDIT):
        if spread_width is None or net_credit is None:
            return None
        return float(spread_width) - float(net_credit)
    return float(entry_price)


def compute_backfill(db_path: str):
    """Return per-row dicts; performs NO writes."""
    out = []
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM trades WHERE status='CLOSED'"):
            ur = _unit_risk(r["strategy_name"] or "", r["entry_price"],
                            r["spread_width"], r["net_credit"])
            if not ur or ur <= 0:
                continue
            qty = capped_quantity(unit_risk=ur, cap_usd=1000.0)
            new_pnl = float(r["pnl_pct"] or 0.0) * float(r["entry_price"] or 0.0) * qty
            out.append({
                "entry_id": r["entry_id"],
                "strategy": r["strategy_name"],
                "old_pnl_usd": float(r["pnl_usd"] or 0.0),
                "new_quantity": qty,
                "new_pnl_usd": round(new_pnl, 4),
            })
    return out


def apply_backfill(db_path: str):
    snap = f"{db_path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(db_path, snap)
    rows = compute_backfill(db_path)
    with sqlite3.connect(db_path) as c:
        for r in rows:
            c.execute("UPDATE trades SET quantity=?, pnl_usd=? WHERE entry_id=?",
                      (r["new_quantity"], r["new_pnl_usd"], r["entry_id"]))
    return snap, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="paper_trades_crypto.db")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (snapshots .bak first); default dry-run")
    a = ap.parse_args(argv)
    rows = compute_backfill(a.db)
    tot_old = sum(r["old_pnl_usd"] for r in rows)
    tot_new = sum(r["new_pnl_usd"] for r in rows)
    print(f"{'id':>4} {'strategy':<22} {'old_pnl':>12} {'new_qty':>9} {'new_pnl':>11}")
    for r in rows:
        print(f"{r['entry_id']:>4} {str(r['strategy'])[:22]:<22} "
              f"{r['old_pnl_usd']:>12,.2f} {r['new_quantity']:>9.4f} "
              f"{r['new_pnl_usd']:>11,.2f}")
    print(f"\nTOTAL  old ${tot_old:,.2f}  →  new ${tot_new:,.2f}  ({len(rows)} rows)")
    if a.apply:
        snap, _ = apply_backfill(a.db)
        print(f"APPLIED. Snapshot: {snap}")
    else:
        print("DRY-RUN — re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
