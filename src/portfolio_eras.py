"""Portfolio P&L split by ERA — before vs after the 2026-06-16 data overhaul.

`pre_data`  = every trade logged before the real-marks / screener overhaul
              (the old scoring that bought contracts without cost-aware EV,
              relative-value, or concentration checks).
`finalized` = trades logged by the overhauled screener (net-of-cost EV ranking +
              quant read + portfolio guard).

Keeping them apart is the only honest way to tell whether the new process trades
better than the old one — never pool them. Realized P&L from CLOSED trades; open
positions are counted, not marked.

CLI:
    python -m src.portfolio_eras                 # both eras from paper_trades.db
    python -m src.portfolio_eras --db paper_trades.db
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

ERAS = ("pre_data", "finalized")


def era_stats(db_path: str = "paper_trades.db") -> Dict[str, Dict[str, Any]]:
    """Per-era summary: closed-trade realized P&L + open count. Robust if the
    `era` column doesn't exist yet (treats everything as pre_data)."""
    out: Dict[str, Dict[str, Any]] = {}
    with sqlite3.connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        era_expr = "era" if "era" in cols else "'pre_data'"
        for era in ERAS:
            closed = conn.execute(
                f"SELECT pnl_usd, pnl_pct FROM trades "
                f"WHERE {era_expr}=? AND status='CLOSED' AND pnl_usd IS NOT NULL", (era,)
            ).fetchall()
            open_n = conn.execute(
                f"SELECT COUNT(*) FROM trades WHERE {era_expr}=? AND status='OPEN'", (era,)
            ).fetchone()[0]
            total_n = conn.execute(
                f"SELECT COUNT(*) FROM trades WHERE {era_expr}=?", (era,)
            ).fetchone()[0]
            pnls = [r[0] for r in closed if r[0] is not None]
            wins = sum(1 for p in pnls if p > 0)
            pcts = [r[1] for r in closed if r[1] is not None]
            out[era] = {
                "total": total_n,
                "closed": len(pnls),
                "open": open_n,
                "realized_pnl_usd": round(sum(pnls), 2),
                "win_rate": round(wins / len(pnls), 4) if pnls else None,
                "avg_pnl_pct": round(sum(pcts) / len(pcts), 4) if pcts else None,
            }
    return out


def by_strategy(db_path: str, era: str) -> List[tuple]:
    """Closed realized P&L by strategy within an era (for the breakdown)."""
    with sqlite3.connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        era_expr = "era" if "era" in cols else "'pre_data'"
        return conn.execute(
            f"SELECT strategy_name, COUNT(*), COALESCE(SUM(pnl_usd),0) FROM trades "
            f"WHERE {era_expr}=? AND status='CLOSED' AND pnl_usd IS NOT NULL "
            f"GROUP BY strategy_name ORDER BY SUM(pnl_usd)", (era,)
        ).fetchall()


def _fmt_money(x):
    return f"${x:,.2f}" if x is not None else "  -  "


def _fmt_pct(x):
    return f"{x:+.1%}" if x is not None else "  -  "


def print_report(db_path: str = "paper_trades.db") -> None:
    stats = era_stats(db_path)
    print(f"Portfolio P&L by era  ({db_path})")
    print("=" * 72)
    labels = {"pre_data": "PRE-DATA ERA (old scoring, before 2026-06-16 overhaul)",
              "finalized": "FINALIZED ERA (cost-aware screener, 2026-06-16+)"}
    for era in ERAS:
        s = stats[era]
        print(f"\n[{labels[era]}]")
        if s["total"] == 0:
            print("  (no trades yet)")
            continue
        print(f"  trades: {s['total']}  ({s['closed']} closed, {s['open']} open)")
        print(f"  realized P&L: {_fmt_money(s['realized_pnl_usd'])}  |  "
              f"win rate: {_fmt_pct(s['win_rate'])}  |  avg/trade: {_fmt_pct(s['avg_pnl_pct'])}")
        rows = by_strategy(db_path, era)
        if rows:
            print("  by strategy (closed):")
            for name, n, pnl in rows:
                print(f"    {name:18} n={n:<4} {_fmt_money(pnl)}")
    print("\n" + "=" * 72)
    print("Note: eras are NEVER pooled — the split is how we judge old vs new process.")


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Portfolio P&L split by era (pre-data vs finalized)")
    ap.add_argument("--db", default="paper_trades.db")
    args = ap.parse_args()
    print_report(args.db)


if __name__ == "__main__":
    _cli()
