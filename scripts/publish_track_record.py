#!/usr/bin/env python3
"""
Publish a public paper-trading track record to reports/TRACK_RECORD.md.

Reads closed trades from paper_trades.db and renders an honest, plainly-caveated
summary: win rate, average return, per-strategy breakdown, the forward-cohort
gate status, and a full timestamped table of closed trades.

Pure rendering (`render_track_record`) is separated from I/O so it is testable
against a seeded in-memory SQLite db. Wired into the weekly startup-maintenance
throttle (see src/maintenance.py) so it refreshes at most once a week.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Allow running as a plain script (python scripts/publish_track_record.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence import load_model_evidence, format_evidence_banner  # noqa: E402

_CLOSED_COLUMNS = [
    "date", "ticker", "strategy_name", "type", "strike", "expiration",
    "entry_price", "exit_price", "pnl_pct", "pnl_usd", "exit_reason",
    "paper_only", "status",
]


def fetch_closed_trades(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all CLOSED trades as a list of plain dicts, oldest first."""
    cols = ", ".join(_CLOSED_COLUMNS)
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT {cols} FROM trades WHERE UPPER(status) = 'CLOSED' ORDER BY date ASC"
        )
    except sqlite3.OperationalError:
        return []
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _fmt_pct(v: Optional[float]) -> str:
    # pnl_pct is stored as a fraction (0.42 == +42%); scale to percent for display.
    try:
        return f"{float(v) * 100.0:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_money(v: Optional[float]) -> str:
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def render_track_record(rows: List[Dict[str, Any]], evidence: Dict[str, Any]) -> str:
    """Render the closed-trade rows + model evidence into a Markdown document."""
    n = len(rows)
    pnls = [r.get("pnl_pct") for r in rows if r.get("pnl_pct") is not None]
    wins = [p for p in pnls if float(p) > 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    avg_pnl = (sum(float(p) for p in pnls) / len(pnls)) if pnls else 0.0

    out: List[str] = []
    out.append("# Paper Trading Track Record")
    out.append("")
    out.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M} • {n} closed trades_")
    out.append("")
    out.append("> **Methodology & caveats.** These are **paper trades**, not live "
               "fills. Entries and exits use **delayed retail data** (Yahoo Finance) "
               "and a **modeled friction** assumption (spread/slippage), so realized "
               "results would differ. The descriptive stats below are real; the "
               "**predictive edge of the ranking model is still under out-of-sample "
               "evaluation** and is *not* established — see "
               "[docs/VALIDATION_POWER.md](../docs/VALIDATION_POWER.md).")
    out.append("")
    out.append(f"_{format_evidence_banner(evidence)}_")
    out.append("")

    # --- Summary -------------------------------------------------------------
    out.append("## Summary")
    out.append("")
    out.append(f"- Closed trades: **{n}**")
    out.append(f"- Win rate: **{win_rate:.1f}%** ({len(wins)}/{len(pnls)} with a recorded return)")
    out.append(f"- Average return: **{_fmt_pct(avg_pnl)}** per trade (paper, pre-tax, modeled friction)")
    out.append("")

    # --- Per-strategy breakdown ----------------------------------------------
    out.append("## By strategy")
    out.append("")
    out.append("| Strategy | Closed | Win rate | Avg return |")
    out.append("|----------|-------:|---------:|-----------:|")
    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_strat.setdefault(r.get("strategy_name") or "Unknown", []).append(r)
    for strat in sorted(by_strat):
        srows = by_strat[strat]
        sp = [float(r["pnl_pct"]) for r in srows if r.get("pnl_pct") is not None]
        swins = [p for p in sp if p > 0]
        swr = (len(swins) / len(sp) * 100.0) if sp else 0.0
        savg = (sum(sp) / len(sp)) if sp else 0.0
        out.append(f"| {strat} | {len(srows)} | {swr:.1f}% | {_fmt_pct(savg)} |")
    out.append("")

    # --- Forward-cohort gate -------------------------------------------------
    out.append("## Forward-cohort gate")
    out.append("")
    out.append(f"- Gate decision: **{evidence.get('gate_decision', 'UNKNOWN')}**")
    out.append(f"- Cohort size: **{evidence.get('cohort_n', 0)}** "
               "(closed cohort trades accumulated since the gate window opened)")
    out.append("")

    # --- Full table ----------------------------------------------------------
    out.append("## Closed trades")
    out.append("")
    out.append("| Date | Ticker | Structure | Entry | Exit | P&L % | Exit reason |")
    out.append("|------|--------|-----------|------:|-----:|------:|-------------|")
    for r in rows:
        out.append(
            f"| {r.get('date', '—')} | {r.get('ticker', '—')} | "
            f"{r.get('strategy_name', '—')} | {_fmt_money(r.get('entry_price'))} | "
            f"{_fmt_money(r.get('exit_price'))} | {_fmt_pct(r.get('pnl_pct'))} | "
            f"{r.get('exit_reason') or '—'} |"
        )
    out.append("")
    return "\n".join(out)


def publish(db_path: str = "paper_trades.db", reports_dir: str = "reports") -> Optional[str]:
    """Read the db, render, and write reports/TRACK_RECORD.md. Returns the path."""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        rows = fetch_closed_trades(conn)
    finally:
        conn.close()
    evidence = load_model_evidence(reports_dir)
    md = render_track_record(rows, evidence)
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "TRACK_RECORD.md")
    with open(out_path, "w") as f:
        f.write(md)
    return out_path


def main() -> int:
    path = publish()
    if path:
        print(f"Wrote {path}")
        return 0
    print("paper_trades.db not found; nothing to publish.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
