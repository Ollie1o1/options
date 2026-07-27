"""Self-calibrating structure league table.

Each structure earns or loses its place by its own realized payoffs. Nothing is
hardcoded about which structure is good - if Long Put's edge decays, its margin
shrinks and it benches itself.

Read-only against paper_trades.db.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np

from .types import StructureMargin

DEFAULT_DB = "paper_trades.db"
DEFAULT_HISTORY = os.path.join("data", "structure_margins.tsv")
_HISTORY_COLS = ("date", "strategy", "n", "breakeven_hit", "realized_hit",
                 "margin", "state")


def breakeven_hit(avg_win: float, avg_loss: float) -> Optional[float]:
    """Hit rate needed to break even given observed payoff asymmetry.

    avg_loss is passed as a positive magnitude. Returns None when the payoffs
    are degenerate (both zero), rather than dividing by zero.
    """
    total = float(avg_win) + float(avg_loss)
    if total <= 0:
        return None
    return float(avg_loss) / total


def _bootstrap_margin_ci(pnl, breakeven, n_boot=2000, seed=0):
    """95% CI on (realized_hit - breakeven) by resampling the trade list."""
    arr = np.asarray(pnl, dtype=float)
    if len(arr) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    hits = [float((rng.choice(arr, len(arr), replace=True) > 0).mean()) - breakeven
            for _ in range(n_boot)]
    return float(np.quantile(hits, 0.025)), float(np.quantile(hits, 0.975))


def compute_league_table(db_path: str = DEFAULT_DB, window_days: int = 90,
                         now: Optional[datetime] = None, min_n: int = 20,
                         min_wins: int = 8, min_losses: int = 8
                         ) -> Dict[str, StructureMargin]:
    """Build the league table from closed trades inside the rolling window.

    A missing or unreadable DB yields an empty table - the engine then reports
    "no structure evidence available" rather than inventing a default.
    """
    if not os.path.exists(db_path):
        return {}
    now = now or datetime.now()
    cutoff = (now - timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT strategy_name, pnl_usd FROM trades "
                "WHERE status='CLOSED' AND pnl_usd IS NOT NULL "
                "AND strategy_name IS NOT NULL AND date >= ?",
                (cutoff,)).fetchall()
    except sqlite3.Error:
        return {}

    by_strategy: Dict[str, list] = {}
    for name, pnl in rows:
        by_strategy.setdefault(str(name), []).append(float(pnl))

    table: Dict[str, StructureMargin] = {}
    for name, pnl in by_strategy.items():
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = abs(float(np.mean(losses))) if losses else 0.0
        be = breakeven_hit(avg_win, avg_loss)
        if be is None:
            continue
        realized = len(wins) / len(pnl)
        margin = realized - be
        lo, hi = _bootstrap_margin_ci(pnl, be)

        # Sufficiency: breakeven needs BOTH sides estimated. A large n with few
        # wins cannot pin down avg_win, so total-n alone is not enough.
        sufficient = (len(pnl) >= min_n and len(wins) >= min_wins
                      and len(losses) >= min_losses)
        state = "ACTIVE" if sufficient else "UNPROVEN"

        table[name] = StructureMargin(
            strategy=name, n=len(pnl), wins=len(wins), losses=len(losses),
            avg_win=avg_win, avg_loss=avg_loss, breakeven_hit=be,
            realized_hit=realized, margin=margin, state=state,
            ci_lo=lo, ci_hi=hi)
    return table


def load_history(path: str = DEFAULT_HISTORY):
    """Read weekly snapshots. Missing file is not an error - it is week one."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path) as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != len(header):
                    continue
                row = dict(zip(header, parts))
                row["n"] = int(float(row.get("n", 0) or 0))
                for key in ("breakeven_hit", "realized_hit", "margin"):
                    row[key] = float(row.get(key, 0) or 0)
                out.append(row)
    except (OSError, ValueError):
        return []
    return out


def append_snapshot(path, table, today: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a") as f:
        if not exists:
            f.write("\t".join(_HISTORY_COLS) + "\n")
        for m in table.values():
            f.write("\t".join([
                today, m.strategy, str(m.n), "{:.6f}".format(m.breakeven_hit),
                "{:.6f}".format(m.realized_hit), "{:.6f}".format(m.margin),
                m.state]) + "\n")


def _last_for(history, strategy):
    rows = [h for h in history if h.get("strategy") == strategy]
    return rows[-1] if rows else None


def apply_states(table, history, today: str):
    """Resolve ACTIVE/BENCHED with two-step hysteresis in both directions.

    UNPROVEN always wins - a structure without enough evidence is never benched
    (that would imply we measured it) and never recommended.
    """
    for name, m in table.items():
        if m.state == "UNPROVEN":
            continue
        prev = _last_for(history, name)
        was_benched = bool(prev) and prev.get("state") == "BENCHED"
        prev_margin = float(prev["margin"]) if prev else None

        if was_benched:
            # Need two consecutive non-negative to earn the way back.
            if m.margin >= 0 and prev_margin is not None and prev_margin >= 0:
                m.state = "ACTIVE"
            else:
                m.state = "BENCHED"
        else:
            # Need two consecutive negative to be benched.
            if m.margin < 0 and prev_margin is not None and prev_margin < 0:
                m.state = "BENCHED"
            else:
                m.state = "ACTIVE"
    return table
