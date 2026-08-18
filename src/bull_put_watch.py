"""Does the restored Bull Put line clear its own bar, measured by the repaired stack?

Bull Put was switched off by accident on 2026-08-01 — removed from
`paper_only_strategies` so the short-premium gate's cohort could authorise
capital, and never added to `allowed_strategies`, so `apply_auto_log_allowlist`
dropped every candidate for seventeen days. Restored 2026-08-18.

Its headline 66.4% over 131 closed trades is the reason it was restored, and it
is NOT evidence about what happens next: every one of those trades was priced
by the pre-repair EV layer, before the spread-EV sign fix, the measured vol
basis and the measured error bar. The trades entered from the restore date are
the first ones the repaired stack has produced, and they are what this tracks.

Two things this deliberately refuses to do:

  * Score open positions. An open trade is not a win, and counting it as one is
    how a cohort flatters itself right up until it closes.
  * Hardcode the bar. The required win rate is the MANAGED figure, computed
    from realised payoffs under the exits actually used, and it drifts as
    trades close. A frozen 50.9% would eventually compare this month's win rate
    against last month's bar. It is read from the ledger.

Read-only on the ledger, no network. Live marks on open positions belong to
`python -m src.check_pnl`, which already prices them properly; duplicating that
here would add a network dependency to a report whose value is the closed-trade
arithmetic.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# The day Bull Put was restored to `allowed_strategies`. Trades before it were
# selected and priced by a different system.
COHORT_START = "2026-08-18"

# Closed trades before the win rate is worth reading. Not a significance
# threshold — it is the point at which the number stops being anecdote. n=3 at
# 100% is noise and the report says so rather than implying a verdict.
TARGET_N = 20

STRATEGY = "Bull Put"


@dataclass(frozen=True)
class Report:
    required: Optional[float]
    n_open: int
    n_closed: int
    n_wins: int
    total_pnl: float
    open_rows: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    closed_rows: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def win_rate(self) -> Optional[float]:
        return (self.n_wins / self.n_closed) if self.n_closed else None

    @property
    def provisional(self) -> bool:
        """True until the sample is large enough to read at all."""
        return self.n_closed < TARGET_N

    @property
    def clears(self) -> bool:
        """Whether the delivered rate is at or above the required one. Says
        nothing about significance — read it with `provisional`."""
        wr, req = self.win_rate, self.required
        return wr is not None and req is not None and wr >= req


def _required_from_ledger(db_path: str) -> Optional[float]:
    try:
        from .candidate_verdict import required_win_rates_from_ledger
        return required_win_rates_from_ledger(db_path).get(STRATEGY)
    except Exception:
        return None


def report(db_path: str = "paper_trades.db",
           *,
           cohort_start: str = COHORT_START,
           required: Optional[float] = None) -> Report:
    """The cohort as it stands. Never raises — a watch that dies is a watch
    that stops being read."""
    if required is None:
        required = _required_from_ledger(db_path)

    open_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM trades WHERE strategy_name = ? AND date >= ? "
            "ORDER BY date, entry_id", (STRATEGY, cohort_start))
        for r in cur:
            row = dict(r)
            if str(row.get("status") or "").upper() == "CLOSED":
                closed_rows.append(row)
            else:
                open_rows.append(row)
        conn.close()
    except Exception:
        pass

    scored = [r for r in closed_rows if r.get("pnl_usd") is not None]
    wins = sum(1 for r in scored if float(r["pnl_usd"]) > 0)
    total = sum(float(r["pnl_usd"]) for r in scored)

    return Report(required=required, n_open=len(open_rows),
                  n_closed=len(scored), n_wins=wins, total_pnl=total,
                  open_rows=tuple(open_rows), closed_rows=tuple(scored))


def render(rep: Report) -> List[str]:
    """Plain lines for a log file. No colour, no theme lookup."""
    req_txt = f"{rep.required * 100:.1f}%" if rep.required is not None else "n/a"
    lines = [
        f"BULL PUT WATCH — cohort from {COHORT_START} (repaired-stack trades only)",
        f"  open {rep.n_open}  |  closed n={rep.n_closed}  |  "
        f"required {req_txt}",
    ]

    if rep.n_closed == 0:
        lines.append("  delivered: n/a — nothing has closed yet.")
    else:
        wr = rep.win_rate or 0.0
        verdict = "at or above" if rep.clears else "BELOW"
        lines.append(
            f"  delivered {wr * 100:.1f}% ({rep.n_wins}/{rep.n_closed}) — "
            f"{verdict} the required {req_txt}  |  P&L ${rep.total_pnl:,.2f}")

    if rep.provisional:
        need = TARGET_N - rep.n_closed
        lines.append(
            f"  PROVISIONAL — {need} more close{'s' if need != 1 else ''} "
            f"before this number is worth acting on (target n={TARGET_N}). "
            f"The 66.4% history was measured on the PRE-repair EV layer.")

    for r in rep.open_rows:
        strike = r.get("strike")
        long_strike = r.get("long_strike")
        legs = (f"{strike:g}/{long_strike:g}p"
                if strike is not None and long_strike is not None else "?")
        lines.append(
            f"    open  {str(r.get('ticker') or '?'):<6} {legs:<12} "
            f"entered {str(r.get('date') or '')[:10]}  "
            f"exp {str(r.get('expiration') or '')[:10]}  "
            f"risk ${float(r.get('capital_at_risk') or 0):,.0f}")

    for r in rep.closed_rows:
        pnl = float(r.get("pnl_usd") or 0.0)
        lines.append(
            f"    closed {str(r.get('ticker') or '?'):<6} "
            f"{'WIN ' if pnl > 0 else 'LOSS'} ${pnl:>9,.2f}  "
            f"{str(r.get('exit_reason') or '')}")

    return lines
