"""Measuring how far stopped-out trades ran PAST their stop.

Extracted from `scripts/overshoot_report.py` so the number is available to the
code that has to *qualify a verdict* with it, not only to a report a human
runs. The short-premium gate's exit-fidelity caveat previously stated the share
as a string literal ("94%"), which meant it could not change when the ledger
changed and could not stop being true once the scheduler was fixed.

Overshoot is defined per trade as:

    overshoot = |realized return| - |stop level|

in the units the stop was stated in (percent of entry premium), so a trade
stopped exactly on its rule scores 0.0 and anything positive is the excess.
Trades whose stop has no numeric level ("strike breached") are counted and
reported separately rather than folded into a distribution they cannot join.

Read-only throughout: the ledger records what happened, and nothing here
edits it. The overshoot is an artifact of WHEN exits were checked, not of the
rules themselves — see `SCHEDULER_DIED`.
"""
from __future__ import annotations

import re
import sqlite3
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


# The scheduler outage was a closed window, not an ongoing condition. It stopped
# firing after 2026-06-15 and resumed on 2026-08-04 (launchd shows the agent
# active with a clean exit; logs/launchagent.log has the gap and the recovery).
# The window matters because exits went unenforced inside it, which is what the
# overshoot below measures — but a caveat written in the present tense would now
# overstate a problem that has been fixed.
SCHEDULER_DIED = "2026-06-15"
SCHEDULER_RECOVERED = "2026-08-04"

# Stop levels are written into the exit reason itself, which makes the rule that
# fired recoverable per trade rather than inferred from config that has changed
# since. Each pattern maps to the stop level as a fraction of entry premium.
_STOP_PATTERNS: Sequence[Tuple[str, float]] = (
    (r"stop loss \((-?\d+(?:\.\d+)?)% of credit\)", 0.01),
    (r"stop loss \(-?(\d+(?:\.\d+)?)%\)", 0.01),
)

_UNLEVELLED = "strike breached"


def parse_stop_level(exit_reason: Optional[str]) -> Optional[float]:
    """The stop level a reason states, as a positive fraction of premium.

    Returns None when the reason is not a stop at all, or is a stop with no
    numeric level (a strike breach). None is never silently treated as zero —
    a missing level must exclude the trade, not score it as a perfect exit.
    """
    if not exit_reason:
        return None
    text = str(exit_reason).strip().lower()
    if not text.startswith("stop loss") and "stop" not in text:
        return None
    for pattern, scale in _STOP_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return abs(float(m.group(1))) * scale
    return None


def is_stop_exit(exit_reason: Optional[str]) -> bool:
    return bool(exit_reason) and "stop" in str(exit_reason).strip().lower()


def overshoot_for(pnl_pct: Optional[float],
                  exit_reason: Optional[str]) -> Optional[float]:
    """Excess loss beyond the stated stop, as a fraction of premium.

    Only losses can overshoot a stop: a trade recorded as a stop exit that
    settled positive is a data oddity, not a 'negative overshoot', so it is
    excluded rather than averaged in as if it offset a real overshoot.
    """
    level = parse_stop_level(exit_reason)
    if level is None or pnl_pct is None:
        return None
    if pnl_pct >= 0:
        return None
    return abs(float(pnl_pct)) - level


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "median": None, "p90": None, "max": None,
                "n_overshot": 0, "share_overshot": None}
    overshot = [v for v in vals if v > 1e-9]
    idx = max(0, int(round(0.90 * (len(vals) - 1))))
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "p90": vals[idx],
        "max": vals[-1],
        "n_overshot": len(overshot),
        "share_overshot": len(overshot) / len(vals),
    }


def fetch_stop_exits(db_path: str) -> List[Dict[str, Any]]:
    """Every closed trade whose exit reason names a stop. Read-only."""
    sql = (
        "SELECT date, exit_date, ticker, strategy_name, pnl_pct, pnl_usd, "
        "       exit_reason, capital_at_risk "
        "FROM trades "
        "WHERE UPPER(status) = 'CLOSED' AND exit_reason IS NOT NULL"
    )
    out: List[Dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql):
            r = dict(row)
            if is_stop_exit(r.get("exit_reason")):
                out.append(r)
    return out


def _exit_day(row: Dict[str, Any]) -> Optional[str]:
    """The date the exit was recorded, falling back to the entry date.

    Rows predating the exit_date column carry only `date`; using it keeps them
    in the split rather than dropping them, and it can only mis-bin a trade
    that was ENTERED before the scheduler died and exited after — which biases
    against the finding, not toward it.
    """
    for key in ("exit_date", "date"):
        v = row.get(key)
        if v:
            return str(v)[:10]
    return None


def summarize(rows: Sequence[Dict[str, Any]],
              cutoff: str = SCHEDULER_DIED) -> Dict[str, Any]:
    """Overshoot distributions, split on the scheduler's death."""
    levelled: List[Tuple[str, float, Dict[str, Any]]] = []
    unlevelled: List[Dict[str, Any]] = []

    for r in rows:
        reason = str(r.get("exit_reason") or "").lower()
        ov = overshoot_for(r.get("pnl_pct"), r.get("exit_reason"))
        if ov is None:
            if _UNLEVELLED in reason:
                unlevelled.append(r)
            continue
        day = _exit_day(r)
        if not day:
            continue
        levelled.append((day, ov, r))

    before = [ov for day, ov, _ in levelled if day < cutoff]
    after = [ov for day, ov, _ in levelled if day >= cutoff]

    worst = sorted(levelled, key=lambda t: t[1], reverse=True)[:10]

    return {
        "cutoff": cutoff,
        "n_stop_exits": len(rows),
        "n_levelled": len(levelled),
        "n_unlevelled": len(unlevelled),
        "all": _distribution([ov for _, ov, _ in levelled]),
        "before": _distribution(before),
        "after": _distribution(after),
        "worst": [
            {"ticker": r.get("ticker"), "strategy": r.get("strategy_name"),
             "exit_day": day, "pnl_pct": r.get("pnl_pct"),
             "exit_reason": r.get("exit_reason"), "overshoot": ov}
            for day, ov, r in worst
        ],
    }
