#!/usr/bin/env python3
"""Measure how far stopped-out trades ran PAST their stop before anyone looked.

A stop rule says "exit at -50% of premium" or "at 100% of the credit". The
ledger holds exits at -157.5%, -110.3%, -104.1%. The rules were not wrong and
the market did not gap that far — the exits were simply CHECKED late, because
`update_positions` only runs when the screener is opened. The LaunchAgents have
been dead since 2026-06-15, so through that window "when the operator opened the
app" was the entire exit schedule.

That matters beyond tidiness: the overshoot is not spread evenly. Defined-risk
credit structures are the ones whose stops are stated as a multiple of a small
credit, so they absorb most of it — which makes every credit line look worse and
noisier than its own rules imply, and credit-vs-debit is the comparison the whole
backlog turns on.

This script measures the artifact. It does not correct it, and nothing here
edits the ledger: the record stays as-traded and the methodology note explains
it. The fix is prospective and not code — the Login Items toggle that lets the
scheduler run (see ideas.json step_0_user_only).

Overshoot is defined per trade as:

    overshoot = |realized return| - |stop level|

in the units the stop was stated in (percent of entry premium), so a trade
stopped exactly on its rule scores 0.0 and anything positive is the excess.
Trades whose stop has no fixed level ("strike breached") carry a real stop rule
but not a numeric one, so they are counted and reported separately rather than
folded into a distribution they cannot belong to.

Read-only: opens the ledger through a `mode=ro` URI. Usage:

    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/overshoot_report.py
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The date the LaunchAgents stopped running (logs/launchagent.log last entry).
# Exits before it were checked on a timer; exits after it were checked whenever
# the operator happened to open the app.
SCHEDULER_DIED = "2026-06-15"

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


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%"


def format_summary(s: Dict[str, Any]) -> str:
    """The methodology note, as markdown. Also the script's stdout report."""
    lines: List[str] = []
    lines.append(
        f"Exit checks run inside `update_positions`, which runs when the "
        f"screener is opened — not on a timer. The scheduled LaunchAgents "
        f"stopped running on **{s['cutoff']}**, so from that date exits were "
        f"checked at irregular, manual intervals. A stop rule cannot fire on a "
        f"day nobody looked, so stopped-out trades in that window record the "
        f"loss they had drifted to by the next check, not the loss the rule "
        f"specified."
    )
    lines.append("")
    lines.append(
        f"Measured over {s['n_levelled']} closed trades whose exit reason "
        f"states a numeric stop level (of {s['n_stop_exits']} stop exits in "
        f"total; {s['n_unlevelled']} more stopped on a strike breach, which "
        f"has no numeric level to overshoot). Overshoot is the realized loss "
        f"minus the stated stop, in percent of entry premium:"
    )
    lines.append("")
    lines.append("| window | trades | median overshoot | p90 | worst | share past their stop |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, key in (("Before " + s["cutoff"], "before"),
                       ("After " + s["cutoff"] + " (manual cadence)", "after"),
                       ("All", "all")):
        d = s[key]
        share = ("n/a" if d["share_overshot"] is None
                 else f"{d['share_overshot'] * 100:.0f}%")
        lines.append(
            f"| {label} | {d['n']} | {_pct(d['median'])} | {_pct(d['p90'])} | "
            f"{_pct(d['max'])} | {share} |"
        )
    lines.append("")
    lines.append(
        "**The recorded exits are not corrected for this and never will be.** "
        "The record stays as-traded; this note is how it is read. Losses on "
        "stopped trades in the manual-cadence window are overstated relative "
        "to the rules that were supposed to govern them, and because "
        "defined-risk credit structures state their stops as a multiple of a "
        "small credit, the overstatement falls hardest on exactly the lines "
        "the credit-vs-debit comparison depends on."
    )
    lines.append("")
    lines.append(
        "This note is removed only once the scheduler has been verifiably "
        "alive for a full window — not when it is merely fixed."
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure stop-overshoot from sparse exit checks")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--cutoff", default=SCHEDULER_DIED,
                    help="date the scheduler died (YYYY-MM-DD)")
    args = ap.parse_args()

    rows = fetch_stop_exits(args.db)
    summary = summarize(rows, cutoff=args.cutoff)
    print(format_summary(summary))
    print()
    print("Worst overshoots:")
    for w in summary["worst"]:
        print(f"  {w['exit_day']}  {w['ticker']:<6} {w['strategy']:<14} "
              f"{_pct(w['pnl_pct'])} vs rule → overshoot {_pct(w['overshoot'])}"
              f"   [{w['exit_reason']}]")


if __name__ == "__main__":
    main()
