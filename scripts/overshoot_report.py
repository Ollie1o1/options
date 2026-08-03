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
from datetime import date
from typing import Any, Dict, Optional

# The measurement itself lives in `src/overshoot` so the short-premium gate can
# state a MEASURED overshoot share in its exit-fidelity caveat rather than a
# remembered one. This module keeps the CLI and the prose.
from src.overshoot import (  # noqa: E402
    SCHEDULER_DIED,
    fetch_stop_exits,
    is_stop_exit,
    overshoot_for,
    parse_stop_level,
    summarize,
)


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
