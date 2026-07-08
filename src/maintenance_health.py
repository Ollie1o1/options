"""Automation staleness guard.

The screener's gate and track record only advance when the daily maintenance
(auto-log / auto-close / checkpoint / archive) actually runs — and it only runs
when something launches it (the interactive screener or the headless
LaunchAgent). When the Mac is asleep or the LaunchAgent isn't approved, nothing
runs and *nothing warns you*: the "truth machine" quietly stops being true
(observed 2026-07-07 — auto-log had not run for 11 calendar days).

This module reads the maintenance state file and reports, per job, how many
*business* days it has been since it last ran (so a normal weekend never trips
it), with an escalating severity and a plain-English estimate of the damage
(trading days of cohort filling missed). It is pure — it computes and renders;
callers decide where to surface it and whether to force a catch-up.

`health_banner` returns "" when everything is fresh, so a healthy system is
silent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

try:  # colour when a TTY formatter is available; plain text otherwise
    from src import formatting as _fmt
except Exception:  # pragma: no cover
    _fmt = None

# The working auto-log windows that must run every RTH weekday (mirrors
# maintenance.WORKING_AUTOLOG_WINDOWS). If any is missing from state, the
# cohort feeder has not fully run and auto-log is treated as not-run.
WORKING_WINDOWS = ("ds", "sps", "ss", "ics")

# stale-day thresholds per cadence: (<=days -> severity); anything past the last
# tier is CRITICAL. Daily jobs should run every trading day; weekly jobs refresh
# roughly every 7 calendar (~5 business) days with buffer.
_THRESH = {
    "daily": [(1, "OK"), (2, "WARN"), (4, "STALE")],
    "weekly": [(6, "OK"), (9, "WARN"), (14, "STALE")],
}
_RANK = {"OK": 0, "WARN": 1, "STALE": 2, "CRITICAL": 3}
_NEVER = 99  # sentinel business-day age when a job has no recorded run

# Per-job severity ceiling. Only auto-log — the cohort/gate filler — can reach
# CRITICAL; the read-only refreshes (checkpoint/archive) top out at STALE and the
# cosmetic public publish at WARN, so a stale archive never shouts as loudly as a
# starving cohort.
_MAX_SEV = {"auto-log": "CRITICAL", "checkpoint": "STALE",
            "track-record": "WARN", "chain-archive": "STALE"}


@dataclass(frozen=True)
class JobHealth:
    name: str
    cadence: str
    last_run: Optional[str]        # "YYYY-MM-DD" or None (never ran)
    business_days_stale: int
    severity: str


@dataclass(frozen=True)
class HealthReport:
    jobs: List[JobHealth]
    worst: str
    autolog_missed_days: int
    now: date


def business_days_between(d1: date, d2: date) -> int:
    """Number of weekdays strictly after d1 up to and including d2. 0 when d2
    is on or before d1. Weekends (Sat/Sun) are not counted, so a Friday run
    checked on Monday reads as 1 business day, not 3."""
    if d2 <= d1:
        return 0
    from datetime import timedelta
    n = 0
    cur = d1
    while cur < d2:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:  # Mon-Fri
            n += 1
    return n


def _severity(stale_days: Optional[int], cadence: str) -> str:
    if stale_days is None:
        return "CRITICAL"
    for thresh, sev in _THRESH[cadence]:
        if stale_days <= thresh:
            return sev
    return "CRITICAL"


def _autolog_last_run(state: dict) -> Optional[str]:
    """Oldest run date across the working windows, or None if any never ran."""
    al = (state or {}).get("last_autolog") or {}
    dates = []
    for w in WORKING_WINDOWS:
        d = al.get(w)
        if not d:
            return None
        dates.append(d)
    return min(dates) if dates else None


def _job(name: str, cadence: str, last_run: Optional[str], today: date) -> JobHealth:
    if last_run:
        try:
            stale = business_days_between(date.fromisoformat(last_run), today)
        except ValueError:
            stale = _NEVER
            last_run = None
    else:
        stale = _NEVER
    sev = _severity(stale, cadence)
    cap = _MAX_SEV.get(name, "CRITICAL")
    if _RANK[sev] > _RANK[cap]:
        sev = cap
    return JobHealth(name=name, cadence=cadence, last_run=last_run,
                     business_days_stale=stale, severity=sev)


def compute_health(state: dict, now) -> HealthReport:
    """Build the health report from a maintenance state dict. `now` may be a
    date or datetime."""
    today = now.date() if isinstance(now, datetime) else now
    jobs = [
        _job("auto-log", "daily", _autolog_last_run(state), today),
        _job("checkpoint", "weekly", (state or {}).get("last_checkpoint"), today),
        _job("track-record", "weekly", (state or {}).get("last_track_record"), today),
        _job("chain-archive", "daily", (state or {}).get("last_chain_archive"), today),
    ]
    worst = "OK"
    for j in jobs:
        if _RANK[j.severity] > _RANK[worst]:
            worst = j.severity
    autolog = next(j for j in jobs if j.name == "auto-log")
    return HealthReport(jobs=jobs, worst=worst,
                        autolog_missed_days=autolog.business_days_stale, now=today)


def _c(text: str, attr: str, bold: bool = False) -> str:
    if _fmt is None:
        return text
    try:
        return _fmt.colorize(text, getattr(_fmt.Colors, attr), bold=bold)
    except Exception:
        return text


_SEV_COLOR = {"WARN": "BRIGHT_YELLOW", "STALE": "BRIGHT_YELLOW",
              "CRITICAL": "BRIGHT_RED"}


def health_banner(report: HealthReport) -> str:
    """A loud, escalating banner — empty string when everything is fresh."""
    if report.worst == "OK":
        return ""
    autolog = next(j for j in report.jobs if j.name == "auto-log")
    color = _SEV_COLOR.get(report.worst, "BRIGHT_YELLOW")
    rule = "=" * 78
    lines = [_c(rule, color)]

    if autolog.severity != "OK":
        # The important case: the cohort filler itself is behind.
        head = _c(f"⚠ AUTOMATION {report.worst} — cohort auto-log is behind", color, bold=True)
        lines += [
            "  " + head,
            f"  auto-log last ran {autolog.last_run or 'never'} "
            f"({autolog.business_days_stale} business days ago)",
            f"  ≈ {report.autolog_missed_days} trading day(s) of cohort filling missed "
            f"— the n≥50 gate is that far behind",
        ]
    else:
        # Auto-log is current; only the background refreshes have fallen behind.
        head = _c(f"⚠ MAINTENANCE {report.worst} — background jobs behind "
                  f"(cohort auto-log is current)", color, bold=True)
        lines.append("  " + head)

    others = [j for j in report.jobs
              if j.name != "auto-log" and j.severity != "OK"]
    for j in others:
        lines.append(f"  {j.name} last ran {j.last_run or 'never'} "
                     f"({j.business_days_stale} business days ago) [{j.severity}]")
    lines.append("  Launch during market hours to catch up, or run "
                 "`python -m src.maintenance --health` for detail.")
    lines.append(_c(rule, color))
    return "\n".join(lines)


def health_lines(report: HealthReport) -> List[str]:
    """Per-job breakdown for the standalone `--health` command (all jobs, even
    healthy ones)."""
    out = [f"Maintenance health as of {report.now.isoformat()} "
           f"(worst: {report.worst})", "-" * 60]
    for j in report.jobs:
        last = j.last_run or "never"
        out.append(f"  {j.name:<14} last {last:<12} "
                   f"{j.business_days_stale:>3} business days ago  [{j.severity}]")
    return out
