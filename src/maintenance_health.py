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

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

try:  # colour when a TTY formatter is available; plain text otherwise
    from src import formatting as _fmt
except Exception:  # pragma: no cover
    _fmt = None

try:  # the shared component kit; banner falls back to bare rules without it
    from src import ui as _ui
except Exception:  # pragma: no cover
    _ui = None

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
            "track-record": "WARN", "chain-archive": "STALE",
            "morning-briefing": "STALE"}


@dataclass(frozen=True)
class JobHealth:
    name: str
    cadence: str
    last_run: Optional[str]        # "YYYY-MM-DD" or None (never ran)
    business_days_stale: int
    severity: str


@dataclass(frozen=True)
class LaunchdJob:
    label: str
    pid: Optional[int]
    last_exit_status: int

    @property
    def failed(self) -> bool:
        return self.last_exit_status != 0


# EX_CONFIG. On macOS this is what a LaunchAgent returns when the OS refuses to
# run it, which is a permissions decision the user has to reverse by hand — not
# a bug in the job. Reporting it as a generic failure sends the reader hunting
# through the script instead of into System Settings.
_EX_CONFIG = 78


def parse_launchctl_list(output: str, prefix: str) -> List["LaunchdJob"]:
    """Parse `launchctl list` into our jobs. Unparseable lines are skipped.

    The state file cannot answer this question: the interactive path stamps it
    too, so a scheduler that has not fired in six weeks still looks recent
    whenever someone opens the screener by hand.
    """
    jobs: List[LaunchdJob] = []
    for line in (output or "").splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        pid_raw, status_raw, label = parts[0], parts[1], parts[-1]
        if not label.startswith(prefix):
            continue
        try:
            status = int(status_raw)
        except ValueError:
            continue
        pid = int(pid_raw) if pid_raw.isdigit() else None
        jobs.append(LaunchdJob(label=label, pid=pid, last_exit_status=status))
    return jobs


def launchd_failure_message(jobs: List["LaunchdJob"]) -> Optional[str]:
    """One line naming dead schedulers, or None when they are all healthy."""
    failed = [j for j in jobs if j.failed]
    if not failed:
        return None
    names = ", ".join(j.label.rsplit(".", 1)[-1] for j in failed)
    msg = (f"{len(failed)} scheduled job(s) are not running: {names} "
           f"(last exit {', '.join(str(j.last_exit_status) for j in failed)})")
    if any(j.last_exit_status == _EX_CONFIG for j in failed):
        msg += (". Exit 78 is macOS refusing to launch them — fix in "
                "System Settings > General > Login Items & Extensions > "
                "Allow in the Background. Nothing scheduled runs until then")
    return msg


def read_launchd_status(prefix: str = "com.options-screener") -> List["LaunchdJob"]:
    """Live `launchctl list`. Returns [] if launchctl is unavailable — this is a
    diagnostic, and must never be the reason a run fails."""
    import subprocess
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_launchctl_list(out, prefix)


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
        _job("morning-briefing", "daily", (state or {}).get("last_morning_briefing"), today),
    ]
    worst = "OK"
    for j in jobs:
        if _RANK[j.severity] > _RANK[worst]:
            worst = j.severity
    autolog = next(j for j in jobs if j.name == "auto-log")
    return HealthReport(jobs=jobs, worst=worst,
                        autolog_missed_days=autolog.business_days_stale, now=today)


def _style(text: str, name: str, bold: bool = None) -> str:
    """Semantic style, or plain text when formatting is unavailable."""
    if _fmt is None:
        return text
    try:
        return _fmt.style(text, name, bold=bold)
    except Exception:
        return text


def _glyph(name: str, fallback: str = "") -> str:
    if _fmt is None:
        return fallback
    return _fmt.GLYPHS.get(name, fallback)


_SEV_STYLE = {"WARN": "warn", "STALE": "warn", "CRITICAL": "bad"}


def _wrap_body(lines: List[str], width: int) -> List[str]:
    """Wrap body lines to the card's inner width.

    ``ui.card`` pads to a fixed width but never wraps, so any line longer than
    the box pushes the right border off screen. The launchd message is the one
    that gets there — three job names plus the exit-78 remedy is ~280 chars —
    and it renders on every startup while the agents are down.

    Body lines are plain text at this point (only the title carries ANSI), so a
    plain textwrap is safe. Long tokens are left intact so paths and the
    `python -m src.maintenance --health` command stay copy-pasteable.
    """
    import textwrap

    inner = max(20, width - 4)  # "│ " + body + " │"
    out: List[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, inner, break_long_words=False,
                                 break_on_hyphens=False) or [""])
    return out


def health_banner(report: HealthReport, width: int = 100,
                  launchd_jobs: Optional[List["LaunchdJob"]] = None) -> str:
    """A loud, escalating banner — empty string when everything is fresh.

    Rendered as a boxed `ui.card` so the banner sits on the same component kit
    as every other panel. Falls back to bare rules if `ui` is unavailable.

    ``launchd_jobs`` is checked independently of state freshness. A dead
    scheduler and a fresh state file happen together all the time — opening the
    screener by hand stamps the file — and that combination is precisely how
    three agents sat dead for six weeks without the banner ever firing.
    """
    launchd_msg = launchd_failure_message(launchd_jobs or [])
    if report.worst == "OK" and not launchd_msg:
        return ""
    if report.worst == "OK" and launchd_msg:
        warn = _glyph("warn", "!")
        title = _style(f"{warn} SCHEDULER DEAD — nothing is running on a timer",
                       "err", bold=True)
        body = _wrap_body(
            [launchd_msg,
             "State files look current because interactive runs stamp them too."],
            width)
        if _ui is not None:
            return _ui.card(title, body, width, boxed=True)
        rule = _style("─" * width, "err")
        return "\n".join([rule, "  " + title] + ["  " + b for b in body] + [rule])
    autolog = next(j for j in report.jobs if j.name == "auto-log")
    sev = _SEV_STYLE.get(report.worst, "warn")
    warn = _glyph("warn", "!")

    body: List[str] = []
    if autolog.severity != "OK":
        # The important case: the cohort filler itself is behind.
        title = f"{warn} AUTOMATION {report.worst} — cohort auto-log is behind"
        body += [
            f"auto-log last ran {autolog.last_run or 'never'} "
            f"({autolog.business_days_stale} business days ago)",
            f"≈ {report.autolog_missed_days} trading day(s) of cohort filling missed "
            f"— the n≥50 gate is that far behind",
        ]
    else:
        # Auto-log is current; only the background refreshes have fallen behind.
        title = (f"{warn} MAINTENANCE {report.worst} — background jobs behind "
                 f"(cohort auto-log is current)")

    for j in report.jobs:
        if j.name == "auto-log" or j.severity == "OK":
            continue
        body.append(f"{j.name} last ran {j.last_run or 'never'} "
                    f"({j.business_days_stale} business days ago) [{j.severity}]")
    if launchd_msg:
        body.append(launchd_msg)
    body.append("Launch during market hours to catch up, or run "
                "`python -m src.maintenance --health` for detail.")

    body = _wrap_body(body, width)
    styled_title = _style(title, sev, bold=True)
    if _ui is not None:
        # ui.card pads on visible width, so a pre-styled (ANSI) title aligns.
        return _ui.card(styled_title, body, width, boxed=True)
    rule = _style("─" * width, sev)
    return "\n".join([rule, "  " + styled_title] + ["  " + b for b in body] + [rule])


# Days a dead scheduler must persist before the interactive path hard-blocks
# with a required acknowledgement (`dead_scheduler_ack_banner`). `launchctl
# list` carries no timestamps, so duration is tracked by stamping
# first-observed-dead into the maintenance state (`launchd_dead_since`) and
# diffing it against `today` on every later read — see `launchd_dead_days`
# and `next_launchd_dead_state`.
DEAD_SCHEDULER_ACK_DAYS = 7


def launchd_dead_days(jobs: Optional[List["LaunchdJob"]], state: dict,
                      today: date) -> Optional[int]:
    """Calendar days the scheduler has been observed dead, or None while
    healthy (or when launchctl gave us nothing to judge, e.g. it isn't
    available — see `read_launchd_status`).

    Pure: reads the `launchd_dead_since` marker already in `state` but never
    touches the filesystem itself. Pair with `next_launchd_dead_state` to
    persist the marker forward; the caller owns the actual write.
    """
    if not any(j.failed for j in (jobs or [])):
        return None
    since_str = (state or {}).get("launchd_dead_since")
    if not since_str:
        return 0
    try:
        since = date.fromisoformat(since_str)
    except ValueError:
        return 0
    return max(0, (today - since).days)


DEFAULT_LAUNCHAGENT_LOG_PATH = os.path.join("logs", "launchagent.log")

_LOG_LINE_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}:\d{2}\]")


def seed_dead_since_date(log_path: str = DEFAULT_LAUNCHAGENT_LOG_PATH) -> Optional[str]:
    """Best lower-bound evidence for when the scheduler was last alive.

    Reads `logs/launchagent.log`, written *only* by the LaunchAgents
    themselves — unlike the maintenance state file, nothing about an
    interactive run touches it, so its last write carries none of the
    "opened the screener by hand" contamination `health_banner`'s own
    docstring warns about. Prefers the last parsed `[YYYY-MM-DD HH:MM:SS]`
    line; falls back to the file's own mtime if it exists but has no
    parseable line. Returns None when the file doesn't exist at all (a
    fresh clone, or a machine that never had the agents installed) — the
    caller should fall back to "today" rather than invent a date from
    nothing.
    """
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = _LOG_LINE_TS_RE.match(line)
        if m:
            return m.group(1)
    try:
        mtime = os.path.getmtime(log_path)
    except OSError:
        return None
    return datetime.fromtimestamp(mtime).date().isoformat()


def next_launchd_dead_state(jobs: Optional[List["LaunchdJob"]], state: dict,
                            today: date, seed_date: Optional[str] = None) -> dict:
    """The state dict to persist after this read.

    Stamps `launchd_dead_since` the first time the scheduler is observed
    dead, and clears it the moment it recovers — so fixing the Login Items
    toggle resets the clock instead of leaving the ack permanently primed.

    A first observation is seeded from `seed_date` (pass
    `seed_dead_since_date()` — the LaunchAgent log's own last-write
    evidence) rather than `today`, so a scheduler that has actually been
    dead for weeks doesn't get handed a fresh one-day-old marker the moment
    this code first runs. Falls back to `today` when `seed_date` is None
    (no log to read evidence from). An *existing* marker is never
    overwritten by seeding, whatever `seed_date` says.

    Returns a new dict; the input `state` is never mutated. Callers own the
    actual file write (and should skip it when the marker is unchanged).
    """
    new_state = dict(state or {})
    if any(j.failed for j in (jobs or [])):
        if not new_state.get("launchd_dead_since"):
            new_state["launchd_dead_since"] = seed_date or today.isoformat()
    else:
        new_state.pop("launchd_dead_since", None)
    return new_state


def dead_scheduler_ack_banner(dead_days: int, width: int = 100) -> str:
    """The escalated card shown once the scheduler has been dead longer than
    `DEAD_SCHEDULER_ACK_DAYS` — interactive path only, a required Enter
    keypress, no bypass (see `options_screener._dead_scheduler_ack`).

    States plainly what degrades without a running scheduler: exit checks
    (stop-loss / take-profit) fall to manual cadence, and cohort accrual for
    the gate stops advancing on its own.
    """
    warn = _glyph("warn", "!")
    title = _style(f"{warn} SCHEDULER DEAD {dead_days} DAYS — confirm to continue",
                   "bad", bold=True)
    body = _wrap_body([
        f"No scheduled job has run in {dead_days} days. Two things are "
        "silently degrading:",
        "exit checks (stop-loss / take-profit) are on manual cadence — a "
        "position can run past its stop between visits.",
        "cohort accrual for the gate is manual-only — the n>=50 threshold "
        "advances only when you open the screener by hand.",
        "Fix in System Settings > General > Login Items & Extensions > "
        "Allow in the Background.",
    ], width)
    if _ui is not None:
        return _ui.card(title, body, width, boxed=True)
    rule = _style("─" * width, "bad")
    return "\n".join([rule, "  " + title] + ["  " + b for b in body] + [rule])


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
