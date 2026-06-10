"""Startup maintenance: run the jobs the (now-retired) cron used to run —
auto-log, exit-enforcement, weekly checkpoint — idempotently, crash-isolated,
when the user opens the screener. Replaces cron, which kept silently dying.

Design:
- Pure throttle-decision helpers (``due_*``, ``autolog_window``) decide *what* to
  run; they are trivially testable and never touch the filesystem.
- Side-effecting steps go through an injectable ``runner`` (subprocess by default)
  so tests never spawn processes.
- Every step is wrapped so a failure can never stop the screener from starting.
- Throttle state persists in ``logs/.maintenance_state.json``.
- The cohort-progress line reuses ``phase1_checkpoint.compute_checkpoint`` so the
  cohort filter lives in exactly one place.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime
from typing import Callable, Optional

from src import phase1_checkpoint

VENV_PY = os.path.expanduser("~/.venvs/options/bin/python")
DEFAULT_STATE_PATH = os.path.join("logs", ".maintenance_state.json")


# ── Throttle decisions (pure) ───────────────────────────────────────────────

def _days_between(a: str, b: str) -> int:
    da = datetime.strptime(a, "%Y-%m-%d")
    db = datetime.strptime(b, "%Y-%m-%d")
    return (db - da).days


def due_checkpoint(state: dict, today: str, min_days: int = 7) -> bool:
    """Weekly checkpoint runs only if >= min_days since the last one."""
    last = (state or {}).get("last_checkpoint")
    if not last:
        return True
    try:
        return _days_between(last, today) >= min_days
    except ValueError:
        return True


def due_track_record(state: dict, today: str, min_days: int = 7) -> bool:
    """Public track-record refresh runs at most weekly (same pattern as checkpoint)."""
    last = (state or {}).get("last_track_record")
    if not last:
        return True
    try:
        return _days_between(last, today) >= min_days
    except ValueError:
        return True


def autolog_window(weekday: int, hhmm: int):
    """weekday: 1=Mon..7=Sun (datetime.isoweekday()). hhmm: e.g. 1430.
    Returns (key, mode_flag) for the active scan window, or None.
    Windows mirror the retired cron's clock gates."""
    if weekday > 5:
        return None
    if 1015 <= hhmm <= 1130:
        return ("ds", "-ds")
    if 1215 <= hhmm <= 1330:
        return ("sps", "-sps")
    if 1400 <= hhmm <= 1500:
        return ("ics", "-ics")
    return None


def due_autolog(state: dict, window_key: str, today: str) -> bool:
    """Auto-log runs at most once per (window, calendar day)."""
    last = ((state or {}).get("last_autolog") or {}).get(window_key)
    return (not last) or last != today


# ── State persistence (failure-safe) ────────────────────────────────────────

def load_state(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1)


# ── Cohort progress (reuses the checkpoint cohort filter) ───────────────────

def _open_cohort_count(db_path: str, phase1_start: str) -> int:
    sql = ("SELECT COUNT(*) FROM trades WHERE strategy_name='Long Call' "
           "AND status='OPEN' AND COALESCE(paper_only,0)=0 AND date >= ?")
    try:
        with sqlite3.connect(db_path) as conn:
            return int(conn.execute(sql, (phase1_start,)).fetchone()[0])
    except sqlite3.Error:
        return 0


def cohort_progress_line(db_path: str, phase1_start: str, today: Optional[str] = None) -> str:
    r = phase1_checkpoint.compute_checkpoint(db_path, phase1_start, today=today)
    n_open = _open_cohort_count(db_path, phase1_start)
    return (f"Forward cohort: {r['n_trades']}/50 closed clean | open: {n_open} | "
            f"weeks: {r['weeks_elapsed']} | gate: {r['decision']}")


# ── Orchestrator ────────────────────────────────────────────────────────────

def _default_runner(cmd) -> int:
    """Run a subprocess, capturing output to logs/maintenance.log."""
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "maintenance.log"), "a") as logf:
        logf.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] $ {' '.join(cmd)}\n")
        logf.flush()
        return subprocess.call(cmd, stdout=logf, stderr=logf)


def _run_checkpoint(db_path: str, phase1_start: str) -> None:
    result = phase1_checkpoint.compute_checkpoint(db_path, phase1_start)
    phase1_checkpoint.write_checkpoint(result, output_dir="reports")


def _run_track_record(db_path: str) -> None:
    from scripts.publish_track_record import publish
    publish(db_path=db_path, reports_dir="reports")


def run_startup_maintenance(db_path: str = "paper_trades.db",
                            phase1_start: Optional[str] = None,
                            state_path: str = DEFAULT_STATE_PATH,
                            now: Optional[datetime] = None,
                            runner: Optional[Callable] = None,
                            checkpoint_fn: Optional[Callable] = None,
                            track_record_fn: Optional[Callable] = None) -> dict:
    """Run due maintenance jobs, crash-isolated. Returns {'cohort': line, 'ran': [...]}.
    Never raises.

    Note: exit-rule enforcement is NOT run here. The interactive screener startup
    already enforces exits synchronously via ``PaperManager.update_positions()``;
    duplicating it would mean a second ~60s market scan on every boot.
    """
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    runner = runner or _default_runner
    state = load_state(state_path)
    ran = []

    # 1. Auto-log (once per window/day, weekdays, in-window only).
    try:
        win = autolog_window(now.isoweekday(), now.hour * 100 + now.minute)
        if win and due_autolog(state, win[0], today):
            rc = runner([VENV_PY, "run.py", win[1], "--5", "--no-ai"])
            if rc == 0:
                state.setdefault("last_autolog", {})[win[0]] = today
                ran.append(f"auto-log:{win[0]}")
    except Exception:
        pass

    # 2. Weekly checkpoint (>=7 days) — read-only gate refresh.
    try:
        if phase1_start and due_checkpoint(state, today):
            fn = checkpoint_fn or _run_checkpoint
            fn(db_path=db_path, phase1_start=phase1_start)
            state["last_checkpoint"] = today
            ran.append("checkpoint")
    except Exception:
        pass

    # 3. Weekly public track-record refresh (>=7 days), read-only over the db.
    try:
        if due_track_record(state, today):
            fn = track_record_fn or _run_track_record
            fn(db_path=db_path)
            state["last_track_record"] = today
            ran.append("track_record")
    except Exception:
        pass

    try:
        save_state(state_path, state)
    except Exception:
        pass

    # 4. Cohort progress (always; failure-safe).
    try:
        cohort = cohort_progress_line(db_path, phase1_start, today=today) if phase1_start else ""
    except Exception:
        cohort = ""
    return {"cohort": cohort, "ran": ran}


# ── Headless entry point (LaunchAgent / manual) ──────────────────────────────

def run_headless(db_path: str = "paper_trades.db",
                 config_path: str = "config.json",
                 state_path: str = DEFAULT_STATE_PATH,
                 now: Optional[datetime] = None,
                 runner: Optional[Callable] = None,
                 checkpoint_fn: Optional[Callable] = None,
                 track_record_fn: Optional[Callable] = None) -> dict:
    """Run startup maintenance without the interactive screener.

    The LaunchAgent entry: reads phase1_start from config.json itself,
    delegates to ``run_startup_maintenance`` (which is already crash-isolated
    per job), and never raises — a scheduler must not crashloop on a bad day.
    Exit-enforcement is covered transitively: the auto-log subprocess runs the
    screener, whose startup enforces exits.
    """
    try:
        with open(config_path) as f:
            phase1_start = (json.load(f).get("auto_log") or {}).get("phase1_start_date")
    except (OSError, ValueError):
        phase1_start = None
    try:
        return run_startup_maintenance(
            db_path=db_path, phase1_start=phase1_start, state_path=state_path,
            now=now, runner=runner, checkpoint_fn=checkpoint_fn,
            track_record_fn=track_record_fn)
    except Exception:
        return {"cohort": "", "ran": []}


def main() -> None:
    summary = run_headless()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ran = ", ".join(summary.get("ran") or []) or "nothing due"
    print(f"[{stamp}] maintenance: {ran}")
    if summary.get("cohort"):
        print(f"  {summary['cohort']}")


if __name__ == "__main__":
    main()
