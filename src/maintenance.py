"""Startup maintenance: run the jobs the (now-retired) cron used to run —
auto-log, exit-enforcement, weekly checkpoint — idempotently, crash-isolated,
when the user opens the screener. Replaces cron, which kept silently dying.

Design:
- Pure throttle-decision helpers (``due_*``, ``due_autolog_windows``) decide *what*
  to run; they are trivially testable and never touch the filesystem.
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

# The auto-log job spawns `run.py <mode>` — which boots the screener, which
# calls run_startup_maintenance again. The parent records the window as done
# only AFTER the child exits, so without a marker the child sees the window
# still due and spawns another full scan: confirmed as a ~170-deep process
# bomb on 2026-06-10. Children are marked via this env var and skip all
# maintenance (the parent owns it).
CHILD_ENV_MARKER = "OPTIONS_MAINTENANCE_CHILD"


def _child_env() -> dict:
    env = dict(os.environ)
    env[CHILD_ENV_MARKER] = "1"
    return env


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


# The working strategies — the ones that have earned their place in the ledger —
# each logged once per day. (key, run.py-flag). Order is the run order.
#   ds  long calls (cohort feeder, carries the DTE floor)
#   sps credit spreads (Bull Put / Bear Call)
#   ss  short puts (short-premium single leg)
#   ics iron condors
WORKING_AUTOLOG_WINDOWS = [
    ("ds", "-ds"),
    ("sps", "-sps"),
    ("ss", "-ss"),
    ("ics", "-ics"),
]
# Catch-up runs only inside regular trading hours: outside this band yfinance
# returns 0/0 bid-ask and every contract fails the liquidity filter.
AUTOLOG_RTH_BAND = (1015, 1600)


def due_autolog_windows(state: dict, weekday: int, hhmm: int, today: str):
    """Every working auto-log window not yet logged today, when inside the
    weekday RTH band. Decoupled from per-clock slots (the retired cron's model)
    so a single market-hours launch logs ALL working strategies — long calls,
    spreads, short puts, iron condors — instead of just one per day."""
    if weekday > 5:
        return []
    if not (AUTOLOG_RTH_BAND[0] <= hhmm <= AUTOLOG_RTH_BAND[1]):
        return []
    return [w for w in WORKING_AUTOLOG_WINDOWS if due_autolog(state, w[0], today)]


def due_autolog(state: dict, window_key: str, today: str) -> bool:
    """Auto-log runs at most once per (window, calendar day)."""
    last = ((state or {}).get("last_autolog") or {}).get(window_key)
    return (not last) or last != today


def due_morning_briefing(state: dict, today: str, weekday: int) -> bool:
    """Once per business day: write the morning-briefing HTML/JSON pair."""
    if weekday > 5:
        return False
    return (state or {}).get("last_morning_briefing") != today


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
        return subprocess.call(cmd, stdout=logf, stderr=logf, env=_child_env())


def _autolog_cmd(win) -> list:
    """run.py command for one working window. The ds (long-call) feeder carries
    the cohort DTE floor so its picks are gate-eligible, not paper_only data."""
    cmd = [VENV_PY, "run.py", win[1], "--5", "--no-ai"]
    if win[0] == "ds":
        cmd += ["--min-dte", str(_cohort_min_dte())]
    return cmd


def _spawn_catchup_detached() -> None:
    """Fire-and-forget the multi-window catch-up so an interactive launch never
    blocks on 1-4 full scans. The detached child marks state per window as each
    finishes; start_new_session keeps it alive after the screener exits."""
    subprocess.Popen(
        [VENV_PY, "-m", "src.maintenance", "--catchup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=_child_env(), start_new_session=True,
    )


def run_catchup(db_path: str = "paper_trades.db",
                state_path: str = DEFAULT_STATE_PATH,
                now: Optional[datetime] = None,
                runner: Optional[Callable] = None) -> dict:
    """Run every due working auto-log window for today, blocking, marking state
    after each success so a crash mid-way keeps the windows already done.
    Invoked detached by interactive startup (``-m src.maintenance --catchup``)
    and inline by the headless path. Idempotent per window/day; safe to run
    twice (the auto-log itself dedups at the DB layer)."""
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    runner = runner or _default_runner
    ran = []
    for win in due_autolog_windows(load_state(state_path), now.isoweekday(),
                                   now.hour * 100 + now.minute, today):
        try:
            rc = runner(_autolog_cmd(win))
        except Exception:
            continue
        if rc == 0:
            # Re-read before write: the foreground run_startup_maintenance writes
            # this same file concurrently (checkpoint / track-record / chain-
            # archive marks). Merge our autolog mark into the latest on-disk
            # state so a stale in-memory copy can't clobber those marks.
            cur = load_state(state_path)
            cur.setdefault("last_autolog", {})[win[0]] = today
            save_state(state_path, cur)
            ran.append(win[0])
    # Crypto swing paper track rides the same heartbeat, but fully isolated:
    # it runs only AFTER the options windows are marked, and any failure (crypto
    # exchange down, import error) is swallowed so it can never touch the options
    # cohort. Crypto is a satellite — it must never jeopardise the options gate.
    try:
        if _run_swing_paper() is not None:
            ran.append("swing-paper")
    except Exception:
        pass
    return {"ran": ran}


def _run_swing_paper() -> Optional[dict]:
    """Best-effort: log/resolve the daily swing-breakout paper track. Returns the
    summary dict, or None if nothing happened / it could not run. Never raises to
    the caller (the caller also guards) — crypto is isolated from options."""
    from src.leverage.paper import PaperLedger
    from src.leverage.swing_paper import run_swing_paper, _SYMBOLS
    from src.leverage import data as D
    summ = run_swing_paper(list(_SYMBOLS.keys()), 1500.0, PaperLedger(),
                           lambda k: D.load_daily(_SYMBOLS[k], days=1000))
    return summ if (summ["opened"] or summ["closed"]) else None


def _run_checkpoint(db_path: str, phase1_start: str) -> None:
    result = phase1_checkpoint.compute_checkpoint(db_path, phase1_start)
    phase1_checkpoint.write_checkpoint(result, output_dir="reports")


def _run_track_record(db_path: str) -> None:
    from scripts.publish_track_record import publish
    publish(db_path=db_path, reports_dir="reports")


def _default_enforce_exits(db_path: str, config_path: str = "config.json") -> None:
    """Mark-to-market open paper trades and close any that hit a take-profit,
    stop-loss, time-exit, or expiry threshold. Same path the interactive
    screener runs at startup — surfaced here so the headless scheduler closes
    trades on its own, independent of whether an auto-log window is open."""
    from src.paper_manager import PaperManager
    PaperManager(db_path=db_path, config_path=config_path).update_positions()


def _cohort_min_dte(config_path: str = "config.json", default: int = 30) -> int:
    """The gate cohort's DTE floor from config (auto_log.cohort_min_dte)."""
    try:
        with open(config_path) as f:
            return int((json.load(f).get("auto_log") or {}).get("cohort_min_dte")
                       or default)
    except (OSError, ValueError, TypeError):
        return default


_MORNING_CMD = [VENV_PY, "-m", "src.morning"]


def _run_chain_archive() -> int:
    """Snapshot today's option chains (free CBOE) per config → data_archive."""
    import json as _json

    from src import chain_archive
    try:
        with open("config.json") as f:
            cfg = (_json.load(f).get("data_archive") or {})
    except (OSError, ValueError):
        cfg = {}
    if not cfg.get("enabled", False):
        return 0
    symbols = cfg.get("symbols") or []
    if not symbols:
        return 0
    return chain_archive.archive_symbols(
        symbols,
        max_dte=int(cfg.get("max_dte", 120)),
        moneyness_band=float(cfg.get("moneyness_band", 0.15)),
        min_open_interest=float(cfg.get("min_open_interest", 1)))


def run_startup_maintenance(db_path: str = "paper_trades.db",
                            phase1_start: Optional[str] = None,
                            state_path: str = DEFAULT_STATE_PATH,
                            now: Optional[datetime] = None,
                            runner: Optional[Callable] = None,
                            background: bool = False,
                            spawn_fn: Optional[Callable] = None,
                            checkpoint_fn: Optional[Callable] = None,
                            track_record_fn: Optional[Callable] = None,
                            chain_archive_fn: Optional[Callable] = None,
                            morning_fn: Optional[Callable] = None) -> dict:
    """Run due maintenance jobs, crash-isolated. Returns {'cohort': line, 'ran': [...]}.
    Never raises.

    Note: exit-rule enforcement is NOT run here. The interactive screener startup
    already enforces exits synchronously via ``PaperManager.update_positions()``;
    duplicating it would mean a second ~60s market scan on every boot.
    """
    # Recursion guard: inside an auto-log child, do nothing — the parent owns
    # maintenance. Without this, in-window startups spawn scans forever.
    if os.environ.get(CHILD_ENV_MARKER):
        return {"cohort": "", "ran": []}

    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    runner = runner or _default_runner
    state = load_state(state_path)
    ran = []

    # 1. Auto-log every working strategy not yet logged today (weekdays, RTH).
    #    One launch logs long calls, spreads, short puts, and iron condors —
    #    the ds feeder carries the cohort DTE floor so its Long Calls are
    #    gate-eligible (without it, 7-29 DTE picks land paper_only=1 and the
    #    forward cohort never fills — observed 2026-06-11 at n=3 after 2 weeks).
    #    Interactive startup (background=True) spawns the catch-up detached so it
    #    never blocks on 1-4 full scans; headless/inline runs them via runner.
    try:
        windows = due_autolog_windows(state, now.isoweekday(),
                                      now.hour * 100 + now.minute, today)
        if windows and background:
            try:
                (spawn_fn or _spawn_catchup_detached)()
                ran.append(f"auto-log:queued({len(windows)})")
            except Exception:
                pass
        else:
            for win in windows:
                try:
                    rc = runner(_autolog_cmd(win))
                except Exception:
                    continue
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

    # 4. Daily chain archive (weekday afternoons, once/day) — free CBOE
    #    snapshots that compound into a real backtest dataset.
    try:
        from src import chain_archive as _ca
        if _ca.due_chain_archive(state, today, now.isoweekday(),
                                 now.hour * 100 + now.minute):
            fn = chain_archive_fn or _run_chain_archive
            n = fn()
            state["last_chain_archive"] = today
            if n:
                ran.append(f"chain-archive:{n}rows")
    except Exception:
        pass

    # 5. Morning briefing (business days, once/day) — HTML/JSON pair under
    #    reports/briefings/ so a fresh page is waiting every morning. Headless
    #    heartbeat only: interactive startup (background=True) must never block
    #    on ~30s of fetches — the INTEL menu covers on-demand builds there.
    #    Goes through `runner` so stubbed-runner tests never spawn a real build.
    try:
        if not background and due_morning_briefing(state, today, now.isoweekday()):
            rc = morning_fn() if morning_fn else runner(_MORNING_CMD)
            if rc == 0:
                state["last_morning_briefing"] = today
                ran.append("morning-briefing")
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
                 enforce_exits_fn: Optional[Callable] = None,
                 checkpoint_fn: Optional[Callable] = None,
                 track_record_fn: Optional[Callable] = None,
                 chain_archive_fn: Optional[Callable] = None,
                 morning_fn: Optional[Callable] = None) -> dict:
    """Run startup maintenance without the interactive screener.

    The LaunchAgent entry: reads phase1_start from config.json itself,
    enforces exit rules directly, then delegates to ``run_startup_maintenance``
    (which is already crash-isolated per job), and never raises — a scheduler
    must not crashloop on a bad day.

    Exit-enforcement runs HERE, directly, on every invocation. It used to be
    only transitive (the auto-log subprocess runs the screener, whose startup
    enforces exits), which meant trades hitting a stop/take-profit/time-exit on
    a day with no open auto-log window never closed until the user next opened
    the screener by hand. ``update_positions`` is idempotent, so the redundant
    pass on auto-log days is harmless.
    """
    try:
        with open(config_path) as f:
            phase1_start = (json.load(f).get("auto_log") or {}).get("phase1_start_date")
    except (OSError, ValueError):
        phase1_start = None

    # Enforce exits first so a later failure in auto-log/checkpoint can't keep
    # a stopped-out position open. Crash-isolated: a yfinance hiccup must not
    # abort the rest of maintenance.
    try:
        (enforce_exits_fn or _default_enforce_exits)(db_path=db_path, config_path=config_path)
    except Exception:
        pass

    try:
        return run_startup_maintenance(
            db_path=db_path, phase1_start=phase1_start, state_path=state_path,
            now=now, runner=runner, checkpoint_fn=checkpoint_fn,
            track_record_fn=track_record_fn, chain_archive_fn=chain_archive_fn,
            morning_fn=morning_fn)
    except Exception:
        return {"cohort": "", "ran": []}


def main() -> None:
    import sys
    if "--health" in sys.argv[1:]:
        # On-demand staleness report — no scan, no network.
        from src.maintenance_health import compute_health, health_lines, health_banner
        rep = compute_health(load_state(DEFAULT_STATE_PATH), datetime.now())
        print("\n".join(health_lines(rep)))
        banner = health_banner(rep)
        if banner:
            print("\n" + banner)
        return
    if "--catchup" in sys.argv[1:]:
        # Detached multi-window auto-log fired by an interactive launch.
        summary = run_catchup()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ran = ", ".join(summary.get("ran") or []) or "nothing due"
        print(f"[{stamp}] catch-up auto-log: {ran}")
        return
    summary = run_headless()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ran = ", ".join(summary.get("ran") or []) or "nothing due"
    print(f"[{stamp}] maintenance: {ran}")
    if summary.get("cohort"):
        print(f"  {summary['cohort']}")


if __name__ == "__main__":
    main()
