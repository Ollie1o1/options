"""Read-only first-run environment self-check.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.doctor

Prints a pass/warn/fail table covering the things that silently break a fresh
checkout: wrong Python version, drifted dependencies, no network path to the
two data sources, unwritable state paths, a broken config.json, a dead
scheduler, and which optional integrations are configured. Every failure gets
a one-line fix.

Design rules, mirroring ``src.execution.preflight``:
  - Every check is a pure function over injected inputs, so the logic is
    trivially testable; only the real-input gathering in ``run_doctor`` and
    ``main`` touches the filesystem/network/environment.
  - The doctor NEVER mutates anything. It does not install packages, does not
    create files or directories, and does not construct a PaperManager
    (which would run schema migrations on `paper_trades.db`) — DB schema is
    read via a read-only sqlite URI instead.
  - Network checks hit one cheap endpoint each with a short timeout, and
    every exception is caught and classified rather than allowed to raise —
    a dead network degrades the report, it never hangs or crashes the doctor.
    Tests always inject a fake `fetch`; nothing here is exercised over a real
    socket in the suite.
  - Rendering goes through ``src.ui`` / ``fmt.style`` (the quant-desk theme),
    never raw ANSI.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

RUN_COMMAND = "python run.py"

MIN_PYTHON = (3, 11)  # CI matrix is 3.11/3.12; floor, not ceiling

# Packages this repo cannot run its core paths without. Not every line of
# requirements-lock.txt — most of that is transitive (streamlit's own deps,
# etc.) and drifting there is not actionable for an operator.
KEY_DEPENDENCIES = [
    "numpy", "pandas", "scipy", "requests", "yfinance", "python-dotenv",
    "beautifulsoup4", "lxml", "tenacity", "jsonschema", "curl_cffi", "tqdm",
]

OPTIONAL_ENV_KEYS = ["OPENROUTER_API_KEY", "POLYGON_API_KEY", "SEC_EDGAR_CONTACT"]

_ENV_KEY_PURPOSE = {
    "OPENROUTER_API_KEY": "enables the AI ranking layer",
    "POLYGON_API_KEY": "enables the Polygon data provider fallback",
    "SEC_EDGAR_CONTACT": "enables the EDGAR insider-buys signal (SEC requires a contact UA)",
}

YAHOO_PROBE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=1d&interval=1d"
CBOE_PROBE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
_PROBE_HEADERS = {"User-Agent": "options-screener-doctor/1.0"}
_PROBE_TIMEOUT = 3.0


@dataclass
class CheckResult:
    name: str
    status: str          # "PASS" | "WARN" | "FAIL"
    detail: str
    fix: str = field(default="")


_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


# ── Python / dependencies ────────────────────────────────────────────────────

def check_python_version(version_info: Optional[Tuple[int, int, int]] = None) -> CheckResult:
    vi = version_info if version_info is not None else tuple(sys.version_info[:3])
    ver_str = ".".join(str(p) for p in vi)
    min_str = ".".join(str(p) for p in MIN_PYTHON)
    if tuple(vi[:2]) >= MIN_PYTHON:
        return CheckResult("python version", "PASS", f"{ver_str} (>= {min_str})")
    return CheckResult("python version", "FAIL", f"{ver_str} (need >= {min_str})",
                        f"install Python {min_str}+ — CI runs 3.11/3.12")


def parse_lock_file(text: str) -> Dict[str, str]:
    """'numpy==2.4.3' lines -> {'numpy': '2.4.3'}. Comments/blank lines skipped."""
    pins: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, ver = line.partition("==")
        pins[name.strip().lower()] = ver.strip()
    return pins


def _installed_version(pkg: str) -> Optional[str]:
    import importlib.metadata as im
    try:
        return im.version(pkg)
    except im.PackageNotFoundError:
        return None


def check_dependencies(lock_text: Optional[str],
                       get_version: Optional[Callable[[str], Optional[str]]] = None,
                       packages: Optional[List[str]] = None) -> CheckResult:
    """Installed vs pinned versions for the key packages. Never installs
    anything — reports drift only. `get_version` is injectable for tests."""
    if lock_text is None:
        return CheckResult("dependencies", "FAIL", "requirements-lock.txt not found",
                            "run the doctor from the repo root (requirements-lock.txt is the pin file)")
    pins = parse_lock_file(lock_text)
    get_version = get_version or _installed_version
    packages = packages if packages is not None else KEY_DEPENDENCIES
    missing, drift = [], []
    for pkg in packages:
        pinned = pins.get(pkg.lower())
        if pinned is None:
            continue  # not pinned in this lock file — nothing to compare
        installed = get_version(pkg)
        if installed is None:
            missing.append(pkg)
        elif installed != pinned:
            drift.append(f"{pkg} {installed}!={pinned}")
    if missing:
        return CheckResult("dependencies", "FAIL",
                            f"not installed: {', '.join(missing)}",
                            "pip install -r requirements-lock.txt (the doctor will never do this for you)")
    if drift:
        return CheckResult("dependencies", "WARN",
                            f"version drift: {', '.join(drift)}",
                            "pip install -r requirements-lock.txt to match the pinned versions")
    return CheckResult("dependencies", "PASS",
                        f"{len(packages)} key packages match requirements-lock.txt")


# ── Network reachability ─────────────────────────────────────────────────────

def classify_network_result(exc: Optional[BaseException],
                            status_code: Optional[int]) -> Tuple[str, str, str]:
    """Pure classifier -> (status, detail, fix). Distinguishes a real outage
    (connection/timeout error) from rate-limiting/bot-blocking (a non-2xx HTTP
    response) so an operator doesn't chase a 429 like it's a dead network."""
    if exc is not None:
        cls = type(exc).__name__
        if "Timeout" in cls:
            return ("FAIL", f"timed out after {_PROBE_TIMEOUT:.0f}s — offline or host unreachable",
                    "check your network connection")
        if "ConnectionError" in cls:
            return ("FAIL", "unreachable — connection failed (offline or host down)",
                    "check your network connection")
        return ("FAIL", f"error: {exc}", "check your network connection")
    if status_code == 200:
        return ("PASS", "reachable (HTTP 200)", "")
    if status_code == 429:
        return ("WARN", "rate-limited (HTTP 429)",
                "not a config problem — back off and retry later")
    if status_code == 999:
        return ("WARN", "blocked (HTTP 999 — bot-detection, not a real outage)",
                "the app's own fetch path uses different headers than this probe")
    if status_code is not None and 500 <= status_code < 600:
        return ("WARN", f"server error (HTTP {status_code})", "host-side issue; retry later")
    return ("FAIL", f"unexpected response (HTTP {status_code})", "investigate manually")


def _probe(url: str, timeout: float,
          fetch: Optional[Callable[[], object]] = None) -> Tuple[Optional[BaseException], Optional[int]]:
    if fetch is None:
        import requests

        def fetch():
            return requests.get(url, timeout=timeout, headers=_PROBE_HEADERS)
    try:
        resp = fetch()
        return None, getattr(resp, "status_code", None)
    except Exception as e:  # noqa: BLE001 — any failure classifies as unreachable
        return e, None


def check_yahoo_network(fetch: Optional[Callable[[], object]] = None,
                        timeout: float = _PROBE_TIMEOUT) -> CheckResult:
    exc, status = _probe(YAHOO_PROBE_URL, timeout, fetch)
    sev, detail, fix = classify_network_result(exc, status)
    return CheckResult("network: yahoo", sev, detail, fix)


def check_cboe_network(fetch: Optional[Callable[[], object]] = None,
                       timeout: float = _PROBE_TIMEOUT) -> CheckResult:
    exc, status = _probe(CBOE_PROBE_URL, timeout, fetch)
    sev, detail, fix = classify_network_result(exc, status)
    return CheckResult("network: cboe", sev, detail, fix)


# ── Writability ───────────────────────────────────────────────────────────────

def check_dir_writable(label: str, path: str) -> CheckResult:
    """A directory is fine either way: writable already, or absent with a
    writable parent (creatable on demand by the app later — the doctor never
    creates it itself)."""
    name = f"{label} dir"
    if os.path.isdir(path):
        writable = os.access(path, os.W_OK)
        detail = f"{path} ({'writable' if writable else 'exists but NOT writable'})"
        bad_target = path
    else:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        writable = os.access(parent, os.W_OK)
        detail = f"{path} does not exist yet; parent {parent} " \
                 f"({'writable, will be creatable' if writable else 'NOT writable'})"
        bad_target = parent
    status = "PASS" if writable else "FAIL"
    fix = "" if writable else f"fix permissions on {bad_target}"
    return CheckResult(name, status, detail, fix)


def _sqlite_user_version_ro(db_path: str) -> Optional[int]:
    """PRAGMA user_version through a read-only URI — never opens for write,
    never runs paper_manager's migration path."""
    if not os.path.exists(db_path):
        return None
    import sqlite3
    try:
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def check_db_writable(db_path: str = "paper_trades.db",
                      expected_schema: Optional[int] = None) -> CheckResult:
    if os.path.exists(db_path):
        writable = os.access(db_path, os.W_OK)
        detail = f"{db_path} ({'writable' if writable else 'NOT writable'})"
        ver = _sqlite_user_version_ro(db_path)
        if ver is not None:
            detail += f", schema v{ver}"
            if expected_schema is not None and ver != expected_schema:
                detail += f" (code expects v{expected_schema} — will migrate on next real write)"
        bad_target = db_path
    else:
        parent = os.path.dirname(os.path.abspath(db_path)) or "."
        writable = os.access(parent, os.W_OK)
        detail = f"{db_path} does not exist yet; parent {parent} " \
                 f"({'writable, will be creatable' if writable else 'NOT writable'})"
        bad_target = parent
    status = "PASS" if writable else "FAIL"
    fix = "" if writable else f"fix permissions on {bad_target}"
    return CheckResult("db writable", status, detail, fix)


def _expected_schema_version() -> Optional[int]:
    try:
        from src.paper_manager import _SCHEMA_VERSION
        return _SCHEMA_VERSION
    except Exception:
        return None


# ── config.json ───────────────────────────────────────────────────────────────

def check_config(config_path: str = "config.json") -> CheckResult:
    if not os.path.exists(config_path):
        return CheckResult("config.json", "FAIL", f"{config_path} not found",
                            "restore config.json from git (`git checkout config.json`) or copy a known-good one")
    import json
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return CheckResult("config.json", "FAIL", f"invalid JSON: {e}",
                            f"fix the syntax error in {config_path}")
    from src.config_validator import validate_core_config
    warnings = validate_core_config(cfg)
    if warnings:
        return CheckResult("config.json", "WARN", "; ".join(warnings),
                            "fix the flagged key(s) above in " + config_path)
    return CheckResult("config.json", "PASS", f"{config_path} valid")


# ── Scheduler state ──────────────────────────────────────────────────────────

def check_scheduler(state_path: Optional[str] = None, now=None,
                    jobs: Optional[List] = None) -> CheckResult:
    """Scheduler health. `jobs` is injectable so tests never depend on the
    launchctl state of the machine running them; None reads it live."""
    from datetime import datetime as _dt

    from src.maintenance import DEFAULT_STATE_PATH, load_state
    from src.maintenance_health import compute_health

    path = state_path or DEFAULT_STATE_PATH
    state = load_state(path)  # pure read: {} on missing/corrupt file
    if not state:
        return CheckResult("scheduler", "WARN", "no maintenance state yet (fresh install / never run)",
                            f"run `{RUN_COMMAND}` once to seed {path} via startup maintenance")
    # A dead scheduler outranks a stale one, and is a different problem with a
    # different fix: staleness means "nothing has run lately, open the app";
    # exit 78 means macOS is refusing to launch the jobs at all, and opening
    # the app will never fix it. Reporting only staleness sends the operator
    # to the one action that cannot work.
    from src.maintenance_health import (launchd_dead_days, launchd_failure_message,
                                        read_launchd_status, seed_dead_since_date)
    if jobs is None:
        try:
            jobs = read_launchd_status()
        except Exception:  # pragma: no cover - diagnostics never fail a run
            jobs = []
    failure = launchd_failure_message(jobs) if jobs else None
    if failure:
        _now = now or _dt.now()
        # Callers pass either a date or a datetime; launchd_dead_days wants a date.
        today = _now.date() if hasattr(_now, "date") else _now
        days = launchd_dead_days(
            jobs, {"launchd_dead_since": seed_dead_since_date()}, today)
        span = f" for ~{days} days" if days else ""
        return CheckResult(
            "scheduler", "FAIL", f"scheduled jobs are not running{span}",
            "System Settings > General > Login Items & Extensions > Allow in "
            "the Background (exit 78 is macOS refusing to launch them)")

    report = compute_health(state, now or _dt.now())
    sev_map = {"OK": "PASS", "WARN": "WARN", "STALE": "WARN", "CRITICAL": "FAIL"}
    status = sev_map.get(report.worst, "WARN")
    detail = f"worst job: {report.worst} (auto-log {report.autolog_missed_days}bd stale)"
    fix = "" if status == "PASS" else "open the launcher/screener to trigger catch-up maintenance"
    return CheckResult("scheduler", status, detail, fix)


# ── Optional integrations ────────────────────────────────────────────────────

def read_env_keys(dotenv_path: str = ".env",
                  keys: Optional[List[str]] = None,
                  environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Read the given keys from a .env file plus the process environment,
    without mutating either. Read-only; tolerates a missing/unreadable file."""
    keys = keys if keys is not None else OPTIONAL_ENV_KEYS
    environ = environ if environ is not None else os.environ
    values: Dict[str, str] = {}
    try:
        if os.path.exists(dotenv_path):
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    if k in keys:
                        values[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    for k in keys:
        if k not in values and environ.get(k):
            values[k] = environ[k]
    return values


def check_optional_env(name: str, values: Dict[str, str]) -> CheckResult:
    present = bool((values.get(name) or "").strip())
    purpose = _ENV_KEY_PURPOSE.get(name, "")
    detail = f"{name}: {'set' if present else 'not set'} (optional — {purpose})"
    return CheckResult(f"optional: {name}", "PASS", detail)


# ── Aggregation / gathering / rendering ──────────────────────────────────────

def run_doctor(config_path: str = "config.json",
               db_path: str = "paper_trades.db",
               reports_dir: str = "reports",
               logs_dir: str = "logs",
               lock_path: str = "requirements-lock.txt",
               env_path: str = ".env",
               state_path: Optional[str] = None,
               network_probe: bool = True) -> List[CheckResult]:
    checks: List[CheckResult] = [check_python_version()]

    lock_text = None
    if os.path.exists(lock_path):
        with open(lock_path) as f:
            lock_text = f.read()
    checks.append(check_dependencies(lock_text))

    if network_probe:
        checks.append(check_yahoo_network())
        checks.append(check_cboe_network())
    else:
        # A skipped check is unknown, not healthy — so it stays a WARN, and it
        # carries a fix like every other non-PASS row: "run it without the flag"
        # is the action, and a row with no action is the thing this table exists
        # to avoid.
        skip_fix = "re-run without --no-network to actually test reachability"
        checks.append(CheckResult("network: yahoo", "WARN",
                                  "skipped (--no-network)", skip_fix))
        checks.append(CheckResult("network: cboe", "WARN",
                                  "skipped (--no-network)", skip_fix))

    checks.append(check_dir_writable("reports", reports_dir))
    checks.append(check_dir_writable("logs", logs_dir))
    checks.append(check_db_writable(db_path, expected_schema=_expected_schema_version()))
    checks.append(check_config(config_path))
    checks.append(check_scheduler(state_path))

    env_values = read_env_keys(env_path)
    for key in OPTIONAL_ENV_KEYS:
        checks.append(check_optional_env(key, env_values))

    return checks


def render(checks: List[CheckResult], width: int = 96) -> str:
    from src import formatting as fmt
    from src import ui

    style_map = {"PASS": "good", "WARN": "warn", "FAIL": "bad"}
    glyph_map = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}

    cols = [{"h": "CHECK", "w": 20}, {"h": "STATUS", "w": 9}, {"h": "DETAIL", "w": 62}]
    rows = []
    for c in checks:
        status_cell = fmt.style(f"{glyph_map[c.status]} {c.status}", style_map[c.status], bold=True)
        rows.append([c.name, status_cell, c.detail])

    lines = [ui.rule(width, "FIRST-RUN DOCTOR"), ui.table(cols, rows)]

    fails = [c for c in checks if c.status == "FAIL"]
    warns = [c for c in checks if c.status == "WARN"]
    if fails:
        lines.append("")
        lines.append(fmt.style("Fix these:", "bad", bold=True))
        for c in fails:
            if c.fix:
                lines.append(f"  - {c.name}: {c.fix}")
    if warns:
        lines.append("")
        lines.append(fmt.style("Worth a look:", "warn"))
        for c in warns:
            if c.fix:
                lines.append(f"  - {c.name}: {c.fix}")

    lines.append("")
    if fails:
        lines.append(fmt.style("Some checks failed — see fixes above before trading real money.", "bad"))
    elif warns:
        lines.append(fmt.style("All checks pass or degrade gracefully; a few things are worth a look.", "warn"))
    else:
        lines.append(fmt.style("Everything checks out.", "good"))
    lines.append(fmt.style(f"Run the screener with:  {RUN_COMMAND}", "accent"))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Read-only first-run environment self-check")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--lock", default="requirements-lock.txt")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--no-network", action="store_true",
                    help="skip the live Yahoo/CBOE reachability probes")
    args = ap.parse_args(argv)

    checks = run_doctor(config_path=args.config, db_path=args.db,
                        reports_dir=args.reports_dir, logs_dir=args.logs_dir,
                        lock_path=args.lock, env_path=args.env,
                        network_probe=not args.no_network)
    print(render(checks))
    raise SystemExit(1 if any(c.status == "FAIL" for c in checks) else 0)


if __name__ == "__main__":
    main()
