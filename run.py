#!/usr/bin/env python3
"""Launcher that auto-activates the venv before running the screener.

Usage (Mac/Linux):  python3 run.py [OPTIONS]
Usage (Windows):    python  run.py [OPTIONS]

Shortcut flags (expanded before forwarding):
  --default-scoring   Discovery scan + baseline weights + auto-log top 5 + unattended.
                      Equivalent to:
                        --mode discover --weights baseline --auto-log --log-top 5 --auto
                      Aliases: --default_scoring, -ds

  --spreads-scoring   Credit-spread scan + baseline + auto-log top 5 + unattended.
                      Aliases: --spreads_scoring, -sps
  --iron-scoring      Iron-condor scan + baseline + auto-log top 5 + unattended.
                      Aliases: --iron_scoring, -ics
  --sell-scoring      Short-premium (single-leg) scan + baseline + auto-log top 5 + unattended.
                      Aliases: --sell_scoring, -ss

  --5, --10, --N      Shorthand for --log-top N. Can be combined with any *-scoring
                      shortcut (the later value wins, so `--spreads-scoring --10` logs top 10).

  --logging-help, -lh  Print calibration guidance (what/when to log) and exit.
"""
import os
import re
import sys
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))

# ── Logging-help doc (shown by --logging-help / -lh) ──────────────────────
_LOGGING_HELP = """\
Calibration logging guide                      (see LOGGING_PLAN.md for full plan)
=========================

Goal: build a diverse paper-trade ledger so the weight optimizer and IC
analysis have signal across strategies, tickers, and score ranges — not
just "top-1 short premium on ORCL every day."

Auto-log defaults active (config.json)
--------------------------------------
  auto_log_skip_long_puts: true   — Long Puts (PF 0.69 in the ledger)
                                    are silently filtered from --auto-log.
                                    Auto-log line shows "filtered N Long Put(s)".
                                    Flip to false in config.json to re-enable.

Shortcut commands
-----------------
  python3 run.py --default-scoring   (alias -ds)
      Discovery scan, baseline weights, auto-log top 5. The workhorse.
      With auto_log_skip_long_puts on, this logs Long Calls only.

  python3 run.py --spreads-scoring   (alias -sps)
      Credit-spread scan. Bull Put + Bear Call. Currently the gating
      sample for spread calibration (need 30+ closed; currently 16).

  python3 run.py --iron-scoring      (alias -ics)
      Iron-condor scan. ICs are 49 DTE on opening so first closes don't
      arrive until ~mid-May. Use small batches (--5) to avoid stacking.

  python3 run.py --sell-scoring      (alias -ss)
      Short-premium single-leg scan. NAKED risk — only run when stress
      test is healthy (currently it's not; see #4 below).

  Any of the above accepts --5 / --10 / --N to override top-N.
      e.g.  python3 run.py --default-scoring --10

When to use what
----------------
1. ROTATE MODES DAILY, scan during market hours (10:00-15:30 ET).
   Outside RTH, yfinance returns 0/0 bid-ask and every contract fails
   the liquidity filter. Recommended rotation:
     Mon  -ds --10    (discover, long calls)
     Tue  -sps --10   (spreads — biggest calibration gap)
     Wed  -ics --5    (small IC batch — concentration concern)
     Thu  -ds --10    (more long calls, rotate ticker base)
     Fri  rest, OR -sps --5 if no naked exposure left

2. ONE POSITION PER TICKER PER SCAN — auto-enforced. Per-symbol dedup
   keeps the highest-scored leg only. ORCL+MU concentrated 85% of a
   $15k loss historically; this guard prevents the repeat.

3. LOG ACROSS THE SCORE DISTRIBUTION every 1-2 weeks: -ds --20 and
   manually pick rank 5/10/15/20. Without score variance in the x-axis,
   IC is mathematically flat regardless of strategy quality.

4. CHECK STRESS BEFORE ADDING NAKED RISK.
   Run `python3 -m src.check_pnl` first. If the -20%/-10% IV scenario
   loss exceeds book size, skip --sell-scoring and use defined-risk
   modes (-sps, -ics) only. Concentrated SPY/QQQ/IWM ICs amplify this
   stress — break correlations before adding more.

5. DON'T RE-RUN THE SAME SCAN TWICE IN A DAY. The ledger ends up with
   near-duplicate rows and the IC gets biased by a single day's regime.

6. DON'T --apply WEIGHTS until you cross ALL of:
     - 200+ closed trades
     - 30+ closed PER strategy you'll actually use
     - shrinkage ≥ 0.80 in --calibrate output
     - zero anomaly warnings in logs/calibration_*.warnings
   Preview anytime with `python3 -m src.backtester --calibrate` (read-only).

Daily/weekly automation (install via `crontab -e`)
--------------------------------------------------
  scripts/enforce_exits.sh        — closes anything past TP/stop/time-exit
  scripts/calibrate_snapshot.sh   — appends per-component IC drift to
                                    logs/calibration_history.tsv

Quick reference
---------------
  Healthy book, calibrating?     rotate modes, top 5-10 default
  Stress >> book?                 -sps or -ics only; never -ss
  IC looks flat/inverted?         sample deeper (--20), check warnings
  Where am I in the plan?         see LOGGING_PLAN.md
  Want the full screener help?    python3 run.py --help
"""

# ── Shortcut-flag expansion ────────────────────────────────────────────────
# Each shortcut expands to a full arg list forwarded to the screener.
_SCORING_BASE = ["--weights", "baseline", "--auto-log", "--log-top", "5", "--auto"]
_SHORTCUT_EXPANSIONS: dict = {
    # discovery (existing)
    "--default-scoring":  ["--mode", "discover"] + _SCORING_BASE,
    "--default_scoring":  ["--mode", "discover"] + _SCORING_BASE,
    "-ds":                ["--mode", "discover"] + _SCORING_BASE,
    # credit spreads
    "--spreads-scoring":  ["--mode", "spreads"] + _SCORING_BASE,
    "--spreads_scoring":  ["--mode", "spreads"] + _SCORING_BASE,
    "-sps":               ["--mode", "spreads"] + _SCORING_BASE,
    # iron condors
    "--iron-scoring":     ["--mode", "iron"] + _SCORING_BASE,
    "--iron_scoring":     ["--mode", "iron"] + _SCORING_BASE,
    "-ics":               ["--mode", "iron"] + _SCORING_BASE,
    # short premium (single-leg)
    "--sell-scoring":     ["--mode", "sell"] + _SCORING_BASE,
    "--sell_scoring":     ["--mode", "sell"] + _SCORING_BASE,
    "-ss":                ["--mode", "sell"] + _SCORING_BASE,
}
_LOGGING_HELP_ALIASES = {"--logging-help", "--logging_help", "-lh"}
# --5 / --10 / --25 / etc. → --log-top N
_TOPN_PATTERN = re.compile(r"^--(\d+)$")

# Intercept --logging-help before venv checks / forwarding.
for _arg in sys.argv[1:]:
    if _arg in _LOGGING_HELP_ALIASES:
        sys.stdout.write(_LOGGING_HELP)
        sys.exit(0)

_argv = []
for _arg in sys.argv[1:]:
    if _arg in _SHORTCUT_EXPANSIONS:
        _argv.extend(_SHORTCUT_EXPANSIONS[_arg])
        continue
    _m = _TOPN_PATTERN.match(_arg)
    if _m:
        _argv.extend(["--log-top", _m.group(1)])
        continue
    _argv.append(_arg)

# Locate venv Python (Windows first, then Unix)
_venv_python = os.path.join(_project_root, "venv", "Scripts", "python.exe")
if not os.path.isfile(_venv_python):
    _venv_python = os.path.join(_project_root, "venv", "bin", "python")

# ── macOS venv-poisoning guard ────────────────────────────────────────────
# If the venv was created by a sandboxed shell (e.g. a coding-assistant CLI
# running under App Sandbox), every site-packages file inherits the kernel-set
# `com.apple.provenance` xattr AND the launching process lacks the entitlement
# macOS needs to approve those reads cheaply. The result: a ~1.5s Gatekeeper
# check per file on import, so `import pandas` hangs for minutes with zero
# output. The xattr cannot be removed (kernel-protected) — the only fix is
# to recreate the venv from a shell with proper entitlements (Terminal.app /
# iTerm / Ghostty). Note: presence of the xattr alone is NOT a failure —
# what matters is whether imports actually run fast. So we only abort if a
# real timed probe confirms the hang; a healthy venv sails through.
_STAMP_PATH = os.path.join(_project_root, "venv", ".provenance_probe_ok")
# Strictness knobs:
#   _PROBE_BUDGET_S — if the bundled imports take longer than this, treat the venv as
#                     poisoned. Healthy native venv imports pandas+numpy+yfinance
#                     in ~1-3s; a provenance-poisoned venv takes 15-60s+. Anything
#                     above 6s is already a poor user experience.
#   _PROBE_TIMEOUT_S — hard ceiling. Beyond this we abort the subprocess.
_PROBE_BUDGET_S = 6.0
_PROBE_TIMEOUT_S = 15.0

_FALLBACK_VENV = os.path.expanduser("~/.venvs/options/bin/python")

def _probe_imports(python_path: str) -> tuple[str, float]:
    """Probe how the given interpreter handles the screener's heaviest imports.

    Returns ("ok"|"error"|"slow"|"timeout", elapsed_seconds).
    """
    import time as _time
    t0 = _time.monotonic()
    try:
        completed = subprocess.run(
            [python_path, "-c", "import pandas, numpy, yfinance"],
            timeout=_PROBE_TIMEOUT_S,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout", _time.monotonic() - t0
    except OSError:
        return "error", _time.monotonic() - t0
    elapsed = _time.monotonic() - t0
    if completed.returncode != 0:
        return "error", elapsed
    if elapsed > _PROBE_BUDGET_S:
        return "slow", elapsed
    return "ok", elapsed

def _venv_status() -> str:
    """Return "ok"|"error"|"slow"|"timeout" for the project venv (cached)."""
    if sys.platform != "darwin":
        # Non-Darwin: skip the cache + slow check, just probe import success.
        status, _ = _probe_imports(_venv_python)
        return status
    if not os.path.isdir(os.path.join(_project_root, "venv", "lib")):
        return "error"
    try:
        stamp_mtime = os.path.getmtime(_STAMP_PATH)
        venv_mtime = os.path.getmtime(os.path.join(_project_root, "venv"))
        runpy_mtime = os.path.getmtime(__file__)
        if stamp_mtime >= venv_mtime and stamp_mtime >= runpy_mtime:
            return "ok"
    except OSError:
        pass
    status, _ = _probe_imports(_venv_python)
    if status == "ok":
        try:
            open(_STAMP_PATH, "w").close()
        except OSError:
            pass
    return status

def _print_rebuild_instructions():
    print()
    print("Fix (run from Terminal.app / iTerm / Ghostty — NOT a coding-CLI):")
    print(f"  cd {_project_root}")
    print("  rm -rf venv")
    print("  python3 -m venv venv")
    print("  venv/bin/pip install -r requirements-lock.txt")
    print()
    print("Verify by re-running: python3 run.py")

_project_venv_status = _venv_status() if os.path.isfile(_venv_python) else "missing"

# If the project venv is unhealthy in any way, try the auto-logger fallback
# (~/.venvs/options) before giving up. That venv is what cron uses for
# auto_log_*.sh, so it's typically healthy when the project venv isn't.
if _project_venv_status != "ok":
    fb_ok = False
    if os.path.isfile(_FALLBACK_VENV):
        fb_status, _ = _probe_imports(_FALLBACK_VENV)
        fb_ok = fb_status == "ok"
    if fb_ok:
        reason = {
            "missing": "missing",
            "error":   "broken (imports fail)",
            "slow":    "slow imports (provenance poisoning)",
            "timeout": "imports timed out",
        }.get(_project_venv_status, _project_venv_status)
        print(f"NOTICE: project venv is {reason} — falling back to {_FALLBACK_VENV}")
        print("        Rebuild venv/ from your own Terminal when convenient:")
        print(f"          cd {_project_root} && rm -rf venv && python3 -m venv venv")
        print("          venv/bin/pip install -r requirements-lock.txt")
        _venv_python = _FALLBACK_VENV
    else:
        if _project_venv_status == "missing":
            print("ERROR: Virtual environment not found and no fallback at ~/.venvs/options.")
        elif _project_venv_status in ("slow", "timeout"):
            print("ERROR: venv imports are slow or hanging — macOS provenance poisoning.")
            print()
            print("Cause: this venv was created by a sandboxed shell, so every")
            print("site-packages file carries the kernel-set com.apple.provenance xattr.")
            print("macOS Gatekeeper does a per-file security check on import; with")
            print("~10k files in the venv that adds up to 5-60s+ of pure overhead at")
            print("every startup. The xattr cannot be removed (kernel-protected) — the")
            print("only durable fix is to recreate the venv from a non-sandboxed shell.")
        else:
            print("ERROR: project venv imports fail (broken install).")
        _print_rebuild_instructions()
        sys.exit(1)

# If any equity-side flag was provided, dispatch directly to options_screener
# (preserves cron + every power-user shortcut). With no flags, route through
# src.launcher which shows the [1] STOCKS [2] CRYPTO menu.
_target_module = "src.launcher" if not _argv else "src.options_screener"

# If not already inside the venv, re-launch with the venv interpreter
if sys.prefix == sys.base_prefix:
    sys.exit(subprocess.call(
        [_venv_python, "-m", _target_module] + _argv,
        cwd=_project_root,
    ))

sys.argv = [sys.argv[0]] + _argv
if _argv:
    from src.options_screener import main
else:
    from src.launcher import main
main()
