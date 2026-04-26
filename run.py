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
Calibration logging guide
=========================

Goal: build a diverse paper-trade ledger so the weight optimizer and IC
analysis have signal across strategies, tickers, and score ranges — not
just "top-1 short premium on ORCL every day."

Shortcut commands
-----------------
  python3 run.py --default-scoring   (alias -ds)
      Discovery scan, baseline weights, auto-log top 5. The workhorse.

  python3 run.py --spreads-scoring   (alias -sps)
      Credit-spread scan. Logs defined-risk two-leg trades.

  python3 run.py --iron-scoring      (alias -ics)
      Iron-condor scan. Logs range-bound, defined-risk trades.

  python3 run.py --sell-scoring      (alias -ss)
      Short-premium single-leg scan. Naked risk — size carefully.

  Any of the above accepts --5 / --10 / --N to override top-N.
      e.g.  python3 run.py --default-scoring --10

When to use what
----------------
1. ROTATE MODES DAILY — don't stack the same scan all week.
   Recommended weekly rotation:
     Mon  --default-scoring   (discover)
     Tue  --spreads-scoring
     Wed  --iron-scoring
     Thu  --default-scoring
     Fri  --sell-scoring      (only if stress test is healthy)
   Why: the optimizer can't learn which weights work for spreads if 90%
   of the ledger is single-leg discover picks.

2. ONE POSITION PER TICKER PER SCAN — enforced automatically.
   All *-scoring shortcuts now dedupe by symbol before taking top-N, so
   if MU shows 6 candidates the auto-log keeps only the highest-scored
   one. Concentration tanked the last ledger (ORCL + MU = 85% of a
   $15k loss) — this guard prevents the repeat.

3. LOG ACROSS THE SCORE DISTRIBUTION when calibrating.
   Auto-log top 5 is fine as a default, but every 1-2 weeks sample
   deeper: `--default-scoring --20` and manually pick rank 5, 10, 15, 20.
   Why: IC regression needs variance in the x-axis. If every logged
   trade has score 0.82-0.88 the correlation is meaningless.

4. CHECK STRESS BEFORE ADDING NAKED RISK.
   Run `python3 -m src.check_pnl` first. If `Stress -1σ` loss exceeds
   your open book, CLOSE OR ROLL existing naked shorts before adding
   `--sell-scoring` trades. Defined-risk modes (--spreads, --iron) are
   safe to keep logging.

5. DON'T RE-RUN THE SAME SCAN TWICE IN A DAY.
   The ledger ends up with near-duplicate rows and the IC gets biased
   by a single day's regime.

Quick reference
---------------
  Healthy book, calibrating?     rotate modes, top 5 default
  Stress >> book?                 --spreads-scoring or --iron-scoring only
  IC looks flat/inverted?         sample deeper (--20) and manually vary ranks
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
def _venv_import_is_hanging() -> bool:
    if sys.platform != "darwin":
        return False
    probe_py = os.path.join(
        _project_root, "venv", "lib"
    )
    if not os.path.isdir(probe_py):
        return False
    # Cached pass: if we've already verified this venv imports cleanly, trust
    # the stamp (invalidated when the stamp is older than the venv).
    try:
        stamp_mtime = os.path.getmtime(_STAMP_PATH)
        venv_mtime = os.path.getmtime(os.path.join(_project_root, "venv"))
        if stamp_mtime >= venv_mtime:
            return False
    except OSError:
        pass
    # Time a real pandas import with a 15s budget. Normal is ~1-2s; a poisoned
    # venv takes many minutes (we've measured 37s just for pandas.io.api).
    try:
        completed = subprocess.run(
            [_venv_python, "-c", "import pandas"],
            timeout=15,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return True
    except OSError:
        return False
    # Only stamp on clean success; a non-zero exit is a real import error
    # (unrelated to the hang) that the screener's own traceback will surface.
    if completed.returncode == 0:
        try:
            open(_STAMP_PATH, "w").close()
        except OSError:
            pass
    return False

if os.path.isfile(_venv_python) and _venv_import_is_hanging():
    print("ERROR: venv import is hanging — likely macOS provenance poisoning.")
    print()
    print("Cause: this venv was created by a sandboxed shell, so every")
    print("site-packages file has the kernel-set com.apple.provenance xattr,")
    print("which triggers a slow Gatekeeper check on each import (~1.5s/file).")
    print()
    print("Fix: recreate the venv from Terminal.app / iTerm / Ghostty")
    print("(NOT a coding-assistant CLI running under App Sandbox):")
    print(f"  cd {_project_root}")
    print("  rm -rf venv")
    print("  python3 -m venv venv")
    print("  venv/bin/pip install -r requirements-lock.txt")
    sys.exit(1)

# If not already inside the venv, re-launch with the venv interpreter
if os.path.isfile(_venv_python) and sys.prefix == sys.base_prefix:
    sys.exit(subprocess.call(
        [_venv_python, "-m", "src.options_screener"] + _argv,
        cwd=_project_root,
    ))

# If the venv doesn't exist yet, give a clear error instead of a cryptic import fail
if not os.path.isfile(_venv_python):
    print("ERROR: Virtual environment not found.")
    print("Set it up with:")
    if sys.platform == "win32":
        print("  python -m venv venv && venv\\Scripts\\pip install -r requirements-lock.txt")
    else:
        print("  python3 -m venv venv && venv/bin/pip install -r requirements-lock.txt")
    sys.exit(1)

sys.argv = [sys.argv[0]] + _argv
from src.options_screener import main
main()
