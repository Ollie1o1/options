#!/usr/bin/env python3
"""Launcher that auto-activates the venv before running the screener.

Usage (Mac/Linux):  python3 run.py [OPTIONS]
Usage (Windows):    python  run.py [OPTIONS]

Shortcut flags (expanded before forwarding):
  --default-scoring   Discovery scan + baseline weights + auto-log top 5 + unattended.
                      Equivalent to:
                        --mode discover --weights baseline --auto-log --log-top 5 --auto
                      Aliases: --default_scoring, -ds

  --5, --10, --N      Shorthand for --log-top N. Can be combined with --default-scoring
                      (the later value wins, so `--default-scoring --10` logs top 10).
"""
import os
import re
import sys
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))

# ── Shortcut-flag expansion ────────────────────────────────────────────────
# Accept --default-scoring / --default_scoring / -ds as shorthand for the full
# weight-optimization data-collection command.
_DEFAULT_SCORING_EXPANSION = [
    "--mode", "discover",
    "--weights", "baseline",
    "--auto-log",
    "--log-top", "5",
    "--auto",
]
_shortcut_aliases = {"--default-scoring", "--default_scoring", "-ds"}
# --5 / --10 / --25 / etc. → --log-top N
_TOPN_PATTERN = re.compile(r"^--(\d+)$")

_argv = []
for _arg in sys.argv[1:]:
    if _arg in _shortcut_aliases:
        _argv.extend(_DEFAULT_SCORING_EXPANSION)
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
