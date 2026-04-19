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
        print("  python -m venv venv && venv\\Scripts\\pip install -r requirements.txt")
    else:
        print("  python3 -m venv venv && venv/bin/pip install -r requirements.txt")
    sys.exit(1)

sys.argv = [sys.argv[0]] + _argv
from src.options_screener import main
main()
