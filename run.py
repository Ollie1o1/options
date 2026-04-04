#!/usr/bin/env python3
"""Launcher that auto-activates the venv before running the screener.

Usage (Mac/Linux):  python3 run.py [OPTIONS]
Usage (Windows):    python  run.py [OPTIONS]
"""
import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))

# Locate venv Python (Windows first, then Unix)
_venv_python = os.path.join(_project_root, "venv", "Scripts", "python.exe")
if not os.path.isfile(_venv_python):
    _venv_python = os.path.join(_project_root, "venv", "bin", "python")

# If not already inside the venv, re-launch with the venv interpreter
if os.path.isfile(_venv_python) and sys.prefix == sys.base_prefix:
    sys.exit(subprocess.call(
        [_venv_python, "-m", "src.options_screener"] + sys.argv[1:],
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

from src.options_screener import main
main()
