#!/usr/bin/env python3
"""Launcher that auto-activates the venv before running the screener.

Usage:
    python3 run.py              # interactive screener
    python3 run.py --no-ai      # skip AI scoring
    python3 run.py --help       # show options
"""
import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_project_root, "venv", "Scripts", "python.exe")
if not os.path.isfile(_venv_python):
    _venv_python = os.path.join(_project_root, "venv", "bin", "python")

if os.path.isfile(_venv_python) and sys.prefix == sys.base_prefix:
    sys.exit(subprocess.call(
        [_venv_python, "-m", "src.options_screener"] + sys.argv[1:],
        cwd=_project_root,
    ))

from src.options_screener import main
main()
