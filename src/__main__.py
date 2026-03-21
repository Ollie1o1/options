"""Entry point for `python -m src` and `python3 -m src`.

Auto-activates the project venv if running under system Python so that
all dependencies (pandas, openai, rich, etc.) are available.
"""
import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_python = os.path.join(_project_root, "venv", "bin", "python")

if os.path.isfile(_venv_python) and sys.prefix == sys.base_prefix:
    # Running under system Python — re-launch under the venv
    sys.exit(subprocess.call(
        [_venv_python, "-m", "src.options_screener"] + sys.argv[1:],
        cwd=_project_root,
    ))

from src.options_screener import main
main()
