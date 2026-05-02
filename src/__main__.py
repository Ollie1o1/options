"""Entry point for `python -m src` and `python3 -m src`.

Auto-activates the project venv if running under system Python so that
all dependencies (pandas, openai, rich, etc.) are available.
"""
import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_python = os.path.join(_project_root, "venv", "bin", "python")

# With no CLI args, route through the [1]/[2] launcher menu. With args,
# preserve the historical direct-dispatch to options_screener so cron and
# power-user shortcuts behave exactly as before.
_target = "src.launcher" if len(sys.argv) <= 1 else "src.options_screener"

if os.path.isfile(_venv_python) and sys.prefix == sys.base_prefix:
    # Running under system Python — re-launch under the venv
    sys.exit(subprocess.call(
        [_venv_python, "-m", _target] + sys.argv[1:],
        cwd=_project_root,
    ))

if len(sys.argv) <= 1:
    from src.launcher import main
else:
    from src.options_screener import main
main()
