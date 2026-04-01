"""
Launch all three processes: FastAPI server, Discord bot, Telegram bot.

Usage:
    python start_all.py

Press Ctrl+C to shut everything down cleanly.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Detect virtual-environment Python (Windows venv layout)
_venv_win = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
_venv_unix = PROJECT_ROOT / "venv" / "bin" / "python"
if _venv_win.exists():
    PYTHON = str(_venv_win)
elif _venv_unix.exists():
    PYTHON = str(_venv_unix)
else:
    PYTHON = sys.executable

API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000")

processes: list = []


def _start(name: str, cmd: list) -> subprocess.Popen:
    print(f"[start_all] Starting {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    processes.append((name, proc))
    return proc


def _shutdown(signum=None, frame=None):
    print("\n[start_all] Shutting down all processes...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"  Terminating {name} (pid={proc.pid})")
            proc.terminate()
    # Give them a moment to exit cleanly
    time.sleep(2)
    for name, proc in processes:
        if proc.poll() is None:
            print(f"  Force-killing {name} (pid={proc.pid})")
            proc.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def main():
    # 1 — FastAPI server
    _start("api", [
        PYTHON, "-m", "uvicorn",
        "src.api:app",
        "--host", API_HOST,
        "--port", API_PORT,
    ])

    # Wait for the socket to be ready before starting bots
    print(f"[start_all] Waiting 2s for API to bind on {API_HOST}:{API_PORT}...")
    time.sleep(2)

    # 2 — Discord bot
    _start("discord", [PYTHON, "-m", "src.bots.discord_bot"])

    # 3 — Telegram bot
    _start("telegram", [PYTHON, "-m", "src.bots.telegram_bot"])

    print("[start_all] All processes running. Press Ctrl+C to stop.")

    # Monitor processes; restart if one dies unexpectedly
    while True:
        time.sleep(5)
        for name, proc in list(processes):
            rc = proc.poll()
            if rc is not None:
                print(f"[start_all] WARNING: {name} exited with code {rc}")


if __name__ == "__main__":
    main()
