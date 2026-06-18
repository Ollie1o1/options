"""CLI: python -m src.macro_pulse"""
from __future__ import annotations

from src.macro_pulse.orchestrator import run


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
