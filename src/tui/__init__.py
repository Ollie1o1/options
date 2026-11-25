"""Interactive 6-tab Textual Daily Trading Cockpit (`opt` / `opt desk`)."""
from __future__ import annotations

__all__ = ["DeskApp", "run_desk"]


def __getattr__(name: str):
    if name in ("DeskApp", "run_desk"):
        from .app import DeskApp, run_desk

        return DeskApp if name == "DeskApp" else run_desk
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
