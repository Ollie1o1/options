"""Modern composable `opt` subcommand CLI (`click` + `rich`)."""
from __future__ import annotations

__all__ = ["cli"]


def __getattr__(name: str):
    if name == "cli":
        from .main import cli

        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
