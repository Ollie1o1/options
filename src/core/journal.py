"""Append-only markdown journal (calibration-journal pattern)."""
from __future__ import annotations
import os


def append_entry(path: str, title: str, body: str) -> None:
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("# Journal\n\n")
        f.write(f"## {title}\n\n{body}\n\n---\n\n")
