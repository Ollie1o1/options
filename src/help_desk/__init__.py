"""The built-in manual.

Display-only, and structurally so: this package imports the theme layer and the
standard library, and nothing from any scan, ledger, gate, or execution path.
tests/help_desk/test_isolation.py enforces that by parsing the imports.

Every number it prints is a literal that names the document which measured it.
A manual that can fail to render because a fetch was rate-limited is not a
manual.
"""
from .menu import run_menu

__all__ = ["run_menu"]
