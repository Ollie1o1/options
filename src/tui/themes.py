"""Unified theme mapping from src/formatting.py palettes to Textual Theme objects.

Color Discipline Invariant:
  success (#5ec98d) and error (#e06c75) are identical muted WCAG-readable tones
  across all 4 themes and reserved strictly for directional sign, P&L, EV, and PASS/REFUSE.
"""
from __future__ import annotations

from typing import Dict, List

from textual.theme import Theme

THEMES: Dict[str, Theme] = {
    "quant_desk": Theme(
        name="quant_desk",
        primary="#82aad2",
        secondary="#5c7fa3",
        accent="#82aad2",
        foreground="#f0f0f0",
        background="#0f1318",
        surface="#161c24",
        panel="#1c242e",
        success="#5ec98d",
        warning="#d6a452",
        error="#e06c75",
        dark=True,
    ),
    "cyberpunk_neon": Theme(
        name="cyberpunk_neon",
        primary="#50e6ff",
        secondary="#9b5de5",
        accent="#ff3cdc",
        foreground="#fff0fa",
        background="#120b1a",
        surface="#1b1128",
        panel="#241736",
        success="#5ec98d",
        warning="#d6a452",
        error="#e06c75",
        dark=True,
    ),
    "matrix_terminal": Theme(
        name="matrix_terminal",
        primary="#50ff78",
        secondary="#28a745",
        accent="#3cdc5a",
        foreground="#d2ffdc",
        background="#08110a",
        surface="#0e1d12",
        panel="#142919",
        success="#5ec98d",
        warning="#d6a452",
        error="#e06c75",
        dark=True,
    ),
    "amber_crt": Theme(
        name="amber_crt",
        primary="#ffb03b",
        secondary="#b8731a",
        accent="#e69628",
        foreground="#ffdcaa",
        background="#141008",
        surface="#1e180c",
        panel="#292010",
        success="#5ec98d",
        warning="#d6a452",
        error="#e06c75",
        dark=True,
    ),
}

THEME_ORDER: List[str] = ["quant_desk", "cyberpunk_neon", "matrix_terminal", "amber_crt"]


def saved_theme_name() -> str:
    try:
        from src import settings as _settings

        t = _settings.get_theme()
        if t in THEMES:
            return t
    except Exception:
        pass
    return "quant_desk"


def persist_theme_name(name: str) -> None:
    if name not in THEMES:
        return
    try:
        from src import settings as _settings

        _settings.set_theme(name)
    except Exception:
        pass
