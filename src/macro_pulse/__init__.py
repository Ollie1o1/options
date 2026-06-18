"""Macro Pulse — on-demand AI market-weather panel.

Reuses src/worldnews (fetch + theme-scored aggregate) and adds: historical
percentiles for each theme/pulse (is this news flow unusually hot?), a
theme->sector exposure map, and an AI synthesis pass with a deterministic
fallback. Display-only situational-awareness overlay — never touches scoring.

See docs/superpowers/specs/2026-06-18-macro-pulse-design.md.
"""

from src.macro_pulse.orchestrator import run  # noqa: E402,F401
