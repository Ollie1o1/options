"""Pharma catalyst calendar — dated Ph2/Ph3 events for small/mid-cap biotech.

See docs/superpowers/specs/2026-08-25-pharma-catalyst-calendar-design.md.

This package ranks NOTHING. It sorts by event date and shows objective facts.
Nothing here predicts whether a drug works, and no column should ever be added
that implies it does.
"""
from __future__ import annotations

from src.catalyst.models import CatalystEvent, Coverage, Trial

__all__ = ["CatalystEvent", "Coverage", "Trial"]
