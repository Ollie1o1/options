"""Compatibility shim. The desk-wide token source lives in desk_kit.theme;
this module re-exports it so older imports and sidecar-era tooling keep
working. New code should import src.desk_kit.theme directly.
"""
from src.desk_kit.theme import (  # noqa: F401
    DARK, LIGHT, css_tokens, heat_inks,
)
