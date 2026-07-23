"""User preference persistence — theme (and future settings) to/from JSON.

Follows the same tolerant-JSON pattern as src/watchlist.py.
"""
import json

from . import formatting as fmt

_SETTINGS_PATH = "settings.json"

DEFAULT_SETTINGS = {
    "theme": "quant_desk",
}


def load_settings() -> dict:
    """Load settings from JSON, filling in any missing/invalid keys with
    defaults. Never raises — a missing or corrupt file just means defaults."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in DEFAULT_SETTINGS:
                if key in data:
                    settings[key] = data[key]
    except Exception:
        pass
    return settings


def save_settings(settings: dict) -> None:
    """Persist settings to JSON."""
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def get_theme() -> str:
    """Currently persisted theme key."""
    return load_settings()["theme"]


def set_theme(name: str) -> bool:
    """Persist and immediately apply a theme. Returns False (no-op) for an
    unknown theme name."""
    if name not in fmt.THEMES:
        return False
    settings = load_settings()
    settings["theme"] = name
    save_settings(settings)
    fmt.set_theme(name)
    return True


def apply_saved_theme() -> None:
    """Apply the persisted theme to fmt's live palette. Call once near
    process start, before any themed output is printed."""
    fmt.set_theme(load_settings()["theme"])
