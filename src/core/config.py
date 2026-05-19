"""Single config loader. Replaces crypto/auto_logger._load_config and
crypto/exit_enforcer._load_config."""
from __future__ import annotations
import json
from typing import Any, Optional


class Config:
    def __init__(self, data: dict):
        self._data = data or {}

    def section(self, name: str, default: Optional[Any] = None) -> Any:
        val = self._data.get(name)
        if val is None:
            return {} if default is None else default
        return val


def load_config(path: str = "config.json") -> Config:
    try:
        with open(path) as f:
            return Config(json.load(f))
    except (OSError, json.JSONDecodeError):
        return Config({})
