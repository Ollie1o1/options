"""Stores strategies and counts how many configurations have been tried.

The count is family-wide and cumulative: every distinct configuration ever
evaluated against this dataset, including abandoned and losing ones. Re-running
an unchanged spec is not a new trial.

Deflated Sharpe reads this number, so an undercount silently inflates every
result the engine produces. That is the whole reason the registry exists rather
than a folder of JSON files — before this, the number of configurations tried
lived in nobody's head and could not be recovered after the fact.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .spec import StrategySpec

_TRIALS_FILE = "_trials.json"


class Registry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ── trials ──────────────────────────────────────────────────────────
    @property
    def _trials_path(self) -> str:
        return os.path.join(self.root, _TRIALS_FILE)

    def _read_trials(self) -> Dict[str, Any]:
        if not os.path.exists(self._trials_path):
            return {"count": 0, "fingerprints": []}
        try:
            with open(self._trials_path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return {"count": 0, "fingerprints": []}
        d.setdefault("count", 0)
        d.setdefault("fingerprints", [])
        return d

    def _write_trials(self, d: Dict[str, Any]) -> None:
        with open(self._trials_path, "w") as f:
            json.dump(d, f, indent=2)

    @property
    def trial_count(self) -> int:
        return int(self._read_trials()["count"])

    def record_trial(self, fingerprint: Optional[str] = None) -> int:
        """Count a configuration. Abandoned and losing configs count too."""
        d = self._read_trials()
        if fingerprint is not None:
            if fingerprint in d["fingerprints"]:
                return int(d["count"])
            d["fingerprints"].append(fingerprint)
        d["count"] = int(d["count"]) + 1
        self._write_trials(d)
        return int(d["count"])

    # ── specs ───────────────────────────────────────────────────────────
    def _spec_path(self, spec_id: str) -> str:
        return os.path.join(self.root, f"{spec_id}.json")

    def save(self, spec: StrategySpec) -> StrategySpec:
        count = self.record_trial(spec.fingerprint())
        stored = StrategySpec(**{**spec.to_dict(), "trial_count": count})
        with open(self._spec_path(spec.id), "w") as f:
            json.dump(stored.to_dict(), f, indent=2)
        return stored

    def load(self, spec_id: str) -> StrategySpec:
        with open(self._spec_path(spec_id)) as f:
            return StrategySpec.from_dict(json.load(f))

    def list(self) -> List[StrategySpec]:
        out = []
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".json") and fn != _TRIALS_FILE:
                out.append(self.load(fn[:-5]))
        return out
