"""A strategy is declarative data, not code.

Storing it as JSON means a parameter change is a data change — reviewable in a
diff, with no code edit and no silent behaviour drift.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StrategySpec:
    id: str
    version: int
    structure: str
    universe: Dict[str, Any]
    entry: Dict[str, Any]
    exit: Dict[str, Any]
    sizing: Dict[str, Any]
    created: str
    trial_count: int = 0

    def holding_days(self) -> int:
        """Longest a position can be held. Purging and embargo read this.

        A 7-DTE and a 45-DTE strategy leak over different horizons, so the purge
        window must come from the spec rather than a constant.
        """
        dte = self.entry.get("dte", [0, 45])
        max_dte = int(dte[1])
        if self.exit.get("hold_to_expiry"):
            return max_dte
        time_exit = self.exit.get("time_exit_dte")
        if time_exit is None:
            return max_dte
        return max(0, max_dte - int(time_exit))

    def fingerprint(self) -> str:
        """Identity for trial counting: everything except the count itself."""
        d = asdict(self)
        d.pop("trial_count", None)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategySpec":
        return cls(**d)
