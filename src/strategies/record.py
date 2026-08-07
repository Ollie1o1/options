"""A setup: the parameters, plus when to use it, where it can be traded, and why.

Account eligibility is required and validated. A registered account generally
cannot sell naked options, so filing a naked-call setup under `tfsa` is not a
preference — it is a setup that cannot legally be executed. Better to fail at
construction than to discover it at the broker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

from .spec import StrategySpec

STATUSES = ("idea", "specified", "backtesting", "validated",
            "promoted", "live", "retired", "dead")
ACCOUNTS = ("tfsa", "taxable", "both")
SETTLED = ("dead", "retired")

_AMENDABLE = ("name", "hypothesis", "signal", "accounts", "capital_note",
              "status", "evidence", "cost_profile", "verdict", "provenance",
              "links")


@dataclass(frozen=True)
class StrategyRecord:
    spec: StrategySpec
    name: str
    hypothesis: str
    signal: Dict[str, Any] = field(default_factory=dict)
    accounts: List[str] = field(default_factory=list)
    capital_note: str = ""
    status: str = "idea"
    evidence: Dict[str, Any] = field(default_factory=dict)
    cost_profile: Dict[str, Any] = field(default_factory=dict)
    verdict: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    amendments: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if not self.accounts:
            raise ValueError("every setup must declare at least one account")
        for a in self.accounts:
            if a not in ACCOUNTS:
                raise ValueError(f"unknown account {a!r}; expected {ACCOUNTS}")

    def tradeable_in(self, account: str) -> bool:
        return account in self.accounts or "both" in self.accounts

    def is_settled(self) -> bool:
        return self.status in SETTLED

    def amend(self, field_name: str, value: Any, reason: str,
              date: str) -> "StrategyRecord":
        if field_name not in _AMENDABLE:
            raise ValueError(f"{field_name!r} is not amendable")
        entry = {"field": field_name, "from": getattr(self, field_name),
                 "to": value, "reason": reason, "date": date}
        return replace(self, **{field_name: value,
                                "amendments": list(self.amendments) + [entry]})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyRecord":
        d = dict(d)
        d["spec"] = StrategySpec.from_dict(d["spec"])
        return cls(**d)
