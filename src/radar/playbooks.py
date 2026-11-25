"""Loader and tier ordering for configs/playbooks.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.paths import repo_path

TIER_RANK: Dict[str, int] = {
    "VALIDATED": 0,
    "LEAD": 1,
    "CONTEXT": 2,
    "RESEARCH": 3,
    "CLOSED": 4,
}

DEFAULT_RADAR_TIERS = frozenset({"VALIDATED", "LEAD", "CONTEXT"})


@dataclass(frozen=True)
class PlaybookMeta:
    id: str
    label: str
    asset_class: str
    horizon_days: Tuple[int, int]
    accounts: Tuple[str, ...]
    evidence_tier: str
    prereg: Optional[str] = None
    result: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    default_friction_bps: Optional[int] = None
    notes: str = ""

    @property
    def horizon_label(self) -> str:
        lo, hi = self.horizon_days
        return f"{lo}-{hi}d" if lo != hi else f"{lo}d"

    @property
    def tier_rank(self) -> int:
        return TIER_RANK.get(self.evidence_tier, 99)

    @property
    def is_default_radar_eligible(self) -> bool:
        return self.evidence_tier in DEFAULT_RADAR_TIERS


def load_playbooks(path: str = "configs/playbooks.json") -> Dict[str, PlaybookMeta]:
    """Load the machine-readable playbook registry keyed by playbook id."""
    full_path = repo_path(path)
    with open(full_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[str, PlaybookMeta] = {}
    for item in raw.get("playbooks") or []:
        hd = item.get("horizon_days") or [1, 1]
        pb = PlaybookMeta(
            id=str(item["id"]),
            label=str(item["label"]),
            asset_class=str(item.get("asset_class", "stock")),
            horizon_days=(int(hd[0]), int(hd[1])),
            accounts=tuple(item.get("accounts") or ("tfsa", "taxable")),
            evidence_tier=str(item.get("evidence_tier", "RESEARCH")).upper(),
            prereg=item.get("prereg"),
            result=item.get("result"),
            stats=dict(item.get("stats") or {}),
            default_friction_bps=item.get("default_friction_bps"),
            notes=str(item.get("notes") or ""),
        )
        out[pb.id] = pb
    return out


def get_playbook(playbook_id: str, registry: Optional[Dict[str, PlaybookMeta]] = None) -> Optional[PlaybookMeta]:
    reg = registry if registry is not None else load_playbooks()
    return reg.get(playbook_id)


def list_playbooks(
    *,
    include_closed: bool = True,
    path: str = "configs/playbooks.json",
) -> List[PlaybookMeta]:
    reg = load_playbooks(path)
    items = list(reg.values())
    if not include_closed:
        items = [p for p in items if p.evidence_tier != "CLOSED"]
    items.sort(key=lambda p: (p.tier_rank, p.id))
    return items
