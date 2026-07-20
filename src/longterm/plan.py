"""Plan file (longterm_plan.json) — pure intent: names, ladders, cash pool.

The plan file records what the user intends to buy and at which levels.
What actually happened lives in data/longterm.db (see fills.py) — a tranche
is "filled" when a fill row references it, never via flags here.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_PATH = "longterm_plan.json"


@dataclass
class Tranche:
    level: float
    weight: float


@dataclass
class PlanName:
    ticker: str
    tranches: List[Tranche]
    thesis: str = ""
    allocation: Optional[float] = None


@dataclass
class Plan:
    cash_pool_usd: float = 0.0
    names: List[PlanName] = field(default_factory=list)


def load_plan(path: str = DEFAULT_PATH) -> Plan:
    if not os.path.exists(path):
        empty = Plan()
        save_plan(empty, path)
        return empty
    with open(path) as f:
        raw = json.load(f)
    names: List[PlanName] = []
    for n in raw.get("names", []):
        tranches = [Tranche(float(t["level"]), float(t.get("weight", 0.0)))
                    for t in n.get("tranches", [])]
        if tranches and all(t.weight == 0.0 for t in tranches):
            for t in tranches:
                t.weight = 1.0 / len(tranches)
        names.append(PlanName(
            ticker=str(n["ticker"]).upper(),
            tranches=tranches,
            thesis=str(n.get("thesis", "")),
            allocation=(float(n["allocation"]) if n.get("allocation") is not None else None),
        ))
    return Plan(cash_pool_usd=float(raw.get("cash_pool_usd", 0.0)), names=names)


def save_plan(plan: Plan, path: str = DEFAULT_PATH) -> None:
    raw = {"cash_pool_usd": plan.cash_pool_usd, "names": []}
    for n in plan.names:
        entry = {"ticker": n.ticker, "thesis": n.thesis,
                 "tranches": [{"level": t.level, "weight": t.weight} for t in n.tranches]}
        if n.allocation is not None:
            entry["allocation"] = n.allocation
        raw["names"].append(entry)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(raw, f, indent=2)
    os.replace(tmp, path)


def allocations(plan: Plan) -> dict:
    """Per-ticker fraction of the cash pool: explicit allocations honored,
    names without one split whatever fraction is left, equally."""
    explicit = {n.ticker: float(n.allocation) for n in plan.names if n.allocation is not None}
    implicit = [n.ticker for n in plan.names if n.allocation is None]
    leftover = max(0.0, 1.0 - sum(explicit.values()))
    share = (leftover / len(implicit)) if implicit else 0.0
    out = dict(explicit)
    for t in implicit:
        out[t] = share
    return out


def tranche_size_usd(plan: Plan, name: PlanName, tranche: Tranche) -> float:
    return plan.cash_pool_usd * allocations(plan).get(name.ticker, 0.0) * tranche.weight
