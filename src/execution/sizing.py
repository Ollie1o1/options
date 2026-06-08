"""Account-aware position sizing for long-call mirror-mode execution.

Fractional-risk sizing with hard caps. The risk fraction can be informed by a
half-Kelly estimate when win-probability and payoff are supplied, but it is always
clamped by ``max_risk_pct`` (default 2% of account at the stop) and the position
cost is clamped by ``max_position_pct`` (default 10% of account). Pure function —
no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

CONTRACT_MULTIPLIER = 100  # US equity options


@dataclass
class Sizing:
    contracts: int
    dollar_risk: float        # $ lost if the stop is hit
    cost_basis: float         # $ paid to open
    risk_pct: float           # dollar_risk / account_value
    fraction_used: float      # risk fraction actually applied
    kelly_fraction: float     # raw full-Kelly (0 if not supplied)
    notes: str = ""


def _half_kelly(win_prob: Optional[float], payoff_ratio: Optional[float]) -> Optional[float]:
    """Full-Kelly f* = p - (1-p)/b, returned halved and floored at 0.
    None if inputs are missing/invalid."""
    if win_prob is None or payoff_ratio is None or payoff_ratio <= 0:
        return None
    f = win_prob - (1.0 - win_prob) / payoff_ratio
    return max(0.0, f / 2.0)


def size_position(account_value: float,
                  entry_price: float,
                  stop_price: float,
                  win_prob: Optional[float] = None,
                  payoff_ratio: Optional[float] = None,
                  max_risk_pct: float = 0.02,
                  max_position_pct: float = 0.10) -> Sizing:
    """Size a long-call position. ``entry_price``/``stop_price`` are option premiums.

    contracts = min(risk-capped, cost-capped), each a hard ceiling.
    """
    kelly_full = 0.0
    hk = _half_kelly(win_prob, payoff_ratio)
    if hk is not None:
        kelly_full = (hk * 2.0)

    def _zero(note: str) -> Sizing:
        frac = min(max_risk_pct, hk) if hk is not None else max_risk_pct
        return Sizing(0, 0.0, 0.0, 0.0, frac, kelly_full, note)

    if account_value <= 0:
        return _zero("account_value <= 0")

    risk_per_contract = (entry_price - stop_price) * CONTRACT_MULTIPLIER
    if risk_per_contract <= 0:
        return _zero("stop_price >= entry_price (non-positive risk per contract)")

    # Risk fraction: half-Kelly if available, else the max; always capped.
    fraction = min(max_risk_pct, hk) if hk is not None else max_risk_pct
    if fraction <= 0:
        return _zero("non-positive risk fraction (no Kelly edge)")

    risk_budget = account_value * fraction
    contracts_by_risk = math.floor(risk_budget / risk_per_contract)

    cost_per_contract = entry_price * CONTRACT_MULTIPLIER
    contracts_by_cost = math.floor((account_value * max_position_pct) / cost_per_contract) \
        if cost_per_contract > 0 else 0

    contracts = max(0, min(contracts_by_risk, contracts_by_cost))
    if contracts == 0:
        return _zero("caps round position down to 0 contracts")

    dollar_risk = contracts * risk_per_contract
    cost_basis = contracts * cost_per_contract
    note = "risk-capped" if contracts_by_risk <= contracts_by_cost else "cost-capped"
    return Sizing(
        contracts=contracts,
        dollar_risk=round(dollar_risk, 2),
        cost_basis=round(cost_basis, 2),
        risk_pct=round(dollar_risk / account_value, 6),
        fraction_used=round(fraction, 6),
        kelly_fraction=round(kelly_full, 6),
        notes=note,
    )
