"""Candidates expressed per dollar of capital at risk.

A $127 spread and a $34,680 cash-secured put cannot be compared on raw premium
or raw EV — the large one wins by construction. Dividing by capital at risk
puts them on one axis, which is what makes a small budget and a large budget
answerable by the same board.

This matters because size did NOT buy better outcomes in this book. Return on
capital by position size over 877 closed trades showed no monotonic
relationship, and on credit structures the small buckets did better: +16.0% at
$250-500 against -0.3% above $15,000.

EVERYTHING HERE IS DISPLAY-ONLY. These columns must not enter quality_score,
any weight, or any gate, and must never re-sort the board — ranking was
disproven out of sample (Wilcoxon p=0.89), which is why the board refuses
rather than ranks.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .capital_risk import capital_at_risk_for_pick, within_budget


def per_risk(value, risk) -> Optional[float]:
    """``value`` per dollar of capital at risk, or None if that is meaningless.

    None rather than 0 or infinity when risk is unknown or non-positive: a
    blank cell says "not answerable", a 0 says "answered, and the answer is
    zero". They are different claims.
    """
    try:
        v, r = float(value), float(risk)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    return v / r


def annotate(df, strategy_name: str):
    """Add capital_at_risk, reward_per_risk and net_ev_per_risk. Order kept."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    risks, rewards, evs = [], [], []
    for _, row in out.iterrows():
        risk = capital_at_risk_for_pick(row, strategy_name)
        risks.append(risk)
        rewards.append(per_risk(row.get("max_profit"), risk))
        evs.append(per_risk(row.get("ev_per_contract"), risk))
    # Assign as object-dtype Series, not plain lists: when a list mixes
    # ``None`` with real floats, pandas upcasts the column to float64 and
    # silently turns ``None`` into ``NaN``. That is a THIRD, undocumented
    # state on top of the module's None-vs-0 contract, and callers checking
    # ``is None`` to decide whether to print a blank cell would instead
    # render a literal "nan". Object dtype keeps ``None`` as ``None``.
    out["capital_at_risk"] = pd.Series(risks, index=out.index, dtype=object)
    out["reward_per_risk"] = pd.Series(rewards, index=out.index, dtype=object)
    out["net_ev_per_risk"] = pd.Series(evs, index=out.index, dtype=object)
    return out


def affordable(df, budget: Optional[float], strategy_name: str):
    """Rows whose capital at risk fits ``budget``. None budget = no filter."""
    if df is None or len(df) == 0 or budget is None:
        return df
    keep = [within_budget(capital_at_risk_for_pick(row, strategy_name), budget)
            for _, row in df.iterrows()]
    return df[keep]


def budget_use_line(df, budget: Optional[float]) -> Optional[str]:
    """How many of the cheapest survivor the budget buys, or None.

    At a small budget this is usually the most useful line on the board — the
    honest answer is often "you could hold four of these" — and the display
    cannot say it today because it never knows the budget.
    """
    if budget is None or df is None or len(df) == 0:
        return None
    if "capital_at_risk" not in df.columns:
        return None
    risks = [r for r in df["capital_at_risk"].tolist()
             if r is not None and r > 0]
    if not risks:
        return None
    cheapest = min(risks)
    return (f"Budget use: cheapest survivor ${cheapest:,.0f} — "
            f"you could hold {int(budget // cheapest)} of these.")
