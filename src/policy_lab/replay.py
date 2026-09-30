"""Replay a recorded price path under an alternative exit policy.

Pure: no I/O, no globals, no clock. The engine walks the path forward and may
only look at the point it is standing on and the ones behind it — see
`test_no_look_ahead`, which poisons the future with NaN and demands the same
answer.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from src.policy_lab.costs import CostModel
from src.policy_lab.policies import ExitPolicy
from src.policy_lab.types import Outcome, PathPoint, PricePath


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


def _pnl_frac(path: PricePath, close_price: float) -> float:
    """P&L as a fraction of capital at risk.

    A credit structure earns `entry_price - close_price`; a debit one earns
    `close_price - entry_price`. Both are per-share, so x100 for a contract.
    """
    edge = (path.entry_price - close_price) if path.is_credit else (
        close_price - path.entry_price)
    return edge * 100.0 / path.capital_at_risk


def _triggered(path: PricePath, policy: ExitPolicy, point: PathPoint,
               costs: CostModel) -> Optional[str]:
    """Which rule fires on this point, or None.

    The stop is checked before the take-profit: when a day's range could have
    hit both, assuming the good one is how a backtest flatters itself.
    """
    close = costs.close_price(point, path.is_credit)

    if policy.stop_mult is not None:
        if path.is_credit:
            # Credit doubled against us at stop_mult = 2.0.
            if close >= path.entry_price * policy.stop_mult:
                return "stop_loss"
        else:
            # Debit lost stop_mult of its value.
            if close <= path.entry_price * (1.0 - policy.stop_mult):
                return "stop_loss"

    if policy.take_profit_frac is not None:
        if path.is_credit:
            if close <= path.entry_price * (1.0 - policy.take_profit_frac):
                return "take_profit"
        else:
            if close >= path.entry_price * (1.0 + policy.take_profit_frac):
                return "take_profit"

    if policy.time_exit_dte is not None and point.dte <= policy.time_exit_dte:
        return "time_exit"

    if policy.max_hold_days is not None:
        if _days_between(path.entry_date, point.date) >= policy.max_hold_days:
            return "max_hold"

    return None


def replay(path: PricePath, policy: ExitPolicy, costs: CostModel) -> Outcome:
    """What `policy` would have earned on `path` under `costs`."""
    if len(path.points) < 2:
        raise ValueError(
            f"path {path.position_id} has {len(path.points)} point(s); "
            "replay needs an entry and at least one later quote")
    if path.entry_price == 0.0 and (
            policy.take_profit_frac is not None or policy.stop_mult is not None):
        raise ValueError(
            f"path {path.position_id} opened at 0.00; a take-profit or stop "
            "expressed as a fraction of it is undefined")
    if path.capital_at_risk <= 0.0:
        raise ValueError(
            f"path {path.position_id} has capital_at_risk="
            f"{path.capital_at_risk}; returns are undefined")

    interior = path.interior_points
    for point in path.points[1:]:
        reason = _triggered(path, policy, point, costs)
        if reason is not None:
            return Outcome(
                position_id=path.position_id, exit_date=point.date,
                exit_reason=reason,
                pnl_frac_car=_pnl_frac(
                    path, costs.close_price(point, path.is_credit)),
                decision_points=len(interior),
            )

    last = path.points[-1]
    return Outcome(
        position_id=path.position_id, exit_date=last.date,
        exit_reason="hold_to_end",
        pnl_frac_car=_pnl_frac(path, costs.close_price(last, path.is_credit)),
        decision_points=len(interior),
    )
