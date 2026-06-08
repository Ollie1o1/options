"""Long-call exit levels for mirror-mode execution.

Single source of truth: reuses ``paper_manager._normalize_exit_rules`` so the live
exit levels are identical to what the paper ledger enforces. Pure function.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.paper_manager import _normalize_exit_rules


@dataclass
class Exits:
    take_profit_price: float   # premium target to close for profit
    stop_price: float          # premium at which to stop out
    time_exit_dte: int         # close when DTE falls to/below this
    time_exit_date: str        # YYYY-MM-DD = expiration - time_exit_dte days
    min_days_held: int         # honor a minimum hold before time-exit fires


def compute_exits(entry_price: float,
                  expiration: str,
                  today: Optional[str] = None,
                  config: Optional[dict] = None) -> Exits:
    """Compute long-call exit levels from config exit rules.

    ``entry_price`` is the option premium paid. ``expiration`` is YYYY-MM-DD.
    """
    rules = _normalize_exit_rules(config or {})
    long_r = rules["long"]
    tp_pct = long_r["tp"]     # e.g. 1.0 => +100%
    sl_pct = long_r["sl"]     # e.g. -0.5 => -50%
    tdte = rules["time_exit_dte"]

    take_profit_price = round(entry_price * (1.0 + tp_pct), 2)
    stop_price = round(entry_price * (1.0 + sl_pct), 2)

    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    time_exit_date = (exp_dt - timedelta(days=tdte)).strftime("%Y-%m-%d")

    return Exits(
        take_profit_price=take_profit_price,
        stop_price=stop_price,
        time_exit_dte=tdte,
        time_exit_date=time_exit_date,
        min_days_held=rules["min_days_held"],
    )
