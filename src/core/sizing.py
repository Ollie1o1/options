"""Position sizing. capped_quantity is used by the crypto paper ledger and
the backfill; the leverage effective-leverage/Kelly logic lands in Plan 2
in this same module."""
from __future__ import annotations
import math

_CAP_TARGET = 999.0  # target just under the cap, never at/over it


def capped_quantity(unit_risk: float, cap_usd: float = 1000.0) -> float:
    """Fractional size so unit_risk * qty stays just under cap_usd.

    unit_risk: per-1-unit capital at risk - premium paid for a debit
    structure, or (spread_width - net_credit) for a credit structure.
    Rounded DOWN to 4dp so it can never exceed the cap.
    """
    if unit_risk <= 0:
        return 0.0
    target = min(_CAP_TARGET, cap_usd) if cap_usd < _CAP_TARGET else _CAP_TARGET
    raw = target / unit_risk
    q = math.floor(raw * 1e4) / 1e4
    return q
