"""Single source of truth for realized P&L. Replaces the forked math in
crypto/exit_enforcer.py, crypto/check_pnl.py, crypto/backtester.py and the
screener log paths. Pure: no project imports, no I/O."""
from __future__ import annotations
from typing import Optional


def realized_pnl(entry: float, exit_price: float, qty: Optional[float],
                  side: str, structure: str) -> dict:
    """Return {'pnl_usd', 'pnl_pct'} for one closed position.

    entry/exit_price: per-1-unit prices (premium for debit, net credit for credit).
    qty: fractional position size; None -> 1.0.
    side: 'long' or 'short'. structure: 'debit' or 'credit'.

    pnl_pct is the per-unit return ratio (size-independent).
    pnl_usd is the per-unit P&L * qty.

    Note: `entry` is assumed > 0 (caller responsibility); a non-positive `entry`
    yields pnl_pct = 0.0 via the denom guard and is treated as a no-signal case.
    """
    q = 1.0 if qty is None else float(qty)
    e = float(entry)
    x = float(exit_price)
    if structure == "credit":
        per_unit = e - x
        denom = e
    else:
        per_unit = x - e
        denom = e
        if side == "short":
            per_unit = -per_unit
    pnl_usd = per_unit * q
    pnl_pct = (per_unit / denom) if denom else 0.0
    return {"pnl_usd": pnl_usd, "pnl_pct": pnl_pct}
