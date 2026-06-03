"""Liquidation math + hard risk rules for perp positions. Pure where possible.

The primary constraint is "don't get liquidated": a position is only allowed
when the stop is at least 3x closer than the liquidation price.
"""
from __future__ import annotations
import math


def liquidation_price(entry: float, side: str, leverage: float,
                      maint_margin: float = 0.005) -> float:
    """Approximate isolated-margin liquidation price.

    long:  entry * (1 - 1/L + maint)
    short: entry * (1 + 1/L - maint)
    Conservative (ignores fees in the margin buffer, which makes liq appear
    slightly closer than reality -> safer rejections).
    """
    inv_l = 1.0 / leverage
    if side == "long":
        return entry * (1.0 - inv_l + maint_margin)
    return entry * (1.0 + inv_l - maint_margin)


def passes_liquidation_safety(stop_dist: float, liq_dist: float,
                              buffer: float = 3.0) -> bool:
    """True iff stop is at least `buffer`x closer than liquidation.

    stop_dist / liq_dist are fractional distances from entry (both > 0).
    Rule: stop_dist <= liq_dist / buffer.
    """
    if liq_dist <= 0:
        return False
    return stop_dist <= (liq_dist / buffer) + 1e-12


class DailyLossLimit:
    """Tracks intraday realized P&L; blocks trading once the daily loss cap
    is breached. Caller resets at the UTC day boundary."""

    def __init__(self, equity: float, max_daily_loss_frac: float = 0.06):
        self.limit_usd = -abs(equity * max_daily_loss_frac)
        self.day_pnl = 0.0

    def record_pnl(self, pnl_usd: float) -> None:
        self.day_pnl += pnl_usd

    def can_trade(self) -> bool:
        return self.day_pnl > self.limit_usd

    def reset(self) -> None:
        self.day_pnl = 0.0


def expected_worst_streak(n_trades: int, p_win: float) -> float:
    """Expected longest losing streak over n_trades.
    E[streak] ~= ln(N) / ln(1/(1-p_win)). Used by the ruin monitor to size the
    account against routine losing runs."""
    p_loss = 1.0 - p_win
    if n_trades <= 1 or not (0.0 < p_loss < 1.0):
        return 0.0
    return math.log(n_trades) / math.log(1.0 / p_loss)
