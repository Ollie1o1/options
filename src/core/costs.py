"""Leverage cost model (spec Section B). Friction scales with NOTIONAL,
i.e. with leverage. Pure."""
from __future__ import annotations


def round_turn_cost_pct_equity(taker_fee: float, slippage: float,
                                funding_share: float, leverage: float) -> float:
    """Cost of one open+close round turn as a fraction of equity.

    cost = (2*taker_fee + slippage + funding_share) * leverage
    All inputs are fractions (0.00055 == 0.055%).
    """
    per_notional = (2.0 * taker_fee) + slippage + funding_share
    return per_notional * leverage
