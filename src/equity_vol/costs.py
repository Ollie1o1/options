"""Equity transaction-cost model. Real bid/ask crossing lives in the engine
(short straddle is filled at the real dolt bid); this model adds per-leg option
commissions and per-rebalance stock-hedge slippage. Parameterized for stress."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    option_commission: float = 0.65   # per contract, per leg
    hedge_slippage_bps: float = 1.0   # bps of |traded notional| per stock rebalance

    def option_commissions(self, n_legs: int = 2, contracts: int = 1) -> float:
        return abs(self.option_commission) * n_legs * contracts

    def hedge_cost(self, traded_notional: float) -> float:
        return abs(traded_notional) * self.hedge_slippage_bps / 1e4
