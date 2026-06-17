"""Transaction-cost model. Parameterized so cost can be stress-tested. Pure."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    option_spread_frac: float = 0.04   # round-trip bid/ask as fraction of premium (~4%, calibrated)
    hedge_slippage_bps: float = 2.0    # per spot rebalance, bps of traded notional

    def option_entry(self, premium_usd: float) -> float:
        """Cost paid entering the short straddle: half the round-trip spread."""
        return abs(premium_usd) * self.option_spread_frac / 2.0

    def option_exit(self, premium_usd: float) -> float:
        """Cost paid closing early. Zero if held to cash settlement (caller decides)."""
        return abs(premium_usd) * self.option_spread_frac / 2.0

    def hedge_trade(self, notional_usd: float) -> float:
        """Cost of one spot rebalance: slippage bps on absolute traded notional."""
        return abs(notional_usd) * self.hedge_slippage_bps / 1e4
