"""Delta-hedged short-straddle simulator over real dolt entries.

We SELL the ATM straddle at the real dolt bid (premium received), then hold the
underlying to neutralize delta, rebalancing once per available daily close using
a Black-Scholes-repriced straddle delta (entry IV held fixed — a documented
approximation; the traded option marks are real, which is what governs the
cost-wall question). At expiry we settle to intrinsic.

    pnl = premium - terminal_intrinsic + hedge_pnl - costs

Stock hedge P&L over a step is (shares held entering the step) x (price change).
Decisions at step t use only prices at or before t (no look-ahead)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.equity_vol.data import Entry, days_between, straddle_entries
from src.equity_vol.pricing import straddle_delta
from src.equity_vol.costs import CostModel

R = 0.04
Q = 0.0


@dataclass
class TradeResult:
    symbol: str
    date: str
    dte: int
    premium: float
    terminal_intrinsic: float
    hedge_pnl: float
    costs: float
    pnl: float
    ret: float


def simulate_straddle(entry: Entry, closes: Dict[str, float], r: float = R,
                      q: float = Q, cost: CostModel = CostModel()) -> Optional[TradeResult]:
    K, sigma, exp = entry.strike, entry.iv, entry.expiration
    # daily path from entry date through expiry, only dates we have a close for
    path = sorted((d, s) for d, s in closes.items() if entry.date <= d <= exp)
    if len(path) < 2 or entry.straddle_bid <= 0:
        return None
    premium = entry.straddle_bid
    hedge_pnl = 0.0
    hedge_costs = 0.0
    shares = 0.0  # current hedge position
    for i, (d, S) in enumerate(path):
        # P&L of the hedge held coming into this step
        if i > 0:
            prev_S = path[i - 1][1]
            hedge_pnl += shares * (S - prev_S)
        # rebalance using info available at d (no future leak)
        dte = max(0, days_between(d, exp))
        T = dte / 365.0
        target = straddle_delta(S, K, T, r, sigma, q)  # shares to hold = +straddle delta
        delta_shares = target - shares
        hedge_costs += cost.hedge_cost(delta_shares * S)
        shares = target
    # liquidate residual hedge at expiry
    S_T = path[-1][1]
    hedge_costs += cost.hedge_cost(-shares * S_T)
    terminal_intrinsic = abs(S_T - K)
    costs = cost.option_commissions(n_legs=2, contracts=1) + hedge_costs
    pnl = premium - terminal_intrinsic + hedge_pnl - costs
    return TradeResult(symbol=entry.symbol, date=entry.date, dte=entry.dte,
                       premium=premium, terminal_intrinsic=terminal_intrinsic,
                       hedge_pnl=hedge_pnl, costs=costs, pnl=pnl,
                       ret=(pnl / premium if premium else 0.0))


def run_backtest(db_path: str, symbols: List[str], target_dte: int = 30,
                 freq_days: int = 28, r: float = R,
                 cost: CostModel = CostModel()) -> List[TradeResult]:
    results: List[TradeResult] = []
    for sym in symbols:
        px = {}
        from src.equity_vol.data import closes as _closes
        px = _closes(db_path, sym)
        for e in straddle_entries(db_path, sym, target_dte, freq_days):
            tr = simulate_straddle(e, px, r=r, cost=cost)
            if tr is not None:
                results.append(tr)
    return results
