"""Delta-hedged short-straddle path simulator. Pure given inputs (no network).

Cash-accounting contract: sell `q` straddles at entry IV; each hedge step reprice
at current spot + current IV + remaining T and rebalance spot to flatten net delta;
at expiry the short straddle cash-settles at q*|S_T-K| (no exit option spread) and
the spot hedge is closed. net_pnl == final cash after everything is flat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import pricing as P
from .costs import CostModel


@dataclass
class TradeResult:
    entry_spot: float
    strike: float
    dte: int
    qty: float
    premium: float            # premium collected (gross, USD)
    terminal_payout: float    # q*|S_T-K| paid back at expiry
    hedge_pnl: float          # spot-hedge mark-to-market (residual of the identity)
    option_cost: float        # option spread paid
    hedge_cost: float         # spot slippage paid
    net_pnl: float
    pnl_pct_premium: float    # net / premium


@dataclass(frozen=True)
class BacktestResult:
    trades: List[TradeResult]
    params: dict


def simulate_trade(spot: Sequence[float], dvol_pct: Sequence[float], dte: int,
                   r: float, premium_notional: float, cost: CostModel,
                   hedge_step: int = 1, strike: Optional[float] = None) -> TradeResult:
    n = min(len(spot), len(dvol_pct))
    if n < 2 or dte <= 0:
        raise ValueError("need >=2 path points and dte>0")
    S0 = float(spot[0])
    K = float(strike) if strike else S0
    iv0 = float(dvol_pct[0]) / 100.0
    T0 = dte / 365.0
    straddle0 = P.straddle(S0, K, T0, r, iv0)
    if straddle0 <= 0:
        raise ValueError("non-positive entry premium")
    qty = premium_notional / straddle0
    premium = qty * straddle0

    cash = premium
    option_cost = cost.option_entry(premium)
    cash -= option_cost
    hedge_cost = 0.0

    # initial hedge: hold spot to flatten the short straddle's delta.
    hedge_units = qty * P.straddle_delta(S0, K, T0, r, iv0)
    cash -= hedge_units * S0
    hc = cost.hedge_trade(abs(hedge_units) * S0)
    hedge_cost += hc
    cash -= hc

    for u in range(hedge_step, dte, hedge_step):
        if u >= n:
            break
        Su = float(spot[u])
        ivu = float(dvol_pct[u]) / 100.0
        Tu = (dte - u) / 365.0
        target = qty * P.straddle_delta(Su, K, Tu, r, ivu)
        d = target - hedge_units
        cash -= d * Su
        hc = cost.hedge_trade(abs(d) * Su)
        hedge_cost += hc
        cash -= hc
        hedge_units = target

    # expiry: cash-settle short straddle at intrinsic; close spot hedge.
    ST = float(spot[min(dte, n - 1)])
    terminal_payout = qty * abs(ST - K)
    cash -= terminal_payout
    cash += hedge_units * ST
    hc = cost.hedge_trade(abs(hedge_units) * ST)
    hedge_cost += hc
    cash -= hc

    net = cash
    # hedge_pnl is the residual that makes the attribution identity hold:
    # net = premium - terminal_payout + hedge_pnl - option_cost - hedge_cost
    hedge_pnl = net - premium + terminal_payout + option_cost + hedge_cost
    return TradeResult(
        entry_spot=S0, strike=K, dte=dte, qty=qty, premium=premium,
        terminal_payout=terminal_payout, hedge_pnl=hedge_pnl,
        option_cost=option_cost, hedge_cost=hedge_cost, net_pnl=net,
        pnl_pct_premium=net / premium if premium else 0.0,
    )


def run_backtest(spot: Sequence[float], dvol_pct: Sequence[float], dte: int,
                 freq: int, r: float, premium_notional: float, cost: CostModel,
                 hedge_step: int = 1, dates: Optional[Sequence] = None) -> BacktestResult:
    n = min(len(spot), len(dvol_pct))
    trades: List[TradeResult] = []
    entry_dates: List = []
    start = 0
    while start + dte < n:
        sub_s = spot[start:start + dte + 1]
        sub_v = dvol_pct[start:start + dte + 1]
        t = simulate_trade(sub_s, sub_v, dte, r, premium_notional, cost, hedge_step)
        trades.append(t)
        if dates is not None and start < len(dates):
            entry_dates.append(dates[start])
        start += freq
    return BacktestResult(trades=trades, params=dict(
        dte=dte, freq=freq, r=r, premium_notional=premium_notional,
        hedge_step=hedge_step, n_points=n, entry_dates=entry_dates))
