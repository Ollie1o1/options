"""CLI for the delta-hedged vol-carry backtester.

    python -m src.crypto.volbacktest --currency BTC --dte 14 --freq 7 \
        --hedge 1 --cost-stress --oos

All data sources are free / no-auth (Deribit DVOL + option trades, yfinance spot).
See docs/superpowers/specs/2026-06-17-crypto-volbacktest-design.md
"""
from __future__ import annotations

import argparse
import warnings
from typing import Dict, List, Optional

import pandas as pd

from . import data as D
from . import metrics as M
from . import validate as V
from .costs import CostModel
from .engine import run_backtest
from .report import format_report

warnings.filterwarnings("ignore")


def _aligned_series(currency: str, days: int):
    """Merge DVOL (implied) and spot on date; return (S list, V list, dates list)."""
    dvol = D.load_dvol(currency, days=days)
    spot = D.load_spot(currency, days=days + 60)
    m = spot.merge(dvol, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    return (m["Close"].astype(float).tolist(),
            m["dvol"].astype(float).tolist(),
            m["Date"].tolist())


def _regime_split(S: List[float], dates, trades, dte: int, freq: int) -> Dict[str, float]:
    """Mean net P&L grouped by the regime in force at each trade's entry."""
    from src.crypto.regime import classify_btc
    buckets: Dict[str, List[float]] = {}
    start = 0
    for t in trades:
        hist = pd.DataFrame({"Close": S[:start + 1]})
        reg = classify_btc(hist)
        label = reg.label if reg else "n/a"
        buckets.setdefault(label, []).append(t.net_pnl)
        start += freq
    return {k: float(sum(v) / len(v)) for k, v in buckets.items() if v}


def _cost_breakeven(S, Vv, dte, freq, r, premium, base_spread, base_slip,
                    hedge_step, wing_pct=0.0) -> Optional[float]:
    """Smallest cost multiple of (spread, slip) at which mean net P&L turns <= 0.
    Returns None if still positive at 5x."""
    prev_mean = None
    for mult in [x * 0.25 for x in range(0, 21)]:  # 0.00 .. 5.00
        cost = CostModel(base_spread * mult, base_slip * mult)
        res = run_backtest(S, Vv, dte, freq, r, premium, cost,
                           hedge_step=hedge_step, wing_pct=wing_pct)
        mean = M.summarize([t.net_pnl for t in res.trades]).get("mean", 0.0)
        if mean <= 0 and prev_mean is not None and prev_mean > 0:
            return mult
        prev_mean = mean
    return float("inf")  # survives even at 5x quoted spread


def _mark_rmse(currency: str, dvol_dates, dvol_vals) -> Optional[float]:
    """Recent-window check: real ATM trade IVs vs the DVOL path used for marking."""
    try:
        trades = D.load_option_trades(currency, days=7, count=1000)
        spot = D.load_spot(currency, days=30)
        spot_by_day = {pd.Timestamp(d).normalize(): float(c)
                       for d, c in zip(spot["Date"], spot["Close"])}
        real = V.atm_trade_iv_by_day(trades, spot_by_day, band=0.05)
        if not real:
            return None
        dvol_by_day = {pd.Timestamp(d).normalize(): float(v)
                       for d, v in zip(dvol_dates, dvol_vals)}
        model_iv, real_iv = [], []
        for day, riv in real.items():
            if day in dvol_by_day:
                model_iv.append(dvol_by_day[day])
                real_iv.append(riv)
        if not model_iv:
            return None
        return V.mark_rmse_vol_pts(model_iv, real_iv)
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.crypto.volbacktest")
    p.add_argument("--currency", default="BTC")
    p.add_argument("--dte", type=int, default=14)
    p.add_argument("--freq", type=int, default=7)
    p.add_argument("--hedge", type=int, default=1, help="hedge step in days")
    p.add_argument("--premium", type=float, default=1000.0)
    p.add_argument("--spread", type=float, default=0.04, help="round-trip option spread frac")
    p.add_argument("--slip-bps", type=float, default=2.0, help="spot hedge slippage bps")
    p.add_argument("--r", type=float, default=0.045)
    p.add_argument("--days", type=int, default=1000, help="history window (DVOL caps ~1000d)")
    p.add_argument("--wing", type=float, default=0.0,
                   help="defined-risk wings at K*(1+/-wing); 0 = naked straddle")
    p.add_argument("--cost-stress", action="store_true")
    p.add_argument("--oos", action="store_true")
    a = p.parse_args(argv)

    S, Vv, dates = _aligned_series(a.currency, a.days)
    if len(S) <= a.dte + 1:
        print("no trades (insufficient aligned data)")
        return 1

    cost = CostModel(option_spread_frac=a.spread, hedge_slippage_bps=a.slip_bps)
    res = run_backtest(S, Vv, a.dte, a.freq, a.r, a.premium, cost,
                       hedge_step=a.hedge, wing_pct=a.wing)
    pnl = [t.net_pnl for t in res.trades]
    if not pnl:
        print("no trades (insufficient aligned data)")
        return 1

    stats = M.summarize(pnl)
    lag = max(1, a.dte // a.freq)
    tstat = M.newey_west_tstat(pnl, lag=lag)
    ci = M.block_bootstrap_ci(pnl, block=max(2, lag), iters=2000, seed=1)
    tpy = 365.0 / a.freq

    breakeven = None
    if a.cost_stress:
        breakeven = _cost_breakeven(S, Vv, a.dte, a.freq, a.r, a.premium,
                                    a.spread, a.slip_bps, a.hedge, wing_pct=a.wing)

    oos = None
    if a.oos:
        k = int(len(pnl) * 0.6)
        oos = {"IS 60%": M.summarize(pnl[:k]), "OOS 40%": M.summarize(pnl[k:])}

    regime = _regime_split(S, dates, res.trades, a.dte, a.freq)
    rmse = _mark_rmse(a.currency, dates, Vv)

    structure = f"wings±{a.wing:.0%}" if a.wing > 0 else "naked"
    header = f"({structure} dte={a.dte} freq={a.freq} hedge={a.hedge}d cost={a.spread:.0%}/{a.slip_bps:.0f}bps) "
    print(format_report(a.currency, stats, tstat, ci, breakeven, rmse, regime,
                        trades_per_year=tpy, oos=oos, header_extra=header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
