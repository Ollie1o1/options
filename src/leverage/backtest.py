"""Event-driven intraday backtest over 5m bars. Conservative by construction:
- entry at the breakout-bar close (signal.entry),
- when a single bar touches BOTH stop and target, the STOP fills (worst case),
- funding charged at each 8h boundary the position spans,
- every round turn pays the core.costs fee+slippage model.
These biases make the edge verdict pessimistic, which is the point."""
from __future__ import annotations
from dataclasses import dataclass, field, replace, fields as _dc_fields
from typing import Optional
import numpy as np
import pandas as pd
from .signals import Signal, Params, generate_signals

_MAX_HOLD = pd.Timedelta(hours=4)


def simulate_trade(sig: Signal, future: pd.DataFrame, funding: Optional[pd.DataFrame],
                   taker: float = 0.00055, slippage: float = 0.0002) -> dict:
    """Walk a single trade forward over `future` (bars at/after entry).
    Returns {pnl_pct, exit_reason, exit_price}. pnl_pct is per-unit price return
    on the side, net of round-turn cost and funding."""
    long = sig.side == "long"
    armed_trail = False
    trail_stop = sig.stop
    exit_price, reason = None, "time"
    for ts, bar in future.iterrows():
        if ts == sig.ts:
            continue
        hi, lo = bar["high"], bar["low"]
        # worst-case ordering: check stop before target
        if long:
            if lo <= trail_stop:
                exit_price, reason = trail_stop, "stop"
                break
            if hi >= sig.target:
                exit_price, reason = sig.target, "target"
                break
            if not armed_trail and hi >= sig.trail_trigger:
                armed_trail = True
            if armed_trail:
                trail_stop = max(trail_stop, hi - sig.atr * 1.0)
        else:
            if hi >= trail_stop:
                exit_price, reason = trail_stop, "stop"
                break
            if lo <= sig.target:
                exit_price, reason = sig.target, "target"
                break
            if not armed_trail and lo <= sig.trail_trigger:
                armed_trail = True
            if armed_trail:
                trail_stop = min(trail_stop, lo + sig.atr * 1.0)
        if ts - sig.ts >= _MAX_HOLD:
            exit_price, reason = bar["close"], "time"
            break
    if exit_price is None:
        exit_price = future["close"].iloc[-1]
    gross = (exit_price - sig.entry) / sig.entry
    if not long:
        gross = -gross
    cost = 2.0 * taker + slippage  # per-notional round turn (entry+exit)
    fund = _funding_cost(sig, future, funding) if funding is not None else 0.0
    return {"pnl_pct": gross - cost - fund, "exit_reason": reason,
            "exit_price": exit_price}


def _funding_cost(sig: Signal, future: pd.DataFrame,
                  funding: pd.DataFrame) -> float:
    """Sum funding paid over the hold (long pays positive funding)."""
    if funding is None or funding.empty:
        return 0.0
    last_ts = future.index[-1]
    window = funding[(funding.index > sig.ts) & (funding.index <= last_ts)]
    rate = window["rate"].sum()
    return rate if sig.side == "long" else -rate


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    n: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0      # mean net pnl_pct per trade
    payoff: float = 0.0          # avg win / avg loss
    max_dd: float = 0.0
    equity_curve: list = field(default_factory=list)
    sides: list = field(default_factory=list)         # 'long'/'short' per trade
    exit_reasons: list = field(default_factory=list)  # 'stop'/'target'/'time'


def run_backtest(df5: pd.DataFrame, df15: pd.DataFrame, params,
                 funding: Optional[pd.DataFrame] = None,
                 taker: float = 0.00055, slippage: float = 0.0002,
                 signal_fn=generate_signals) -> BacktestResult:
    """`signal_fn(df5, df15, params)` selects the strategy (breakout vs
    reversion). Everything downstream is identical, so results are comparable."""
    sigs = signal_fn(df5, df15, params)
    pnls, eq, curve, sides, reasons = [], 1.0, [], [], []
    peak, max_dd = 1.0, 0.0
    for s in sigs:
        future = df5[df5.index >= s.ts]
        if len(future) < 2:
            continue
        r = simulate_trade(s, future, funding, taker, slippage)
        pnls.append(r["pnl_pct"])
        sides.append(s.side)
        reasons.append(r["exit_reason"])
        eq *= (1.0 + r["pnl_pct"])
        curve.append(eq)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
    if not pnls:
        return BacktestResult()
    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0
    return BacktestResult(trades=pnls, n=len(pnls),
                          win_rate=float((arr > 0).mean()),
                          expectancy=float(arr.mean()), payoff=float(payoff),
                          max_dd=float(max_dd), equity_curve=curve,
                          sides=sides, exit_reasons=reasons)


def walk_forward_windows(index: pd.DatetimeIndex, train_months: int = 6,
                         test_months: int = 2):
    """Return list of (train_start, train_end, test_start, test_end) tuples,
    rolled forward by test_months. test_start == train_end."""
    start = index[0]
    end = index[-1]
    out = []
    tr_start = start
    while True:
        tr_end = tr_start + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        out.append((tr_start, tr_end, tr_end, te_end))
        tr_start = tr_start + pd.DateOffset(months=test_months)
    return out


def robustness_params(base: Params, pct: float = 0.20):
    """Baseline + each numeric param perturbed +/- pct (one at a time)."""
    variants = [base]
    for f in _dc_fields(base):
        val = getattr(base, f.name)
        for mult in (1 - pct, 1 + pct):
            nv = type(val)(round(val * mult)) if isinstance(val, int) else val * mult
            variants.append(replace(base, **{f.name: nv}))
    return variants


def walk_forward(df5: pd.DataFrame, df15: pd.DataFrame, params,
                 funding=None, train_months: int = 6, test_months: int = 2,
                 signal_fn=generate_signals):
    """Run run_backtest on each OOS test window. Returns list of
    (test_start, test_end, BacktestResult)."""
    results = []
    for _, _, te_start, te_end in walk_forward_windows(df5.index, train_months,
                                                       test_months):
        seg5 = df5[(df5.index >= te_start) & (df5.index < te_end)]
        seg15 = df15[(df15.index >= te_start) & (df15.index < te_end)]
        if len(seg5) < 50:
            continue
        seg5.attrs["symbol"] = df5.attrs.get("symbol", "BTCUSDT")
        results.append((te_start, te_end,
                        run_backtest(seg5, seg15, params, funding, signal_fn=signal_fn)))
    return results


def robustness_sweep(df5: pd.DataFrame, df15: pd.DataFrame, base,
                     funding=None, pct: float = 0.20, signal_fn=generate_signals):
    """Run a full backtest for each perturbed param set. Returns list of
    (Params, BacktestResult)."""
    return [(p, run_backtest(df5, df15, p, funding, signal_fn=signal_fn))
            for p in robustness_params(base, pct)]
