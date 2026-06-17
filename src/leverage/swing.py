"""Daily trend-breakout swing strategy for BTC/ETH perps.

Research basis (memory/project_leverage_swing_research): mean-reversion / fading
support-resistance is dead in crypto (price trends THROUGH levels); the trend-
FOLLOWING breakout that fails at 5-15min (cost wall) becomes tradeable at the
DAILY horizon, where the ~13bps round-trip cost is a negligible fraction of the
move. Long bias above the 100d MA, short bias below it.

Risk is NOT hardcoded. The stop is a volatility (ATR) chandelier whose multiple
`k` is CALIBRATED from data — the q-th percentile of the adverse excursion that
historically *winning* breakouts survived ("put the stop beyond where q% of
winners dipped"). There is no fixed take-profit: a winner rides the trailing stop
until that stop — or a regime flip (close back through the MA) — closes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 20      # Donchian breakout window (days)
DEFAULT_MA = 100           # regime filter (days)
DEFAULT_ATR = 14
DEFAULT_COST = 0.0013      # round-trip perp cost (taker + slippage), fraction
DEFAULT_HORIZON = 10       # forward window used to label winners for calibration
DEFAULT_PCT = 80           # winners'-MAE percentile -> stop multiple k
MAX_HOLD = 30              # max trading days a trade can stay open
K_FALLBACK = 2.0           # used when there is too little data to calibrate


def atr(df: pd.DataFrame, period: int = DEFAULT_ATR) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK,
                     ma: int = DEFAULT_MA, atr_period: int = DEFAULT_ATR) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, atr_period)
    out["hi_n"] = out["high"].rolling(lookback).max().shift(1)
    out["lo_n"] = out["low"].rolling(lookback).min().shift(1)
    out["ma"] = out["close"].rolling(ma).mean()
    out["long_brk"] = (out["close"] > out["hi_n"]) & (out["close"] > out["ma"])
    out["short_brk"] = (out["close"] < out["lo_n"]) & (out["close"] < out["ma"])
    return out


@dataclass
class Signal:
    date: object
    side: str            # "long" | "short"
    price: float
    trigger_level: float  # the broken Donchian level
    atr: float
    stop: float          # initial stop = price -/+ k*ATR (k calibrated)
    stop_k: float
    risk_pct: float      # initial stop distance as a fraction of price
    note: str


def latest_signal(df: pd.DataFrame, stop_k: float, lookback: int = DEFAULT_LOOKBACK,
                  ma: int = DEFAULT_MA, atr_period: int = DEFAULT_ATR) -> Optional[Signal]:
    """Return a Signal only if the most recent bar is a fresh breakout, else None
    (the 'show up only when it does' contract). `stop_k` is the calibrated ATR
    multiple for the initial/trailing chandelier stop."""
    f = compute_features(df, lookback, ma, atr_period)
    row = f.iloc[-1]
    a = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
    if a <= 0:
        return None
    price = float(row["close"])
    if bool(row["long_brk"]):
        stop = price - stop_k * a
        return Signal(row.name, "long", price, float(row["hi_n"]), a, stop, stop_k,
                      (price - stop) / price, f"close>{lookback}d-high & >MA{ma}")
    if bool(row["short_brk"]):
        stop = price + stop_k * a
        return Signal(row.name, "short", price, float(row["lo_n"]), a, stop, stop_k,
                      (stop - price) / price, f"close<{lookback}d-low & <MA{ma}")
    return None


def _excursions(f: pd.DataFrame, side: str, horizon: int, max_i: Optional[int] = None):
    """For each breakout entry, return (mae_atr, is_winner): the max adverse
    excursion in ATR units over `horizon` bars, and whether the horizon-forward
    return was favorable. Winner labelling is independent of the stop (no
    circularity). `max_i` caps entries to keep calibration within a train slice."""
    c = f["close"].values
    h = f["high"].values
    l = f["low"].values
    a = f["atr"].values
    flag = f["long_brk"].values if side == "long" else f["short_brk"].values
    n = len(c)
    cap = n if max_i is None else min(max_i, n)
    rows = []
    for i in range(n):
        if i >= cap or not flag[i] or i + horizon >= n:
            continue
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        e = c[i]
        win_slice = slice(i + 1, i + 1 + horizon)
        if side == "long":
            mae = (e - np.min(l[win_slice])) / a[i]
            winner = (c[i + horizon] - e) > 0
        else:
            mae = (np.max(h[win_slice]) - e) / a[i]
            winner = (e - c[i + horizon]) > 0
        rows.append((max(0.0, float(mae)), bool(winner)))
    return rows


def calibrate_stop_k(f: pd.DataFrame, side: str, horizon: int = DEFAULT_HORIZON,
                     pct: float = DEFAULT_PCT, max_i: Optional[int] = None,
                     fallback: float = K_FALLBACK) -> float:
    """Statistically-derived stop multiple: the `pct`-th percentile of the adverse
    excursion (in ATR) that *winning* breakouts survived. Not optimized on the
    P&L itself, so it is far less overfit-prone than a returns grid-search."""
    ex = _excursions(f, side, horizon, max_i)
    winners_mae = [m for m, w in ex if w]
    if len(winners_mae) < 8:
        return fallback
    return float(round(np.percentile(winners_mae, pct), 3))


@dataclass
class SwingTrade:
    side: str
    entry_date: object
    exit_date: object
    entry: float
    exit: float
    stop_k: float
    r_multiple: float    # P&L in units of initial risk (k*ATR)
    ret: float           # net fractional return after cost
    bars_held: int
    reason: str          # "stop" | "regime" | "max_hold"


def backtest(f: pd.DataFrame, k_long: float, k_short: float,
             cost: float = DEFAULT_COST, max_hold: int = MAX_HOLD,
             allow_short: bool = True) -> List[SwingTrade]:
    """Walk each breakout forward with an ATR chandelier trailing stop and a
    regime-flip exit. No overlapping trades. No fixed take-profit."""
    c = f["close"].values
    h = f["high"].values
    l = f["low"].values
    a = f["atr"].values
    ma = f["ma"].values
    long_brk = f["long_brk"].values
    short_brk = f["short_brk"].values
    idx = list(f.index)
    n = len(c)
    trades: List[SwingTrade] = []
    i = DEFAULT_LOOKBACK
    while i < n - 1:
        is_long = bool(long_brk[i])
        is_short = allow_short and bool(short_brk[i])
        if not (is_long or is_short) or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        side = "long" if is_long else "short"
        k = k_long if is_long else k_short
        e = c[i]
        risk = k * a[i]
        if risk <= 0:
            i += 1
            continue
        extreme = e  # highest close (long) / lowest close (short) since entry
        exit_px = c[min(i + max_hold, n - 1)]
        exit_j = min(i + max_hold, n - 1)
        reason = "max_hold"
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if side == "long":
                extreme = max(extreme, c[j])
                stop = extreme - risk
                if l[j] <= stop:
                    exit_px, exit_j, reason = stop, j, "stop"
                    break
                if np.isfinite(ma[j]) and c[j] < ma[j]:
                    exit_px, exit_j, reason = c[j], j, "regime"
                    break
            else:
                extreme = min(extreme, c[j])
                stop = extreme + risk
                if h[j] >= stop:
                    exit_px, exit_j, reason = stop, j, "stop"
                    break
                if np.isfinite(ma[j]) and c[j] > ma[j]:
                    exit_px, exit_j, reason = c[j], j, "regime"
                    break
        if side == "long":
            r = (exit_px - e) / risk
            ret = (exit_px - e) / e - cost
        else:
            r = (e - exit_px) / risk
            ret = (e - exit_px) / e - cost
        trades.append(SwingTrade(side, idx[i], idx[exit_j], e, exit_px, k,
                                 round(r, 3), round(ret, 5), exit_j - i, reason))
        i = exit_j + 1  # no overlapping positions
    return trades


def summarize(trades: List[SwingTrade]) -> Dict[str, float]:
    if not trades:
        return {"n": 0}
    rets = np.array([t.ret for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = float((1 - eq / peak).max()) if len(eq) else 0.0
    return {
        "n": len(trades),
        "win_rate": float((rets > 0).mean()),
        "mean_R": float(rs.mean()),
        "expectancy_R": float(rs.mean()),  # mean R-multiple == expectancy per trade
        "mean_ret": float(rets.mean()),
        "total_ret": float(eq[-1] - 1.0),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": mdd,
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
    }


def walk_forward(df: pd.DataFrame, train_frac: float = 0.6,
                 lookback: int = DEFAULT_LOOKBACK, ma: int = DEFAULT_MA,
                 atr_period: int = DEFAULT_ATR, horizon: int = DEFAULT_HORIZON,
                 cost: float = DEFAULT_COST, allow_short: bool = True) -> Dict:
    """Calibrate the stop on the first `train_frac`, then trade the held-out tail
    with that fixed k. Reports in-sample and out-of-sample stats side by side."""
    f = compute_features(df, lookback, ma, atr_period)
    n = len(f)
    split = int(n * train_frac)
    k_long = calibrate_stop_k(f, "long", horizon, max_i=split)
    k_short = calibrate_stop_k(f, "short", horizon, max_i=split)
    is_trades = backtest(f.iloc[:split], k_long, k_short, cost, allow_short=allow_short)
    oos_trades = backtest(f.iloc[split:], k_long, k_short, cost, allow_short=allow_short)
    return {
        "k_long": k_long, "k_short": k_short, "split": split,
        "in_sample": summarize(is_trades), "out_sample": summarize(oos_trades),
        "oos_trades": oos_trades,
    }
