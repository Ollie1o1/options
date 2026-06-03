"""Mean-reversion signal — the empirically-supported counterpart to the
breakout rule in `signals.py`. Research (2026-06-02, cached BTC/ETH 5m, OOS on
both) found breakout/momentum is sub-coinflip at 5-15min (47-49%) while fading a
>2-sigma dislocation is ~56% directional and stable across time. Same PURE
contract as `generate_signals` (no I/O), and it emits the same `Signal` so the
backtest, ticket, paper ledger, and sizing all work unchanged.

Long rule (short = mirror):
  z = (close - SMA(lookback)) / STD(lookback)
  z <= -z_entry  -> long,  target = SMA (the reversion point), stop atr_stop_mult*ATR lower.
  z >= +z_entry  -> short, target = SMA,                       stop atr_stop_mult*ATR higher.
Exit-on-reversion: target IS the mean, so winners run the full snap-back rather
than a fixed time stop; the 4h max-hold in the backtest still backstops it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd
from .signals import Signal
from .indicators import atr


@dataclass(frozen=True)
class ReversionParams:
    lookback: int = 20          # SMA/STD window (bars)
    z_entry: float = 2.0        # enter when |z| >= this
    atr_stop_mult: float = 1.5  # stop = entry +/- this*ATR (further dislocation)
    atr_period: int = 14
    min_edge_atr: float = 0.25  # require target at least this many ATR away


def generate_reversion_signals(df5: pd.DataFrame, df15: pd.DataFrame,
                               params: ReversionParams) -> List[Signal]:
    """Emit a fade Signal for every bar whose |z| clears the threshold. df15 is
    accepted (signature parity with generate_signals) but unused."""
    symbol = df5.attrs.get("symbol", "BTCUSDT")
    c = df5["close"]
    mean = c.rolling(params.lookback).mean()
    std = c.rolling(params.lookback).std()
    a = atr(df5, period=params.atr_period)
    out: List[Signal] = []
    for i in range(len(df5)):
        m = mean.iloc[i]
        sd = std.iloc[i]
        atr_i = a.iloc[i]
        if pd.isna(m) or pd.isna(sd) or sd <= 0 or not (atr_i > 0):
            continue
        close = c.iloc[i]
        z = (close - m) / sd
        if abs(z) < params.z_entry:
            continue
        # target is the mean; skip if the reversion is too small to matter
        if abs(close - m) < params.min_edge_atr * atr_i:
            continue
        ts = df5.index[i]
        conf = min(1.0, abs(z) / (params.z_entry + 1.0))
        if z >= params.z_entry:   # stretched high -> fade short toward the mean
            stop = close + params.atr_stop_mult * atr_i
            out.append(Signal(symbol, "short", ts, close, atr_i, stop, m, m,
                              "reversion", conf))
        else:                      # stretched low -> fade long toward the mean
            stop = close - params.atr_stop_mult * atr_i
            out.append(Signal(symbol, "long", ts, close, atr_i, stop, m, m,
                              "reversion", conf))
    return out
