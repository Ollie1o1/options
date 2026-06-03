#!/usr/bin/env python
"""Leverage signal research harness — reproduces the 2026-06-02/03 study that
found breakout/momentum is sub-coinflip at 5-15min while mean-reversion (fade
>2 sigma) is ~56% directional OOS. Read-only over the cached 5m data; no orders.

Run:  PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/leverage_signal_research.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.leverage import data as D

HOR = (1, 2, 3)  # bars are 5m -> 5/10/15 min forward


def _feats(df):
    c, o, h, l, v = (df[k] for k in ("close", "open", "high", "low", "volume"))
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change()
    f["mom3"] = c.pct_change(3)
    e12, e48 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=48, adjust=False).mean()
    f["ema_diff"] = e12 - e48
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["barshape"] = (c - o) / (h - l).replace(0, np.nan)
    return f


def _rules(f):
    z = f["z"]
    return {
        "always_long": pd.Series(1, index=f.index),
        "cont_ret1": np.sign(f["ret1"]),
        "cont_mom3": np.sign(f["mom3"]),
        "trend_ema": np.sign(f["ema_diff"]),
        "revert_ret1": -np.sign(f["ret1"]),
        "revert_z15": pd.Series(np.where(z > 1.5, -1, np.where(z < -1.5, 1, 0)), index=f.index),
        "revert_z2": pd.Series(np.where(z > 2, -1, np.where(z < -2, 1, 0)), index=f.index),
        "rsi_revert": pd.Series(np.where(f["rsi"] > 70, -1, np.where(f["rsi"] < 30, 1, 0)), index=f.index),
        "barshape_revert": -np.sign(f["barshape"]),
    }


def study(symbol, split=0.70):
    df5, _ = D.load_history(symbol)
    f = _feats(df5)
    c = df5["close"]
    cut = int(len(df5) * split)
    print(f"\n############### {symbol}  ({c.index[0].date()}..{c.index[-1].date()}, "
          f"{(c.iloc[-1]/c.iloc[0]-1)*100:+.0f}% drift) ###############")
    for h in HOR:
        fwd = (c.shift(-h) - c) / c
        sgn = np.sign(fwd)
        print(f"\n-- forward {h*5}min --  base P(up) OOS={ (sgn.iloc[cut:]>0).mean()*100:.1f}%")
        print(f"  {'rule':18}{'IS hit%':>9}{'OOS hit%':>10}{'OOScov%':>9}")
        rows = []
        for name, pred in _rules(f).items():
            pred = pd.Series(pred, index=f.index)
            valid = (pred != 0) & (~fwd.isna()) & (sgn != 0)
            corr = (pred == sgn) & valid
            def hit(sl):
                m = valid.iloc[sl]
                return (corr.iloc[sl].sum() / m.sum() * 100) if m.sum() else float("nan"), \
                       m.sum() / len(m) * 100
            ih, _ = hit(slice(0, cut))
            oh, oc = hit(slice(cut, None))
            rows.append((name, ih, oh, oc))
        for name, ih, oh, oc in sorted(rows, key=lambda x: -(x[2] if x[2] == x[2] else 0)):
            star = " <-- beats 51% OOS" if (oh == oh and oh >= 51) else ""
            print(f"  {name:18}{ih:>9.1f}{oh:>10.1f}{oc:>9.1f}{star}")


if __name__ == "__main__":
    for s in ("BTCUSDT", "ETHUSDT"):
        study(s)
