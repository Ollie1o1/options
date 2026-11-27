"""Realised-variance building blocks for the sleeve vol-forecast test (V1).

Definitions: docs/PREREG_SLEEVE_INTRADAY_VOL_20261127.md.

* `interval_returns` -> per full session, the 78 five-minute interval returns of one
  ticker (the first runs from the previous full session's 15:55-bar close to today's
  09:30-bar close, absorbing the overnight gap). Rows are NaN when the session is
  invalid (< 75 of 78 bars, missing 09:30 bar, or no previous full-session close).
* basket RV = sum over intervals of (sum_i w_i r_i)^2, on sessions where every
  basket ticker is valid.
* E1 / E2 / E3 estimators and the QLIKE loss.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.intraday.sessions import with_clock

SLOTS = [9 * 60 + 30 + 5 * k for k in range(78)]   # 09:30 .. 15:55 bar starts
MIN_BARS = 75
ANN = 252.0


def interval_returns(bars: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """78 interval returns per date of `sessions` (bool full-session flags, calendar order)."""
    c = with_clock(bars)
    c = c[c["mod"].isin(SLOTS)]
    closes = c.pivot_table(index="date", columns="mod", values="c", aggfunc="last")
    closes = closes.reindex(index=sessions.index, columns=SLOTS)
    count = closes.notna().sum(axis=1)
    first_ok = closes[SLOTS[0]].notna()
    filled = closes.ffill(axis=1)
    last = filled[SLOTS[-1]]
    full = sessions.astype(bool)
    prev_last = last.shift(1).where(full.shift(1, fill_value=False).astype(bool))
    valid = full & (count >= MIN_BARS) & first_ok & prev_last.notna()
    arr = filled.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    out[:, 0] = arr[:, 0] / prev_last.to_numpy(dtype=float) - 1.0
    out[:, 1:] = arr[:, 1:] / arr[:, :-1] - 1.0
    out[~valid.to_numpy()] = np.nan
    res = pd.DataFrame(out, index=sessions.index, columns=range(78))
    return res[full.values]


def daily_rv(returns: pd.DataFrame) -> pd.Series:
    """Single-ticker daily RV (NaN on invalid sessions)."""
    rv = (returns ** 2).sum(axis=1, min_count=78)
    return rv


def basket_daily_rv(rets: Mapping[str, pd.DataFrame], weights: Mapping[str, float]) -> pd.Series:
    """Daily RV of the weight-normalised basket; NaN unless every ticker is valid that day."""
    w_sum = float(sum(weights.values()))
    acc: Optional[np.ndarray] = None
    idx = None
    for tk, w in weights.items():
        r = rets[tk]
        if idx is None:
            idx = r.index
        a = r.reindex(idx).to_numpy(dtype=float) * (w / w_sum)
        acc = a if acc is None else acc + a
    if acc is None or idx is None:
        return pd.Series(dtype=float)
    rv = np.where(np.isnan(acc).any(axis=1), np.nan, (acc ** 2).sum(axis=1))
    return pd.Series(rv, index=idx)


def _last_n_sessions(series: pd.Series, as_of, n: int) -> pd.Series:
    s = series[[d <= as_of for d in series.index]]
    return s.iloc[-n:]


def e1_intraday_rv20(basket_rv: pd.Series, as_of, n: int = 20, min_valid: int = 18) -> Optional[float]:
    win = _last_n_sessions(basket_rv, as_of, n)
    v = win.dropna()
    if len(win) < n or len(v) < min_valid:
        return None
    return math.sqrt(ANN * float(v.mean()))


# ---------------------------------------------------------------- HAR (E2)
def har_frame(rv: pd.Series, horizon: int = 21) -> pd.DataFrame:
    """Rows per session s: RV_d, RV_w (5), RV_m (22), target y = mean RV over s+1..s+horizon,
    and `end` = the session date at s+horizon (the target is known only after it)."""
    v = rv.to_numpy(dtype=float)
    s = pd.Series(v)
    rv_w = s.rolling(5, min_periods=5).mean()
    rv_m = s.rolling(22, min_periods=22).mean()
    fwd = s[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)
    dates = list(rv.index)
    end = [dates[i + horizon] if i + horizon < len(dates) else None for i in range(len(dates))]
    f = pd.DataFrame({"rv_d": v, "rv_w": rv_w.values, "rv_m": rv_m.values, "y": fwd.values, "end": end},
                     index=rv.index)
    return f.dropna(subset=["rv_d", "rv_w", "rv_m", "y", "end"])


def fit_har(frames: Sequence[pd.DataFrame], as_of, min_pairs: int = 1000) -> Optional[np.ndarray]:
    """Pooled OLS of y on [1, rv_d, rv_w, rv_m] over pairs whose target ended on or before as_of."""
    parts = [f[[e <= as_of for e in f["end"]]] for f in frames]
    data = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
    if data is None or len(data) < min_pairs:
        return None
    X = np.column_stack([np.ones(len(data)), data["rv_d"], data["rv_w"], data["rv_m"]])
    beta, *_ = np.linalg.lstsq(X, data["y"].to_numpy(dtype=float), rcond=None)
    return beta


def e2_har(beta: Optional[np.ndarray], basket_rv: pd.Series, as_of) -> Optional[float]:
    if beta is None:
        return None
    win = _last_n_sessions(basket_rv, as_of, 22)
    if len(win) < 22 or win.isna().any():
        return None
    rv_d, rv_w, rv_m = float(win.iloc[-1]), float(win.iloc[-5:].mean()), float(win.mean())
    f = float(beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m)
    f = max(f, 0.25 * rv_m)
    return math.sqrt(ANN * f) if f > 0 else None


# ---------------------------------------------------------------- E3 and loss
def e3_ewma(daily_returns: Sequence[float], lam: float = 0.94, warmup: int = 60,
            max_len: int = 250) -> Optional[float]:
    r = np.asarray(daily_returns, dtype=float)[-max_len:]
    if r.size < warmup or not np.all(np.isfinite(r)):
        return None
    w = (1 - lam) * lam ** np.arange(r.size)[::-1]
    w /= w.sum()
    return math.sqrt(ANN * float(np.sum(w * r ** 2)))


def qlike(real_var: float, forecast_var: float) -> float:
    x = real_var / forecast_var
    return float(x - math.log(x) - 1.0)
