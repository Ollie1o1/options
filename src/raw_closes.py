"""Official closes on the strikes' own (unadjusted) basis, for option backtests.

Validated 2026-11-27 (docs/AUDIT_DOLT_SPOT_RESULT_20261127.md): yfinance `Close`
(auto_adjust=False, split-adjusted only) multiplied back by every later split matches
Alpaca SIP raw daily closes on 97-100% of days 2019-26 for AAPL/AMZN/DIA/GOOGL/JPM/META/
MSFT/NVDA/SPY/XLE, with every split bracketed exactly. Alpaca's own daily closes are
NOT used as truth: in 2019-20 about 3% of them carry post-market prints (SPY 2020-03-12
off by 2.9%).

The put-call-parity spot used by `src/selector_study/dolt_history.py` runs +0.09%
high on average (it is a forward) with p95 |error| 0.37%; prefer this module for any
absolute-return claim.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def unsplit(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Raw close from a split-adjusted close: multiply by every split strictly after the date.

    `close` and `splits` are indexed by datetime.date; split values are ratios (4.0 = 4-for-1).
    """
    out = close.astype(float).copy()
    for sd, ratio in splits.items():
        if ratio and ratio > 0:
            out[[d < sd for d in out.index]] *= float(ratio)
    return out


def official_raw_closes(symbols: Sequence[str], start: str = "2018-12-01",
                        end: Optional[str] = None) -> Dict[str, pd.Series]:
    """{symbol: Series(date -> official raw close)}; `.attrs['splits']` = {iso date: ratio}."""
    import yfinance as yf
    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    out: Dict[str, pd.Series] = {}
    for s in symbols:
        h = yf.Ticker(s).history(start=start, end=end, auto_adjust=False, actions=True)
        h.index = [d.date() for d in h.index]
        splits = h["Stock Splits"][h["Stock Splits"] > 0]
        out[s] = unsplit(h["Close"], splits)
        out[s].attrs["splits"] = {str(k): float(v) for k, v in splits.items()}
    return out


def on_or_before(series: pd.Series, day: dt.date, max_days: int = 4) -> Optional[float]:
    """Close on `day`, else the latest within `max_days` before it (weekend-stamped chains)."""
    for k in range(max_days + 1):
        v = series.get(day - dt.timedelta(days=k))
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def settlement_close(series: pd.Series, session: str, expiration: str) -> Optional[float]:
    """Close at expiration on the CONTRACT's basis.

    A split between entry and expiry adjusts the option (strike / r, deliverable x r), so the
    economically equivalent price is the raw close x every split with session < ex-date <= expiration.
    """
    tc = on_or_before(series, dt.date.fromisoformat(expiration))
    if tc is None:
        return None
    s0, s1 = dt.date.fromisoformat(session), dt.date.fromisoformat(expiration)
    for sd, ratio in series.attrs.get("splits", {}).items():
        d = dt.date.fromisoformat(sd[:10])
        if s0 < d <= s1:
            tc *= float(ratio)
    return tc
