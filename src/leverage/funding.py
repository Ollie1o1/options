"""Funding-rate features for the daily leverage signals.

`data.py` fetches raw funding (8h cadence); nothing consumed it until now.
`align_daily` collapses intraday funding onto the daily bar index (summing the
prints within each UTC day, missing days = 0). `zscore` standardises the daily
funding against a trailing window so "crowded" extremes are comparable across
assets and regimes. All pure; no network.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def align_daily(funding: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Sum intraday funding prints into daily totals reindexed onto `index`.
    Missing days become 0.0 (no funding paid/known that day)."""
    if funding is None or len(funding) == 0:
        return pd.Series(0.0, index=index)
    f = funding.copy()
    f.index = pd.to_datetime(f.index, utc=True)
    daily = f.groupby(f.index.normalize()).sum()
    out = daily.reindex(index.normalize().unique(), fill_value=0.0)
    out.index = index
    return out.astype(float)


def zscore(daily_funding: pd.Series, window: int = 30) -> pd.Series:
    """Trailing-window z-score of daily funding. NaN until `window` samples."""
    mean = daily_funding.rolling(window).mean()
    std = daily_funding.rolling(window).std(ddof=0)
    z = (daily_funding - mean) / std.replace(0.0, np.nan)
    return z
