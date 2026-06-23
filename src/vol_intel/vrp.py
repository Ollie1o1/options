"""Implied-vs-realized volatility (VRP) layer. Pairs ATM IV with trailing
realized vol (reusing vol_analytics.realized_vol over the cached daily closes)
and classifies RICH (sell-vol) / FAIR / CHEAP (buy-vol). Live VRP is a trailing
proxy; the forward VRP is the Track-4 backtest."""
from __future__ import annotations
from typing import Optional
import math
import pandas as pd

from src.vol_analytics import realized_vol
from src.breakout.data import load_series

RICH_VP = 0.04
CHEAP_VP = -0.04


def classify(vrp: float, rich: float = RICH_VP, cheap: float = CHEAP_VP) -> str:
    if vrp >= rich:
        return "RICH"
    if vrp <= cheap:
        return "CHEAP"
    return "FAIR"


def vrp_row(symbol: str, atm_iv: float, rv: float) -> dict:
    v = atm_iv - rv
    return {"symbol": symbol, "iv": atm_iv, "rv": rv, "vrp": v, "label": classify(v)}


def realized_vol_for(symbol: str, ohlcv_db: str, window: int = 20) -> Optional[float]:
    s = load_series(ohlcv_db, symbol)
    if s is None or len(s.close) < window + 1:
        return None
    rv = realized_vol(pd.Series(s.close), window=window)
    return None if (rv is None or math.isnan(rv)) else float(rv)


def rv_percentile(symbol: str, ohlcv_db: str, window: int = 20,
                  lookback: int = 252) -> Optional[float]:
    s = load_series(ohlcv_db, symbol)
    if s is None or len(s.close) < window + 5:
        return None
    closes = pd.Series(s.close)
    start = max(window, len(closes) - lookback)
    hist = []
    for t in range(start, len(closes)):
        rv = realized_vol(closes.iloc[: t + 1], window=window)
        if rv is not None and not math.isnan(rv):
            hist.append(rv)
    if len(hist) < 5:
        return None
    cur = hist[-1]
    return sum(1 for x in hist if x <= cur) / len(hist)
