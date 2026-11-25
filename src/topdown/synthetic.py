"""Deterministic synthetic market for offline tests and demos of the Discover view.

Never used on a live path. Builds SPY, the 16 Outlook ETFs, the rotation-ratio
ETFs, and a handful of real-ticker single names (so the GICS sector map applies)
as seeded random walks with distinct drifts, plus a Cboe-format VIX directory.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.radar.scanners import OUTLOOK_ETFS

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

STOCKS: Dict[str, Tuple[float, float]] = {
    # ticker: (daily drift, beta-ish loading on SPY)
    "XOM": (0.0012, 0.6), "CVX": (0.0011, 0.6), "COP": (0.0010, 0.7), "OXY": (-0.0008, 0.8),
    "AAPL": (0.0006, 1.1), "MSFT": (0.0005, 1.0), "ORCL": (-0.0006, 1.2),
    "NVDA": (0.0015, 1.6), "AMD": (0.0009, 1.7), "MU": (-0.0010, 1.5),
    "JPM": (0.0004, 1.0), "BAC": (0.0003, 1.1), "LLY": (0.0010, 0.6), "PFE": (-0.0007, 0.6),
}
RATIO_ETFS = ("RSP", "HYG", "IEF", "LQD", "EEM")


def business_days(n: int, end: dt.date = dt.date(2026, 11, 24)) -> List[str]:
    out: List[str] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= dt.timedelta(days=1)
    return out[::-1]


def synthetic_ohlcv(n: int = 320, seed: int = 20261125, *, extra: Sequence[str] = (),
                    lag: Optional[Dict[str, int]] = None) -> OhlcvMap:
    """{ticker: (dates, closes, volumes)}; `lag` drops the last k bars of a ticker (stale archive)."""
    rng = np.random.default_rng(seed)
    dates = business_days(n)
    mkt = rng.normal(0.0004, 0.010, n)
    out: OhlcvMap = {}

    def _put(sym: str, drift: float, beta: float, vol: float) -> None:
        r = drift + beta * mkt + rng.normal(0.0, vol, n)
        px = 100.0 * np.cumprod(1.0 + r)
        k = (lag or {}).get(sym, 0)
        ds = dates[: n - k] if k else list(dates)
        cs = px[: n - k] if k else px
        out[sym] = (ds, np.asarray(cs, dtype=float), np.full(len(cs), 1.0e6))

    _put("SPY", 0.0, 1.0, 0.0)
    for i, etf in enumerate(OUTLOOK_ETFS):
        _put(etf, 0.0003 * ((i % 7) - 3), 0.5 + 0.06 * i, 0.006)
    for etf in RATIO_ETFS:
        _put(etf, 0.0001, 0.8, 0.004)
    for sym, (drift, beta) in STOCKS.items():
        _put(sym, drift, beta, 0.012)
    for sym in extra:
        _put(sym, 0.0, 1.0, 0.01)
    return out


def write_cboe_dir(path: str, *, vix: float = 22.0, vix3m: float = 20.0, n: int = 300,
                   end: dt.date = dt.date(2026, 11, 24)) -> str:
    """Write VIX/VIX3M/VVIX/SKEW `_History.csv` files in Cboe's format; returns the directory."""
    os.makedirs(path, exist_ok=True)
    days = business_days(n, end)
    rng = np.random.default_rng(7)
    for name, last in (("VIX", vix), ("VIX3M", vix3m)):
        vals = list(np.clip(rng.normal(last, 3.0, n - 1), 9, 80)) + [last]
        with open(os.path.join(path, f"{name}_History.csv"), "w", encoding="utf-8") as fh:
            fh.write("DATE,OPEN,HIGH,LOW,CLOSE\n")
            for d, v in zip(days, vals):
                y, m, dd = d.split("-")
                fh.write(f"{m}/{dd}/{y},{v:.2f},{v:.2f},{v:.2f},{v:.2f}\n")
    for name, last in (("VVIX", 95.0), ("SKEW", 135.0)):
        with open(os.path.join(path, f"{name}_History.csv"), "w", encoding="utf-8") as fh:
            fh.write(f"DATE,{name}\n")
            for d in days:
                y, m, dd = d.split("-")
                fh.write(f"{m}/{dd}/{y},{last:.2f}\n")
    return path


__all__ = ["STOCKS", "RATIO_ETFS", "business_days", "synthetic_ohlcv", "write_cboe_dir"]
