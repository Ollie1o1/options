"""Trust-the-marks check: compare the DVOL implied-vol path used for marking
against real Deribit ATM option-trade IVs. Reports RMSE in vol points. Pure
given the two aligned IV series; a fetch helper builds them for the CLI.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def mark_rmse_vol_pts(model_iv: Sequence[float], real_iv: Sequence[float]) -> float:
    m = np.asarray([float(x) for x in model_iv])
    r = np.asarray([float(x) for x in real_iv])
    n = min(m.size, r.size)
    if n == 0:
        return float("nan")
    return float(math.sqrt(((m[:n] - r[:n]) ** 2).mean()))


def atm_trade_iv_by_day(trades, spot_by_day: Dict, band: float = 0.05) -> Dict:
    """Mean IV of near-ATM option trades per day, from a Deribit trades frame.

    band = |strike/spot - 1| tolerance. Returns {date: mean_iv_pct}. Deribit
    instrument names are like 'BTC-25SEP26-72000-C'; strike is the 3rd field.
    """
    if trades is None or len(trades) == 0:
        return {}
    df = trades.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()

    def _strike(name):
        try:
            return float(str(name).split("-")[2])
        except Exception:
            return float("nan")

    df["strike"] = df["instrument_name"].map(_strike)
    out: Dict = {}
    for d, g in df.groupby("date"):
        sp = spot_by_day.get(pd.Timestamp(d).normalize())
        if not sp:
            continue
        atm = g[(g["strike"] / sp - 1.0).abs() <= band]
        if len(atm):
            out[pd.Timestamp(d).normalize()] = float(atm["iv"].mean())
    return out
