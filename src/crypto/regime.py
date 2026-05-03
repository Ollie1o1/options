"""Crypto regime classifier: BULL / CHOP / BEAR for BTC.

Uses 200-day MA position + realized-vol percentile. Different from the equity
VIX regime because crypto has no central vol index — RVOL has to do the job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Regime:
    label: str            # "bull" | "chop" | "bear"
    rvol_30d: float       # annualized 30-day realized vol
    rvol_pct: float       # 30d rvol percentile rank vs 1yr (0..1)
    above_200d: bool      # spot above 200-day MA
    spot: float
    ma_200: float

    def __str__(self) -> str:
        loc = "above" if self.above_200d else "below"
        return (f"{self.label.upper()}  "
                f"(rvol30d {self.rvol_30d:.0%} @ p{int(self.rvol_pct*100)}, "
                f"spot {loc} 200dMA)")


def classify_btc(history: pd.DataFrame) -> Optional[Regime]:
    """Classify BTC regime from a daily OHLC history dataframe (from yfinance)."""
    if history is None or history.empty or "Close" not in history.columns:
        return None
    close = history["Close"].astype(float).dropna()
    if len(close) < 200:
        return None
    spot = float(close.iloc[-1])
    ma_200 = float(close.tail(200).mean())
    above = spot > ma_200
    log_returns = np.log(close).diff().dropna()
    if len(log_returns) < 60:
        return None
    rvol_30 = float(log_returns.tail(30).std() * math.sqrt(365))
    # Rolling 30-day rvol over the last year for percentile rank
    rolling = log_returns.rolling(30).std() * math.sqrt(365)
    rolling = rolling.dropna().tail(365)
    if rolling.empty:
        return None
    rvol_pct = float((rolling < rvol_30).sum() / len(rolling))
    # Classification:
    # - bull: above 200d AND rvol < 60th percentile
    # - bear: below 200d OR rvol > 80th percentile
    # - chop: everything else (low-vol mean-reverting)
    if above and rvol_pct < 0.60:
        label = "bull"
    elif (not above) or rvol_pct > 0.80:
        label = "bear"
    else:
        label = "chop"
    return Regime(label=label, rvol_30d=rvol_30, rvol_pct=rvol_pct,
                  above_200d=above, spot=spot, ma_200=ma_200)


# Per-regime weight multipliers — applied on top of base scoring component
# weights. Keep close to 1.0 until we have closed paper trades to calibrate.
REGIME_WEIGHT_MULTIPLIERS = {
    "bull":  {"vrp": 0.85, "iv_rank": 0.85, "term_structure": 1.10, "skew": 1.00,
              "funding_z": 1.10, "basis": 1.00, "funding_divergence": 1.00,
              "oi_surge": 1.20, "liquidity": 1.00},
    "chop":  {"vrp": 1.20, "iv_rank": 1.20, "term_structure": 1.10, "skew": 1.00,
              "funding_z": 0.90, "basis": 1.00, "funding_divergence": 1.20,
              "oi_surge": 1.10, "liquidity": 1.00},
    "bear":  {"vrp": 1.10, "iv_rank": 1.05, "term_structure": 0.90, "skew": 1.20,
              "funding_z": 1.10, "basis": 0.95, "funding_divergence": 1.10,
              "oi_surge": 1.20, "liquidity": 1.10},
}
