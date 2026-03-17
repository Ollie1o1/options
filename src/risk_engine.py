#!/usr/bin/env python3
"""
Risk engine: OI wall detection and gamma-ramp warning.

Extracted from calculate_metrics() to keep options_screener.py focused on
orchestration.  Only depends on pandas, numpy, and typing.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_oi_wall_warning(df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Detect OI walls and annotate each row with a warning string.

    For each (expiration, type) pair, finds the strike with the highest open
    interest.  Rows at or adjacent to that strike receive:
      - "LIMITED UPSIDE"   for call walls
      - "LIMITED DOWNSIDE" for put walls
    """
    df["oi_wall_warning"] = ""

    max_oi_idx = df.groupby(["expiration", "type"])["openInterest"].idxmax()

    for (expiry, opt_type), idx in max_oi_idx.items():
        if pd.isna(idx):
            continue

        wall_strike = df.loc[idx, "strike"]
        mask = (df["expiration"] == expiry) & (df["type"] == opt_type)
        strikes_sorted = df.loc[mask, "strike"].sort_values().unique()
        wall_pos = np.searchsorted(strikes_sorted, wall_strike)

        if opt_type == "call":
            adjacent_strike = strikes_sorted[wall_pos - 1] if wall_pos > 0 else None
            warning_strikes = [wall_strike] + ([adjacent_strike] if adjacent_strike is not None else [])
            warning_mask = mask & df["strike"].isin(warning_strikes)
            df.loc[warning_mask, "oi_wall_warning"] = "LIMITED UPSIDE"
        else:
            adjacent_strike = strikes_sorted[wall_pos + 1] if wall_pos < len(strikes_sorted) - 1 else None
            warning_strikes = [wall_strike] + ([adjacent_strike] if adjacent_strike is not None else [])
            warning_mask = mask & df["strike"].isin(warning_strikes)
            df.loc[warning_mask, "oi_wall_warning"] = "LIMITED DOWNSIDE"

    return df


def calculate_gamma_ramp(df: pd.DataFrame) -> pd.DataFrame:
    """Flag near-expiry ATM/NTM options with explosive gamma.

    Sets df["gamma_ramp"] = True when:
      - DTE < 7
      - gamma > 0.04
      - abs_delta > 0.20  (not deep OTM)
    """
    dte_vals = df["T_years"].values * 365.0
    df["gamma_ramp"] = (
        (dte_vals < 7)
        & (df["gamma"].values > 0.04)
        & (df["abs_delta"].values > 0.20)
    )
    return df


def run_risk_checks(df: pd.DataFrame, current_price: float, config: Dict) -> pd.DataFrame:
    """Run all structural risk checks and return the annotated DataFrame."""
    df = calculate_oi_wall_warning(df, current_price)
    df = calculate_gamma_ramp(df)
    return df
