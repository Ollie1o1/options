#!/usr/bin/env python3
"""Filtering and selection utilities for options contracts.

This module provides functions to filter options chains based on 
liquidity, spread, delta, and other thresholds defined in configuration,
as well as helpers for bucketing and picking top results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def filter_options(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filtering thresholds from config to the options DataFrame.
    
    Args:
        df: DataFrame containing options data
        config: Configuration dictionary containing filtering thresholds
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
        
    f_config = config.get("filters", {})
    
    # 1. Bid-Ask Spread Filter
    # Externalized: max_bid_ask_spread_pct (default 0.40)
    max_spread = f_config.get("max_bid_ask_spread_pct", 0.40)
    if "spread_pct" in df.columns:
        # Filter where spread_pct is within limits and is finite
        df = df[df["spread_pct"] <= max_spread].copy()
        
    # 2. Liquidity Filters (Volume and OI)
    # Externalized: min_volume (default 50) and min_open_interest (default 10)
    min_vol = f_config.get("min_volume", 50)
    min_oi = f_config.get("min_open_interest", 10)
    
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype(float)
        df = df[df["volume"] >= min_vol].copy()
        
    if "openInterest" in df.columns:
        df["openInterest"] = df["openInterest"].fillna(0).astype(float)
        df = df[df["openInterest"] >= min_oi].copy()
        
    # 3. Delta Filters
    # Externalized: delta_min (default 0.15) and delta_max (default 0.35)
    d_min = f_config.get("delta_min", 0.15)
    d_max = f_config.get("delta_max", 0.35)
    
    if "abs_delta" in df.columns:
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
        
    # 4. IV Percentile Filter
    # Externalized: min_iv_percentile (default 20)
    min_iv_pct = f_config.get("min_iv_percentile", 20) / 100.0
    iv_pct_col = next((c for c in ["iv_percentile_30", "iv_percentile"] if c in df.columns), None)
    if iv_pct_col:
        df = df[df[iv_pct_col].fillna(0) >= min_iv_pct].copy()
        
    return df


def categorize_by_premium(df: pd.DataFrame, budget: Optional[float] = None) -> pd.DataFrame:
    """Categorize by premium using quantiles (single-stock) or budget-based (multi-stock)."""
    if df.empty:
        return df
    
    # Calculate contract cost
    df["contract_cost"] = df["premium"] * 100
    
    if budget is not None:
        # Budget mode: categorize based on % of budget
        # LOW: 0-33% of budget, MEDIUM: 33-66%, HIGH: 66-100%
        def cat_budget(cost):
            pct = cost / budget
            if pct <= 0.33:
                return "LOW"
            elif pct <= 0.66:
                return "MEDIUM"
            else:
                return "HIGH"
        df["price_bucket"] = df["contract_cost"].apply(cat_budget)
    else:
        # Single-stock mode: use quantiles
        premiums = df["premium"].astype(float)
        q1 = premiums.quantile(1/3)
        q2 = premiums.quantile(2/3)

        def cat(p):
            if p <= q1:
                return "LOW"
            elif p <= q2:
                return "MEDIUM"
            else:
                return "HIGH"

        df["price_bucket"] = premiums.apply(cat)
    
    return df


def pick_top_per_bucket(df: pd.DataFrame, per_bucket: int = 5, diversify_tickers: bool = False) -> pd.DataFrame:
    """Pick top options per bucket, optionally diversifying across tickers."""
    picks = []
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["price_bucket"] == bucket].copy()
        if sub.empty:
            continue
        
        if diversify_tickers and "symbol" in sub.columns:
            # Try to get diverse tickers in budget mode
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            
            # Pick best from each ticker, then fill remaining slots
            selected = []
            tickers_used = set()
            
            # First pass: best option from each unique ticker
            for _, row in sub.iterrows():
                if row["symbol"] not in tickers_used:
                    selected.append(row)
                    tickers_used.add(row["symbol"])
                    if len(selected) >= per_bucket:
                        break
            
            # Second pass: fill remaining slots with next best regardless of ticker
            if len(selected) < per_bucket:
                for _, row in sub.iterrows():
                    if len(selected) >= per_bucket:
                        break
                    # Check if this exact row is already selected
                    is_duplicate = any(
                        (s["symbol"] == row["symbol"] and 
                         s["strike"] == row["strike"] and 
                         s["expiration"] == row["expiration"] and
                         s["type"] == row["type"]) 
                        for s in selected
                    )
                    if not is_duplicate:
                        selected.append(row)
            
            picks.append(pd.DataFrame(selected))
        else:
            # Standard sorting for single-stock mode
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            picks.append(sub.head(per_bucket))
    
    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True)
    return out

__all__ = [
    "categorize_by_premium",
    "pick_top_per_bucket",
    "filter_options",
]
