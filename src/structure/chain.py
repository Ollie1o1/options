"""Fetch a live chain and turn it into expression candidates.

Isolated from candidates.py so that the pure math stays network-free and
unit-testable; everything that can touch yfinance lives here.
"""
import json
from typing import Dict, Optional, Tuple

from .candidates import build_candidates

DEFAULT_MIN_DTE = 30
DEFAULT_MAX_DTE = 60


def _exit_rules(config_path: str = "config.json") -> dict:
    try:
        with open(config_path) as f:
            return json.load(f).get("exit_rules") or {}
    except (OSError, ValueError):
        return {}


def fetch_candidates(symbol: str, min_dte: int = DEFAULT_MIN_DTE,
                     max_dte: int = DEFAULT_MAX_DTE,
                     config_path: str = "config.json",
                     capital_usd: float = 511.0
                     ) -> Tuple[Dict[str, dict], Optional[str]]:
    """Return (candidates, error). Never raises - a dead feed yields ({}, msg)
    so the report says "no candidate contracts" instead of a stack trace."""
    try:
        from src.data_fetching import fetch_options_yfinance
        res = fetch_options_yfinance(symbol.upper(), max_expiries=6,
                                     min_dte=min_dte, max_dte=max_dte)
    except Exception as exc:
        return {}, "chain fetch failed: {}".format(exc)

    df = (res or {}).get("df")
    if df is None or df.empty:
        return {}, "no options chain for {}".format(symbol.upper())

    try:
        spot = float(df["underlying"].iloc[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return {}, "no spot price for {}".format(symbol.upper())

    # One expiry at a time: mixing expiries would price a spread across
    # different maturities, which is not the structure we mean.
    expiries = sorted(df["expiration"].dropna().unique())
    if not expiries:
        return {}, "no expirations in range for {}".format(symbol.upper())
    exp_df = df[df["expiration"] == expiries[0]].copy()

    cands = build_candidates(exp_df, spot, _exit_rules(config_path),
                             capital_usd=capital_usd)
    if not cands:
        return {}, "chain for {} {} had no usable quotes".format(
            symbol.upper(), expiries[0])
    return cands, None
