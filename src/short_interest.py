"""Enriched short-interest detail derived from yfinance ``ticker.info``.

The screener already reads ``shortPercentOfFloat`` for scoring, but a quant
wants the two fields that turn a level into a read:

    days to cover (shortRatio)  — how crowded the exit is
    month-over-month trend      — is the short building or covering

Both come free from the same already-cached ``info`` dict, so this is a pure
transform with no extra network calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_TREND_TOL = 0.05  # ±5% MoM change counts as "flat"


@dataclass
class ShortInterest:
    pct_float: Optional[float]        # fraction 0-1
    days_to_cover: Optional[float]    # shortRatio (days)
    pct_float_prior: Optional[float]  # prior-month fraction, if derivable
    shares_short: Optional[int]
    trend: Optional[str]              # "rising" | "falling" | "flat" | None


def _f(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_pct(value: Optional[float]) -> Optional[float]:
    """Normalize a short-percent that may arrive as 0-1 or 0-100 into 0-1."""
    if value is None:
        return None
    if not (0 <= value <= 200):
        return None
    return value / 100.0 if value > 1 else value


def short_interest_detail(info: dict) -> ShortInterest:
    info = info or {}
    pct_float = _norm_pct(_f(info.get("shortPercentOfFloat")))
    days_to_cover = _f(info.get("shortRatio"))
    shares_short = _f(info.get("sharesShort"))
    shares_prior = _f(info.get("sharesShortPriorMonth"))
    float_shares = _f(info.get("floatShares"))

    pct_prior = None
    if shares_prior is not None and float_shares and float_shares > 0:
        pct_prior = shares_prior / float_shares

    trend = None
    if shares_short is not None and shares_prior and shares_prior > 0:
        ratio = shares_short / shares_prior
        if ratio > 1 + _TREND_TOL:
            trend = "rising"
        elif ratio < 1 - _TREND_TOL:
            trend = "falling"
        else:
            trend = "flat"

    return ShortInterest(
        pct_float=pct_float,
        days_to_cover=days_to_cover,
        pct_float_prior=pct_prior,
        shares_short=int(shares_short) if shares_short is not None else None,
        trend=trend,
    )


def format_short_interest(si: ShortInterest) -> str:
    """One-line factual SI summary; empty string if nothing is known."""
    if si is None or (si.pct_float is None and si.days_to_cover is None):
        return ""
    parts = []
    if si.pct_float is not None:
        parts.append(f"{si.pct_float * 100:.1f}% of float")
    if si.days_to_cover is not None:
        parts.append(f"{si.days_to_cover:.1f}d to cover")
    if si.trend:
        arrow = {"rising": "↑", "falling": "↓", "flat": "→"}.get(si.trend, "")
        parts.append(f"{arrow} {si.trend} MoM")
    return "  Short interest: " + "  ·  ".join(parts)
