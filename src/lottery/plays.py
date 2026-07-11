"""Play-archetype classifier for the lottery board.

Turns a far-OTM candidate row into a named, direction-aware "play" so the board
reads like trades a human recognises (BOUNCE / CRASH / BREAKOUT / CATALYST /
SQUEEZE) instead of an anonymous delta soup. Uses only columns the scan pipeline
already computes; every field is read defensively and a row that fits no clean
archetype falls back to LONGSHOT.

The contract's own side matters: a call on an oversold, high-rvol name is a
BOUNCE; a put on the same name is not. So classification takes the option type
into account, not just the underlying's technical state.
"""
from __future__ import annotations

from typing import Any, Optional

BOUNCE = "BOUNCE"
CRASH = "CRASH"
BREAKOUT = "BREAKOUT"
CATALYST = "CATALYST"
SQUEEZE = "SQUEEZE"
LONGSHOT = "LONGSHOT"


def _f(row: Any, *keys, default=None) -> Optional[float]:
    for k in keys:
        try:
            if hasattr(row, "get"):
                v = row.get(k, None)
            elif k in getattr(row, "index", []):
                v = row[k]
            else:
                v = None
        except Exception:
            v = None
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv == fv:  # not NaN
            return fv
    return default


def _s(row: Any, *keys, default="") -> str:
    for k in keys:
        try:
            v = row.get(k) if hasattr(row, "get") else (row[k] if k in getattr(row, "index", []) else None)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v
    return default


def _is_call(row: Any) -> bool:
    return _s(row, "type", "option_type", "opt_type", default="call").lower().startswith("c")


def classify_play(row: Any, catalyst_dte_max: float = 45.0, cheap_iv_rank: float = 0.40) -> str:
    """Return the play archetype for one option row (see module docstring)."""
    call = _is_call(row)
    rsi = _f(row, "rsi_14", "rsi")
    adx = _f(row, "adx_14", "adx")
    rvol = _f(row, "rvol", default=1.0)
    spot = _f(row, "underlying", "spot")
    sma20 = _f(row, "sma_20")
    sma50 = _f(row, "sma_50")
    hi20 = _f(row, "high_20d", "hh_20", "high_20")
    si = _f(row, "short_interest", default=0.0)
    whale = bool(_f(row, "Unusual_Whale", default=0.0))
    iv_rank = _f(row, "iv_rank", "iv_rank_score", "iv_percentile_30")
    earn_dte = _f(row, "earnings_dte", "days_to_earnings", "catalyst_dte")
    earn_play = _s(row, "Earnings Play").upper() == "YES"

    has_event = (earn_dte is not None and 0 <= earn_dte <= catalyst_dte_max) or earn_play

    # CATALYST: an event in the option's life AND IV still cheap enough that the
    # long isn't a pre-crush overpay. Rich IV into an event is a trap, not a play.
    if has_event and (iv_rank is None or iv_rank <= cheap_iv_rank):
        return CATALYST

    # SQUEEZE: high short interest + unusual call flow — calls only.
    if call and si is not None and si >= 0.20 and (whale or (rvol and rvol >= 2.0)):
        return SQUEEZE

    # BREAKOUT: trending up, above the 50d, pressing the 20d high — calls only.
    if call and adx is not None and adx > 25 and spot and sma50 and spot > sma50:
        near_high = hi20 is not None and spot >= 0.98 * hi20
        if near_high or (rsi is not None and rsi >= 60):
            return BREAKOUT

    # BOUNCE: oversold, stretched below the 20d, on elevated volume — calls only.
    if call and rsi is not None and rsi <= 32:
        stretched = spot and sma20 and spot <= sma20
        if stretched or (rvol and rvol >= 1.3):
            return BOUNCE

    # CRASH: rolling over / breaking down — puts only.
    if not call:
        rolling_over = rsi is not None and rsi >= 68
        breakdown = spot and sma20 and spot < sma20 and rvol and rvol >= 1.3
        if rolling_over or breakdown:
            return CRASH

    return LONGSHOT


def classify_plays(df, catalyst_dte_max: float = 45.0, cheap_iv_rank: float = 0.40):
    """Vectorised wrapper: returns a list of play labels aligned to df rows."""
    return [classify_play(df.iloc[i], catalyst_dte_max, cheap_iv_rank) for i in range(len(df))]


__all__ = [
    "BOUNCE", "CRASH", "BREAKOUT", "CATALYST", "SQUEEZE", "LONGSHOT",
    "classify_play", "classify_plays",
]
