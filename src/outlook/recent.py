"""Recent price action for the outlook panel — context, never a scored factor.

Deliberately separate from `factors.py`. Everything in that module is z-scored
and weighted into the composite; nothing here ever is. The outlook score is
validated as it stands (relative IC +0.05 at 2mo to +0.08 at 3mo, see
docs/OUTLOOK_FINDINGS.md), and these numbers exist only to stop the panel
asserting an instrument "has been doing well" when it has just fallen.

The need is concrete: on 2026-07-28 the panel ranked SMH first with "12m
momentum +, trend +" while SMH was -13.0% over 21 days. 65% of the composite's
weight cannot see the last month by construction — mom_12_1 ends its window a
month back, trend_score measures against a 200-day average — and reversal_1m,
the one factor that does see it, is negated, so the selloff read as *bullish*.

Same conventions as factors.py: pure functions over close series, reading only
up to `t`, returning None on insufficient history, never looking ahead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

TRADING_WEEK = 5
TRADING_MONTH = 21

# Measured, not guessed: the 10th percentile of 21-day excess returns vs SPY
# across the outlook universe (n=32,256 instrument-months, mean -0.10pp,
# stdev 4.52pp). The flag therefore fires on the worst decile of
# instrument-months. See the design doc's Threshold section.
DEFAULT_LAG_THRESHOLD_PP = -5.2


def _at(closes: Sequence[float], t: Optional[int]) -> int:
    return (len(closes) - 1) if t is None else t


def trailing_return(closes: Sequence[float], t: Optional[int] = None,
                    lookback: int = TRADING_MONTH) -> Optional[float]:
    """Simple return over the trailing `lookback` bars ending at `t`."""
    t = _at(closes, t)
    if t < lookback or t >= len(closes):
        return None
    prev = closes[t - lookback]
    if prev <= 0:
        return None
    return closes[t] / prev - 1.0


def recent_context(closes: Sequence[float], bench: Sequence[float],
                   t: Optional[int] = None,
                   lag_threshold_pp: float = DEFAULT_LAG_THRESHOLD_PP,
                   ) -> Dict[str, Any]:
    """Trailing 5d/21d return and excess vs benchmark, in percentage points.

    `lagging` is deliberately RELATIVE: in a market-wide drawdown every excess
    return sits near zero, so the panel stays clean instead of flagging every
    row at once.
    """
    t = _at(closes, t)
    out: Dict[str, Any] = {"ret_5d": None, "ret_21d": None,
                           "excess_5d": None, "excess_21d": None,
                           "lagging": False}
    if t >= len(bench):
        return out
    for key, lb in (("5d", TRADING_WEEK), ("21d", TRADING_MONTH)):
        inst = trailing_return(closes, t, lb)
        b = trailing_return(bench, t, lb)
        out[f"ret_{key}"] = inst
        if inst is not None and b is not None:
            out[f"excess_{key}"] = (inst - b) * 100.0
    ex = out["excess_21d"]
    out["lagging"] = ex is not None and ex <= lag_threshold_pp
    return out


__all__ = ["trailing_return", "recent_context",
           "DEFAULT_LAG_THRESHOLD_PP", "TRADING_WEEK", "TRADING_MONTH"]
