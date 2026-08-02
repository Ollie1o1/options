"""Synthetic call return under the ladder, in percent of premium paid.

Entry is 90 calendar DTE against a 42-trading-day (~59 calendar) hold, and that
gap is the whole point: a ~60 DTE call held to the horizon expires AT the
negative-median endpoint and pays intrinsic only, while a 90 DTE call sold on
day 59 still carries ~30 days of time value. On a flat losing path ATM value
scales roughly as sqrt(T), so the loser recovers about 58% of remaining time
value — worst-case decay near -42% of premium instead of -100%. Since roughly
half the cohort never touches +20%, that is worth more expected value than any
tuning of the winner leg.

Two variants because the gate reads them asymmetrically: GO is judged on the
conservative one (intrinsic only) so the authorisation is hard to earn, STOP on
the central one (Black-Scholes at remaining tenor) so a strategy is never killed
by its own conservatism buffer.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from src.squeeze.sleeve import ladder
from src.utils import bs_call

ENTRY_DTE_DAYS = 90       # calendar days at entry
RISK_FREE = 0.04
TRADING_DAYS = 252
CALENDAR_PER_TRADING = 365.0 / 252.0


def synthetic_call_return(path: Sequence[float], spot0: float, sigma_d: float,
                          iv: float, horizon_bars: int = ladder.HOLD_BARS,
                          entry_dte_days: int = ENTRY_DTE_DAYS,
                          strike_mult: float = 1.0,
                          variant: str = "central") -> Optional[float]:
    """Fractional return on premium for one synthetic call trade.

    ``sigma_d`` is the trailing daily realised vol used to scale the ladder
    (the study's own normaliser). ``iv`` prices the option and is deliberately
    separate: the gap between them is exactly what the live measurement is for.
    """
    if not path or spot0 <= 0 or sigma_d <= 0 or iv <= 0:
        return None
    if not (math.isfinite(spot0) and math.isfinite(sigma_d) and math.isfinite(iv)):
        return None

    strike = spot0 * strike_mult
    t_entry = entry_dte_days / 365.0
    premium = float(bs_call(spot0, strike, t_entry, RISK_FREE, iv))
    if not math.isfinite(premium) or premium <= 0:
        return None

    sigma_h = sigma_d * math.sqrt(horizon_bars)
    fills = ladder.simulate(path, spot0, sigma_h, hold_bars=horizon_bars)

    proceeds = 0.0
    for fill in fills:
        days_left = entry_dte_days - fill.bar * CALENDAR_PER_TRADING
        intrinsic = max(0.0, fill.price - strike)
        if variant == "conservative" or days_left <= 0:
            value = intrinsic
        else:
            value = float(bs_call(fill.price, strike, days_left / 365.0,
                                  RISK_FREE, iv))
            if not math.isfinite(value):
                value = intrinsic
            value = max(value, intrinsic)
        proceeds += fill.fraction * value

    return proceeds / premium - 1.0
