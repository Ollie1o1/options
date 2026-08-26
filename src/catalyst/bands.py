"""Time bands and deep-fetch budget allocation.

Pure arithmetic — no rendering, no network, no sibling imports.

Bands say WHEN, never how good. The pre-registered rank race of 2026-08-24
found no candidate key ordered a board better than chance, and the catalyst
backtest of 2026-08-26 found no evidence that funded-through, endpoint
amendments or phase predict returns. Time-to-event and date precision are
properties of the DATE, not judgements about the trial — which is the only
reason it is legitimate to let them drive what a reader sees first.

Allocation exists because `collapsed[:limit]` was a lie of omission. Measured
2026-08-26: a 6-month window deep-fetched 40 names, every one of which fell
between 2026-08-30 and 2026-10-31, and 57 later names were never fetched. The
board answered a 2-month question while presenting itself as a 6-month one.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

FIRM = "FIRM"
NEXT_30 = "NEXT_30"
D31_90 = "D31_90"
BEYOND_90 = "BEYOND_90"

#: The bands a trial row can land in. FIRM is excluded: PDUFA dates come from
#: an 8-K, are day-precision and are never estimates, so they render in their
#: own section rather than competing with estimated completion dates.
TRIAL_BANDS: Tuple[str, str, str] = (NEXT_30, D31_90, BEYOND_90)

BAND_TITLES: Dict[str, str] = {
    FIRM: "REGULATORY DECISIONS — FDA decision dates, firm",
    NEXT_30: "NEXT 30 DAYS — estimated primary completion",
    D31_90: "31–90 DAYS",
    BEYOND_90: "BEYOND 90 DAYS",
}

SHORT_TITLES: Dict[str, str] = {
    FIRM: "FDA",
    NEXT_30: "NEXT 30 DAYS",
    D31_90: "31–90 DAYS",
    BEYOND_90: "BEYOND 90",
}

NEAR_DAYS = 30
MID_DAYS = 90


def _as_date(text: str) -> dt.date:
    """Parse an event date, resolving month precision to the 15th.

    "2026-09" means sometime in September. The 15th is the central estimate:
    first-of-month would systematically pull every month-precision row one
    band NEARER than the source supports, end-of-month one band further. The
    convention is stated in the board's legend because it decides the band.
    """
    parts = text.split("-")
    if len(parts) == 2:
        return dt.date(int(parts[0]), int(parts[1]), 15)
    return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))


def days_until(event_date: str, today: str) -> int:
    """Whole days from `today` to `event_date`. Negative if already elapsed."""
    return (_as_date(event_date) - dt.date.fromisoformat(today)).days


def band_for(event_date: str, today: str) -> str:
    """Which trial band a date falls in.

    An elapsed date bands as NEXT_30 rather than crashing or sorting to the
    far end — the sweep can return a date that slipped past today between
    fetch and render.
    """
    days = days_until(event_date, today)
    if days <= NEAR_DAYS:
        return NEXT_30
    if days <= MID_DAYS:
        return D31_90
    return BEYOND_90


def allocate(counts: Dict[str, int], budget: int,
             floor: int = 5) -> Dict[str, int]:
    """Split a deep-fetch budget across trial bands.

    Order of claims: the near band in full (it is small and it is what a
    reader came for), then a floor for each later band so neither is starved,
    then a proportional top-up of what remains. Never exceeds `budget`, never
    allocates a band more names than it has.
    """
    out: Dict[str, int] = {band: 0 for band in TRIAL_BANDS}
    if budget <= 0:
        return out

    remaining = budget
    out[NEXT_30] = min(counts.get(NEXT_30, 0), remaining)
    remaining -= out[NEXT_30]

    later = [band for band in (D31_90, BEYOND_90) if counts.get(band, 0) > 0]
    for band in later:
        take = min(floor, counts[band], remaining)
        out[band] += take
        remaining -= take

    short = {band: counts[band] - out[band] for band in later}
    total_short = sum(short.values())
    if remaining > 0 and total_short > 0:
        for band in later:
            out[band] += min(int(remaining * short[band] / total_short),
                             short[band])
        # Integer division leaves a remainder; hand it out greedily so the
        # budget is actually spent rather than quietly rounded away.
        for band in later:
            spare = budget - sum(out.values())
            if spare <= 0:
                break
            out[band] += min(spare, counts[band] - out[band])
    return out
