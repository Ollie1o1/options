"""Which listed expirations a scan fetches — spread across the DTE window, not bunched at its front.

The screener can afford only N chain requests per symbol (Yahoo rate limits), and it
used to spend them on the N *nearest* expirations inside the DTE window. On names that
list weeklies or dailies that left the back of the window unsearched. Measured on
`data/candidates.db` (scans 2026-11-10 .. 11-24, Credit Spreads mode, window 7-45 DTE,
N = 4): SPY / QQQ / IWM were only ever priced at 9-15 DTE, and weekly single names
stopped at ~24 DTE, so the 30-45 DTE region — where per-contract friction is lowest —
was never on the board.

`select_expirations(..., method="spread")` spends the same N requests on expirations
spaced evenly across the expirations actually listed in the window (first and last
always included). Near each target it prefers a standard monthly (third-Friday)
expiration: measured on `data/chain_archive.db` (2026-09-01 .. 11-18, |delta|
0.15-0.40, 37 symbols), a monthly quoted a median 1.6 percentage points of mid tighter
than the average of the weeklies one week either side of it, and was tighter in
93.5% of 646 such comparisons (36 of 37 symbols).

`method="nearest"` is the legacy behaviour, kept for reproducing old scans
(`config.json` → `expiry_selection: "nearest"`).
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional, Sequence, Tuple

METHODS = ("spread", "nearest")
DEFAULT_METHOD = "spread"


def is_standard_monthly(day: dt.date) -> bool:
    """Third Friday of the month (the standard monthly equity-option expiration)."""
    return day.weekday() == 4 and 15 <= day.day <= 21


def _parse(expirations: Sequence[str], today: dt.date) -> List[Tuple[int, str, dt.date]]:
    out: List[Tuple[int, str, dt.date]] = []
    for e in expirations:
        try:
            d = dt.date.fromisoformat(str(e)[:10])
        except ValueError:
            continue
        out.append(((d - today).days, str(e), d))
    out.sort()
    return out


def select_expirations(
    expirations: Sequence[str],
    *,
    n: int,
    today: Optional[dt.date] = None,
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
    method: str = DEFAULT_METHOD,
    prefer_monthly: bool = True,
) -> List[str]:
    """Pick at most `n` expirations (returned in date order).

    Window: expirations with min_dte <= DTE <= max_dte. If none are listed inside it,
    falls back to the nearest `n` listed (unchanged legacy behaviour — the caller's
    DTE filter downstream still applies).
    """
    if n <= 0:
        return []
    today = today or dt.date.today()
    parsed = _parse(expirations, today)
    lo = min_dte if min_dte is not None else -10_000
    hi = max_dte if max_dte is not None else 10_000
    if method not in METHODS:
        raise ValueError(f"expiry selection method must be one of {METHODS}, got {method!r}")
    window = [p for p in parsed if lo <= p[0] <= hi]
    if not window:
        return [p[1] for p in parsed[:n]]
    if method == "nearest" or len(window) <= n:
        return [p[1] for p in window[:n]]
    if n == 1:
        return [window[0][1]]

    first, last = window[0][0], window[-1][0]
    step = (last - first) / (n - 1)
    tol = step / 2.0
    used: set = set()
    picked: List[Tuple[int, str, dt.date]] = []
    for i in range(n):
        target = first + i * step
        free = [p for p in window if p[1] not in used]
        if not free:
            break
        near = [p for p in free if abs(p[0] - target) <= tol]
        monthlies = [p for p in near if is_standard_monthly(p[2])] if prefer_monthly else []
        # The window's two ends are always covered exactly; interior targets may
        # move to a nearby monthly. Ties break toward the earlier expiration.
        if monthlies and 0 < i < n - 1:
            pool = monthlies
        else:
            pool = free
        choice = min(pool, key=lambda p: (abs(p[0] - target), p[0]))
        used.add(choice[1])
        picked.append(choice)
    picked.sort()
    return [p[1] for p in picked]


__all__ = ["METHODS", "DEFAULT_METHOD", "is_standard_monthly", "select_expirations"]
