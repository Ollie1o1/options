"""Capacity: what a real account could actually have deployed.

Per-trade edge is not the same as money made. An index put spread in this repo's
earlier research showed a profit factor of 4.29 and, sized responsibly, produced
about **0.3% CAGR on 31 trades in three years** — the per-trade number concealed
a capacity wall.

So capacity is reported beside edge, never after it. A strategy that is
wonderful per trade and can only be entered eleven times a year is a hobby, not
an allocation.
"""
from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def _d(date: str) -> _dt.date:
    return _dt.date.fromisoformat(str(date)[:10])


def apply_capacity(trades: Sequence[Any], max_concurrent: int,
                   max_capital: float) -> Tuple[List[Any], Dict[str, int]]:
    """Keep only the trades an account of this size could actually have held.

    Walks entries in time order holding a book of live positions. A candidate is
    rejected when accepting it would breach either the concurrency cap or the
    capital cap. Rejections are counted, because "the strategy works but you
    could not have been in it" is a result, not a footnote.
    """
    stats = {"kept": 0, "dropped_concurrency": 0, "dropped_capital": 0}
    live: List[Any] = []
    kept: List[Any] = []

    for t in sorted(trades, key=lambda x: str(x.entry_date)):
        entry = _d(t.entry_date)
        live = [o for o in live
                if o.exit_date is None or _d(o.exit_date) > entry]
        if len(live) >= max_concurrent:
            stats["dropped_concurrency"] += 1
            continue
        deployed = sum(float(o.capital_at_risk) for o in live)
        if deployed + float(t.capital_at_risk) > max_capital:
            stats["dropped_capital"] += 1
            continue
        live.append(t)
        kept.append(t)
        stats["kept"] += 1
    return kept, stats


def capacity_stats(trades: Sequence[Any], max_capital: float) -> Dict[str, Any]:
    """Trades per year, peak deployment, and return on the capital committed."""
    closed = [t for t in trades if t.exit_date]
    if not closed:
        return {"trades_per_year": 0.0, "max_concurrent": 0,
                "peak_deployed": 0.0, "total_pnl": 0.0,
                "return_on_cap": 0.0, "years": 0.0}

    first = min(_d(t.entry_date) for t in closed)
    last = max(_d(t.exit_date) for t in closed)
    years = max((last - first).days / 365.25, 1e-9)

    # Peak simultaneous exposure, walked as an event timeline.
    events: Dict[_dt.date, float] = defaultdict(float)
    counts: Dict[_dt.date, int] = defaultdict(int)
    for t in closed:
        events[_d(t.entry_date)] += float(t.capital_at_risk)
        events[_d(t.exit_date)] -= float(t.capital_at_risk)
        counts[_d(t.entry_date)] += 1
        counts[_d(t.exit_date)] -= 1
    running = peak = 0.0
    n_running = n_peak = 0
    for day in sorted(events):
        running += events[day]
        n_running += counts[day]
        peak = max(peak, running)
        n_peak = max(n_peak, n_running)

    total = sum(float(t.pnl or 0.0) for t in closed)
    return {
        "trades_per_year": len(closed) / years,
        "max_concurrent": n_peak,
        "peak_deployed": round(peak, 2),
        "total_pnl": round(total, 2),
        # Against the account, not against the peak: idle capital is a real cost.
        "return_on_cap": round(total / max_capital / years, 4) if max_capital else 0.0,
        "years": round(years, 2),
    }
