"""Join the squeeze panel, the price book, and shares outstanding into the row
shape ``dhist.compute`` consumes.

Paths are zero-copy NumPy VIEWS into the arrays ``PriceBook`` already holds.
Materialising a few hundred thousand of them as Python lists costs about 2.4GB;
as views it is roughly 30MB, and the price book has to stay resident anyway.

``spot0`` is taken from the panel's own ``spot`` — the close at the entry index —
and never from the first bar of the path, which sits one day later. That
distinction is invisible in any output: strikes would still be struck and
ladders would still fire, on a trade entered a day after the one intended.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from src.squeeze.sleeve import cohort


def build(records: Sequence[dict], book, shares, horizon: int
          ) -> Tuple[List[dict], Dict[str, int]]:
    """``dhist`` rows for one horizon, plus counts of what did not make it."""
    # short_path is split by arm because the censoring is not arm-neutral: a
    # heavily-shorted microcap that just ran +10% is more likely to delist
    # mid-window than a bottom-half-SI control, and one shared counter cannot
    # show a differential. The total is kept for compatibility.
    stats = {"ungradeable": 0, "short_path": 0,
             "short_path_treated": 0, "short_path_control": 0,
             "treated": 0, "control": 0, "excluded": 0}

    by_date: Dict[str, List[dict]] = {}
    for rec in records:
        if rec.get("si_ratio") is None:
            stats["ungradeable"] += 1
            continue
        by_date.setdefault(rec["date"], []).append(rec)

    out: List[dict] = []
    for date in sorted(by_date):
        day = by_date[date]
        arms = cohort.label(day)
        for rec, arm in zip(day, arms):
            if arm is None:
                stats["excluded"] += 1
                continue
            row = _row(rec, arm, book, shares, horizon)
            if row is None:
                stats["short_path"] += 1
                stats["short_path_" + arm] += 1
                continue
            stats[arm] += 1
            out.append(row)
    return out, stats


def _row(rec: dict, arm: str, book, shares, horizon: int):
    sym = rec["symbol"]
    closes = book._close.get(sym)
    if closes is None:
        return None
    i = int(rec["entry_index"])
    if i + horizon >= closes.size:
        return None
    path = closes[i + 1: i + 1 + horizon]
    if path.size != horizon:
        return None

    spot = float(rec["spot"])
    shares_out = shares.get(sym, rec["date"])
    if not shares_out or spot <= 0:
        return None
    sigma_d = float(rec["sigma_d"])

    return {
        "date": rec["date"], "symbol": sym, "arm": arm,
        "rv": sigma_d * math.sqrt(252.0),
        "log_mcap": math.log(float(shares_out) * spot),
        "log_price": math.log(spot),
        # cohort.label gives arm None to any row without a ret_5d, and only
        # labelled rows reach here — so this is never None, and no value is
        # ever fabricated for a matching covariate.
        "ret_5d": float(rec["ret_5d"]),
        "sigma_d": sigma_d,
        "spot0": spot,
        "path": path,
    }
