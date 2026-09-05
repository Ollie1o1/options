"""outlook's factor construction, applied point-in-time to an arbitrary
(potentially single-name) universe instead of the 16 sector/asset ETFs it
was built and backtested on (`docs/OUTLOOK_FINDINGS.md`: bullish calls 66-72%
right, relative IC +0.05-0.08 — on that 16-instrument universe only).

This is a TRANSFER TEST, not a re-validation: the factor formulas
(`src/outlook/factors.py`) and the cross-sectional z-score/composite logic
(`src/outlook/engine.py::rank_universe`) are reused UNCHANGED. Nothing here
is re-tuned for the wider universe — the whole point of
`docs/PREREG_OUTLOOK_FEATURE_20260905.md` is testing whether an already-frozen
construction, built and validated on 16 broad, comparatively low-idiosyncratic
-vol instruments, carries any skill onto individual stocks, which are noisier.
It may not. That is a real, informative possible answer.

Point-in-time, explicitly: for a target date D, a symbol's factor row uses
only that symbol's OWN close history up to and including D — never a future
close, and never another symbol's calendar position (a shared positional
index across series with different gap patterns would silently misalign
`relative_strength`'s benchmark lookback; this looks up both series
independently BY DATE instead).
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Sequence, Tuple

from src.outlook.engine import load_outlook_config, rank_universe
from src.outlook.factors import mom_12_1, reversal_1m, trend_score

BENCH = "SPY"
RELATIVE_STRENGTH_LOOKBACK = 63


def _index_asof(sorted_dates: Sequence[str], target: str) -> Optional[int]:
    """Position of the latest date <= target — point-in-time, never a future
    index. None if `target` is before every date on file."""
    i = bisect.bisect_right(sorted_dates, target) - 1
    return i if i >= 0 else None


def _series(closes_by_date: Dict[str, float]) -> Tuple[List[str], List[float]]:
    dates = sorted(closes_by_date)
    return dates, [closes_by_date[d] for d in dates]


def _factor_row(sdates: Sequence[str], scloses: Sequence[float],
                bdates: Sequence[str], bcloses: Sequence[float],
                date: str) -> Optional[Dict[str, Optional[float]]]:
    st, bt = _index_asof(sdates, date), _index_asof(bdates, date)
    if st is None or bt is None:
        return None
    row: Dict[str, Optional[float]] = {
        "mom_12_1": mom_12_1(scloses, st),
        "trend_score": trend_score(scloses, st),
        "reversal_1m": reversal_1m(scloses, st),
        "mkt_trend": trend_score(bcloses, bt),
    }
    lb = RELATIVE_STRENGTH_LOOKBACK
    if (st >= lb and bt >= lb
            and scloses[st - lb] > 0 and bcloses[bt - lb] > 0):
        inst = scloses[st] / scloses[st - lb] - 1.0
        b = bcloses[bt] / bcloses[bt - lb] - 1.0
        row["relative_strength"] = inst - b
    else:
        row["relative_strength"] = None
    return row


def composite_lookup(
    closes: Dict[str, Dict[str, float]], dates: Sequence[str],
    cfg: Optional[Dict] = None,
) -> Dict[Tuple[str, str], float]:
    """{(symbol, date): composite_score} for every symbol/date where the
    cross-sectional rank on that date could be computed.

    A date is skipped entirely if fewer than 2 symbols have enough history —
    z-scoring needs a cross-section, not a single point. A symbol/date with
    insufficient OWN history (too early, missing from `closes`) is simply
    absent from the output, matching how every other unmeasurable feature in
    this repo is reported: missing, never zero-filled.
    """
    cfg = cfg or load_outlook_config()
    series = {sym: _series(c) for sym, c in closes.items() if c}
    if BENCH not in series:
        return {}
    bdates, bcloses = series[BENCH]

    out: Dict[Tuple[str, str], float] = {}
    for date in dates:
        features_by_ticker: Dict[str, Dict[str, Optional[float]]] = {}
        for sym, (sdates, scloses) in series.items():
            if sym == BENCH:
                continue
            row = _factor_row(sdates, scloses, bdates, bcloses, date)
            if row is not None and any(v is not None for v in row.values()):
                features_by_ticker[sym] = row
        if len(features_by_ticker) < 2:
            continue
        for r in rank_universe(features_by_ticker, cfg):
            out[(r["ticker"], date)] = r["score"]
    return out
