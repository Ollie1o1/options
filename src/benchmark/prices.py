"""Stitched SPY price series, from two read-only sources.

`data/equity_ohlcv.db::ohlcv` (ticker='SPY') is a settled daily close, but
stale past 2026-08-14. `data/chain_archive.db::chain_snapshots` (symbol='SPY')
carries an intraday `spot` alongside each archived option chain and is
current, but only exists on the 52 dates the archive happened to snapshot a
chain. Neither source alone covers the closed book's date range, so the two
must be stitched — and validated on every run, not trusted once and
forgotten: a source that started disagreeing would silently poison every
downstream return calculation.
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

# Above this threshold on the median ABSOLUTE relative difference between the
# two sources on their common dates, refuse to stitch rather than silently
# blend disagreeing data. Measured on the real databases (2026-09-28, 28
# common dates): median +0.000%, mean +0.053%, max 1.21% — comfortably under
# this bar. 0.50% is 40x the measured median, so this fires on a real source
# defect, not on ordinary intraday-vs-close noise.
MAX_MEDIAN_ABS_DIFF = 0.0050

# A trade whose entry or exit date has no SPY price within this many calendar
# days is dropped rather than matched further afield. Measured on the real
# 968 closed trades: 827 (85%) match exactly, all 968 (100%) match within
# this tolerance.
DEFAULT_TOLERANCE_DAYS = 4


def _ro(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_ohlcv_close(ohlcv_db_path: str, ticker: str = "SPY") -> Dict[str, float]:
    """`{date: close}` for `ticker` from `equity_ohlcv.db::ohlcv`."""
    con = _ro(ohlcv_db_path)
    try:
        rows = con.execute(
            "SELECT date, close FROM ohlcv WHERE ticker = ? AND close IS NOT NULL",
            (ticker,)).fetchall()
    finally:
        con.close()
    return {d: float(c) for d, c in rows}


def load_archive_spot(archive_db_path: str, symbol: str = "SPY") -> Dict[str, float]:
    """`{snap_date: spot}` for `symbol` from `chain_archive.db::chain_snapshots`.

    Every observed date carries exactly one distinct `spot` across all of that
    date's contract rows (verified on the real archive: 52/52 dates), so
    `MIN(spot)` recovers it without needing a join against a single "the"
    contract row.
    """
    con = _ro(archive_db_path)
    try:
        rows = con.execute(
            """SELECT snap_date, MIN(spot) FROM chain_snapshots
                WHERE symbol = ? AND spot IS NOT NULL
                GROUP BY snap_date""",
            (symbol,)).fetchall()
    finally:
        con.close()
    return {d: float(s) for d, s in rows}


@dataclass(frozen=True)
class StitchValidation:
    """The overlap check between the two SPY sources, computed fresh each run."""
    n_common: int
    median_diff: Optional[float]       # signed, (archive - ohlcv) / ohlcv
    mean_diff: Optional[float]
    max_abs_diff: Optional[float]
    median_abs_diff: Optional[float]
    threshold: float
    ok: bool


def validate_stitch(ohlcv: Dict[str, float], archive: Dict[str, float],
                     threshold: float = MAX_MEDIAN_ABS_DIFF) -> StitchValidation:
    """Compare the two sources on their common dates. Never mutates either input.

    Raises `ValueError` when the median absolute relative difference exceeds
    `threshold` — refusing to stitch rather than silently blending sources
    that have started to disagree.
    """
    common = sorted(set(ohlcv) & set(archive))
    if not common:
        v = StitchValidation(0, None, None, None, None, threshold, ok=True)
        return v
    diffs = [(archive[d] - ohlcv[d]) / ohlcv[d] for d in common if ohlcv[d] != 0.0]
    if not diffs:
        v = StitchValidation(len(common), None, None, None, None, threshold, ok=True)
        return v
    abs_diffs = [abs(x) for x in diffs]
    median_diff = statistics.median(diffs)
    mean_diff = statistics.mean(diffs)
    max_abs_diff = max(abs_diffs)
    median_abs_diff = statistics.median(abs_diffs)
    ok = median_abs_diff <= threshold
    v = StitchValidation(len(common), median_diff, mean_diff, max_abs_diff,
                          median_abs_diff, threshold, ok=ok)
    if not ok:
        raise ValueError(
            f"SPY stitch validation failed: median absolute difference "
            f"{median_abs_diff:.4%} across {len(common)} common dates exceeds "
            f"the {threshold:.4%} threshold. Refusing to stitch disagreeing "
            f"sources.")
    return v


def stitch_spy_series(ohlcv_db_path: str, archive_db_path: str,
                       threshold: float = MAX_MEDIAN_ABS_DIFF
                       ) -> "Tuple[Dict[str, float], StitchValidation]":
    """The combined `{date: price}` series, plus the validation that gated it.

    `ohlcv.close` wins on any date both sources cover — it is a settled
    close, `spot` an intraday snapshot. `spot` only extends the series past
    `ohlcv`'s last date (2026-08-14 as measured) in practice, since every
    earlier archive date is also an `ohlcv` date and is overridden.

    Raises via `validate_stitch` if the two sources disagree beyond
    `threshold`; the caller gets no series in that case, deliberately —
    there is no safe partial stitch to fall back to. Returns the
    `StitchValidation` alongside the series so a caller (the CLI's
    manifest) never has to recompute it.
    """
    ohlcv = load_ohlcv_close(ohlcv_db_path)
    archive = load_archive_spot(archive_db_path)
    validation = validate_stitch(ohlcv, archive, threshold=threshold)
    stitched = dict(archive)
    stitched.update(ohlcv)  # ohlcv wins on overlapping dates
    return stitched, validation


@dataclass(frozen=True)
class PriceMatch:
    """The result of matching one target date against a price series."""
    target_date: str
    matched_date: str
    price: float
    offset_days: int   # 0 = exact date match


def nearest_price(series: Dict[str, float], target_date: str,
                   tolerance_days: int = DEFAULT_TOLERANCE_DAYS
                   ) -> Optional[PriceMatch]:
    """The price on `target_date`, or the nearest date within `tolerance_days`.

    Exact match is always preferred. When there is no exact match, offsets
    are searched in increasing order (1, 2, 3, ... `tolerance_days`), and at
    each offset the earlier date is tried before the later one — an
    arbitrary but deterministic tie-break, so the same input always resolves
    to the same matched date. Returns `None` when nothing is found within
    tolerance; the caller must count that as a drop, not silently skip it.
    """
    if target_date in series:
        return PriceMatch(target_date, target_date, series[target_date], 0)
    d = date.fromisoformat(target_date[:10])
    for offset in range(1, tolerance_days + 1):
        for candidate in (d - timedelta(days=offset), d + timedelta(days=offset)):
            cs = candidate.isoformat()
            if cs in series:
                return PriceMatch(target_date, cs, series[cs], offset)
    return None
