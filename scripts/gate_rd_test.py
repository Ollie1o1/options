"""scripts/gate_rd_test.py

Runs the regression discontinuity design frozen in
docs/PREREG_GATE_RD_20260902.md: does the candidate_verdict friction gate
(round_trip_pct > 0.25 -> refused) separate genuinely worse trades from
genuinely better ones? Every design parameter below (cutoff, bandwidth,
horizon, clustering, estimator, decision rule) is LOCKED by that document;
this module implements it, it does not choose it.

Reporting only. Reads data/candidates.db, writes nothing anywhere.

THE RUNNING VARIABLE IS RECOMPUTED, NOT READ FROM THE STORED COLUMN.
candidates.round_trip_pct is NULL on 100% of refused_by='friction' rows (a
mode-dependent recording gap: the modes that produce a friction refusal never
persist the flat column, and the modes that persist it never produce one).
features_json carries every per-leg quote candidate_verdict.verdict_for
needs, so this recomputes from there instead - verified in the prereg to
reproduce the recorded gate decision exactly on a sample.

ENTRY PRICING AND OUTCOMES REUSE candidate_marks, NOT A REIMPLEMENTATION.
entry_price_for/_blob/pnl_pct already compute exactly the quantities this
design needs (a structure's net entry price at the limit fill, and a return
derived from the sign of that entry) - see that module's own docstring for
why those conventions are the ones comparable to the real book.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import candidate_verdict as cv  # noqa: E402
from src.candidate_marks import _blob  # noqa: E402

#: The friction gate's own threshold (candidate_verdict.DEFAULT_MAX_FRICTION),
#: restated here rather than imported so this module's own constant survives
#: a refactor of candidate_verdict that renames or removes that name - the
#: prereg locked the NUMBER, not a pointer to wherever it currently lives.
CUTOFF = 0.25


def running_variable(row: dict) -> Tuple[Optional[float], str]:
    """This candidate's round-trip friction, and where the number came from.

    Status is one of:
      "measured"   - candidates.round_trip_pct was already populated.
      "recomputed" - recomputed from features_json via candidate_verdict.
                     verdict_for, because the stored column was NULL.
      "credit_gone"  - a different economic condition (net credit vanishes
                     once crossed), independent of the friction margin;
                     excluded from the RD by the prereg (see doc SS3).
      "unpriceable"  - could not be priced at all (missing leg quote);
                     excluded.
    """
    stored = row.get("round_trip_pct")
    if stored is not None:
        return float(stored), "measured"

    blob = _blob(row)
    merged = dict(blob)
    merged["strategy_name"] = row.get("strategy_name")
    v = cv.verdict_for(merged)
    if not v.priced:
        return None, "unpriceable"
    reason = (v.reason or "").lower()
    if "credit disappears" in reason:
        return None, "credit_gone"
    return v.round_trip_pct, "recomputed"


def relative_spread(row: dict) -> Optional[float]:
    """(ask - bid) / mid for the structure's short leg, from features_json.
    The short leg is the one whose own liquidity is crossed to enter, and
    the quantity Task 5's stratified-matching design compares candidates on.

    Defined here (not beside its only caller in Task 5) so Task 2's
    fetch_band_rows can populate every row's `rel_spread` field without a
    forward reference to code that task hasn't written yet."""
    blob = _blob(row)
    bid, ask = blob.get("short_bid"), blob.get("short_ask")
    if bid is None or ask is None:
        return None
    bid, ask = float(bid), float(ask)
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


_MULTILEG_SQL = """
    SELECT rowid AS row_id, contract_key, symbol, ts, date(ts) AS day,
           expiration, strategy_name, round_trip_pct, features_json,
           julianday(expiration) - julianday(ts) AS dte
    FROM candidates
    WHERE strategy_name IN ('Bull Put', 'Bear Call', 'Iron Condor')
"""


def fetch_band_rows(candidates_db: str, bandwidth: float) -> list:
    """Every multi-leg candidate within `bandwidth` of the friction cutoff,
    excluding credit_gone and unpriceable rows. No outcome attached yet.

    `x` is round_trip_pct - CUTOFF, centered so a local-linear fit's
    intercept at x=0 is the fitted value exactly at the cutoff."""
    import sqlite3

    from src.candidate_marks import entry_price_for

    con = sqlite3.connect(candidates_db)
    con.row_factory = sqlite3.Row
    try:
        raw = con.execute(_MULTILEG_SQL).fetchall()
    finally:
        con.close()

    out = []
    for r in raw:
        row = dict(r)
        rtp, status = running_variable(row)
        if status in ("credit_gone", "unpriceable"):
            continue
        x = rtp - CUTOFF
        if abs(x) > bandwidth:
            continue
        blob = _blob(row)
        abs_delta = blob.get("entry_delta")
        if abs_delta is None:
            continue
        entry_signed = entry_price_for(row)
        if entry_signed is None:
            continue
        out.append({
            "row_id": row["row_id"], "contract_key": row["contract_key"],
            "symbol": row["symbol"], "day": row["day"],
            "strategy_name": row["strategy_name"], "x": x,
            "round_trip_pct": rtp, "status": status,
            "abs_delta": abs(float(abs_delta)), "dte": float(row["dte"]),
            "entry_signed": entry_signed,
            "rel_spread": relative_spread(row),
        })
    return out


def attach_outcome(rows: list, candidates_db: str, horizon_days: int) -> list:
    """Add `outcome`: the forward return to the FIRST candidate_marks row
    for this contract_key dated at or after day + horizon_days, or None if
    no such mark exists. Never uses a mark dated before `day` (no lookahead
    the other way either - this only ever looks forward)."""
    import sqlite3
    from datetime import date, timedelta

    from src.candidate_marks import pnl_pct as _pnl_pct

    if not rows:
        return []

    con = sqlite3.connect(candidates_db)
    try:
        keys = sorted({r["contract_key"] for r in rows})
        placeholders = ",".join("?" for _ in keys)
        marks = con.execute(
            f"SELECT contract_key, mark_date, mid FROM candidate_marks "
            f"WHERE contract_key IN ({placeholders}) ORDER BY mark_date",
            keys,
        ).fetchall()
    finally:
        con.close()

    by_key: dict = {}
    for ck, md, mid in marks:
        by_key.setdefault(ck, []).append((md, mid))

    out = []
    for row in rows:
        r = dict(row)
        target = date.fromisoformat(r["day"]) + timedelta(days=horizon_days)
        qualifying = [(md, mid) for md, mid in by_key.get(r["contract_key"], [])
                     if date.fromisoformat(md[:10]) >= target and mid is not None]
        if not qualifying:
            r["outcome"] = None
        else:
            qualifying.sort(key=lambda t: t[0])
            _, mark_mid = qualifying[0]
            r["outcome"] = _pnl_pct(r["entry_signed"], float(mark_mid))
        out.append(r)
    return out


def collapse_to_clusters(rows: list, value_key: str) -> tuple:
    """One (x_mean, value_mean) point per (symbol, day) per side of the
    cutoff. Rows whose `value_key` is None are dropped before averaging -
    this is where a row with no qualifying forward mark leaves the analysis,
    never as a zero.

    This is the literal reading of "one observation = one symbol-day": the
    regressions in rd_estimate never see a raw candidate row, only a
    per-cluster mean, so a symbol-day that happened to contribute five
    candidates cannot outweigh one that contributed one."""
    import collections

    below: dict = collections.defaultdict(lambda: {"x": [], "v": []})
    above: dict = collections.defaultdict(lambda: {"x": [], "v": []})
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        bucket = below if r["x"] < 0 else above
        key = (r["symbol"], r["day"])
        bucket[key]["x"].append(r["x"])
        bucket[key]["v"].append(v)

    def _means(bucket: dict) -> list:
        out = []
        for vals in bucket.values():
            xs, vs = vals["x"], vals["v"]
            out.append((sum(xs) / len(xs), sum(vs) / len(vs)))
        return out

    return _means(below), _means(above)


def local_linear_intercept(points: list) -> float:
    """The fitted value at x=0 of a line through `points` (uniform kernel:
    every point counted equally). A single point is its own intercept - a
    degenerate but well-defined 0-slope fit."""
    import numpy as np

    if len(points) == 1:
        return float(points[0][1])
    xs = np.array([p[0] for p in points], dtype=float)
    vs = np.array([p[1] for p in points], dtype=float)
    if np.all(xs == xs[0]):
        return float(vs.mean())
    slope, intercept = np.polyfit(xs, vs, 1)
    return float(intercept)


def rd_estimate(below: list, above: list) -> float:
    """ITT = intercept(above) - intercept(below) at the cutoff. Negative
    means the refused (above-cutoff) side measures worse - the sign that
    supports the gate, per the prereg's own convention."""
    return local_linear_intercept(above) - local_linear_intercept(below)


def cluster_bootstrap_rd(below: list, above: list, n_boot: int = 4000,
                         seed: int = 20260902, alpha: float = 0.05) -> tuple:
    """Point ITT, percentile 95% CI, and Harvey's-hurdle t-statistic.

    Resamples symbol-day clusters with replacement, independently on each
    side (the two sides are different populations - a below-cutoff cluster
    resampling should never contribute to the above-cutoff fit), refits
    both local-linear regressions, and recomputes ITT each time. `t` is
    None when either side has fewer than 2 clusters - there is no
    resampling variance to measure, so no t-statistic is reported rather
    than one computed from a single point."""
    import random

    point = rd_estimate(below, above)
    if len(below) < 2 or len(above) < 2:
        return point, None, None, None

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        b = [below[rng.randrange(len(below))] for _ in below]
        a = [above[rng.randrange(len(above))] for _ in above]
        draws.append(rd_estimate(b, a))

    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[int((1 - alpha / 2) * len(draws)) - 1]
    se = (sum((d - point) ** 2 for d in draws) / (len(draws) - 1)) ** 0.5
    t = (point / se) if se > 0 else None
    return point, lo, hi, t


def density_check(rows: list) -> tuple:
    """Raw candidate counts each side of the cutoff, within the bandwidth
    already applied by fetch_band_rows. A large asymmetry would suggest the
    running variable is mismeasured or gamed right at the threshold - a
    deterministic arithmetic gate should not exhibit one."""
    below = sum(1 for r in rows if r["x"] < 0)
    above = sum(1 for r in rows if r["x"] >= 0)
    return below, above


def covariate_check(rows: list, covariate_key: str) -> tuple:
    """The same RD machinery applied to a covariate instead of the outcome.
    A jump here means the two sides differ in more than friction, which
    would undermine the "otherwise similar" premise the whole design relies
    on. Reported beside the primary result, never gates running it."""
    below, above = collapse_to_clusters(rows, covariate_key)
    return cluster_bootstrap_rd(below, above)


def negative_control(rows: list, seed: int = 20260902) -> tuple:
    """Shuffle `outcome` across sides WITHIN each symbol-day cell, breaking
    the link between round_trip_pct and outcome while holding the day/
    cluster structure fixed, then recompute the RD estimate. A real effect
    must not survive this - if it does, something other than the
    discontinuity is producing it."""
    import collections
    import random

    rng = random.Random(seed)
    by_day: dict = collections.defaultdict(list)
    for r in rows:
        by_day[(r["symbol"], r["day"])].append(r)

    shuffled = []
    for cell_rows in by_day.values():
        outcomes = [r.get("outcome") for r in cell_rows]
        rng.shuffle(outcomes)
        for r, new_outcome in zip(cell_rows, outcomes):
            shuffled.append({**r, "outcome": new_outcome})

    below, above = collapse_to_clusters(shuffled, "outcome")
    return cluster_bootstrap_rd(below, above)


def sign_consistency(rows: list) -> tuple:
    """ITT computed separately on the first half and second half of the
    window, split at the median entry day. Reported, not a gate.

    Split is `< median_day` / `>= median_day`, not `<=`/`>`: with an EVEN
    number of distinct days (days[len//2] lands on the day that starts the
    second half), a `<=` split would put every day in the first half and
    leave the second half empty. `<`/`>=` gives a real split for any day
    count >= 2. Returns (None, None) when fewer than 2 distinct days exist,
    and None for either half specifically if that half ends up with no
    cluster on one side of the cutoff - never a crash on an empty fit."""
    days = sorted({r["day"] for r in rows})
    if len(days) < 2:
        return None, None
    median_day = days[len(days) // 2]
    first = [r for r in rows if r["day"] < median_day]
    second = [r for r in rows if r["day"] >= median_day]

    def _itt(subset: list) -> Optional[float]:
        below, above = collapse_to_clusters(subset, "outcome")
        if not below or not above:
            return None
        return rd_estimate(below, above)

    return _itt(first), _itt(second)
