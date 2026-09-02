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
