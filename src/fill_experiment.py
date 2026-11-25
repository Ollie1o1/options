"""Real-fill execution harness for measuring empirical limit-order fill rates & concession `k`.

Writes to `data/fills.db::real_fills` so real brokerage fills (Wealthsimple / IBKR)
can replace the placeholder `DEFAULT_LIMIT_K = 0.35` in `src/execution_truth.py`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

DEFAULT_FILLS_DB = "data/fills.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS real_fills (
    fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    broker TEXT NOT NULL,
    ticker TEXT NOT NULL,
    structure TEXT NOT NULL,
    legs_json TEXT NOT NULL,
    side TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    mid REAL NOT NULL,
    limit_posted REAL NOT NULL,
    concession_pct REAL NOT NULL,
    filled INTEGER NOT NULL,
    fill_price REAL,
    wait_seconds INTEGER,
    notes TEXT DEFAULT ''
);
"""


def init_fills_db(db_path: str = DEFAULT_FILLS_DB) -> None:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA_SQL)
        con.commit()
    finally:
        con.close()


def compute_concession_pct(side: str, bid: float, ask: float, limit_posted: float) -> float:
    """Signed fraction of half-spread conceded toward the touch.

    0.0 = posted at exact mid; 1.0 = crossed full touch (ask on BUY, bid on SELL);
    negative = posted passively inside mid toward the resting side.
    """
    mid = 0.5 * (float(bid) + float(ask))
    half = 0.5 * (float(ask) - float(bid))
    if half <= 1e-12:
        return 0.0
    s = str(side).upper()
    if s == "BUY":
        return float((float(limit_posted) - mid) / half)
    if s == "SELL":
        return float((mid - float(limit_posted)) / half)
    raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")


def log_real_fill(
    *,
    broker: str,
    ticker: str,
    structure: str,
    side: str,
    bid: float,
    ask: float,
    limit_posted: float,
    filled: bool,
    legs: Optional[Any] = None,
    fill_price: Optional[float] = None,
    wait_seconds: Optional[int] = None,
    notes: str = "",
    ts_utc: Optional[str] = None,
    db_path: str = DEFAULT_FILLS_DB,
) -> int:
    """Insert one limit-order attempt (filled or cancelled) into `real_fills`."""
    init_fills_db(db_path)
    bid_f, ask_f, lim_f = float(bid), float(ask), float(limit_posted)
    if ask_f < bid_f:
        raise ValueError(f"ask ({ask_f}) < bid ({bid_f})")
    mid_f = 0.5 * (bid_f + ask_f)
    concession = compute_concession_pct(side, bid_f, ask_f, lim_f)
    ts = ts_utc or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    legs_json = json.dumps(legs if legs is not None else [])
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            """
            INSERT INTO real_fills (
                ts_utc, broker, ticker, structure, legs_json, side,
                bid, ask, mid, limit_posted, concession_pct,
                filled, fill_price, wait_seconds, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                str(broker).lower(),
                str(ticker).upper(),
                str(structure),
                legs_json,
                str(side).upper(),
                bid_f,
                ask_f,
                mid_f,
                lim_f,
                concession,
                1 if filled else 0,
                float(fill_price) if fill_price is not None else None,
                int(wait_seconds) if wait_seconds is not None else None,
                str(notes or ""),
            ),
        )
        con.commit()
        return int(cur.lastrowid or 0)
    finally:
        con.close()


def list_real_fills(db_path: str = DEFAULT_FILLS_DB, limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    init_fills_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM real_fills ORDER BY fill_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def fill_rate_summary(db_path: str = DEFAULT_FILLS_DB) -> Dict[str, Any]:
    """Compute empirical fill rate & realized concession `k` grouped by (broker, structure)."""
    if not os.path.exists(db_path):
        return {"total_attempts": 0, "total_filled": 0, "overall_fill_rate": None, "groups": []}
    init_fills_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM real_fills").fetchall()]
    finally:
        con.close()

    if not rows:
        return {"total_attempts": 0, "total_filled": 0, "overall_fill_rate": None, "groups": []}

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["broker"], r["structure"]), []).append(r)

    groups: List[Dict[str, Any]] = []
    for (broker, structure), items in sorted(grouped.items()):
        n = len(items)
        filled_items = [x for x in items if x["filled"]]
        n_fill = len(filled_items)
        realized_ks: List[float] = []
        for x in filled_items:
            px = x["fill_price"] if x["fill_price"] is not None else x["limit_posted"]
            realized_ks.append(compute_concession_pct(x["side"], x["bid"], x["ask"], px))
        avg_wait = [x["wait_seconds"] for x in filled_items if x["wait_seconds"] is not None]
        groups.append(
            {
                "broker": broker,
                "structure": structure,
                "attempts": n,
                "filled": n_fill,
                "fill_rate": round(n_fill / n, 4),
                "mean_posted_k": round(sum(x["concession_pct"] for x in items) / n, 4),
                "mean_realized_k": round(sum(realized_ks) / len(realized_ks), 4) if realized_ks else None,
                "median_wait_s": sorted(avg_wait)[len(avg_wait) // 2] if avg_wait else None,
            }
        )

    tot_fill = sum(1 for r in rows if r["filled"])
    return {
        "total_attempts": len(rows),
        "total_filled": tot_fill,
        "overall_fill_rate": round(tot_fill / len(rows), 4),
        "groups": groups,
    }


# ── Test O1 execution verdict (docs/PREREG_O1_FILL_K_20261125.md) ────────────

O1_TARGET_K = 0.35
O1_MIN_FILLS = 20
O1_CAP_FILLS = 40
O1_PREFIX = "O1"


def is_o1_structure(structure: Any) -> bool:
    """A fill belongs to the Test O1 sample iff its structure starts with 'O1'."""
    return str(structure or "").strip().upper().startswith(O1_PREFIX)


def o1_fill_verdict(db_path: str = DEFAULT_FILLS_DB) -> Dict[str, Any]:
    """Preregistered O1 verdict: is the mean realized concession k <= 0.35?

    COLLECTING (< 20 filled) · PASS (one-sided 95% upper bound <= 0.35) ·
    FAIL (lower bound > 0.35) · INCONCLUSIVE (otherwise; cap 40, then re-run
    the estimator at the measured k). Never raises on a missing DB.
    """
    import math as _m

    rows: List[Dict[str, Any]] = []
    if os.path.exists(db_path):
        init_fills_db(db_path)
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in con.execute("SELECT * FROM real_fills ORDER BY fill_id").fetchall()]
        finally:
            con.close()
    o1 = [r for r in rows if is_o1_structure(r.get("structure"))]
    filled = [r for r in o1 if r.get("filled")]
    ks: List[float] = []
    for r in filled:
        px = r["fill_price"] if r.get("fill_price") is not None else r["limit_posted"]
        ks.append(compute_concession_pct(r["side"], r["bid"], r["ask"], px))
    n = len(ks)
    out: Dict[str, Any] = {
        "attempts": len(o1),
        "filled": n,
        "needed": O1_MIN_FILLS,
        "target_k": O1_TARGET_K,
        "fill_rate": (n / len(o1)) if o1 else None,
        "mean_k": None,
        "median_k": None,
        "k_lo": None,
        "k_hi": None,
        "ks": [round(k, 4) for k in ks],
        "fill_rate_flag": bool(o1) and n < 0.5 * len(o1),
        "prereg": "docs/PREREG_O1_FILL_K_20261125.md",
    }
    if n:
        mean = sum(ks) / n
        srt = sorted(ks)
        med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        out["mean_k"], out["median_k"] = round(mean, 4), round(med, 4)
        if n >= 2:
            from scipy.stats import t as _t

            sd = _m.sqrt(sum((k - mean) ** 2 for k in ks) / (n - 1))
            half = float(_t.ppf(0.95, n - 1)) * sd / _m.sqrt(n)
            out["k_lo"], out["k_hi"] = round(mean - half, 4), round(mean + half, 4)
    if n < O1_MIN_FILLS:
        out["verdict"] = "COLLECTING"
        out["reason"] = f"{n}/{O1_MIN_FILLS} filled O1 orders — no verdict before {O1_MIN_FILLS}"
    elif out["k_hi"] is not None and out["k_hi"] <= O1_TARGET_K:
        out["verdict"] = "PASS"
        out["reason"] = f"mean k {out['mean_k']:.2f}, 95% upper bound {out['k_hi']:.2f} <= {O1_TARGET_K}"
    elif out["k_lo"] is not None and out["k_lo"] > O1_TARGET_K:
        out["verdict"] = "FAIL"
        out["reason"] = (f"mean k {out['mean_k']:.2f}, 95% lower bound {out['k_lo']:.2f} > {O1_TARGET_K}"
                         " — re-run the O1 estimator at the measured k before any further O1 entry")
    else:
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = (f"mean k {out['mean_k']:.2f}, bounds [{out['k_lo']:.2f}, {out['k_hi']:.2f}] straddle "
                         f"{O1_TARGET_K}" + (" — at the 40-fill cap: re-run the O1 estimator at the measured k"
                                            if n >= O1_CAP_FILLS else " — keep collecting (cap 40)"))
    return out


def o1_progress_line(v: Dict[str, Any]) -> str:
    """One line for the LAB tab / web sidebar / CLI."""
    k = "—" if v.get("mean_k") is None else f"{v['mean_k']:.2f}"
    rate = "" if v.get("fill_rate") is None else f" · fill rate {v['fill_rate']:.0%}"
    flag = " · ⚠ under half of attempts filled" if v.get("fill_rate_flag") else ""
    return (f"TEST O1 FILLS {v['filled']}/{v['needed']} · mean k {k} (target ≤ {v['target_k']:.2f})"
            f"{rate} · {v['verdict']}{flag}")
