"""Per-symbol ATM implied vol from the daily chain archive, plus day-over-day
ΔIV. ATM IV = call+put-averaged IV of the strike nearest spot at the expiry
closest to TARGET_DTE. Archive-native (no SVI fit, which is fragile on thin
chains)."""
from __future__ import annotations
import datetime as _dt
import sqlite3
from typing import Any, Dict, List, Optional

TARGET_DTE = 30


def _dte(snap_date: str, expiration: str) -> Optional[int]:
    try:
        s = _dt.datetime.strptime(str(snap_date)[:10], "%Y-%m-%d")
        e = _dt.datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return (e - s).days


def atm_iv(rows: List[Dict[str, Any]], spot: float, target_dte: int = TARGET_DTE) -> Optional[float]:
    valid = [r for r in rows
             if r.get("iv") is not None and r.get("strike") is not None
             and r.get("expiration") and float(r["iv"]) > 0]
    if not valid or not spot:
        return None
    exp_dte: Dict[str, int] = {}
    for r in valid:
        d = _dte(r["snap_date"], r["expiration"])
        if d is not None and d >= 0:
            exp_dte.setdefault(r["expiration"], d)
    if not exp_dte:
        return None
    best_exp = min(exp_dte, key=lambda e: abs(exp_dte[e] - target_dte))
    leg = [r for r in valid if r["expiration"] == best_exp]
    nearest = min(leg, key=lambda r: abs(float(r["strike"]) - float(spot)))
    at_strike = [float(r["iv"]) for r in leg if float(r["strike"]) == float(nearest["strike"])]
    return sum(at_strike) / len(at_strike) if at_strike else None


def latest_snap_date(conn: sqlite3.Connection) -> Optional[str]:
    r = conn.execute("SELECT MAX(snap_date) FROM chain_snapshots").fetchone()
    return r[0] if r and r[0] else None


def prev_snap_date(conn: sqlite3.Connection, symbol: str, date: str) -> Optional[str]:
    r = conn.execute(
        "SELECT MAX(snap_date) FROM chain_snapshots WHERE symbol=? AND snap_date<?",
        (symbol, date)).fetchone()
    return r[0] if r and r[0] else None


def _rows_for(conn: sqlite3.Connection, symbol: str, snap_date: str):
    cur = conn.execute(
        "SELECT type,strike,expiration,iv,spot,snap_date FROM chain_snapshots "
        "WHERE symbol=? AND snap_date=?", (symbol, snap_date))
    rows, spot = [], None
    for t, k, e, iv, sp, sd in cur.fetchall():
        rows.append({"type": t, "strike": k, "expiration": e, "iv": iv, "snap_date": sd})
        if sp:
            spot = sp
    return rows, spot


def atm_iv_for(conn: sqlite3.Connection, symbol: str, snap_date: str) -> Optional[float]:
    rows, spot = _rows_for(conn, symbol, snap_date)
    return atm_iv(rows, spot) if spot else None


def iv_move(db_path: str, snap_date: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        date = snap_date or latest_snap_date(conn)
        if not date:
            return out
        syms = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM chain_snapshots WHERE snap_date=?", (date,))]
        for s in syms:
            iv = atm_iv_for(conn, s, date)
            if iv is None:
                continue
            prev = prev_snap_date(conn, s, date)
            piv = atm_iv_for(conn, s, prev) if prev else None
            out.append({"symbol": s, "snap_date": date, "iv": iv,
                        "prev_iv": piv, "d_iv": (iv - piv) if piv is not None else None})
    return out
