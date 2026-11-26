"""Later paper marks for frozen study entries, from the chain archive. Local files only.

One mark row per entry per trading session after entry, up to the earlier of the
expiration and the newest archived session, so a gap is a row, never an absence:

  marked           both legs found with a usable quote to close
  unpriceable      both legs found, but no ask to buy back the short or a crossed quote
  missing_leg      the symbol was archived that session but a leg was not
  missing_session  nothing archived for the symbol that session
  settled          expiration day: the underlying's close for intrinsic settlement
                   (NULL close once the give-up window passes means no settlement price)

Rows other than `marked` and a priced `settled` are retried on the next run, because the
archive can still be backfilled for that day. A mark is a quote snapshot. It shows the
price a marketable order would have faced, never that a resting order filled.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from typing import Any, Dict, List, Optional

from src.paths import repo_path
from src.selector_study.db import load_settings
from src.selector_study.outcomes import resolve_all

FINAL = frozenset({"marked"})
_SETTLE_GIVE_UP_DAYS = 5


def _trading_days(after: str, through: str) -> List[str]:
    from src.data_quality import is_trading_day
    out: List[str] = []
    d = dt.date.fromisoformat(after) + dt.timedelta(days=1)
    end = dt.date.fromisoformat(through)
    while d <= end:
        if is_trading_day(d):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def entry_marks(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM marks WHERE entry_id=? ORDER BY mark_session", (entry_id,))]


def _adj_close(ohlcv: sqlite3.Connection, symbol: str, day: str, max_days: int = 4) -> Optional[float]:
    lo = (dt.date.fromisoformat(day) - dt.timedelta(days=max_days)).isoformat()
    row = ohlcv.execute("SELECT close FROM ohlcv WHERE ticker=? AND date<=? AND date>=? "
                        "ORDER BY date DESC LIMIT 1", (symbol, day, lo)).fetchone()
    return float(row[0]) if row and row[0] is not None and float(row[0]) > 0 else None


def _close_on(ohlcv: Optional[sqlite3.Connection], symbol: str, expiration: str,
              arch: Optional[sqlite3.Connection] = None) -> Optional[float]:
    """The underlying's close at expiration, on the strikes' (unadjusted) basis.

    The local OHLCV archive stores dividend- and split-adjusted closes. The chain archive's
    own spot is unadjusted, so the ratio of the two on the last archived session before
    expiration converts the adjusted expiration close back. Without an archive spot within
    a week the adjusted close is not used: a settlement on the wrong basis is worse than none."""
    if ohlcv is None or arch is None:
        return None
    adj = _adj_close(ohlcv, symbol, expiration)
    lo = (dt.date.fromisoformat(expiration) - dt.timedelta(days=7)).isoformat()
    row = arch.execute("SELECT snap_date, AVG(spot) FROM chain_snapshots WHERE symbol=? AND snap_date<=? "
                       "AND snap_date>=? AND spot>0 GROUP BY snap_date ORDER BY snap_date DESC LIMIT 1",
                       (symbol, expiration, lo)).fetchone()
    if adj is None or not row or row[1] is None:
        return None
    base_adj = _adj_close(ohlcv, symbol, str(row[0]))
    return adj * float(row[1]) / base_adj if base_adj else None


def mark_entries(conn: sqlite3.Connection, *, chain_db: str = "data/chain_archive.db",
                 ohlcv_db: str = "data/equity_ohlcv.db", today: Optional[str] = None) -> Dict[str, int]:
    """Record marks for every entry whose outcome is not settled in both cost cases."""
    settings = load_settings(conn)
    today = today or dt.date.today().isoformat()
    counts = {"entries": 0, "resolved": 0, "written": 0, "marked": 0, "unpriceable": 0,
              "missing_leg": 0, "missing_session": 0, "settled": 0}
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return counts
    arch = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    o_full = repo_path(ohlcv_db)
    ohlcv = sqlite3.connect(f"file:{o_full}?mode=ro", uri=True) if os.path.exists(o_full) else None
    try:
        latest_row = arch.execute("SELECT MAX(snap_date) FROM chain_snapshots").fetchone()
        latest = min(str(latest_row[0]), today) if latest_row and latest_row[0] else None
        archived = {(str(s), str(d)) for s, d in arch.execute(
            "SELECT DISTINCT symbol, snap_date FROM chain_snapshots")}
        for entry in [dict(r) for r in conn.execute("SELECT * FROM entries ORDER BY session, entry_id")]:
            counts["entries"] += 1
            if all(o["status"] != "open" for o in resolve_all(entry, entry_marks(conn, entry["entry_id"]),
                                                               settings).values()):
                counts["resolved"] += 1
                continue
            prior = entry_marks(conn, entry["entry_id"])
            have = {m["mark_session"]: m["status"] for m in prior}
            settled = any(m["status"] == "settled" and m["underlying_close"] is not None for m in prior)
            sym, exp = entry["symbol"], entry["expiration"]
            if latest is not None:
                for day in _trading_days(entry["session"], min(exp, latest)):
                    if day >= exp or have.get(day) in FINAL:
                        continue
                    rec = _quote_mark(arch, entry, day, (sym, day) in archived)
                    _upsert(conn, rec)
                    counts["written"] += 1
                    counts[rec["status"]] += 1
            still_open = any(o["status"] == "open" for o in resolve_all(
                entry, entry_marks(conn, entry["entry_id"]), settings).values())
            if exp <= today and not settled and still_open:
                close = _close_on(ohlcv, sym, exp, arch)
                stale = (dt.date.fromisoformat(today) - dt.date.fromisoformat(exp)).days > _SETTLE_GIVE_UP_DAYS
                if close is not None or stale:
                    _upsert(conn, {"entry_id": entry["entry_id"], "mark_session": exp, "status": "settled",
                                   "underlying_close": close, "source": "equity_ohlcv" if close else None})
                    counts["written"] += 1
                    counts["settled"] += 1
            conn.commit()
    finally:
        arch.close()
        if ohlcv is not None:
            ohlcv.close()
    return counts


def _quote_mark(arch: sqlite3.Connection, entry: Dict[str, Any], day: str, archived: bool) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"entry_id": entry["entry_id"], "mark_session": day}
    if not archived:
        return {**rec, "status": "missing_session"}
    legs = {}
    for role, strike in (("short", entry["short_strike"]), ("long", entry["long_strike"])):
        legs[role] = arch.execute(
            "SELECT bid, ask, snapshot_ts, source FROM chain_snapshots WHERE symbol=? AND snap_date=? "
            "AND expiration=? AND type='put' AND ABS(strike-?)<1e-6 LIMIT 1",
            (entry["symbol"], day, entry["expiration"], float(strike))).fetchone()
    if legs["short"] is None or legs["long"] is None:
        return {**rec, "status": "missing_leg"}
    (sb, sa, ts, src), (lb, la, _ts2, _src2) = legs["short"], legs["long"]
    rec.update(short_bid=sb, short_ask=sa, long_bid=lb, long_ask=la, observed_at=ts, source=src)
    ok = (None not in (sb, sa, lb, la) and float(sa) > 0 and float(sa) >= float(sb)
          and float(la) >= float(lb) and float(lb) >= 0)
    return {**rec, "status": "marked" if ok else "unpriceable"}


def _upsert(conn: sqlite3.Connection, rec: Dict[str, Any]) -> None:
    cols = ["entry_id", "mark_session", "status", "short_bid", "short_ask", "long_bid", "long_ask",
            "underlying_close", "observed_at", "source"]
    vals = [rec.get(c) for c in cols]
    conn.execute(
        f"INSERT INTO marks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
        "ON CONFLICT(entry_id, mark_session) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in cols[2:])
        + " WHERE marks.status != 'marked' AND NOT (marks.status = 'settled' AND marks.underlying_close IS NOT NULL)",
        vals)
