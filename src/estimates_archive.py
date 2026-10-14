"""Point-in-time archive of analyst estimates, once per trading day.

Estimate REVISIONS are among the better-documented equity signals, but no
free source serves their history: yfinance shows the consensus today and its
value 7/30/60/90 days ago, nothing older. A daily snapshot is the only way to
build the series, so this starts the clock. It records data; nothing reads it
to make a decision.

Two tables in ``data/estimates_archive.db``:

- ``estimate_snapshots`` — one row per (day, symbol, dataset, period, field),
  LONG format on purpose. yfinance renames columns between versions
  (``downLast7days`` vs ``downLast7Days``); a wide schema would break or
  silently drop a column on upgrade, a long one just records the new name.
- ``analyst_actions`` — upgrades/downgrades with firm, grades and price
  targets. Yahoo serves this with its own dated history, so rows are
  de-duplicated on (symbol, grade_date, firm, action) and ``first_seen``
  records when THIS archive first saw each one — the honest point-in-time
  column, the same role ``archived_at`` plays in ``news_archive``.

Run after the close so it never competes with the scans for Yahoo's rate
limit (2026-10-14: the candidate marker lost most of a day to it).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                          "estimates_archive.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS estimate_snapshots (
    snap_date   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    period      TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       REAL,
    archived_at TEXT NOT NULL,
    PRIMARY KEY (snap_date, symbol, dataset, period, field)
);
CREATE INDEX IF NOT EXISTS idx_est_symbol ON estimate_snapshots(symbol, snap_date);
CREATE TABLE IF NOT EXISTS analyst_actions (
    symbol              TEXT NOT NULL,
    grade_date          TEXT NOT NULL,
    firm                TEXT NOT NULL,
    action              TEXT NOT NULL,
    to_grade            TEXT,
    from_grade          TEXT,
    price_target_action TEXT,
    current_pt          REAL,
    prior_pt            REAL,
    first_seen          TEXT NOT NULL,
    PRIMARY KEY (symbol, grade_date, firm, action)
);
CREATE INDEX IF NOT EXISTS idx_actions_seen ON analyst_actions(first_seen);
"""

# yfinance Ticker attributes that are tables indexed by period.
DATASETS = ("eps_trend", "eps_revisions", "earnings_estimate",
            "revenue_estimate", "recommendations")

# Consecutive symbols that may fail before the run gives up on the feed. ETFs
# return empty tables legitimately and do not count as failures.
_ABORT_AFTER_CONSECUTIVE_FAILURES = 10
_PAUSE_SECONDS = 0.5


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.executescript(_SCHEMA)
    return conn


def _default_fetch(symbol: str) -> Dict[str, Any]:
    """Every dataset for one symbol, as yfinance returns them."""
    from . import data_fetching
    data_fetching._init_yfinance()
    data_fetching._init_yf_session()
    tkr = data_fetching.yf.Ticker(symbol, session=data_fetching._yf_session)
    out: Dict[str, Any] = {}
    for name in DATASETS + ("upgrades_downgrades",):
        out[name] = getattr(tkr, name)
    return out


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _snapshot_rows(symbol: str, data: Dict[str, Any], snap_date: str,
                   stamp: str) -> List[tuple]:
    rows: List[tuple] = []
    for dataset in DATASETS:
        df = data.get(dataset)
        if df is None or getattr(df, "empty", True):
            continue
        if "period" in df.columns:          # `recommendations` keeps it as a column
            df = df.set_index("period")
        for period, rec in df.iterrows():
            for field, value in rec.items():
                if field == "period":
                    continue
                rows.append((snap_date, symbol, dataset, str(period), str(field),
                             _num(value), stamp))
    return rows


def _action_rows(symbol: str, data: Dict[str, Any], stamp: str) -> List[tuple]:
    df = data.get("upgrades_downgrades")
    if df is None or getattr(df, "empty", True):
        return []
    rows: List[tuple] = []
    for when, rec in df.iterrows():
        firm, action = rec.get("Firm"), rec.get("Action")
        if not firm or not action:
            continue
        rows.append((symbol, str(when), str(firm), str(action),
                     rec.get("ToGrade") or None, rec.get("FromGrade") or None,
                     rec.get("priceTargetAction") or None,
                     _num(rec.get("currentPriceTarget")),
                     _num(rec.get("priorPriceTarget")), stamp))
    return rows


def archive_symbols(symbols: Iterable[str], *, db_path: Optional[str] = None,
                    today: Optional[str] = None,
                    fetch: Optional[Callable[[str], Dict[str, Any]]] = None,
                    now: Optional[datetime] = None,
                    pause: float = _PAUSE_SECONDS) -> Dict[str, Any]:
    """Snapshot every symbol; never raises. Returns the run's counts.

    ``complete`` is False when the run stopped on a dead feed, so the caller
    can leave the day unstamped and retry it.
    """
    fetch = fetch or _default_fetch
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    today = today or datetime.now().strftime("%Y-%m-%d")
    counts: Dict[str, Any] = {"symbols": 0, "with_estimates": 0, "failed": 0,
                              "rows": 0, "new_actions": 0, "aborted": False}
    consecutive = 0
    with connect(db_path) as conn:
        for sym in symbols:
            if consecutive >= _ABORT_AFTER_CONSECUTIVE_FAILURES:
                counts["aborted"] = True
                log.warning("estimates archive: %d symbols in a row failed — "
                            "stopping; the day will be retried", consecutive)
                break
            counts["symbols"] += 1
            try:
                data = fetch(sym) or {}
            except Exception:
                log.warning("estimates archive: fetch failed for %s", sym,
                            exc_info=True)
                counts["failed"] += 1
                consecutive += 1
                continue
            consecutive = 0
            snap = _snapshot_rows(sym, data, today, stamp)
            if snap:
                counts["with_estimates"] += 1
                conn.executemany(
                    "INSERT OR REPLACE INTO estimate_snapshots VALUES "
                    "(?,?,?,?,?,?,?)", snap)
                counts["rows"] += len(snap)
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO analyst_actions VALUES "
                "(?,?,?,?,?,?,?,?,?,?)", _action_rows(sym, data, stamp))
            counts["new_actions"] += conn.total_changes - before
            conn.commit()
            if pause:
                time.sleep(pause)
    counts["complete"] = not counts["aborted"]
    return counts


def universe(cfg_path: str = "config.json",
             candidates_db: Optional[str] = None, days: int = 30) -> List[str]:
    """Config watchlists plus every symbol the board scanned in ``days``.

    The board's own recent universe is what an estimate signal would ever be
    applied to; the watchlists keep it from shrinking on a quiet month.
    """
    syms: set = set()
    try:
        with open(cfg_path) as fh:
            for v in (json.load(fh).get("watchlists") or {}).values():
                if isinstance(v, list):
                    syms.update(str(s).upper() for s in v)
    except (OSError, ValueError):
        pass
    try:
        from . import candidate_record as cr
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with cr.connect(candidates_db) as conn:
            syms.update(r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM candidates WHERE ts >= ?", (since,))
                if r[0])
    except Exception:
        log.warning("estimates archive: candidate universe unreadable",
                    exc_info=True)
    return sorted(syms)


def due_estimates_archive(state: Optional[dict], today: str, hhmm: int) -> bool:
    """Once per trading day, after the close (>= 16:00)."""
    if hhmm < 1600:
        return False
    from .data_quality import is_trading_day
    if not is_trading_day(today):
        return False
    return (state or {}).get("last_estimates_archive") != today


def archive_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not os.path.exists(path):
        return {"days": 0, "symbols": 0, "rows": 0, "actions": 0}
    with sqlite3.connect(path) as conn:
        days, symbols, rows, first, last = conn.execute(
            "SELECT COUNT(DISTINCT snap_date), COUNT(DISTINCT symbol), COUNT(*),"
            " MIN(snap_date), MAX(snap_date) FROM estimate_snapshots").fetchone()
        actions, = conn.execute("SELECT COUNT(*) FROM analyst_actions").fetchone()
    return {"days": days, "symbols": symbols, "rows": rows, "first": first,
            "last": last, "actions": actions}


if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        logging.basicConfig(level=logging.WARNING)
        print(json.dumps(archive_symbols(universe()), indent=1))
    print(json.dumps(archive_stats(), indent=1))
