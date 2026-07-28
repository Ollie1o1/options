"""FINRA consolidated short-interest — point-in-time universe, cached locally.

The public endpoint needs no auth and returns, per settlement date, the whole
reported US equity universe (~16-20k rows): shares short now, shares short at
the previous settlement, average daily volume, and days-to-cover. That is three
of the squeeze grader's inputs (level, ``dtc``, ``trend``) reconstructed exactly
as they stood, with no survivorship filter — delisted names are simply present
on the dates they were alive and absent afterwards.

Settlement dates are the 15th and the last business day of each month; the API
has no "list the dates" call, so they are discovered by probing the record-total
header and cached.

CLI:
    python -m src.squeeze.backtest.finra --discover --start 2018-01-01
    python -m src.squeeze.backtest.finra --backfill
    python -m src.squeeze.backtest.finra --stats
"""
from __future__ import annotations

import calendar
import datetime as _dt
import json
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

from src.squeeze.backtest import DEFAULT_DB

API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PAGE = 5000
_THROTTLE_S = 0.25
_MAX_RETRIES = 4

# API coverage starts in 2018; earlier probes return 0 rows.
COVERAGE_MIN = "2018-01-01"

_DDL = [
    """CREATE TABLE IF NOT EXISTS si (
        settlement_date TEXT NOT NULL,
        symbol          TEXT NOT NULL,
        shares_short    REAL,
        shares_prior    REAL,
        adv             REAL,
        dtc             REAL,
        market_class    TEXT,
        PRIMARY KEY (settlement_date, symbol)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_si_symbol ON si(symbol)",
    """CREATE TABLE IF NOT EXISTS si_dates (
        settlement_date TEXT PRIMARY KEY,
        row_count       INTEGER,
        fetched         INTEGER DEFAULT 0
    )""",
]


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    return conn


# ── HTTP ────────────────────────────────────────────────────────────────────
def _post(payload: dict, want_header: bool = False):
    """POST to the FINRA API with retry/backoff. Returns rows, or record-total."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    last = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if want_header:
                    total = resp.headers.get("record-total")
                    resp.read()
                    return int(total) if total is not None else 0
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as exc:                      # transient network
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FINRA request failed after {_MAX_RETRIES} tries: {last}")


def _date_filter(date: str) -> dict:
    return {"fieldName": "settlementDate", "fieldValue": date, "compareType": "equal"}


def row_count(date: str) -> int:
    """Rows FINRA holds for *date* (0 = not a settlement date)."""
    return _post({"limit": 1, "offset": 0, "compareFilters": [_date_filter(date)]},
                 want_header=True)


def fetch_date(date: str) -> List[dict]:
    """Every short-interest row for one settlement date, paginated."""
    out: List[dict] = []
    offset = 0
    while True:
        rows = _post({"limit": PAGE, "offset": offset,
                      "compareFilters": [_date_filter(date)]})
        if not rows:
            break
        out.extend(rows)
        if len(rows) < PAGE:
            break
        offset += PAGE
        time.sleep(_THROTTLE_S)
    return out


# ── settlement-date discovery ───────────────────────────────────────────────
def candidate_dates(start: str, end: str) -> List[str]:
    """The 15th and month-end of every month in range, plus 4 prior days each.

    FINRA settles mid-month and month-end, rolling back off weekends/holidays,
    so probing a small window around each nominal date finds the real one.
    """
    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    out: List[str] = []
    y, m = d0.year, d0.month
    while _dt.date(y, m, 1) <= d1:
        last = calendar.monthrange(y, m)[1]
        for nominal in (15, last):
            for back in range(0, 5):
                day = nominal - back
                if day < 1:
                    continue
                cand = _dt.date(y, m, day)
                if d0 <= cand <= d1 and cand.weekday() < 5:
                    out.append(cand.isoformat())
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return sorted(set(out))


def discover_dates(start: str = COVERAGE_MIN, end: Optional[str] = None,
                   db_path: str = DEFAULT_DB, verbose: bool = True) -> List[str]:
    """Probe candidate dates, record the real settlement dates in ``si_dates``."""
    end = end or _dt.date.today().isoformat()
    conn = connect(db_path)
    known = {r[0] for r in conn.execute("SELECT settlement_date FROM si_dates")}
    found: List[str] = sorted(known)
    # Skip a nominal window entirely once one of its days has landed.
    by_month: Dict[str, List[str]] = {}
    for cand in candidate_dates(start, end):
        by_month.setdefault(cand[:7] + ("-A" if int(cand[8:10]) <= 15 else "-B"), []).append(cand)

    for window, cands in sorted(by_month.items()):
        if any(c in known for c in cands):
            continue
        for cand in sorted(cands, reverse=True):      # latest first = the real one
            n = row_count(cand)
            time.sleep(_THROTTLE_S)
            if n > 0:
                conn.execute(
                    "INSERT OR REPLACE INTO si_dates(settlement_date,row_count,fetched) "
                    "VALUES(?,?,COALESCE((SELECT fetched FROM si_dates WHERE settlement_date=?),0))",
                    (cand, n, cand))
                conn.commit()
                found.append(cand)
                if verbose:
                    print(f"  settlement {cand}  rows={n:,}", flush=True)
                break
    conn.close()
    return sorted(set(found))


# ── backfill ────────────────────────────────────────────────────────────────
def _norm(row: dict) -> Optional[tuple]:
    sym = (row.get("symbolCode") or "").strip().upper()
    if not sym:
        return None
    def f(key):
        v = row.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return (row.get("settlementDate"), sym, f("currentShortPositionQuantity"),
            f("previousShortPositionQuantity"), f("averageDailyVolumeQuantity"),
            f("daysToCoverQuantity"), row.get("marketClassCode"))


def backfill(db_path: str = DEFAULT_DB, verbose: bool = True) -> int:
    """Pull every discovered settlement date not yet stored."""
    conn = connect(db_path)
    todo = [r[0] for r in conn.execute(
        "SELECT settlement_date FROM si_dates WHERE fetched=0 ORDER BY settlement_date")]
    total = 0
    for date in todo:
        rows = fetch_date(date)
        recs = [r for r in (_norm(x) for x in rows) if r]
        # Another backfill may hold the write lock; a settlement date costs a
        # paginated round-trip to refetch, so wait it out rather than dying.
        for attempt in range(6):
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO si(settlement_date,symbol,shares_short,"
                    "shares_prior,adv,dtc,market_class) VALUES(?,?,?,?,?,?,?)", recs)
                conn.execute("UPDATE si_dates SET fetched=1 WHERE settlement_date=?", (date,))
                conn.commit()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 5:
                    raise
                conn.rollback()
                time.sleep(5 * (attempt + 1))
        total += len(recs)
        if verbose:
            print(f"  {date}: {len(recs):,} rows", flush=True)
        time.sleep(_THROTTLE_S)
    conn.close()
    return total


def stats(db_path: str = DEFAULT_DB) -> dict:
    conn = connect(db_path)
    dates = conn.execute(
        "SELECT COUNT(*), SUM(fetched), MIN(settlement_date), MAX(settlement_date) "
        "FROM si_dates").fetchone()
    rows = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM si").fetchone()
    conn.close()
    return {"settlement_dates": dates[0], "fetched": dates[1] or 0,
            "first": dates[2], "last": dates[3],
            "rows": rows[0], "symbols": rows[1]}


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="FINRA short-interest backfill")
    p.add_argument("--discover", action="store_true")
    p.add_argument("--backfill", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--start", default=COVERAGE_MIN)
    p.add_argument("--end", default=None)
    p.add_argument("--db", default=DEFAULT_DB)
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.discover:
        found = discover_dates(args.start, args.end, args.db)
        print(f"settlement dates known: {len(found)}")
    if args.backfill:
        n = backfill(args.db)
        print(f"stored {n:,} short-interest rows")
    if args.stats or not (args.discover or args.backfill):
        for k, v in stats(args.db).items():
            print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
