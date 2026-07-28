"""Daily closes for the study universe, survivorship-free, from DoltHub `stocks`.

Access pattern matters here. The ``ohlcv`` table is primary-keyed (date,
act_symbol), so an unbounded per-symbol scan hits the API's ~30s deadline —
verified, not assumed. Pulling a whole cross-section for one date, by contrast,
returns ~9.5k rows in a couple of seconds. So the backfill walks trading days
and filters to the study universe locally.

The reason to pay that cost rather than bulk-download from yfinance: this table
still prices names that later delisted (BBBY, MULN, NKLA ...). yfinance drops
them, which would quietly delete exactly the population whose outcomes the tail
test is trying to measure.

CLI:
    python -m src.squeeze.backtest.prices --backfill --start 2018-01-01
    python -m src.squeeze.backtest.prices --stats
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Iterable, List, Optional, Set

from src.squeeze.backtest import DEFAULT_DB

STOCKS_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master"
_THROTTLE_S = 0.15
_MAX_RETRIES = 4

_DDL = [
    """CREATE TABLE IF NOT EXISTS px (
        date   TEXT NOT NULL,
        symbol TEXT NOT NULL,
        close  REAL,
        volume REAL,
        PRIMARY KEY (symbol, date)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_px_date ON px(date)",
    """CREATE TABLE IF NOT EXISTS px_dates (
        date TEXT PRIMARY KEY,
        rows INTEGER
    )""",
]


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    return conn


def _query(sql: str) -> List[dict]:
    url = STOCKS_API + "?" + urllib.parse.urlencode({"q": sql})
    last = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                data = json.load(resp)
            if data.get("query_execution_status") == "Success":
                return data.get("rows", [])
            msg = data.get("query_execution_message", "")
            last = RuntimeError(msg)
            if "deadline" in msg.lower():
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"dolt stocks query failed: {last}")


def cross_section(date: str) -> List[dict]:
    """Every symbol's close/volume for one trading date ([] on a holiday)."""
    return _query(
        f"select act_symbol, close, volume from ohlcv where date='{date}'")


def _weekdays(start: str, end: str) -> List[str]:
    d = _dt.date.fromisoformat(start)
    last = _dt.date.fromisoformat(end)
    out = []
    while d <= last:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def backfill(start: str, end: str, symbols: Optional[Set[str]] = None,
             db_path: str = DEFAULT_DB, verbose: bool = True) -> int:
    """Store closes for *symbols* (all, if None) across every trading day."""
    conn = connect(db_path)
    done = {r[0] for r in conn.execute("SELECT date FROM px_dates")}
    todo = [d for d in _weekdays(start, end) if d not in done]
    total = 0
    for i, date in enumerate(todo, 1):
        rows = cross_section(date)
        recs = []
        for r in rows:
            sym = (r.get("act_symbol") or "").strip().upper()
            if not sym or (symbols is not None and sym not in symbols):
                continue
            try:
                close = float(r["close"]) if r.get("close") is not None else None
                vol = float(r["volume"]) if r.get("volume") is not None else None
            except (TypeError, ValueError):
                continue
            if close is None or close <= 0:
                continue
            recs.append((date, sym, close, vol))
        conn.executemany(
            "INSERT OR REPLACE INTO px(date,symbol,close,volume) VALUES(?,?,?,?)", recs)
        conn.execute("INSERT OR REPLACE INTO px_dates(date,rows) VALUES(?,?)",
                     (date, len(recs)))
        conn.commit()
        total += len(recs)
        if verbose and (i % 20 == 0 or len(recs) == 0):
            print(f"  [{i}/{len(todo)}] {date}: {len(recs):,} rows", flush=True)
        time.sleep(_THROTTLE_S)
    conn.close()
    return total


CSV_URL = "https://www.dolthub.com/csv/post-no-preference/stocks/master/ohlcv"


def load_csv_stream(start: str, symbols: Optional[Set[str]] = None,
                    db_path: str = DEFAULT_DB, url: str = CSV_URL,
                    batch: int = 100_000, verbose: bool = True) -> int:
    """Stream the whole ``ohlcv`` table over HTTP, filtering as it arrives.

    The row-limited SQL API would need tens of thousands of requests to cover
    this; the CSV export delivers it in one pass. Rows are filtered and inserted
    on the fly and the raw file is never written to disk, which matters because
    the full table is multi-gigabyte.

    The export is ordered by date, so a dropped connection can be resumed
    cheaply: rows already stored are skipped by date on the retry. urllib speaks
    HTTP/1.1, avoiding the HTTP/2 stream resets that killed earlier attempts.
    """
    import csv
    import io

    conn = connect(db_path)
    have = conn.execute("SELECT MAX(date) FROM px").fetchone()[0]
    resume_after = have if have and have >= start else None
    if verbose and resume_after:
        print(f"  resuming: skipping rows through {resume_after}", flush=True)

    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    kept = skipped = 0
    pending: List[tuple] = []
    seen_dates: Set[str] = set()

    with urllib.request.urlopen(req, timeout=300) as resp:
        stream = io.TextIOWrapper(resp, encoding="utf-8", newline="")
        reader = csv.DictReader(stream)
        for row in reader:
            date = row.get("date") or ""
            if date < start:
                skipped += 1
                continue
            if resume_after and date <= resume_after:
                skipped += 1
                continue
            sym = (row.get("act_symbol") or "").strip().upper()
            if not sym or (symbols is not None and sym not in symbols):
                continue
            try:
                close = float(row["close"])
                vol = float(row["volume"]) if row.get("volume") else 0.0
            except (TypeError, ValueError, KeyError):
                continue
            if close <= 0:
                continue
            pending.append((date, sym, close, vol))
            seen_dates.add(date)
            if len(pending) >= batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO px(date,symbol,close,volume) VALUES(?,?,?,?)",
                    pending)
                conn.commit()
                kept += len(pending)
                if verbose:
                    print(f"  ...{kept:,} rows stored (through {date})", flush=True)
                pending.clear()

    if pending:
        conn.executemany(
            "INSERT OR REPLACE INTO px(date,symbol,close,volume) VALUES(?,?,?,?)", pending)
        kept += len(pending)
    conn.executemany("INSERT OR REPLACE INTO px_dates(date,rows) VALUES(?,0)",
                     [(d,) for d in seen_dates])
    conn.commit()
    conn.close()
    if verbose:
        print(f"  done: {kept:,} rows stored, {skipped:,} skipped by date", flush=True)
    return kept


def stats(db_path: str = DEFAULT_DB) -> dict:
    conn = connect(db_path)
    px = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) FROM px").fetchone()
    days = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN rows>0 THEN 1 ELSE 0 END) FROM px_dates").fetchone()
    conn.close()
    return {"rows": px[0], "symbols": px[1], "first": px[2], "last": px[3],
            "days_probed": days[0], "trading_days": days[1] or 0}


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="DoltHub daily close backfill")
    p.add_argument("--backfill", action="store_true")
    p.add_argument("--load-csv", action="store_true",
                   help="bulk-load via the CSV export (preferred; one pass)")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=_dt.date.today().isoformat())
    p.add_argument("--universe-db", default=DEFAULT_DB,
                   help="DB holding the FINRA si table used to scope symbols")
    p.add_argument("--all-symbols", action="store_true",
                   help="store every symbol instead of the study universe")
    p.add_argument("--db", default=DEFAULT_DB)
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.backfill or args.load_csv:
        syms = None
        if not args.all_symbols:
            from src.squeeze.backtest.universe import study_symbols
            syms = study_symbols(args.universe_db)
            print(f"study universe: {len(syms):,} symbols")
        if args.load_csv:
            n = load_csv_stream(args.start, syms, args.db)
        else:
            n = backfill(args.start, args.end, syms, args.db)
        print(f"stored {n:,} price rows")
    if args.stats or not (args.backfill or args.load_csv):
        for k, v in stats(args.db).items():
            print(f"  {k:14s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
