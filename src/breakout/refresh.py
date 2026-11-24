"""Audited refresh of the existing equity OHLCV cache, never an alternate archive.

Daily overlap requests detect adjustment revisions; a revision requires a full
re-fetch before any prices change. Legacy histories and histories not fully
checked in 30 days are re-fetched in full. Old values and provider observations
are retained. No ratio-based split guesses or fallback to option-chain spots.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import zlib
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from src.data_quality import holiday_calendar_covers, is_trading_day
from .data import DEFAULT_DB, _conn

SOURCE = "Yahoo Finance via yfinance"
ADJUSTMENT = "Adj Close; High/Low scaled by Adj Close / Close; provider split-adjusted volume"
OVERLAP_DAYS = 35
FULL_REFRESH_DAYS = 30
PRICE_TOLERANCE = 1e-6
_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_fetches (
 id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, requested_start TEXT,
 end_exclusive TEXT NOT NULL, fetched_at TEXT NOT NULL,
 source TEXT NOT NULL, adjustment TEXT NOT NULL, sha256 TEXT,
 payload BLOB, accepted INTEGER NOT NULL DEFAULT 0, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_price_fetches ON price_fetches(ticker,id);
CREATE TABLE IF NOT EXISTS price_state (
 ticker TEXT PRIMARY KEY, source TEXT NOT NULL, adjustment TEXT NOT NULL,
 last_success TEXT NOT NULL, last_full TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ohlcv_evidence (
 ticker TEXT, date TEXT, fetch_id INTEGER NOT NULL, first_seen TEXT NOT NULL,
 source_close REAL, dividend REAL, split REAL, PRIMARY KEY(ticker,date)
);
CREATE TABLE IF NOT EXISTS ohlcv_revisions (
 id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, date TEXT NOT NULL,
 fetch_id INTEGER NOT NULL, close REAL, high REAL, low REAL, volume REAL
);
"""


def completed_session(now: Optional[datetime] = None) -> Optional[str]:
    """Latest US session safe to collect, after a conservative 18:00 ET delay.

    Half days also wait until 18:00. Unknown holiday years stay unknown.
    A naive injected clock is interpreted as Eastern, like maintenance.
    """
    et = ZoneInfo("America/New_York")
    now = now or datetime.now(timezone.utc)
    local = now.replace(tzinfo=et) if now.tzinfo is None else now.astimezone(et)
    day = local.date()
    if local.hour < 18:
        day -= timedelta(days=1)
    while holiday_calendar_covers(day.year):
        if is_trading_day(day):
            return day.isoformat()
        day -= timedelta(days=1)
    return None


def fetch_yahoo(ticker: str, start: Optional[str], through: str) -> list[dict]:
    """Explicit provider conventions, no automatic price repair or partial day.

    Yahoo Close is already split-adjusted; it is NOT an original as-traded
    close. Preserve it alongside Adj Close and the supplied action fields.
    Dates retain exchange-local labels rather than converting midnight to UTC.
    """
    import yfinance as yf
    end = (date.fromisoformat(through) + timedelta(days=1)).isoformat()
    kwargs: dict = dict(interval="1d", end=end, auto_adjust=False, back_adjust=False,
                  actions=True, repair=False, timeout=20)
    if start:
        kwargs["start"] = start
    else:
        kwargs["period"] = "max"
    frame = yf.Ticker(ticker).history(**kwargs)
    if frame is None or frame.empty:
        return []
    required = {"Close", "Adj Close", "High", "Low", "Volume", "Dividends", "Stock Splits"}
    if not required.issubset(frame.columns):
        raise ValueError("Provider omitted required price/action columns")
    rows = []
    for idx, rec in frame.iterrows():
        source_close, adj_close = float(rec["Close"]), float(rec["Adj Close"])
        ratio = adj_close / source_close if source_close > 0 else math.nan
        rows.append({"date": idx.strftime("%Y-%m-%d"), "close": adj_close,
                     "high": float(rec["High"]) * ratio, "low": float(rec["Low"]) * ratio,
                     "volume": float(rec["Volume"]), "source_close": source_close,
                     "dividend": float(rec["Dividends"]), "split": float(rec["Stock Splits"])})
    return rows


def _validate(rows: list[dict], start: Optional[str], through: str) -> list[dict]:
    if not rows:
        raise ValueError("Empty response; coverage remains unknown")
    seen = set()
    out = []
    for row in rows:
        day = row["date"]
        if date.fromisoformat(day).isoformat() != day or day in seen:
            raise ValueError("Invalid or duplicate session date")
        seen.add(day)
        if day > through or (start and day < start):
            raise ValueError("Response contains a date outside the requested window")
        if date.fromisoformat(day).weekday() >= 5:
            raise ValueError("Daily equity bar on a weekend")
        values = {key: float(row[key]) for key in ("close", "high", "low", "volume")}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite OHLCV at {day}")
        if min(values[k] for k in ("close", "high", "low")) <= 0 or values["volume"] < 0:
            raise ValueError(f"Nonpositive price or negative volume at {day}")
        # Multiplying by Adj Close / Close can differ by one floating-point
        # unit when the close is exactly the day's high or low.
        tolerance = values["close"] * 1e-10
        if values["low"] > values["high"] + tolerance or not values["low"] - tolerance <= values["close"] <= values["high"] + tolerance:
            raise ValueError(f"Inconsistent adjusted high/low/close at {day}")
        rec: dict = {"date": day, **values}
        for key in ("source_close", "dividend", "split"):
            value = row.get(key)
            rec[key] = float(value) if value is not None else None
            if rec[key] is not None and (not math.isfinite(rec[key]) or rec[key] < 0):
                raise ValueError(f"Invalid provider {key} at {day}")
        out.append(rec)
    return sorted(out, key=lambda r: r["date"])


def _different(old: tuple, row: dict) -> bool:
    return any(a is None or not math.isfinite(float(a)) or
               not math.isclose(float(a), row[k], rel_tol=PRICE_TOLERANCE, abs_tol=1e-8)
               for a, k in zip(old, ("close", "high", "low", "volume")))


def refresh_universe(db_path: str, tickers: list[str], *,
                     through: Optional[str] = None, now: Optional[datetime] = None,
                     fetcher: Optional[Callable[[str, Optional[str]], list[dict]]] = None,
                     pause: float = .5, attempts: int = 2) -> dict:
    """Per-symbol atomic writes; bounded retries; partial success is incomplete.

    A full response must retain every previously stored session. Revisions
    outside a 35-day overlap are checked by the monthly full refresh. Intraday
    and future bars are rejected. Latest dates alone never prove continuity.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    fixed_clock = now is not None
    now = now or datetime.now(timezone.utc)
    observed_at = now.replace(tzinfo=ZoneInfo("America/New_York")) if now.tzinfo is None else now
    stamp = observed_at.astimezone(timezone.utc).isoformat()
    through = through or completed_session(now)
    if through is None:
        raise ValueError("Holiday calendar does not cover the completed session")
    date.fromisoformat(through)
    fetcher = fetcher or (lambda ticker, start: fetch_yahoo(ticker, start, through))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    consecutive_failed = 0
    with closing(_conn(db_path)) as con:
        con.executescript(_SCHEMA)
        for ticker in dict.fromkeys(s.strip().upper() for s in tickers if s.strip()):
            result: dict = {"status": "failed", "inserted": 0, "revised": 0, "error": None}
            results[ticker] = result
            if consecutive_failed >= 5:
                result["error"] = "Feed stopped after five consecutive failed symbols; retry required"
                continue
            old = {r[0]: r[1:] for r in con.execute(
                "SELECT date,close,high,low,volume FROM ohlcv WHERE ticker=? ORDER BY date", (ticker,))}
            state = con.execute("SELECT source,adjustment,last_full FROM price_state WHERE ticker=?", (ticker,)).fetchone()
            last = max(old) if old else None
            full = (not state or state[:2] != (SOURCE, ADJUSTMENT) or
                    (date.fromisoformat(stamp[:10]) - date.fromisoformat(state[2][:10])).days >= FULL_REFRESH_DAYS)
            start = None if full or not last else (date.fromisoformat(last) - timedelta(days=OVERLAP_DAYS)).isoformat()
            if last and last > through:
                # A historical --through must not rewrite a newer stored history.
                result["error"] = "Archive extends beyond requested completed session"
                continue
            for request in range(2):  # overlap, then full rebase if needed
                rows = None
                fetch_id = None
                for attempt in range(attempts):
                    payload = None
                    digest = None
                    try:
                        raw = fetcher(ticker, start)
                        # Archive exactly the normalized provider response before validation.
                        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
                        payload, digest = zlib.compress(encoded), hashlib.sha256(encoded).hexdigest()
                        rows = _validate(raw, start, through)
                        error = None
                    except Exception as exc:
                        rows = None
                        error = f"{type(exc).__name__}: {exc}"
                    if not fixed_clock:
                        stamp = datetime.now(timezone.utc).isoformat()
                    with con:
                        cur = con.execute(
                            "INSERT INTO price_fetches(ticker,requested_start,end_exclusive,fetched_at,source,adjustment,sha256,payload,error) "
                            "VALUES(?,?,?,?,?,?,?,?,?)", (ticker, start, (date.fromisoformat(through) + timedelta(days=1)).isoformat(),
                                                         stamp, SOURCE, ADJUSTMENT, digest, payload, error))
                        fetch_id = cur.lastrowid
                    if rows is not None:
                        break
                    result["error"] = error
                    if pause and attempt + 1 < attempts:
                        time.sleep(pause)
                if rows is None:
                    break
                incoming = {r["date"]: r for r in rows}
                required_dates = {d for d in old if start is None or d >= start}
                error = None
                if not required_dates.issubset(incoming):
                    error = "Response omitted previously stored dates; refused partial replacement"
                elif start and not required_dates:
                    error = "No exact-date overlap; source consistency is unknown"
                elif start and any(_different(old[d], incoming[d]) for d in required_dates):
                    with con:
                        con.execute("UPDATE price_fetches SET error=? WHERE id=?",
                                    ("Overlap revision detected; full refresh required", fetch_id))
                    start, full = None, True
                    continue
                if error:
                    with con:
                        con.execute("UPDATE price_fetches SET error=? WHERE id=?", (error, fetch_id))
                    result["error"] = error
                    break
                # Commit prices, provenance, old versions and state as one transaction.
                with con:
                    for row in rows:
                        day = row["date"]
                        if day in old and old[day] != tuple(row[k] for k in ("close", "high", "low", "volume")):
                            con.execute("INSERT INTO ohlcv_revisions(ticker,date,fetch_id,close,high,low,volume) VALUES(?,?,?,?,?,?,?)",
                                        (ticker, day, fetch_id, *old[day]))
                            result["revised"] += 1
                        elif day not in old:
                            result["inserted"] += 1
                        con.execute("INSERT OR REPLACE INTO ohlcv VALUES(?,?,?,?,?,?)",
                                    (ticker, day, row["close"], row["high"], row["low"], row["volume"]))
                        con.execute("INSERT INTO ohlcv_evidence VALUES(?,?,?,?,?,?,?) "
                                    "ON CONFLICT(ticker,date) DO UPDATE SET fetch_id=excluded.fetch_id, "
                                    "source_close=excluded.source_close,dividend=excluded.dividend,split=excluded.split",
                                    (ticker, day, fetch_id, stamp, row["source_close"], row["dividend"], row["split"]))
                    con.execute("INSERT OR REPLACE INTO price_state VALUES(?,?,?,?,?)",
                                (ticker, SOURCE, ADJUSTMENT, stamp, stamp if full or state is None else state[2]))
                    con.execute("UPDATE price_fetches SET accepted=1 WHERE id=?", (fetch_id,))
                # A latest bar is not proof of continuous coverage. Compare
                # exact benchmark dates and maintained-calendar sessions.
                all_dates = set(old) | set(incoming)
                first = min(all_dates)
                reference = {r[0] for r in con.execute(
                    "SELECT date FROM ohlcv WHERE ticker='SPY' AND date>=? AND date<=?", (first, through))}
                day = date.fromisoformat(max(first, through[:4] + "-01-01"))
                while day.isoformat() <= through:
                    if holiday_calendar_covers(day.year) and is_trading_day(day):
                        reference.add(day.isoformat())
                    day += timedelta(days=1)
                values = {d: float(old[d][0]) for d in old}
                values.update({d: r["close"] for d, r in incoming.items()})
                dates = sorted(values)
                large_moves = [{"date": b, "return": values[b] / values[a] - 1}
                               for a, b in zip(dates, dates[1:])
                               if values[a] > 0 and abs(values[b] / values[a] - 1) > .45]
                status = ("stale" if max(incoming) < through else
                          "quality_review" if reference - all_dates else "updated")
                result.update(status=status, error=None,
                              last=max(incoming), full_refresh=full, large_moves=large_moves)
                break
            consecutive_failed = consecutive_failed + 1 if result["status"] == "failed" else 0
            if pause:
                time.sleep(pause)
    return {"through": through, "symbols": results,
            "complete": bool(results) and all(r["status"] == "updated" for r in results.values()),
            "inserted": sum(r["inserted"] for r in results.values()),
            "revised": sum(r["revised"] for r in results.values())}
