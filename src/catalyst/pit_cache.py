"""Content cache for point-in-time reconstruction.

THE RULE, ONE SENTENCE: **an entry may answer a question whose ``as_of`` is at
or before the time the entry was fetched.**

That is not a guessed TTL, it is the semantics of the sources themselves. A
version list fetched on the 27th can answer "what was true on the 25th"
forever, and can NEVER answer "what is true on the 28th" — a trial amended on
the 28th is invisible to it. Deriving expiry from the question rather than from
a timer means the cache is exactly as fresh as it needs to be and no fresher.

Four DIFFERENT caching arguments live here. Conflating them would be a bug.

  * ``(nct_id, version)`` — IMMUTABLE. A study version is a frozen historical
    record. Cache forever, no ``as_of``, no expiry. The only one with no
    freshness question at all.
  * ``(nct_id)`` version LIST — APPEND-ONLY; it grows with every amendment.
    Served only for ``as_of <= fetched_at``.
  * ``(cik)`` companyfacts — APPEND-ONLY; it grows as filings arrive. The
    earlier argument here ("consumers filter to ``filed <= as_of`` so a
    fresher fetch only adds rows that are discarded") is true ONLY when
    ``as_of`` is at or before the fetch. Past that it silently omits filings
    that existed. Same rule as the version list.
  * ``(sweep window)`` universe — a DELIBERATE FREEZE, and the one thing here
    that must NOT auto-renew: re-sweeping changes which rows the study
    contains, and a population that moves under a study destroys any
    comparison with its own earlier runs. It ages loudly instead, and is
    refreshed only when explicitly asked.

Prices are a fifth case, append-AND-revise: new bars arrive and recent ones are
restated. A CLOSED window (ending before the fetch) is safe indefinitely — a
later split rescales both endpoints of a return equally, leaving the ratio
unchanged. An OPEN window (ending at or after the fetch) is good only for the
day it was taken.

A cache MISS is None. That is distinct from an entry holding an empty result,
which means "we looked and there was nothing".

BUT AN EMPTY RESULT FROM A SWALLOWING FETCHER IS NOT AN ANSWER. `_fetch_prices`
and `_fetch_versions` both return a falsy value on ANY exception, so {} means
"we could not look" far more often than "there is nothing there". Storing one
poisoned a whole study on 2026-08-28: a rate-limited run cached 145 empty price
series and the next run returned n=0 for every hypothesis, silently. Callers
fetching through a swallowing fetcher must not store a falsy result.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from src.paths import repo_path

DEFAULT_DB = repo_path(os.path.join("data", "catalyst_pit.db"))

_DDL = [
    """CREATE TABLE IF NOT EXISTS pit_versions (
        nct_id   TEXT PRIMARY KEY,
        payload  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS pit_study (
        nct_id   TEXT NOT NULL,
        version  INTEGER NOT NULL,
        payload  TEXT NOT NULL,
        PRIMARY KEY (nct_id, version)
    )""",
    """CREATE TABLE IF NOT EXISTS pit_facts (
        cik      INTEGER PRIMARY KEY,
        payload  TEXT NOT NULL
    )""",
    # A THIRD caching argument, different again from the two above: this one
    # is neither immutable nor append-safe, it is a deliberate FREEZE.
    # ClinicalTrials.gov gains and edits trials, and the cap band is applied
    # with TODAY'S market cap, so re-sweeping silently changes the study
    # population — it moved H3's arms between two runs a day apart with no
    # code change. Refreshed only when explicitly asked.
    """CREATE TABLE IF NOT EXISTS pit_universe (
        key       TEXT PRIMARY KEY,
        swept_at  TEXT NOT NULL,
        payload   TEXT NOT NULL
    )""",
    # Market cap is TODAY'S by construction (board_as_of says so), so it is an
    # open-window value: good for the day it was taken and no longer. `cap` is
    # NOT NULL because an unknown cap must never be frozen — `market_caps`
    # swallows its exceptions and returns None, so None means "we could not
    # look" as often as "there is no cap".
    """CREATE TABLE IF NOT EXISTS pit_caps (
        ticker      TEXT PRIMARY KEY,
        fetched_at  TEXT NOT NULL,
        cap         REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS pit_prices (
        ticker      TEXT NOT NULL,
        start       TEXT NOT NULL,
        end         TEXT NOT NULL,
        fetched_at  TEXT NOT NULL,
        payload     TEXT NOT NULL,
        PRIMARY KEY (ticker, start, end)
    )""",
]

#: Columns added after the tables first shipped, as (table, column, type).
_MIGRATIONS = (
    ("pit_versions", "fetched_at", "TEXT"),
    ("pit_facts", "fetched_at", "TEXT"),
)

#: What to assume a row was fetched at when it predates the `fetched_at`
#: column. The true time is unknown, so this is the EARLIEST it could have
#: been: the day `data/catalyst_pit.db` was created. Assuming anything later
#: would claim a freshness these rows cannot demonstrate. Every backtest
#: vintage is years before this, so the study still runs entirely from cache;
#: a live `as_of` of today correctly misses and refetches.
LEGACY_FETCHED_AT = "2026-08-25"


def _fresh_for(fetched_at: Optional[str], as_of: Optional[str]) -> bool:
    """May an entry fetched at ``fetched_at`` answer a question about ``as_of``?

    Yes when the question is at or before the fetch. `as_of=None` means the
    caller has no point-in-time requirement and accepts any age.
    """
    if as_of is None:
        return True
    return str(as_of) <= str(fetched_at or LEGACY_FETCHED_AT)


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    path = repo_path(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _DDL:
        conn.execute(ddl)
    # Additive only, and idempotent: an existing row keeps a NULL fetched_at
    # and is read through LEGACY_FETCHED_AT rather than being discarded. A
    # migration that threw away a 441MB cache to gain a timestamp would be a
    # worse answer than assuming its earliest possible age.
    for table, column, coltype in _MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    conn.commit()
    return conn


def _today() -> str:
    import datetime as _dt
    return _dt.date.today().isoformat()


def get_versions(conn: sqlite3.Connection, nct_id: str,
                 as_of: Optional[str] = None
                 ) -> Optional[List[Dict[str, Any]]]:
    """The cached version list, or None if absent OR too old for ``as_of``.

    Too-old reads as a MISS on purpose: the caller's next move is to fetch,
    which is exactly right. Returning a stale list with a warning would leave
    the decision to a reader who is not there.
    """
    row = conn.execute(
        "SELECT payload, fetched_at FROM pit_versions WHERE nct_id=?",
        (nct_id,)).fetchone()
    if not row or not _fresh_for(row[1], as_of):
        return None
    return list(json.loads(row[0]))


def put_versions(conn: sqlite3.Connection, nct_id: str,
                 versions: List[Dict[str, Any]],
                 fetched_at: Optional[str] = None) -> None:
    conn.execute(
        "INSERT INTO pit_versions (nct_id, payload, fetched_at) VALUES (?,?,?) "
        "ON CONFLICT(nct_id) DO UPDATE SET payload=excluded.payload, "
        "fetched_at=excluded.fetched_at",
        (nct_id, json.dumps(versions), fetched_at or _today()))
    conn.commit()


def get_prices(conn: sqlite3.Connection, ticker: str, start: str, end: str,
               today: Optional[str] = None) -> Optional[Dict[str, float]]:
    """A cached close series, or None if absent or no longer trustworthy.

    A CLOSED window (``end`` before the fetch) never expires: the bars are
    settled, and a later split rescales both endpoints of a return equally so
    the ratio is unchanged. An OPEN window (``end`` at or after the fetch) is
    served only on the day it was taken, because tomorrow it is missing a bar.
    """
    row = conn.execute(
        "SELECT payload, fetched_at FROM pit_prices "
        "WHERE ticker=? AND start=? AND end=?", (ticker, start, end)).fetchone()
    if not row:
        return None
    fetched_at = str(row[1])
    now = today or _today()
    closed = str(end) < fetched_at
    if not closed and fetched_at != now:
        return None
    return {str(k): float(v) for k, v in json.loads(row[0]).items()}


def put_prices(conn: sqlite3.Connection, ticker: str, start: str, end: str,
               series: Dict[str, float],
               fetched_at: Optional[str] = None) -> None:
    """Store a close series.

    CALLERS MUST NOT PASS AN EMPTY SERIES fetched through a swallowing
    fetcher. `_fetch_prices` returns {} on any exception, so {} means "we
    could not look" far more often than "there is nothing there", and storing
    it made a whole study return n=0 on 2026-08-28. The guard lives at the
    call site because only the caller knows whether its fetcher can report
    failure."""
    conn.execute(
        "INSERT INTO pit_prices (ticker, start, end, fetched_at, payload) "
        "VALUES (?,?,?,?,?) ON CONFLICT(ticker, start, end) DO UPDATE SET "
        "fetched_at=excluded.fetched_at, payload=excluded.payload",
        (ticker, start, end, fetched_at or _today(), json.dumps(series)))
    conn.commit()


def get_study(conn: sqlite3.Connection, nct_id: str,
              version: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT payload FROM pit_study WHERE nct_id=? AND version=?",
        (nct_id, version)).fetchone()
    return dict(json.loads(row[0])) if row else None


def put_study(conn: sqlite3.Connection, nct_id: str, version: int,
              payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO pit_study (nct_id, version, payload) VALUES (?,?,?) "
        "ON CONFLICT(nct_id, version) DO UPDATE SET payload=excluded.payload",
        (nct_id, version, json.dumps(payload)))
    conn.commit()


def get_universe(conn: sqlite3.Connection,
                 key: str) -> Optional[Tuple[str, List[str]]]:
    """(swept_at, nct_ids) for a pinned universe, or None if never swept.

    The date rides with the list because a frozen population that cannot be
    aged or audited is worse than no freeze at all — a reader must be able to
    ask how old the universe is.
    """
    row = conn.execute(
        "SELECT swept_at, payload FROM pit_universe WHERE key=?",
        (key,)).fetchone()
    return (str(row[0]), list(json.loads(row[1]))) if row else None


def put_universe(conn: sqlite3.Connection, key: str, swept_at: str,
                 nct_ids: List[str]) -> None:
    conn.execute(
        "INSERT INTO pit_universe (key, swept_at, payload) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET swept_at=excluded.swept_at, "
        "payload=excluded.payload",
        (key, swept_at, json.dumps(list(nct_ids))))
    conn.commit()


def get_caps(conn: sqlite3.Connection, tickers: Any,
             today: Optional[str] = None) -> Dict[str, float]:
    """Caps fetched TODAY for the requested tickers. Missing keys are misses."""
    wanted = list(dict.fromkeys(str(t) for t in tickers))
    if not wanted:
        return {}
    now = today or _today()
    out: Dict[str, float] = {}
    for chunk_start in range(0, len(wanted), 400):
        chunk = wanted[chunk_start:chunk_start + 400]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT ticker, cap FROM pit_caps "
            f"WHERE fetched_at = ? AND ticker IN ({marks})",
            [now, *chunk]).fetchall()
        out.update({str(r[0]): float(r[1]) for r in rows})
    return out


def put_caps(conn: sqlite3.Connection, caps: Dict[str, Any],
             fetched_at: Optional[str] = None) -> None:
    """Store the caps that are KNOWN. A None is skipped, never stored."""
    stamp = fetched_at or _today()
    rows = [(str(t), stamp, float(v)) for t, v in (caps or {}).items()
            if v is not None]
    if not rows:
        return
    conn.executemany(
        "INSERT INTO pit_caps (ticker, fetched_at, cap) VALUES (?,?,?) "
        "ON CONFLICT(ticker) DO UPDATE SET fetched_at=excluded.fetched_at, "
        "cap=excluded.cap", rows)
    conn.commit()


def get_facts(conn: sqlite3.Connection, cik: int,
              as_of: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Cached companyfacts, or None if absent OR too old for ``as_of``.

    The original argument for caching this forever — consumers filter to
    ``filed <= as_of``, so a fresher fetch only adds rows that get discarded —
    holds ONLY while ``as_of`` is at or before the fetch. Past that the filter
    silently drops filings that genuinely existed by then.
    """
    row = conn.execute(
        "SELECT payload, fetched_at FROM pit_facts WHERE cik=?",
        (cik,)).fetchone()
    if not row or not _fresh_for(row[1], as_of):
        return None
    return dict(json.loads(row[0]))


def put_facts(conn: sqlite3.Connection, cik: int, payload: Dict[str, Any],
              fetched_at: Optional[str] = None) -> None:
    conn.execute(
        "INSERT INTO pit_facts (cik, payload, fetched_at) VALUES (?,?,?) "
        "ON CONFLICT(cik) DO UPDATE SET payload=excluded.payload, "
        "fetched_at=excluded.fetched_at",
        (cik, json.dumps(payload), fetched_at or _today()))
    conn.commit()
