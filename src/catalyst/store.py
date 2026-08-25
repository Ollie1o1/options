"""Persistence for catalyst events and their forward outcomes.

WHY THIS EXISTS. A board that only renders today's state can never be checked.
This repo already learned that the hard way: the trade ledger cannot validate
the thing that selects into it, which is why data/candidates.db records the
REFUSED population. Same move here — every event is stamped with when WE first
saw it, and every later observation lands in catalyst_marks. In twelve months
that makes "did the funded-through flag predict anything?" an answerable
question instead of a matter of taste.

Dates move constantly, so a re-observation is never an update-in-place: the
event row carries the latest date, and the full observation series lives in the
marks table. Slippage is DERIVED from that series, never stored as a fact — a
stored difference is a second copy that can silently disagree with its source.

NULL means not recorded, never zero.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from typing import List, Optional, Tuple

from src.catalyst.models import CatalystEvent
from src.paths import repo_path

DEFAULT_DB = repo_path(os.path.join("data", "catalysts.db"))

_DDL = [
    """CREATE TABLE IF NOT EXISTS catalyst_events (
        event_id       TEXT PRIMARY KEY,
        nct_id         TEXT NOT NULL,
        ticker         TEXT NOT NULL,
        sponsor_name   TEXT NOT NULL,
        event_type     TEXT NOT NULL,
        phase          TEXT NOT NULL,
        event_date     TEXT NOT NULL,
        date_precision TEXT NOT NULL,
        date_type      TEXT NOT NULL,
        brief_title    TEXT,
        enrollment     INTEGER,
        allocation     TEXT,
        masking        TEXT,
        primary_outcome TEXT,
        mcap_at_seen   REAL,
        first_seen     TEXT NOT NULL,
        last_seen      TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS catalyst_marks (
        event_id      TEXT NOT NULL,
        marked_at     TEXT NOT NULL,
        observed_date TEXT,
        status        TEXT,
        spot          REAL,
        PRIMARY KEY (event_id, marked_at)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_date ON catalyst_events(event_date)",
    "CREATE INDEX IF NOT EXISTS idx_events_ticker ON catalyst_events(ticker)",
]


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    path = repo_path(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    return conn


def upsert_event(conn: sqlite3.Connection, event: CatalystEvent,
                 seen_at: str) -> None:
    """Insert, or refresh an existing row. ``first_seen`` is written once and
    never moves — it is the only record of when this became knowable."""
    t = event.trial
    conn.execute(
        """INSERT INTO catalyst_events (
               event_id, nct_id, ticker, sponsor_name, event_type, phase,
               event_date, date_precision, date_type, brief_title, enrollment,
               allocation, masking, primary_outcome, mcap_at_seen,
               first_seen, last_seen)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(event_id) DO UPDATE SET
               event_date=excluded.event_date,
               date_precision=excluded.date_precision,
               date_type=excluded.date_type,
               enrollment=excluded.enrollment,
               mcap_at_seen=excluded.mcap_at_seen,
               last_seen=excluded.last_seen""",
        (event.event_id, t.nct_id, event.ticker, t.sponsor_name,
         event.event_type, t.phase, t.event_date, t.date_precision,
         t.date_type, t.brief_title, t.enrollment, t.allocation, t.masking,
         t.primary_outcome, event.mcap, seen_at, seen_at))
    conn.commit()


def add_mark(conn: sqlite3.Connection, event_id: str, marked_at: str,
             observed_date: Optional[str], status: Optional[str],
             spot: Optional[float]) -> None:
    """Record one observation. Re-marking the same day replaces that day."""
    conn.execute(
        """INSERT INTO catalyst_marks (event_id, marked_at, observed_date,
                                       status, spot)
           VALUES (?,?,?,?,?)
           ON CONFLICT(event_id, marked_at) DO UPDATE SET
               observed_date=excluded.observed_date,
               status=excluded.status,
               spot=excluded.spot""",
        (event_id, marked_at, observed_date, status, spot))
    conn.commit()


def _as_date(text: str) -> dt.date:
    """Parse a CT.gov date. Month precision anchors to the 1st — the same
    convention on both ends of a subtraction, so a month-to-month slip is a
    real interval and not an artefact of where we anchored."""
    parts = text.split("-")
    if len(parts) == 2:
        return dt.date(int(parts[0]), int(parts[1]), 1)
    return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))


def slippage(conn: sqlite3.Connection, event_id: str) -> Optional[int]:
    """Days between the earliest and latest observed date. None with fewer
    than two observations — one point is not a trend."""
    rows = conn.execute(
        """SELECT observed_date FROM catalyst_marks
           WHERE event_id=? AND observed_date IS NOT NULL
           ORDER BY marked_at""", (event_id,)).fetchall()
    if len(rows) < 2:
        return None
    try:
        return (_as_date(rows[-1][0]) - _as_date(rows[0][0])).days
    except (ValueError, IndexError):
        return None


def outstanding(conn: sqlite3.Connection,
                as_of: str) -> List[Tuple[str, str, str]]:
    """(event_id, ticker, event_date) for events whose date has passed."""
    return [(r[0], r[1], r[2]) for r in conn.execute(
        """SELECT event_id, ticker, event_date FROM catalyst_events
           WHERE event_date <= ? ORDER BY event_date""", (as_of,)).fetchall()]
