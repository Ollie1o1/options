"""Point-in-time store for prediction-market quotes.

`archived_at` is the whole point: feeds revise, and a `close_time` or a
back-filled price can move after the fact, but the day WE saw a quote cannot.
Same reasoning as `news_archive.archived_at`, which is the only reason the
news question was answerable at all.

One row per (ticker, archived_at). Re-running a day replaces that day rather
than duplicating it, so the job is safe to retry.

NULL means not recorded. A missing bid is NULL, never 0.0 — on this data a
zero bid is a real and meaningful value (a one-sided book), so conflating the
two would destroy the distinction that matters most.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Sequence

from src.paths import repo_path

DEFAULT_DB = repo_path(os.path.join("data", "predmarkets.db"))

_DDL = [
    """CREATE TABLE IF NOT EXISTS pm_quotes (
        ticker        TEXT NOT NULL,
        archived_at   TEXT NOT NULL,
        series        TEXT NOT NULL,
        title         TEXT,
        yes_bid       REAL,
        yes_ask       REAL,
        last          REAL,
        open_interest REAL,
        close_time    TEXT,
        PRIMARY KEY (ticker, archived_at)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_pm_series ON pm_quotes(series)",
    "CREATE INDEX IF NOT EXISTS idx_pm_archived ON pm_quotes(archived_at)",
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


def record(conn: sqlite3.Connection, quotes: Sequence, archived_at: str) -> int:
    """Store one day's quotes. Returns the number written."""
    rows = [(q.ticker, archived_at, q.series, q.title, q.yes_bid, q.yes_ask,
             q.last, q.open_interest, q.close_time) for q in quotes]
    conn.executemany(
        """INSERT INTO pm_quotes (ticker, archived_at, series, title, yes_bid,
                                  yes_ask, last, open_interest, close_time)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker, archived_at) DO UPDATE SET
               yes_bid=excluded.yes_bid, yes_ask=excluded.yes_ask,
               last=excluded.last, open_interest=excluded.open_interest,
               close_time=excluded.close_time""",
        rows)
    conn.commit()
    return len(rows)
