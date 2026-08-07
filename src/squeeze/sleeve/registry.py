"""Point-in-time registry of treated and control cohorts, frozen at formation.

Each FINRA cycle forms its cohorts once and never revises them. Choosing
controls later, from names that happened to survive, would rebuild exactly the
survivorship bias the asymmetry study was built to avoid — and it would do so
invisibly, because the resulting panel looks well-formed.

Coverage is recorded per cycle rather than inferred. A cycle whose chains were
never snapshotted is MISSING, and missingness is an input to the gate's
validity check: silently skipping those cycles would select for quiet days and
bias implied vol downward, which is the direction that flatters the strategy.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Sequence
from src.paths import repo_path

DEFAULT_DB = repo_path("data/squeeze_sleeve.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cohort (
    cycle_date TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    arm        TEXT NOT NULL,
    si_decile  INTEGER,
    rv         REAL,
    log_mcap   REAL,
    log_price  REAL,
    PRIMARY KEY (cycle_date, symbol)
);
CREATE TABLE IF NOT EXISTS cycle (
    cycle_date TEXT PRIMARY KEY,
    covered    INTEGER NOT NULL DEFAULT 0
);
"""


def ensure_db(db_path: str = DEFAULT_DB) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def open_cycle(cycle_date: str, treated: Sequence[dict],
               controls: Sequence[dict], db_path: str = DEFAULT_DB) -> int:
    """Freeze one cycle's cohorts. Re-opening an existing cycle is a no-op."""
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute("SELECT 1 FROM cycle WHERE cycle_date=?",
                              (cycle_date,)).fetchone()
        if exists:
            return 0
        rows = []
        for members, arm in ((treated, "treated"), (controls, "control")):
            for m in members:
                rows.append((cycle_date, m["symbol"], m.get("arm", arm),
                             m.get("si_decile"), m.get("rv"),
                             m.get("log_mcap"), m.get("log_price")))
        conn.executemany(
            "INSERT OR IGNORE INTO cohort (cycle_date, symbol, arm, si_decile,"
            " rv, log_mcap, log_price) VALUES (?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO cycle (cycle_date, covered) VALUES (?, 0)",
                     (cycle_date,))
        return len(rows)


def cycle_members(cycle_date: str, db_path: str = DEFAULT_DB) -> List[dict]:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, arm, si_decile, rv, log_mcap, log_price"
            " FROM cohort WHERE cycle_date=? ORDER BY arm, symbol",
            (cycle_date,)).fetchall()
    return [dict(r) for r in rows]


def cycles(db_path: str = DEFAULT_DB) -> List[str]:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(
            "SELECT cycle_date FROM cycle ORDER BY cycle_date")]


def mark_coverage(cycle_date: str, covered: bool,
                  db_path: str = DEFAULT_DB) -> None:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE cycle SET covered=? WHERE cycle_date=?",
                     (1 if covered else 0, cycle_date))


def coverage(db_path: str = DEFAULT_DB) -> Dict[str, bool]:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return {r[0]: bool(r[1]) for r in conn.execute(
            "SELECT cycle_date, covered FROM cycle ORDER BY cycle_date")}
