"""Content cache for point-in-time reconstruction.

Two DIFFERENT caching arguments live here, and conflating them would be a bug.

  * ``(nct_id, version)`` is genuinely IMMUTABLE. A study version is a frozen
    historical record; cache forever, never invalidate.
  * ``(cik)`` companyfacts is NOT immutable — it GROWS as new filings arrive.
    It is cached anyway, but for a different reason: every consumer filters to
    ``filed <= as_of``, so a fresher fetch only adds rows that are then
    discarded. Append-safe, not immutable. Nothing may read it without
    applying that filter.

A cache MISS is None. An empty fetched result is []. Collapsing those would
make "we never looked" indistinguishable from "there is nothing there".
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

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


def get_versions(conn: sqlite3.Connection,
                 nct_id: str) -> Optional[List[Dict[str, Any]]]:
    row = conn.execute("SELECT payload FROM pit_versions WHERE nct_id=?",
                       (nct_id,)).fetchone()
    return list(json.loads(row[0])) if row else None


def put_versions(conn: sqlite3.Connection, nct_id: str,
                 versions: List[Dict[str, Any]]) -> None:
    conn.execute(
        "INSERT INTO pit_versions (nct_id, payload) VALUES (?,?) "
        "ON CONFLICT(nct_id) DO UPDATE SET payload=excluded.payload",
        (nct_id, json.dumps(versions)))
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


def get_facts(conn: sqlite3.Connection, cik: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT payload FROM pit_facts WHERE cik=?",
                       (cik,)).fetchone()
    return dict(json.loads(row[0])) if row else None


def put_facts(conn: sqlite3.Connection, cik: int,
              payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO pit_facts (cik, payload) VALUES (?,?) "
        "ON CONFLICT(cik) DO UPDATE SET payload=excluded.payload",
        (cik, json.dumps(payload)))
    conn.commit()
