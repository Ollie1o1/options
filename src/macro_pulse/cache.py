"""Per-day cache for the synthesized macro narrative — one AI call per day per
distinct news state. Keyed on sorted headline titles + the calendar day.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Optional


def default_db_path() -> str:
    return os.path.join("data", "macro_pulse_cache.db")


def bundle_key(top_titles: list[str], day: str) -> str:
    payload = "|".join(sorted(t.strip() for t in top_titles)) + "@" + day
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS narrative_cache "
                 "(key TEXT PRIMARY KEY, payload_json TEXT)")
    return conn


def get(key: str, db_path: str) -> Optional[dict]:
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT payload_json FROM narrative_cache WHERE key = ?",
            (key,)).fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (ValueError, TypeError):
        return None


def put(key: str, payload: dict, db_path: str) -> None:
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO narrative_cache (key, payload_json) "
            "VALUES (?, ?)", (key, json.dumps(payload)))
        conn.commit()
        conn.close()
    except (sqlite3.Error, TypeError):
        pass
