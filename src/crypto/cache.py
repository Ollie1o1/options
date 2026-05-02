"""SQLite-backed disk cache for crypto data fetches.

Mirrors the yf_disk_cache.db pattern from src.data_fetching: each cached blob
is keyed by (source, key) with a TTL. Chains expire fast (5 min), funding
even faster (60s), spot bars hold longer (1h for daily bars).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "crypto_cache.db",
)

_DEFAULT_TTLS = {
    "deribit_chain": 300,        # 5 min
    "deribit_index": 60,         # 1 min
    "binance_funding": 60,       # 1 min
    "yf_history": 3600,          # 1 hour for daily bars
}


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crypto_cache (
            source TEXT NOT NULL,
            key TEXT NOT NULL,
            payload TEXT NOT NULL,
            stored_at INTEGER NOT NULL,
            PRIMARY KEY (source, key)
        )
        """
    )
    return conn


def get(source: str, key: str, ttl: Optional[int] = None) -> Optional[Any]:
    """Return cached payload if it exists and is fresher than TTL, else None."""
    if ttl is None:
        ttl = _DEFAULT_TTLS.get(source, 300)
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT payload, stored_at FROM crypto_cache WHERE source=? AND key=?",
                (source, key),
            ).fetchone()
        if row is None:
            return None
        payload_json, stored_at = row
        if int(time.time()) - int(stored_at) > ttl:
            return None
        return json.loads(payload_json)
    except (sqlite3.Error, json.JSONDecodeError):
        return None


def put(source: str, key: str, payload: Any) -> None:
    """Persist payload under (source, key). Upserts on conflict."""
    try:
        blob = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO crypto_cache(source, key, payload, stored_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source, key) DO UPDATE SET
                    payload = excluded.payload,
                    stored_at = excluded.stored_at
                """,
                (source, key, blob, int(time.time())),
            )
    except sqlite3.Error:
        pass
