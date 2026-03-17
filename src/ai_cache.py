"""Same-day SQLite cache for AI option scores.

Cache key: (symbol, option_type, strike_bucket, expiry, iv_rank_bucket, trade_date)
TTL: current trading day (entries from a prior date are stale).

Usage:
    cache = AIScoreCache()
    result = cache.get(row)         # returns dict or None
    cache.set(row, score_dict)      # stores result
    cache.clear_stale()             # removes prior-day entries
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Any
import math

_DB_PATH = Path(__file__).resolve().parent.parent / ".ai_score_cache.db"


def _safe_json_loads(value: str, default):
    """Parse JSON string, returning default on any error."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return default
_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ai_scores (
    fingerprint   TEXT PRIMARY KEY,
    ai_score      REAL,
    reasoning     TEXT,
    flags         TEXT,
    catalyst_risk TEXT,
    iv_justified  INTEGER,
    ai_confidence REAL,
    trade_date    TEXT,
    scored_at     TEXT
);
"""

_TICKER_CTX_DDL = """
CREATE TABLE IF NOT EXISTS ticker_contexts (
    fingerprint  TEXT PRIMARY KEY,
    context_json TEXT,
    trade_date   TEXT,
    scored_at    TEXT
);
"""

def _trade_date() -> str:
    """Return today's date string in UTC."""
    return date.today().isoformat()

def _make_fingerprint(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol", "")).upper()
    opt_type = str(row.get("type", "")).lower()
    strike = row.get("strike", 0) or 0
    # Bucket to nearest $5 — use int arithmetic to avoid float rounding issues
    strike_bucket = int(round(float(strike) / 5)) * 5
    expiry = str(row.get("expiration", ""))[:10]
    iv_rank = row.get("iv_rank", 0.5) or 0.5
    # Bucket IV rank to nearest 0.05 — round to int step first, then scale back
    iv_bucket = round(round(float(iv_rank) * 20) / 20, 2)
    trade_date = _trade_date()
    return f"{symbol}|{opt_type}|{strike_bucket}|{expiry}|{iv_bucket}|{trade_date}"


class AIScoreCache:
    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_TABLE_DDL)
            conn.execute(_TICKER_CTX_DDL)

    def get(self, row: dict[str, Any]) -> Optional[dict]:
        fp = _make_fingerprint(row)
        today = _trade_date()
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM ai_scores WHERE fingerprint=? AND trade_date=?",
                (fp, today),
            )
            r = cur.fetchone()
        if r is None:
            return None
        return {
            "id": str(row.get("_id", "")),
            "ai_score": float(r["ai_score"]),
            "reasoning": r["reasoning"] or "",
            "flags": _safe_json_loads(r["flags"], []),
            "catalyst_risk": r["catalyst_risk"] or "medium",
            "iv_justified": bool(r["iv_justified"]),
            "ai_confidence": float(r["ai_confidence"] or 5.0),
        }

    def set(self, row: dict[str, Any], result: dict) -> None:
        fp = _make_fingerprint(row)
        today = _trade_date()
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ai_scores
                   (fingerprint, ai_score, reasoning, flags, catalyst_risk,
                    iv_justified, ai_confidence, trade_date, scored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fp,
                    float(result.get("ai_score", 50)),
                    str(result.get("reasoning", "")),
                    json.dumps(result.get("flags", [])),
                    str(result.get("catalyst_risk", "medium")),
                    int(bool(result.get("iv_justified", True))),
                    float(result.get("ai_confidence", 5.0)),
                    today,
                    now,
                ),
            )

    def get_ticker_context(self, symbol: str) -> Optional[dict]:
        fp = f"ticker_ctx:{symbol.upper()}:{_trade_date()}"
        with self._conn() as conn:
            r = conn.execute(
                "SELECT context_json FROM ticker_contexts WHERE fingerprint=?", (fp,)
            ).fetchone()
        if r is None:
            return None
        return _safe_json_loads(r["context_json"], None)

    def set_ticker_context(self, symbol: str, ctx: dict) -> None:
        fp = f"ticker_ctx:{symbol.upper()}:{_trade_date()}"
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ticker_contexts
                   (fingerprint, context_json, trade_date, scored_at)
                   VALUES (?, ?, ?, ?)""",
                (fp, json.dumps(ctx), _trade_date(), now),
            )

    def clear_stale(self) -> int:
        today = _trade_date()
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM ai_scores WHERE trade_date != ?", (today,))
            return cur.rowcount

    def stats(self) -> dict:
        today = _trade_date()
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM ai_scores").fetchone()[0]
            today_count = conn.execute(
                "SELECT COUNT(*) FROM ai_scores WHERE trade_date=?", (today,)
            ).fetchone()[0]
        return {"total": total, "today": today_count}
