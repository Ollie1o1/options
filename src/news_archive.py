"""Point-in-time archive of fetched news items.

The screener stores a scalar ``sentiment_score_norm`` per trade, but the raw
headlines that produced it are discarded after each scan. Without the corpus
you can never retroactively engineer a better sentiment feature, nor audit
what the model "saw" when a trade was opened.

This module persists every news item the fetcher surfaces, stamped with the
fetch time, into ``data/news_archive.db``. It is best-effort and append-only:
de-duplicated on ``(symbol, headline, published-date)`` so re-scanning the
same day does not inflate the corpus.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                          "news_archive.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_archive (
    id          INTEGER PRIMARY KEY,
    dedup_key   TEXT UNIQUE,
    symbol      TEXT,
    headline    TEXT,
    source      TEXT,
    published   TEXT,
    sentiment   REAL,
    relevance   REAL,
    url         TEXT,
    archived_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_archive(symbol);
CREATE INDEX IF NOT EXISTS idx_news_archived ON news_archive(archived_at);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _published_str(published) -> str:
    if isinstance(published, datetime):
        return published.astimezone(timezone.utc).isoformat()
    return str(published) if published is not None else ""


def _dedup_key(symbol: str, headline: str, published) -> str:
    day = _published_str(published)[:10]
    raw = f"{symbol.upper()}|{headline.strip().lower()}|{day}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def archive_items(items, symbol: str, db_path: str = DEFAULT_DB,
                  now: Optional[datetime] = None) -> int:
    """Persist *items* for *symbol*; return the number of NEW rows inserted.

    *items* is any iterable of objects exposing ``headline``, ``source``,
    ``published``, ``sentiment``, ``relevance`` and ``url`` (the fetcher's
    NewsItem). De-duplicated on (symbol, headline, published-date).
    """
    if not items:
        return 0
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    conn = _connect(db_path)
    inserted = 0
    try:
        for it in items:
            headline = (getattr(it, "headline", "") or "").strip()
            if not headline:
                continue
            published = getattr(it, "published", None)
            key = _dedup_key(symbol, headline, published)
            cur = conn.execute(
                "INSERT OR IGNORE INTO news_archive "
                "(dedup_key, symbol, headline, source, published, sentiment, "
                " relevance, url, archived_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (key, symbol.upper(), headline,
                 getattr(it, "source", "") or "",
                 _published_str(published),
                 float(getattr(it, "sentiment", 0.0) or 0.0),
                 float(getattr(it, "relevance", 0.0) or 0.0),
                 getattr(it, "url", "") or "",
                 stamp),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def archive_stats(db_path: str = DEFAULT_DB) -> dict:
    """Return factual coverage stats; zeroed (never raises) if DB absent."""
    base = {"total": 0, "symbols": 0, "archive_days": 0, "latest": None}
    if not os.path.exists(db_path):
        return base
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT symbol), "
                "COUNT(DISTINCT substr(archived_at,1,10)), MAX(archived_at) "
                "FROM news_archive"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return base
    if not row:
        return base
    return {
        "total": row[0] or 0,
        "symbols": row[1] or 0,
        "archive_days": row[2] or 0,
        "latest": row[3],
    }


def format_stats_line(stats: dict) -> str:
    """One-line factual summary of the point-in-time news corpus."""
    if not stats or not stats.get("total"):
        return "  Point-in-time news archive: empty"
    latest = (stats.get("latest") or "")[:10] or "n/a"
    return (f"  Point-in-time news archive: {stats['total']} items across "
            f"{stats['symbols']} tickers, {stats['archive_days']} days "
            f"(latest {latest})")
