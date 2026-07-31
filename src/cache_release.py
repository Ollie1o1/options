"""Release yfinance's on-disk caches while the process sits idle at a menu.

yfinance keeps three peewee-backed SQLite caches (timezones, cookies, ISINs),
each opened with ``journal_mode=wal``. A WAL database that is merely *open*
holds `<db>-wal` and `<db>-shm` alongside it, and those files exist for exactly
as long as the connection does. A launcher parked at a menu prompt is therefore
holding writer-side state for a database it is not using — for hours, if the
window is left open.

Closing them is safe and effectively invisible: peewee's ``SqliteDatabase``
defaults to ``autoconnect=True``, so the next cache read reconnects on its own.
Verified against the installed yfinance (see `docs/IDLE_CACHE_RELEASE.md`):
closing drops both sidecar files, and a subsequent read revives the connection
and recreates them.

Why this is the fix rather than a guess at one: an idle launcher was measured
holding ~41 handles to the tz cache, and **a WAL cannot checkpoint while a
reader is alive**. The WAL therefore grows unbounded and every connection from
a concurrent scan re-scans it under lock contention — a `-ds` window went from
2m22s to 55 minutes with a day-old launcher open. Closing the handles is
exactly what lets the WAL truncate, so it never reaches that size.

Not yet done: an end-to-end confirmation on the operator's machine (idle for
hours, then time a concurrent scan with and without this). See
`docs/IDLE_CACHE_RELEASE.md`, which also records a naive benchmark that failed
to reproduce the slowdown and why it was the wrong test — and that iCloud is a
dead end already chased twice.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Every yfinance cache manager exposing close_db(). Named rather than
# discovered so a yfinance upgrade that renames one shows up as a miss in
# `release_yfinance_caches`'s return value instead of silently doing nothing.
_MANAGER_NAMES = ("_TzDBManager", "_CookieDBManager", "_ISINDBManager")


def release_yfinance_caches() -> List[str]:
    """Close every yfinance cache database. Returns the names actually closed.

    Never raises: this runs on the way into a menu prompt, and a diagnostic
    convenience must not be able to take the launcher down with it.
    """
    closed: List[str] = []
    try:
        import yfinance.cache as _cache
    except Exception as exc:  # noqa: BLE001 - yfinance absent or broken
        logger.debug("yfinance cache module unavailable: %s", exc)
        return closed

    for name in _MANAGER_NAMES:
        manager = getattr(_cache, name, None)
        close = getattr(manager, "close_db", None) if manager else None
        if close is None:
            continue
        try:
            close()
            closed.append(name)
        except Exception as exc:  # noqa: BLE001 - closing is best-effort
            logger.debug("could not close %s: %s", name, exc)
    return closed
