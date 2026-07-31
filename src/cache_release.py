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

Honesty about scope, because this module was written to chase a specific
complaint: an idle launcher has been observed to make scans in other sessions
20-50x slower, and a local WAL contention benchmark did **not** reproduce that
slowdown — holding the connection open measured *faster* than reopening per
read. So this is not a proven fix for that symptom. What it is: the process no
longer holds database handles it has no use for while waiting on a human, which
removes a whole class of cross-session interference from the picture and costs
one reconnect. If the slowdown survives this, the cause is elsewhere and the
next place to look is the enclosing directory (a synced folder makes sidecar
file churn expensive) rather than the connection.
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
