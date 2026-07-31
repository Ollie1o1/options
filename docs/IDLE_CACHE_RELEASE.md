# Idle cache release — what was measured, and what was not

Date: 2026-07-31. Covers idea `idle-launcher-wal-guard`.

## The complaint

A launcher left open at a menu prompt has been observed to make scans in *other*
sessions 20-50x slower. Diagnosed by the operator on 2026-07-25, after the
symptom had twice been misattributed — first to iCloud sync, then to Yahoo rate
limits. The standing mitigation was tribal knowledge: "quit the idle launcher."

## What is definitely true

yfinance keeps three peewee-backed SQLite caches — timezone, cookie, ISIN — and
opens all three with `journal_mode=wal`:

```
pragmas={'journal_mode': 'wal', 'cache_size': -64}
```

A WAL database that is merely *open* keeps `<db>-wal` and `<db>-shm` beside it,
for as long as the connection lives. Measured against the installed yfinance:

| state | `-wal` | `-shm` |
|---|---|---|
| connection open | present | present |
| after `close_db()` | gone | gone |
| after next read (autoconnect) | present | present |
| closed again | gone | gone |

`peewee.SqliteDatabase` defaults to `autoconnect=True` — confirmed at runtime —
so closing is transparent: the next cache read reconnects by itself. Closing
costs one reconnect and nothing else.

So an idle launcher genuinely does hold writer-side state, indefinitely, for
three databases it is not using. That is worth not doing regardless of what it
costs, which is what `src/cache_release.py` now prevents: every menu prompt
releases the caches before waiting on the human.

## What could NOT be reproduced

The 20-50x slowdown itself. A local contention benchmark — one process holding
a WAL connection open while a second process did 300 connect-and-read cycles
against the same file — measured the opposite of the hypothesis:

| holder state | 300 connect+read in a second process |
|---|---|
| connection HELD open | 0.020 s |
| connection RELEASED | 0.104 s |

Holding the WAL open made the second process **five times faster**, because a
released database has to recreate the `-wal`/`-shm` sidecars on each new
connection. Plain WAL contention on a local disk therefore does not explain the
reported slowdown, and this change should not be described as having fixed it.

## What this means

- The release is shipped on handle-hygiene grounds: a process waiting on a
  human should not hold database handles. That is defensible on its own.
- It is **not** an established fix for the 20-50x slowdown. If the slowdown
  persists with this in place, the cause is elsewhere.
- The next place to look is the enclosing directory rather than the connection.
  The repo lives under `~/Desktop`, and on a synced Desktop every create and
  delete of a `-wal`/`-shm` sidecar is a sync event. That would make sidecar
  *churn* expensive in a way a local-disk benchmark cannot show — and it would
  also explain why the symptom was originally mistaken for an iCloud problem.
  Testing that means timing the same scan with the repo on local-only storage,
  which needs the operator's machine and was out of scope here.

## Reproducing the measurements

Both experiments are small and self-contained; the numbers above came from
running peewee against a temporary WAL database and inspecting the sidecar
files, and from `inspect.getsource` over the installed `yfinance.cache`. The
manager names and the WAL pragma are pinned by `tests/test_cache_release.py`,
so a yfinance upgrade that renames a manager or drops WAL fails the suite
rather than silently turning the release into a no-op.
