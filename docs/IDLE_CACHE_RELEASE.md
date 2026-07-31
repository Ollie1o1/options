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

## The mechanism, as actually diagnosed

This was diagnosed on the operator's machine on 2026-07-29, and the detail
matters because it is what makes the release the right fix:

An idle launcher holds **~41 open handles** to
`~/Library/Caches/py-yfinance/tkr-tz.db`. The WAL cannot **checkpoint** while a
reader is alive — the file's mtime freezes at the launcher's start time — so
the WAL grows unbounded and every new connection from a concurrent scan
re-scans the whole thing under lock contention. The stack sits in
`sqlite3_step -> btreeBeginTrans -> sqlite3PagerSharedLock -> pread/fcntl`,
around 58% of samples, with no ESTABLISHED socket — which is why it looks like
a network stall and is not one.

Measured cost at the time: a `-ds` window that ran 2m22s went to **55 minutes
with zero rows** with a day-old launcher open; after quitting it, the same
window finished in ~2.5 minutes.

Diagnose a recurrence with `sample <pid>` and
`lsof ~/Library/Caches/py-yfinance/tkr-tz.db`.

## A benchmark that did NOT reproduce it, and why it was the wrong test

A naive local experiment — one connection held open while a second process did
300 connect-and-read cycles — measured the opposite:

| holder state | 300 connect+read in a second process |
|---|---|
| connection HELD open | 0.020 s |
| connection RELEASED | 0.104 s |

That is a real result but it does not bear on the problem, because it recreates
neither of the conditions that cause it: a single handle rather than ~41, and a
tiny WAL that never had the chance to grow un-checkpointed. Cheap reads against
a small WAL are faster when a connection is already warm. The pathology is a
*large, un-checkpointable* WAL, which takes a long-lived idle session to build.
Recorded here so the negative result is not mistaken for evidence that holding
handles is harmless.

## Why the release is the right fix

Closing the handles is precisely what lets the WAL checkpoint. The release
targets the diagnosed mechanism rather than a guess at it: no reader alive at
the menu prompt means the WAL can truncate, so it never reaches the size that
makes concurrent scans pathological.

What has NOT been done is an end-to-end confirmation on the operator's machine
— leaving a launcher idle for hours, then timing a concurrent scan with and
without the release. That is the check that would close this out, and it needs
a real session and real elapsed time.

**A dead end, already chased twice — do not chase it again.** iCloud is not the
cause. The project lives under a synced `~/Desktop` and `fileproviderd` does
hold a write handle on `yf_disk_cache.db`, but a timed read of a synced copy
against a `/private/tmp` copy is identical (0.005s both).

## Reproducing the measurements

Both experiments are small and self-contained; the numbers above came from
running peewee against a temporary WAL database and inspecting the sidecar
files, and from `inspect.getsource` over the installed `yfinance.cache`. The
manager names and the WAL pragma are pinned by `tests/test_cache_release.py`,
so a yfinance upgrade that renames a manager or drops WAL fails the suite
rather than silently turning the release into a no-op.
