"""Expired rows in the yfinance disk cache are never deleted.

Measured on the real cache 2026-07-30: 4,207 of 4,782 rows expired (88%), and
49.7MB of the 57.5MB file was dead payload against 3.4MB live. Nothing in the
code path ever issued a DELETE, so the file only ever grew and every lookup
carried the weight.

The delete boundary must match what `_yf_disk_get` considers readable: it
selects `expires > now`, so a row at exactly `now` is already a miss and is safe
to remove. Deleting anything with `expires > now` would throw away live data.

VACUUM is what actually returns the space to the filesystem, but it rewrites the
whole file, so it is throttled rather than run on every launch.
"""
import os
import sqlite3
import tempfile
import time
import unittest

from src.data_fetching import prune_yf_disk_cache


class PruneTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "yf_disk_cache.db")
        self.now = int(time.time())
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE yf_cache (key TEXT PRIMARY KEY, value BLOB, expires INTEGER)")
        conn.commit()
        conn.close()

    def add(self, key, expires, payload=b"x" * 1024):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT OR REPLACE INTO yf_cache VALUES (?,?,?)", (key, payload, expires))
        conn.commit()
        conn.close()

    def keys(self):
        conn = sqlite3.connect(self.db)
        out = {r[0] for r in conn.execute("SELECT key FROM yf_cache")}
        conn.close()
        return out


class DeletesOnlyDeadRows(PruneTestBase):
    def test_expired_rows_are_deleted(self):
        self.add("old", self.now - 100)
        prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(self.keys(), set())

    def test_live_rows_are_kept(self):
        self.add("fresh", self.now + 900)
        prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(self.keys(), {"fresh"})

    def test_a_row_expiring_exactly_now_is_dead(self):
        # _yf_disk_get uses `expires > now`, so this row can never be read again.
        self.add("boundary", self.now)
        prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(self.keys(), set())

    def test_a_row_expiring_one_second_out_is_still_live(self):
        self.add("barely", self.now + 1)
        prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(self.keys(), {"barely"})

    def test_mixed_cache_keeps_exactly_the_live_rows(self):
        self.add("a", self.now - 10)
        self.add("b", self.now + 10)
        self.add("c", self.now - 5000)
        self.add("d", self.now + 5000)
        prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(self.keys(), {"b", "d"})

    def test_reports_how_many_rows_it_deleted(self):
        self.add("a", self.now - 10)
        self.add("b", self.now - 10)
        self.add("c", self.now + 10)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(res["deleted"], 2)


class ReclaimsSpace(PruneTestBase):
    def test_vacuum_shrinks_the_file_on_disk(self):
        for i in range(200):
            self.add(f"dead{i}", self.now - 100, payload=b"y" * 4096)
        before = os.path.getsize(self.db)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertTrue(res["vacuumed"])
        self.assertLess(os.path.getsize(self.db), before)

    def test_nothing_to_delete_skips_the_vacuum(self):
        # VACUUM rewrites the entire file; doing that for zero reclaimed bytes
        # is pure cost on a 57MB database.
        self.add("fresh", self.now + 900)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(res["deleted"], 0)
        self.assertFalse(res["vacuumed"])


class Throttled(PruneTestBase):
    def test_a_second_run_inside_the_interval_does_nothing(self):
        self.add("a", self.now - 10)
        prune_yf_disk_cache(db_path=self.db, now=self.now)
        self.add("b", self.now - 10)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now + 60)
        self.assertEqual(res["deleted"], 0)
        self.assertEqual(res.get("skipped"), "throttled")
        self.assertEqual(self.keys(), {"b"})

    def test_a_run_after_the_interval_prunes_again(self):
        self.add("a", self.now - 10)
        prune_yf_disk_cache(db_path=self.db, now=self.now, min_interval_s=3600)
        self.add("b", self.now - 10)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now + 3601,
                                  min_interval_s=3600)
        self.assertEqual(res["deleted"], 1)
        self.assertEqual(self.keys(), set())

    def test_force_bypasses_the_throttle(self):
        self.add("a", self.now - 10)
        prune_yf_disk_cache(db_path=self.db, now=self.now)
        self.add("b", self.now - 10)
        res = prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(res["deleted"], 1)


class NeverBreaksTheApp(PruneTestBase):
    def test_a_missing_database_is_not_an_error(self):
        res = prune_yf_disk_cache(db_path=os.path.join(self.dir, "nope.db"),
                                  now=self.now, force=True)
        self.assertEqual(res["deleted"], 0)

    def test_a_corrupt_database_is_not_an_error(self):
        bad = os.path.join(self.dir, "bad.db")
        with open(bad, "w") as f:
            f.write("this is not a database")
        res = prune_yf_disk_cache(db_path=bad, now=self.now, force=True)
        self.assertEqual(res["deleted"], 0)

    def test_an_empty_cache_is_a_no_op(self):
        res = prune_yf_disk_cache(db_path=self.db, now=self.now, force=True)
        self.assertEqual(res["deleted"], 0)
        self.assertFalse(res["vacuumed"])


class WiredIntoCacheInit(unittest.TestCase):
    """Pruning that nothing calls is pruning that never happens. Cache init is
    the one place guaranteed to run in every process that touches the cache —
    the screener, a scan, the catch-up windows — and it is already lock-guarded
    and idempotent. It must not block: the prune runs off the calling thread."""

    def setUp(self):
        import src.data_fetching as D
        self.D = D
        self._orig = D.prune_yf_disk_cache
        self._orig_init = D._YF_DISK_INITIALIZED[0]
        self.calls = []

    def tearDown(self):
        self.D.prune_yf_disk_cache = self._orig
        self.D._YF_DISK_INITIALIZED[0] = self._orig_init

    def test_cache_init_triggers_a_prune(self):
        import threading
        done = threading.Event()

        def _fake(*a, **k):
            self.calls.append(1)
            done.set()
            return {"deleted": 0, "vacuumed": False, "skipped": None}

        self.D.prune_yf_disk_cache = _fake
        self.D._YF_DISK_INITIALIZED[0] = False
        self.D._yf_disk_init()
        self.assertTrue(done.wait(timeout=5.0), "cache init never pruned")

    def test_cache_init_does_not_block_on_a_slow_prune(self):
        import threading
        import time as _t
        release = threading.Event()

        def _slow(*a, **k):
            release.wait(timeout=10)
            return {"deleted": 0, "vacuumed": False, "skipped": None}

        self.D.prune_yf_disk_cache = _slow
        self.D._YF_DISK_INITIALIZED[0] = False
        t0 = _t.time()
        self.D._yf_disk_init()
        elapsed = _t.time() - t0
        release.set()
        self.assertLess(elapsed, 1.0, "cache init waited on the prune")

    def test_a_raising_prune_cannot_break_cache_init(self):
        import threading
        reached = threading.Event()

        def _boom(*a, **k):
            reached.set()
            raise RuntimeError("vacuum failed")

        self.D.prune_yf_disk_cache = _boom
        self.D._YF_DISK_INITIALIZED[0] = False
        self.D._yf_disk_init()  # must not raise
        self.assertTrue(self.D._YF_DISK_INITIALIZED[0])
        self.assertTrue(reached.wait(timeout=5.0))

    def test_a_raising_prune_does_not_print_a_traceback(self):
        # An exception escaping a thread target dumps a traceback to stderr from
        # a context no caller can catch — alarming noise mid-scan, and it made
        # the test suite's own output dirty.
        import contextlib
        import io
        import threading
        reached = threading.Event()

        def _boom(*a, **k):
            reached.set()
            raise RuntimeError("vacuum failed")

        self.D.prune_yf_disk_cache = _boom
        self.D._YF_DISK_INITIALIZED[0] = False
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.D._yf_disk_init()
            reached.wait(timeout=5.0)
            # Give the thread a moment to unwind past the raise.
            for t in threading.enumerate():
                if t is not threading.current_thread() and not t.daemon:
                    t.join(timeout=1)
            threading.Event().wait(0.2)
        self.assertNotIn("Traceback", err.getvalue())
        self.assertNotIn("vacuum failed", err.getvalue())


if __name__ == "__main__":
    unittest.main()
