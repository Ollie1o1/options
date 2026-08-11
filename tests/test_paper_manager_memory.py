"""`PaperManager(db_path=":memory:")` must be a working database.

`_get_connection` opens and CLOSES a connection per operation. An in-memory
SQLite database lives and dies with its connection, so every operation was
getting a brand-new empty one — `_init_db()` built the schema on a connection
that was then discarded, and the next call saw a database with **zero tables**.
Measured 2026-08-11, before the fix:

    PaperManager(db_path=":memory:")
    tables after init: []

Five test modules pass the literal and only pass because they never query it.
Anything that did would fail with "no such table: trades".

This was masked until 2026-08-11 by a separate bug: `repo_path` resolved
`:memory:` into a real file at the repo root, so the "in-memory" managers were
quietly sharing one database on disk. Fixing that (PR #30) made the keyword
genuine and exposed this.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_paper_manager_memory -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src.paper_manager import PaperManager


class InMemoryLedgerTest(unittest.TestCase):
    def setUp(self):
        self.mgr = PaperManager(db_path=":memory:")

    def _tables(self, mgr):
        with mgr._get_connection() as conn:
            return {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}

    def test_the_schema_survives_past_the_call_that_created_it(self):
        self.assertIn("trades", self._tables(self.mgr))

    def test_a_write_is_visible_to_a_later_read(self):
        with self.mgr._get_connection() as conn:
            conn.execute("INSERT INTO trades (ticker, status) VALUES ('AAA','OPEN')")
        with self.mgr._get_connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM trades WHERE ticker='AAA'").fetchone()[0]
        self.assertEqual(n, 1)

    def test_two_managers_do_not_share_a_database(self):
        # The isolation the literal is chosen FOR. Sharing was the state of the
        # world until PR #30, via a real file at the repo root, and a
        # `file::memory:?cache=shared` fix would have reintroduced it
        # process-wide.
        other = PaperManager(db_path=":memory:")
        with self.mgr._get_connection() as conn:
            conn.execute("INSERT INTO trades (ticker, status) VALUES ('AAA','OPEN')")
        with other._get_connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        self.assertEqual(n, 0)

    def test_a_failed_operation_rolls_back(self):
        with self.assertRaises(sqlite3.OperationalError):
            with self.mgr._get_connection() as conn:
                conn.execute("INSERT INTO trades (ticker, status) VALUES ('BBB','OPEN')")
                conn.execute("SELECT * FROM no_such_table")
        with self.mgr._get_connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM trades WHERE ticker='BBB'").fetchone()[0]
        self.assertEqual(n, 0, "the failed transaction was not rolled back")

    def test_the_connection_stays_usable_after_a_failure(self):
        try:
            with self.mgr._get_connection() as conn:
                conn.execute("SELECT * FROM no_such_table")
        except sqlite3.OperationalError:
            pass
        self.assertIn("trades", self._tables(self.mgr))


class FileBackedLedgerStillWorksTest(unittest.TestCase):
    """The regression guard: the ordinary on-disk path must be untouched.

    Never names the real ledger — see
    `feedback_tests_must_not_name_the_real_ledger`.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sandbox_trades.db")
        self.mgr = PaperManager(db_path=self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_the_file_is_created_and_carries_the_schema(self):
        self.assertTrue(os.path.exists(self.path))
        with self.mgr._get_connection() as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("trades", names)

    def test_a_write_survives_a_new_manager_on_the_same_file(self):
        # The property in-memory deliberately does NOT have, kept pinned so the
        # two paths cannot be confused later.
        with self.mgr._get_connection() as conn:
            conn.execute("INSERT INTO trades (ticker, status) VALUES ('CCC','OPEN')")
        again = PaperManager(db_path=self.path)
        with again._get_connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM trades WHERE ticker='CCC'").fetchone()[0]
        self.assertEqual(n, 1)

    def test_a_file_backed_connection_is_closed_after_use(self):
        with self.mgr._get_connection() as conn:
            pass
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
