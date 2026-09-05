"""Schema v22 records the budget that was in force when a trade was logged.

`NULL` here means "no limit was in force" — not "not recorded". That is
truthful rather than a fudge because the global cap shipped 2026-07-29, so
rows from that date onward really did have a $4,000 budget and earlier rows
really had none (the unbounded-feeder era that produced the $27k and $83k
positions).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import paper_manager as pm


class TestMigration22(unittest.TestCase):

    def test_schema_version_is_22(self):
        # This file is scoped to migration 22's own correctness; the global
        # "current" schema version has since moved on (23: per-leg fill
        # recording, 24: earnings_state) — updated here rather than renamed,
        # to keep this diff about the version bump and nothing else.
        self.assertGreaterEqual(pm._SCHEMA_VERSION, 22)

    def test_migration_22_is_registered(self):
        self.assertIn(22, pm._MIGRATIONS)
        self.assertTrue(pm._MIGRATIONS[22])

    def _apply(self, rows):
        """Build a minimal v21-shaped trades table, apply v22, return contents."""
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "t.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT)")
        conn.executemany("INSERT INTO trades (date, ticker) VALUES (?, ?)", rows)
        conn.commit()
        for sql in pm._MIGRATIONS[22]:
            conn.execute(sql)
        conn.commit()
        out = list(conn.execute(
            "SELECT date, budget_at_entry FROM trades ORDER BY date"))
        conn.close()
        return out

    def test_rows_on_or_after_the_cap_date_are_backfilled_to_4000(self):
        out = self._apply([("2026-07-29", "A"), ("2026-08-01", "B")])
        self.assertEqual([r[1] for r in out], [4000.0, 4000.0])

    def test_rows_before_the_cap_date_stay_null(self):
        """The unbounded-feeder era genuinely had no limit."""
        out = self._apply([("2026-07-01", "A"), ("2026-07-28", "B")])
        self.assertEqual([r[1] for r in out], [None, None])

    def test_the_boundary_is_2026_07_29_inclusive(self):
        out = self._apply([("2026-07-28", "A"), ("2026-07-29", "B")])
        self.assertEqual([r[1] for r in out], [None, 4000.0])

    def test_no_rows_are_added_or_removed(self):
        out = self._apply([("2026-07-01", "A"), ("2026-08-01", "B")])
        self.assertEqual(len(out), 2)

    def test_a_fresh_ledger_has_the_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = pm.PaperManager(db_path=os.path.join(tmp, "fresh.db"))
            with sqlite3.connect(mgr.db_path) as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
                ver = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertIn("budget_at_entry", cols)
        # A fresh ledger runs every migration up to whatever the CURRENT
        # schema version is, not frozen at 22 — see the note on
        # test_schema_version_is_22 above.
        self.assertGreaterEqual(ver, 22)


if __name__ == "__main__":
    unittest.main()
