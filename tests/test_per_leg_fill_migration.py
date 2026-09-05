"""tests/test_per_leg_fill_migration.py

Schema v23 adds per-leg bid/ask columns for multi-leg structures, at entry
and at exit, so the 46% of the closed book that is multi-leg
(docs/SINGLE_LEG_REPRICE_20260902.md) can eventually be repriced under the
measured spread surface the way the single-leg book already was.

NULL means "not recorded" — a legacy row, or a single-leg row for which
these columns never apply — never zero. Nothing here backfills a value.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import paper_manager as pm

_SPREAD_COLS = ("short_bid_entry", "short_ask_entry", "long_bid_entry",
               "long_ask_entry", "short_bid_exit", "short_ask_exit",
               "long_bid_exit", "long_ask_exit")

_CONDOR_COLS = ("short_put_bid_entry", "short_put_ask_entry",
               "long_put_bid_entry", "long_put_ask_entry",
               "short_call_bid_entry", "short_call_ask_entry",
               "long_call_bid_entry", "long_call_ask_entry",
               "short_put_bid_exit", "short_put_ask_exit",
               "long_put_bid_exit", "long_put_ask_exit",
               "short_call_bid_exit", "short_call_ask_exit",
               "long_call_bid_exit", "long_call_ask_exit")


class TestMigration23(unittest.TestCase):
    def test_schema_version_is_23(self):
        # Scoped to migration 23's own correctness, not to it being the tip —
        # see the equivalent note in test_budget_at_entry_migration.py.
        self.assertGreaterEqual(pm._SCHEMA_VERSION, 23)

    def test_migration_23_is_registered(self):
        self.assertIn(23, pm._MIGRATIONS)
        self.assertTrue(pm._MIGRATIONS[23])

    def _apply(self):
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "t.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT)")
        conn.executemany("INSERT INTO trades (date, ticker) VALUES (?, ?)",
                         [("2026-08-01", "A")])
        conn.commit()
        for sql in pm._MIGRATIONS[23]:
            conn.execute(sql)
        conn.commit()
        return conn

    def test_all_24_columns_are_added(self):
        conn = self._apply()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        for name in _SPREAD_COLS + _CONDOR_COLS:
            self.assertIn(name, cols)
        conn.close()

    def test_no_column_is_duplicated_between_spread_and_condor_sets(self):
        self.assertEqual(len(set(_SPREAD_COLS) & set(_CONDOR_COLS)), 0)
        self.assertEqual(len(_SPREAD_COLS) + len(_CONDOR_COLS), 24)

    def test_legacy_row_is_null_on_every_new_column(self):
        conn = self._apply()
        row = conn.execute(
            f"SELECT {', '.join(_SPREAD_COLS + _CONDOR_COLS)} FROM trades"
        ).fetchone()
        self.assertTrue(all(v is None for v in row))
        conn.close()


if __name__ == "__main__":
    unittest.main()
