"""Schema v18 — the execution columns, and the guarantee that adding them
does not disturb anything already in the ledger.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_execution_truth_ledger -v
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import _SCHEMA_VERSION, PaperManager

NEW_COLUMNS = ("entry_price_mid", "entry_price_fill", "entry_price_cross",
               "fill_policy", "fill_source")


class SchemaV18Test(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")

    def _cols(self, db=None):
        conn = sqlite3.connect(db or self.db)
        try:
            return [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        finally:
            conn.close()

    def test_schema_version_is_at_least_eighteen(self):
        self.assertGreaterEqual(_SCHEMA_VERSION, 18)

    def test_the_execution_columns_exist(self):
        PaperManager(db_path=self.db)
        cols = self._cols()
        for c in NEW_COLUMNS:
            self.assertIn(c, cols)

    def test_migration_is_idempotent(self):
        PaperManager(db_path=self.db)
        PaperManager(db_path=self.db)
        cols = self._cols()
        for c in NEW_COLUMNS:
            self.assertEqual(cols.count(c), 1)
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], _SCHEMA_VERSION)
        conn.close()

    def test_existing_rows_survive_the_migration_untouched(self):
        """The whole point of adding columns rather than rewriting any: a row
        written under v17 must read back identically under v18, with the new
        columns NULL rather than invented."""
        conn = sqlite3.connect(self.db)
        conn.execute("PRAGMA user_version = 17")
        conn.execute(
            "CREATE TABLE trades (entry_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "date TEXT, ticker TEXT, expiration TEXT, strike REAL, type TEXT, "
            "entry_price REAL, quality_score REAL, strategy_name TEXT, status TEXT, "
            "exit_price REAL, exit_date TEXT, pnl_pct REAL, pnl_usd REAL, "
            "net_credit REAL, spread_width REAL)")
        conn.execute(
            "INSERT INTO trades (ticker, strategy_name, status, entry_price, "
            "net_credit, spread_width, pnl_usd) "
            "VALUES ('ORCL','Bull Put','CLOSED',1.28,1.28,2.5,44.0)")
        conn.commit()
        conn.close()

        PaperManager(db_path=self.db)

        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades").fetchone()
        self.assertEqual(row["ticker"], "ORCL")
        self.assertAlmostEqual(row["entry_price"], 1.28)
        self.assertAlmostEqual(row["net_credit"], 1.28)
        self.assertAlmostEqual(row["pnl_usd"], 44.0)
        for c in NEW_COLUMNS:
            self.assertIsNone(row[c], f"{c} must be NULL, not invented")
        conn.close()

    def test_entry_price_is_never_redefined_by_this_migration(self):
        """`entry_price` keeps its v17 meaning forever. Analysis that wants the
        honest number reads `entry_price_fill`; nothing downstream breaks."""
        pm = PaperManager(db_path=self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (ticker, strategy_name, status, entry_price) "
            "VALUES ('SPY','Bull Put','OPEN',0.51)")
        conn.commit()
        row = conn.execute(
            "SELECT entry_price, entry_price_fill FROM trades").fetchone()
        self.assertAlmostEqual(row[0], 0.51)
        self.assertIsNone(row[1])
        conn.close()
        del pm


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
