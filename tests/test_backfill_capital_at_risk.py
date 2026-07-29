"""Backfill of capital_at_risk over the existing ledger.

827 rows predate the column. The backfill has to be re-runnable (the
maintenance heartbeat may call it more than once) and must never overwrite a
value already stored at log time.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backfill_capital_at_risk import backfill


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades ("
        "entry_id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, strike REAL,"
        " entry_price REAL, max_loss_usd REAL, quantity REAL,"
        " strategy_name TEXT, capital_at_risk REAL)"
    )
    conn.executemany(
        "INSERT INTO trades (ticker, strike, entry_price, max_loss_usd, quantity,"
        " strategy_name, capital_at_risk) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        conn = sqlite3.connect(self.db)
        out = conn.execute(
            "SELECT strategy_name, capital_at_risk FROM trades ORDER BY entry_id"
        ).fetchall()
        conn.close()
        return out

    def test_fills_each_structure_by_its_own_rule(self):
        _make_db(self.db, [
            ("AAPL", 150.0, 3.50, None, 1.0, "Long Call", None),
            ("WFC", 77.5, 1.52, None, 1.0, "Short Put", None),
            ("INTC", 80.0, 0.50, 50.0, 1.0, "Bull Put", None),
        ])
        result = backfill(self.db)
        self.assertEqual(result["updated"], 3)
        self.assertEqual(
            self.read(),
            [("Long Call", 350.0), ("Short Put", 7598.0), ("Bull Put", 50.0)],
        )

    def test_does_not_overwrite_values_already_stored(self):
        _make_db(self.db, [("AAPL", 150.0, 3.50, None, 1.0, "Long Call", 999.0)])
        result = backfill(self.db)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(self.read()[0][1], 999.0)

    def test_is_idempotent(self):
        _make_db(self.db, [("AAPL", 150.0, 3.50, None, 1.0, "Long Call", None)])
        backfill(self.db)
        second = backfill(self.db)
        self.assertEqual(second["updated"], 0)
        self.assertEqual(self.read()[0][1], 350.0)

    def test_unbounded_rows_are_left_null_and_counted(self):
        # A naked call has no bounded risk; recording 0 would make it look free.
        _make_db(self.db, [("AAPL", 150.0, 1.10, None, 1.0, "Short Call", None)])
        result = backfill(self.db)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["unbounded"], 1)
        self.assertIsNone(self.read()[0][1])

    def test_unmigrated_db_fails_with_an_actionable_message(self):
        # The column arrives with schema v16, applied when PaperManager opens the
        # db. Hitting a raw sqlite "no such column" here tells the reader nothing.
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE trades (entry_id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError) as ctx:
            backfill(self.db)
        self.assertIn("capital_at_risk", str(ctx.exception))

    def test_dry_run_reports_without_writing(self):
        _make_db(self.db, [("AAPL", 150.0, 3.50, None, 1.0, "Long Call", None)])
        result = backfill(self.db, dry_run=True)
        self.assertEqual(result["updated"], 1)
        self.assertIsNone(self.read()[0][1])


if __name__ == "__main__":
    unittest.main()
