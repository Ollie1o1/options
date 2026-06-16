"""Tests for src/portfolio_eras.py — era-split P&L (offline, temp DB)."""
import os
import sqlite3
import tempfile
import unittest

from src import portfolio_eras as pe


def _make_db(path, rows):
    """rows: (era, status, pnl_usd, pnl_pct, strategy)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (era TEXT, status TEXT, pnl_usd REAL, "
                 "pnl_pct REAL, strategy_name TEXT)")
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?)", rows)
    conn.commit(); conn.close()


class EraStatsTest(unittest.TestCase):
    def test_splits_and_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            _make_db(path, [
                ("pre_data", "CLOSED", 100.0, 0.10, "Long Call"),
                ("pre_data", "CLOSED", -50.0, -0.20, "Long Call"),
                ("pre_data", "OPEN", None, None, "Long Call"),
                ("finalized", "CLOSED", 30.0, 0.05, "Short Put"),
            ])
            s = pe.era_stats(path)
        self.assertEqual(s["pre_data"]["total"], 3)
        self.assertEqual(s["pre_data"]["closed"], 2)
        self.assertEqual(s["pre_data"]["open"], 1)
        self.assertAlmostEqual(s["pre_data"]["realized_pnl_usd"], 50.0)
        self.assertAlmostEqual(s["pre_data"]["win_rate"], 0.5)
        self.assertEqual(s["finalized"]["closed"], 1)
        self.assertAlmostEqual(s["finalized"]["realized_pnl_usd"], 30.0)

    def test_empty_era(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            _make_db(path, [("pre_data", "CLOSED", 10.0, 0.1, "Long Call")])
            s = pe.era_stats(path)
        self.assertEqual(s["finalized"]["total"], 0)
        self.assertIsNone(s["finalized"]["win_rate"])

    def test_missing_era_column_treated_as_pre_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE trades (status TEXT, pnl_usd REAL, pnl_pct REAL, "
                         "strategy_name TEXT)")
            conn.execute("INSERT INTO trades VALUES ('CLOSED', 20.0, 0.1, 'Long Call')")
            conn.commit(); conn.close()
            s = pe.era_stats(path)
        self.assertEqual(s["pre_data"]["closed"], 1)
        self.assertEqual(s["finalized"]["total"], 0)

    def test_by_strategy_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            _make_db(path, [
                ("pre_data", "CLOSED", -100.0, -0.5, "Long Put"),
                ("pre_data", "CLOSED", 50.0, 0.2, "Long Call"),
            ])
            rows = pe.by_strategy(path, "pre_data")
        self.assertEqual(rows[0][0], "Long Put")   # worst first (sorted by sum)


if __name__ == "__main__":
    unittest.main()
