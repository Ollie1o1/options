"""Tests for src/alloc/splits.py — split detection off the strike level.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_alloc_splits -v
"""
import os
import sqlite3
import tempfile
import unittest

from src.alloc.splits import detect_splits, split_ratio


def _db(path, rows):
    """rows: (symbol, date, [strikes...])"""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE dolt_chain (symbol TEXT, date TEXT, "
                 "expiration TEXT, strike REAL, type TEXT, bid REAL, ask REAL, "
                 "mid REAL, iv REAL, delta REAL, gamma REAL, theta REAL, "
                 "vega REAL, rho REAL)")
    for sym, date, strikes in rows:
        for k in strikes:
            conn.execute("INSERT INTO dolt_chain (symbol,date,strike) "
                         "VALUES (?,?,?)", (sym, date, k))
    conn.commit()
    conn.close()


class DetectSplitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "c.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detects_an_adjacent_day_split(self):
        # A real 4:1: strikes are re-listed around the new price overnight.
        _db(self.db, [("AAA", "2024-06-06", [400, 420, 440]),
                      ("AAA", "2024-06-07", [100, 105, 110])])
        self.assertEqual(detect_splits(self.db), {"AAA": {"2024-06-07"}})

    def test_ordinary_drift_is_not_a_split(self):
        _db(self.db, [("AAA", "2024-06-06", [100, 105, 110]),
                      ("AAA", "2024-06-07", [101, 106, 111])])
        self.assertEqual(detect_splits(self.db), {})

    def test_a_jump_across_a_long_data_gap_is_drift_not_a_split(self):
        # THE regression. SPY's real series: mean strike 228.9 on 2020-03-20,
        # then the cache's next row is 2022-01-03 at 465.7. That is 21 months
        # in which SPY genuinely doubled, and it was being reported as a split
        # — which would close every open position on a day nothing happened.
        _db(self.db, [("SPY", "2020-03-20", [220, 229, 238]),
                      ("SPY", "2022-01-03", [450, 466, 482])])
        self.assertEqual(detect_splits(self.db), {})

    def test_a_weekend_or_holiday_gap_still_counts_as_adjacent(self):
        # Fri -> Tue after a Monday holiday is 4 days and must not be excused.
        _db(self.db, [("AAA", "2024-05-24", [400, 420, 440]),
                      ("AAA", "2024-05-28", [100, 105, 110])])
        self.assertEqual(detect_splits(self.db), {"AAA": {"2024-05-28"}})

    def test_max_gap_days_is_tunable(self):
        _db(self.db, [("AAA", "2024-01-01", [400]), ("AAA", "2024-02-01", [100])])
        self.assertEqual(detect_splits(self.db), {})
        self.assertEqual(detect_splits(self.db, max_gap_days=60),
                         {"AAA": {"2024-02-01"}})

    def test_symbol_filter(self):
        _db(self.db, [("AAA", "2024-06-06", [400]), ("AAA", "2024-06-07", [100]),
                      ("BBB", "2024-06-06", [400]), ("BBB", "2024-06-07", [100])])
        self.assertEqual(set(detect_splits(self.db, symbols=["AAA"])), {"AAA"})

    def test_split_ratio(self):
        self.assertAlmostEqual(split_ratio(400.0, 100.0), 4.0)
        self.assertEqual(split_ratio(400.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
