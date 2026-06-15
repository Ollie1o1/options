"""Tests for src/dolt_slippage.py — real spread table from cached chains.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_slippage -v
"""
import os
import sqlite3
import tempfile
import unittest

from src import dolt_slippage as ds


class BandTest(unittest.TestCase):
    def test_delta_bands(self):
        self.assertEqual(ds._delta_band(0.05), "deep_otm")
        self.assertEqual(ds._delta_band(0.30), "otm")
        self.assertEqual(ds._delta_band(0.50), "atm")
        self.assertEqual(ds._delta_band(-0.50), "atm")   # uses abs
        self.assertIsNone(ds._delta_band(None))

    def test_dte_bands(self):
        self.assertEqual(ds._dte_band(7), "0-14")
        self.assertEqual(ds._dte_band(30), "14-35")
        self.assertEqual(ds._dte_band(45), "35-70")
        self.assertEqual(ds._dte_band(120), "70+")


class SpreadTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "c.db")
        conn = sqlite3.connect(self.db)
        conn.execute("""CREATE TABLE dolt_chain (
            symbol TEXT, date TEXT, expiration TEXT, strike REAL, type TEXT,
            bid REAL, ask REAL, mid REAL, iv REAL,
            delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL)""")
        # 30 ATM 30-DTE rows with a 10% relative spread (mid 5.0, bid 4.75, ask 5.25)
        for i in range(30):
            conn.execute("INSERT INTO dolt_chain (symbol,date,expiration,strike,type,"
                         "bid,ask,mid,iv,delta,gamma,theta,vega,rho) VALUES "
                         "('X','2024-03-01','2024-03-31',100,'call',4.75,5.25,5.0,0.3,0.5,0,0,0,0)")
        conn.commit(); conn.close()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def test_measure_table_buckets_and_median(self):
        table = ds.measure_spread_table(db_path=self.db)
        self.assertIn("atm|14-35", table)
        cell = table["atm|14-35"]
        self.assertEqual(cell["n"], 30)
        self.assertAlmostEqual(cell["median_rel_spread"], 0.10, places=2)

    def test_half_spread_uses_table_when_enough_data(self):
        table = ds.measure_spread_table(db_path=self.db)
        # median rel spread 0.10 → half-spread 0.05
        self.assertAlmostEqual(ds.half_spread_fraction(0.5, 30, table=table), 0.05, places=2)

    def test_half_spread_falls_back_when_no_data(self):
        table = ds.measure_spread_table(db_path=self.db)
        # deep_itm/70+ has no rows → fallback 7%
        self.assertAlmostEqual(ds.half_spread_fraction(0.95, 120, table=table),
                               ds._FALLBACK_HALF_SPREAD, places=3)


if __name__ == "__main__":
    unittest.main()
