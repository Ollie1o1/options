"""Tests for src/dolt_stocks.py — raw close via split-unadjustment.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_stocks -v
"""
import os
import tempfile
import unittest
from unittest import mock

from src import dolt_stocks as ds


class UnadjustTest(unittest.TestCase):
    def test_unadjust_applies_only_future_splits(self):
        # NVDA-like: 4:1 on 2021-07-20, 10:1 on 2024-06-10
        splits = [("2021-07-20", 4.0), ("2024-06-10", 10.0)]
        adj = {
            "2020-01-02": 6.0,     # before both → ×40
            "2023-06-15": 42.65,   # after 4:1, before 10:1 → ×10
            "2024-12-02": 130.0,   # after both → ×1
        }
        raw = ds.raw_from_adjusted(adj, splits)
        self.assertAlmostEqual(raw["2020-01-02"], 6.0 * 40)
        self.assertAlmostEqual(raw["2023-06-15"], 426.5)   # ≈ real DoltHub $426.53
        self.assertAlmostEqual(raw["2024-12-02"], 130.0)

    def test_no_splits_is_identity(self):
        adj = {"2024-01-02": 100.0, "2024-01-03": 101.0}
        self.assertEqual(ds.raw_from_adjusted(adj, []), adj)


class CloseHistoryCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "c.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fetch_then_cache(self):
        adj = {"2023-06-15": 42.65, "2023-06-16": 43.0}
        with mock.patch("src.dolt_stocks._yf_adjusted", return_value=adj) as fy, \
             mock.patch("src.dolt_stocks._fetch_splits", return_value=[("2024-06-10", 10.0)]) as fs:
            h1 = ds.close_history("NVDA", db_path=self.db)
            h2 = ds.close_history("NVDA", db_path=self.db)
        self.assertEqual(fy.call_count, 1, "second call must hit cache")
        self.assertEqual(fs.call_count, 1)
        self.assertAlmostEqual(h1["2023-06-15"], 426.5)
        self.assertAlmostEqual(h2["2023-06-16"], 430.0)


if __name__ == "__main__":
    unittest.main()
