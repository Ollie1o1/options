"""The spread surface: measured friction conditioned on the contract.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_spread_surface -v
"""
from __future__ import annotations

import unittest

from src.spread_surface import (DELTA_EDGES, DTE_EDGES, MIN_CELL_OBS,
                                OI_EDGES, bucket_index, cell_key)


class BucketIndexTest(unittest.TestCase):
    def test_value_below_first_edge_is_bucket_zero(self):
        self.assertEqual(bucket_index(0.05, DELTA_EDGES), 0)

    def test_value_above_last_edge_is_the_final_bucket(self):
        self.assertEqual(bucket_index(0.95, DELTA_EDGES), len(DELTA_EDGES))

    def test_edges_are_upper_exclusive(self):
        # 0.10 is the first edge, so it belongs to the SECOND bucket, not the
        # first. An off-by-one here silently reclassifies every ATM contract.
        self.assertEqual(bucket_index(0.10, DELTA_EDGES), 1)
        self.assertEqual(bucket_index(0.0999, DELTA_EDGES), 0)

    def test_buckets_are_monotone_in_value(self):
        prev = 0
        for v in (0.0, 0.05, 0.10, 0.24, 0.25, 0.39, 0.40, 0.59, 0.60, 1.0):
            idx = bucket_index(v, DELTA_EDGES)
            self.assertGreaterEqual(idx, prev)
            prev = idx

    def test_five_buckets_from_four_edges(self):
        for edges in (DELTA_EDGES, DTE_EDGES, OI_EDGES):
            self.assertEqual(len(edges), 4)
            self.assertEqual(bucket_index(float("inf"), edges), 4)


class CellKeyTest(unittest.TestCase):
    def test_key_is_three_bucket_indices(self):
        self.assertEqual(cell_key(0.05, 3.0, 5.0), (0, 0, 0))

    def test_key_uses_absolute_delta_bucket(self):
        self.assertEqual(cell_key(0.50, 30.0, 500.0), (3, 2, 2))

    def test_missing_open_interest_is_treated_as_zero_not_dropped(self):
        # NULL open interest means "not recorded". Treating it as the most
        # illiquid bucket is the conservative reading; treating it as liquid
        # would understate cost.
        self.assertEqual(cell_key(0.50, 30.0, None)[2], 0)


import os
import sqlite3
import tempfile

from src.spread_surface import Cell, SpreadSurface, fit_surface


def _make_archive(path, rows):
    """rows: (symbol, snap_date, strike, expiration, bid, ask, delta, oi,
    bid_size, ask_size)"""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE chain_snapshots (
        symbol TEXT, snap_date TEXT, contract TEXT, type TEXT, strike REAL,
        expiration TEXT, bid REAL, ask REAL, bid_size REAL, ask_size REAL,
        iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
        open_interest REAL, volume REAL, last_trade_time TEXT, spot REAL,
        snapshot_ts TEXT, source TEXT)""")
    for (sym, snap, strike, exp, bid, ask, delta, oi, bs, asz) in rows:
        con.execute(
            "INSERT INTO chain_snapshots (symbol, snap_date, type, strike, "
            "expiration, bid, ask, bid_size, ask_size, delta, open_interest, "
            "spot, source) VALUES (?,?,'call',?,?,?,?,?,?,?,?,100.0,'test')",
            (sym, snap, strike, exp, bid, ask, bs, asz, delta, oi))
    con.commit()
    con.close()


class FitSurfaceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "archive.db")

    def _rows(self, n, bid, ask, delta, oi, depth=50):
        return [("TEST", "2026-06-10", 100.0, "2026-07-10", bid, ask, delta,
                 oi, depth, depth) for _ in range(n)]

    def test_a_cell_records_the_median_relative_half_spread(self):
        # mid = 1.00, half-spread = 0.10 => relative 0.10
        _make_archive(self.db, self._rows(40, 0.90, 1.10, 0.50, 500))
        s = fit_surface(self.db)
        cell = s.cells[cell_key(0.50, 30.0, 500.0)]
        self.assertEqual(cell.n, 40)
        self.assertAlmostEqual(cell.rel_half_spread, 0.10, places=6)

    def test_a_cell_below_the_observation_floor_is_not_recorded(self):
        _make_archive(self.db, self._rows(MIN_CELL_OBS - 1, 0.90, 1.10,
                                          0.50, 500))
        s = fit_surface(self.db)
        self.assertNotIn(cell_key(0.50, 30.0, 500.0), s.cells)

    def test_crossed_and_one_sided_quotes_are_excluded(self):
        rows = self._rows(40, 0.90, 1.10, 0.50, 500)
        rows += self._rows(100, 0.0, 1.10, 0.50, 500)    # zero bid
        rows += self._rows(100, 1.20, 1.10, 0.50, 500)   # crossed
        _make_archive(self.db, rows)
        s = fit_surface(self.db)
        # Only the 40 two-sided quotes count. Averaging in a zero bid or a
        # crossed book would understate the real cost of crossing.
        self.assertEqual(s.cells[cell_key(0.50, 30.0, 500.0)].n, 40)

    def test_median_depth_is_the_tighter_side(self):
        rows = [("TEST", "2026-06-10", 100.0, "2026-07-10", 0.90, 1.10, 0.50,
                 500, 7, 200) for _ in range(40)]
        _make_archive(self.db, rows)
        s = fit_surface(self.db)
        self.assertEqual(s.cells[cell_key(0.50, 30.0, 500.0)].median_depth, 7)

    def test_stamp_names_the_command_that_refits_the_model(self):
        _make_archive(self.db, self._rows(40, 0.90, 1.10, 0.50, 500))
        s = fit_surface(self.db)
        self.assertIn("spread_surface", s.stamp["refit_command"])
        self.assertIn("--fit", s.stamp["refit_command"])
        self.assertEqual(s.stamp["rows"], 40)
        self.assertEqual(s.stamp["symbols"], ["TEST"])
        self.assertEqual(s.stamp["date_range"], ["2026-06-10", "2026-06-10"])

    def test_fit_is_deterministic(self):
        _make_archive(self.db, self._rows(40, 0.90, 1.10, 0.50, 500))
        a, b = fit_surface(self.db), fit_surface(self.db)
        self.assertEqual(a.cells, b.cells)
