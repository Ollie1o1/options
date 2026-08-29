"""The spread surface: measured friction conditioned on the contract.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_spread_surface -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from statistics import median

from src.spread_surface import (DEFAULT_SURFACE_PATH, DELTA_EDGES, DTE_EDGES,
                                MIN_CELL_OBS, OI_EDGES, Cell, SpreadSurface,
                                bucket_index, cell_key, fit_surface,
                                load_surface, save_surface)


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

    def test_median_depth_with_one_side_missing_is_conservative(self):
        # bid_size is recorded (7) but ask_size is missing. Falling back to
        # min() over the single surviving value would silently use the
        # *other* side's size (200 from test_median_depth_is_the_tighter_side
        # would leak in as 7 here too if we only ever set bid_size) as if it
        # were the tighter side — that overstates liquidity. A missing side
        # is not evidence of depth, so it must resolve to the worst case (0),
        # not the favorable one.
        rows = [("TEST", "2026-06-10", 100.0, "2026-07-10", 0.90, 1.10, 0.50,
                 500, 7, None) for _ in range(40)]
        _make_archive(self.db, rows)
        s = fit_surface(self.db)
        self.assertEqual(s.cells[cell_key(0.50, 30.0, 500.0)].median_depth, 0)

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


class LookupProvenanceTest(unittest.TestCase):
    def _surface(self, cells):
        return SpreadSurface(cells, {"fit_date": "2026-08-28"})

    def test_an_exact_cell_reports_cell_provenance(self):
        s = self._surface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)})
        value, prov = s.relative(abs_delta=0.50, dte=30.0, open_interest=500.0)
        self.assertAlmostEqual(value, 0.02)
        self.assertEqual(prov, "cell")

    def test_a_missing_cell_collapses_open_interest_first(self):
        # Same delta/DTE, different OI bucket. Collapsing OI keeps the two
        # dimensions the caller is most likely to have right.
        s = self._surface({cell_key(0.50, 30.0, 5.0): Cell(40, 0.05, 50)})
        value, prov = s.relative(abs_delta=0.50, dte=30.0,
                                 open_interest=500.0)
        self.assertAlmostEqual(value, 0.05)
        self.assertEqual(prov, "oi_collapsed")

    def test_it_collapses_dte_when_no_oi_match_exists(self):
        s = self._surface({cell_key(0.50, 120.0, 5.0): Cell(40, 0.07, 50)})
        value, prov = s.relative(abs_delta=0.50, dte=30.0,
                                 open_interest=500.0)
        self.assertAlmostEqual(value, 0.07)
        self.assertEqual(prov, "dte_collapsed")

    def test_it_falls_back_to_the_global_median(self):
        s = self._surface({cell_key(0.05, 3.0, 5.0): Cell(40, 0.09, 50)})
        value, prov = s.relative(abs_delta=0.50, dte=30.0,
                                 open_interest=500.0)
        self.assertAlmostEqual(value, 0.09)
        self.assertEqual(prov, "global")

    def test_an_empty_surface_returns_the_caller_default(self):
        value, prov = self._surface({}).relative(
            abs_delta=0.50, dte=30.0, open_interest=500.0, default=0.03)
        self.assertAlmostEqual(value, 0.03)
        self.assertEqual(prov, "caller_default")

    def test_an_empty_surface_without_a_default_refuses(self):
        # Returning 0.0 would report a free trade. Refuse instead.
        with self.assertRaises(ValueError):
            self._surface({}).relative(abs_delta=0.5, dte=30.0,
                                       open_interest=500.0)

    def test_provenance_is_never_a_bare_float(self):
        s = self._surface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)})
        self.assertIsInstance(
            s.relative(abs_delta=0.5, dte=30.0, open_interest=500.0), tuple)


class HalfSpreadTest(unittest.TestCase):
    def test_dollars_are_relative_times_mid(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)}, {})
        self.assertAlmostEqual(
            s.half_spread(2.50, abs_delta=0.50, dte=30.0,
                          open_interest=500.0), 0.05)

    def test_a_positive_mid_never_costs_zero(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)}, {})
        self.assertGreater(
            s.half_spread(0.05, abs_delta=0.50, dte=30.0,
                          open_interest=500.0), 0.0)

    def test_a_non_positive_mid_refuses(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 50)}, {})
        with self.assertRaises(ValueError):
            s.half_spread(0.0, abs_delta=0.50, dte=30.0, open_interest=500.0)


class DepthTest(unittest.TestCase):
    def test_an_order_inside_displayed_depth_is_ok(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 42)}, {})
        self.assertTrue(s.depth_ok(3, abs_delta=0.50, dte=30.0,
                                   open_interest=500.0))

    def test_an_order_exceeding_displayed_depth_is_not_ok(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 4)}, {})
        self.assertFalse(s.depth_ok(10, abs_delta=0.50, dte=30.0,
                                    open_interest=500.0))

    def test_an_unknown_cell_is_not_ok(self):
        # No measurement is not permission.
        self.assertFalse(SpreadSurface({}, {}).depth_ok(
            1, abs_delta=0.50, dte=30.0, open_interest=500.0))


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "surface.json")

    def test_a_saved_surface_reloads_identically(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 42)},
                          {"fit_date": "2026-08-28", "rows": 40})
        save_surface(s, self.path)
        back = load_surface(self.path)
        self.assertEqual(back.cells, s.cells)
        self.assertEqual(back.stamp, s.stamp)

    def test_the_saved_file_is_readable_json_with_a_stamp(self):
        s = SpreadSurface({cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 42)},
                          {"fit_date": "2026-08-28"})
        save_surface(s, self.path)
        with open(self.path) as fh:
            blob = json.load(fh)
        self.assertIn("stamp", blob)
        self.assertIn("cells", blob)

    def test_loading_a_missing_file_returns_an_empty_surface(self):
        s = load_surface(os.path.join(self.dir, "absent.json"))
        self.assertEqual(s.cells, {})


@unittest.skipUnless(os.path.exists("data/chain_archive.db"),
                     "archive not present")
class RealArchivePropertyTest(unittest.TestCase):
    """Properties the measured surface must hold. These are the claims the
    design rests on; if the archive stops supporting them, that is a finding."""

    @classmethod
    def setUpClass(cls):
        cls.surface = fit_surface("data/chain_archive.db")

    def test_relative_spread_is_non_increasing_in_open_interest(self):
        # Averaged across delta/DTE cells, deeper open interest is cheaper.
        by_oi = {}
        for (d, t, o), cell in self.surface.cells.items():
            by_oi.setdefault(o, []).append(cell.rel_half_spread)
        meds = [median(by_oi[o]) for o in sorted(by_oi)]
        for lo, hi in zip(meds, meds[1:]):
            self.assertLessEqual(hi, lo + 1e-9)

    def test_deep_otm_is_worse_than_at_the_money(self):
        otm = median([c.rel_half_spread for k, c in self.surface.cells.items()
                      if k[0] == 0])
        atm = median([c.rel_half_spread for k, c in self.surface.cells.items()
                      if k[0] == 3])
        self.assertGreater(otm, atm)

    def test_every_recorded_cell_clears_the_observation_floor(self):
        for cell in self.surface.cells.values():
            self.assertGreaterEqual(cell.n, MIN_CELL_OBS)

    def test_no_cell_reports_a_free_or_negative_spread(self):
        for cell in self.surface.cells.values():
            self.assertGreater(cell.rel_half_spread, 0.0)
