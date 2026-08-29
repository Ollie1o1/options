"""The spread surface: measured friction conditioned on the contract.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_spread_surface -v
"""
from __future__ import annotations

import unittest

from src.spread_surface import (DELTA_EDGES, DTE_EDGES, OI_EDGES,
                                bucket_index, cell_key)


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
