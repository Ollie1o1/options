"""Tests for the VRP layer — offline, using a synthetic equity_ohlcv DB."""
from __future__ import annotations
import os, tempfile, unittest
import numpy as np
from src.breakout.data import upsert_ohlcv
from src.vol_intel.vrp import classify, vrp_row, realized_vol_for, rv_percentile, RICH_VP, CHEAP_VP


class ClassifyTests(unittest.TestCase):
    def test_rich_fair_cheap(self):
        self.assertEqual(classify(0.05), "RICH")
        self.assertEqual(classify(0.0), "FAIR")
        self.assertEqual(classify(-0.05), "CHEAP")

    def test_boundaries(self):
        self.assertEqual(classify(RICH_VP), "RICH")
        self.assertEqual(classify(CHEAP_VP), "CHEAP")

    def test_vrp_row_sign(self):
        r = vrp_row("AAA", 0.30, 0.18)
        self.assertAlmostEqual(r["vrp"], 0.12, places=6)
        self.assertEqual(r["label"], "RICH")


class RealizedTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "ohlcv.db")
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        rows = [{"date": f"d{i:05d}", "close": float(c), "high": float(c) + 1,
                 "low": float(c) - 1, "volume": 1000.0} for i, c in enumerate(close)]
        upsert_ohlcv(self.db, "AAA", rows)

    def test_realized_vol_positive(self):
        rv = realized_vol_for("AAA", self.db, window=20)
        self.assertIsNotNone(rv)
        self.assertGreater(rv, 0.0)

    def test_realized_vol_missing_symbol_none(self):
        self.assertIsNone(realized_vol_for("NOPE", self.db))

    def test_rv_percentile_in_unit_interval(self):
        p = rv_percentile("AAA", self.db, window=20, lookback=200)
        self.assertIsNotNone(p)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
