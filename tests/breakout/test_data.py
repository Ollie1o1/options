"""Tests for the breakout OHLCV cache — offline, synthetic."""
from __future__ import annotations
import os, tempfile, unittest
import numpy as np
from src.breakout.data import upsert_ohlcv, load_series, update_universe, HORIZONS, THRESHOLDS


def _rows(dates, base=100.0):
    return [{"date": d, "close": base + i, "high": base + i + 1,
             "low": base + i - 1, "volume": 1000 + i} for i, d in enumerate(dates)]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ohlcv.db")

    def test_constants(self):
        self.assertEqual(HORIZONS, {"EOW": 5, "EOM": 21, "3M": 63})
        self.assertEqual(THRESHOLDS, (0.05, 0.10, 0.20))

    def test_upsert_and_load_roundtrip(self):
        n = upsert_ohlcv(self.db, "AAA", _rows(["2020-01-02", "2020-01-03"]))
        self.assertEqual(n, 2)
        s = load_series(self.db, "AAA")
        self.assertEqual(s.dates, ["2020-01-02", "2020-01-03"])
        self.assertTrue(np.allclose(s.close, [100.0, 101.0]))
        self.assertEqual(len(s.high), 2)

    def test_upsert_is_idempotent(self):
        upsert_ohlcv(self.db, "AAA", _rows(["2020-01-02"]))
        n = upsert_ohlcv(self.db, "AAA", _rows(["2020-01-02"]))
        self.assertEqual(n, 0)
        self.assertEqual(len(load_series(self.db, "AAA").dates), 1)

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_series(self.db, "NOPE"))

    def test_update_universe_fetches_only_new(self):
        upsert_ohlcv(self.db, "AAA", _rows(["2020-01-02"]))
        calls = {}
        def fetcher(ticker, start_date):
            calls[ticker] = start_date
            return _rows(["2020-01-03"]) if start_date else _rows(["2020-01-02", "2020-01-03"])
        res = update_universe(self.db, ["AAA", "BBB"], fetcher)
        self.assertEqual(calls["AAA"], "2020-01-02")   # had data -> incremental
        self.assertIsNone(calls["BBB"])                 # no data -> full
        self.assertEqual(res["AAA"], 1)
        self.assertEqual(res["BBB"], 2)


if __name__ == "__main__":
    unittest.main()
