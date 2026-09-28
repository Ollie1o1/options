"""Tests for the stitched SPY series: source loading, overlap validation
(and its refusal to stitch disagreeing sources), and date-tolerant matching.

Every database here is a throwaway file built in `setUp`; no test names or
opens `data/equity_ohlcv.db`, `data/chain_archive.db` or `paper_trades.db`.
"""
import os
import sqlite3
import tempfile
import unittest

from src.benchmark.prices import (
    MAX_MEDIAN_ABS_DIFF,
    load_archive_spot,
    load_ohlcv_close,
    nearest_price,
    stitch_spy_series,
    validate_stitch,
)


def _make_ohlcv(rows):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL, "
                "high REAL, low REAL, volume REAL)")
    for ticker, date, close in rows:
        con.execute("INSERT INTO ohlcv (ticker, date, close) VALUES (?,?,?)",
                    (ticker, date, close))
    con.commit()
    con.close()
    return path


def _make_archive(rows):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT, "
                "contract TEXT, spot REAL)")
    for i, (symbol, snap_date, spot) in enumerate(rows):
        con.execute("INSERT INTO chain_snapshots (symbol, snap_date, "
                    "contract, spot) VALUES (?,?,?,?)",
                    (symbol, snap_date, f"c{i}", spot))
    con.commit()
    con.close()
    return path


class TestLoadSources(unittest.TestCase):
    def setUp(self):
        self.ohlcv_path = _make_ohlcv([
            ("SPY", "2026-01-01", 400.0),
            ("SPY", "2026-01-02", 401.0),
            ("QQQ", "2026-01-01", 300.0),  # different ticker, must not leak in
        ])
        self.archive_path = _make_archive([
            ("SPY", "2026-01-02", 402.0),
            ("SPY", "2026-01-03", 403.0),
            ("AAPL", "2026-01-03", 200.0),  # different symbol
        ])

    def tearDown(self):
        os.remove(self.ohlcv_path)
        os.remove(self.archive_path)

    def test_load_ohlcv_close_filters_ticker(self):
        series = load_ohlcv_close(self.ohlcv_path, "SPY")
        self.assertEqual(series, {"2026-01-01": 400.0, "2026-01-02": 401.0})

    def test_load_archive_spot_filters_symbol(self):
        series = load_archive_spot(self.archive_path, "SPY")
        self.assertEqual(series, {"2026-01-02": 402.0, "2026-01-03": 403.0})

    def test_archive_spot_none_excluded(self):
        path = _make_archive([("SPY", "2026-01-05", None)])
        try:
            series = load_archive_spot(path, "SPY")
            self.assertEqual(series, {})
        finally:
            os.remove(path)


class TestValidateStitch(unittest.TestCase):
    def test_agreeing_sources_pass(self):
        ohlcv = {"2026-01-01": 400.0, "2026-01-02": 401.0}
        archive = {"2026-01-01": 400.1, "2026-01-02": 401.2}
        v = validate_stitch(ohlcv, archive)
        self.assertTrue(v.ok)
        self.assertEqual(v.n_common, 2)
        self.assertIsNotNone(v.median_abs_diff)
        self.assertLessEqual(v.median_abs_diff, MAX_MEDIAN_ABS_DIFF)

    def test_disagreeing_sources_raise(self):
        # A 5% divergence on every common date is far past the 0.50% bar.
        ohlcv = {"2026-01-01": 400.0, "2026-01-02": 401.0, "2026-01-03": 402.0}
        archive = {"2026-01-01": 420.0, "2026-01-02": 421.0, "2026-01-03": 422.0}
        with self.assertRaises(ValueError):
            validate_stitch(ohlcv, archive)

    def test_no_common_dates_is_ok(self):
        v = validate_stitch({"2026-01-01": 400.0}, {"2026-02-01": 500.0})
        self.assertTrue(v.ok)
        self.assertEqual(v.n_common, 0)


class TestStitchSpySeries(unittest.TestCase):
    def setUp(self):
        self.ohlcv_path = _make_ohlcv([
            ("SPY", "2026-01-01", 400.0),
            ("SPY", "2026-01-02", 401.0),
        ])
        self.archive_path = _make_archive([
            ("SPY", "2026-01-02", 401.5),  # overlap: ohlcv must win
            ("SPY", "2026-01-03", 403.0),  # extension beyond ohlcv
        ])

    def tearDown(self):
        os.remove(self.ohlcv_path)
        os.remove(self.archive_path)

    def test_ohlcv_wins_on_overlap_archive_extends(self):
        series, validation = stitch_spy_series(self.ohlcv_path, self.archive_path)
        self.assertEqual(series["2026-01-01"], 400.0)
        self.assertEqual(series["2026-01-02"], 401.0)   # ohlcv, not 401.5
        self.assertEqual(series["2026-01-03"], 403.0)   # archive extension
        self.assertTrue(validation.ok)

    def test_raises_when_sources_disagree(self):
        ohlcv_path = _make_ohlcv([("SPY", "2026-01-01", 400.0)])
        archive_path = _make_archive([("SPY", "2026-01-01", 500.0)])
        try:
            with self.assertRaises(ValueError):
                stitch_spy_series(ohlcv_path, archive_path)
        finally:
            os.remove(ohlcv_path)
            os.remove(archive_path)


class TestNearestPrice(unittest.TestCase):
    def setUp(self):
        self.series = {
            "2026-01-01": 100.0,
            "2026-01-05": 105.0,
            "2026-01-10": 110.0,
        }

    def test_exact_match(self):
        m = nearest_price(self.series, "2026-01-01")
        self.assertEqual(m.offset_days, 0)
        self.assertEqual(m.matched_date, "2026-01-01")
        self.assertEqual(m.price, 100.0)

    def test_tolerant_match_within_window(self):
        m = nearest_price(self.series, "2026-01-03", tolerance_days=4)
        self.assertIsNotNone(m)
        self.assertIn(m.matched_date, ("2026-01-01", "2026-01-05"))
        self.assertLessEqual(m.offset_days, 4)

    def test_no_match_beyond_tolerance(self):
        m = nearest_price(self.series, "2026-06-01", tolerance_days=4)
        self.assertIsNone(m)

    def test_deterministic_tie_break_prefers_earlier_date(self):
        # 2026-01-03 is 2 days from both 2026-01-01 and 2026-01-05.
        series = {"2026-01-01": 100.0, "2026-01-05": 105.0}
        m = nearest_price(series, "2026-01-03", tolerance_days=4)
        self.assertEqual(m.matched_date, "2026-01-01")
        self.assertEqual(m.offset_days, 2)


if __name__ == "__main__":
    unittest.main()
