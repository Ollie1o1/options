"""Tests for src/av_options.py — Alpha Vantage historical options (offline)."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import av_options as av


GOOD_PAYLOAD = {
    "endpoint": "Historical Options",
    "message": "success",
    "data": [
        {
            "contractID": "IBM240105C00150000",
            "symbol": "IBM",
            "expiration": "2024-01-05",
            "strike": "150.00",
            "type": "call",
            "last": "12.00",
            "mark": "12.05",
            "bid": "11.90", "bid_size": "10",
            "ask": "12.20", "ask_size": "12",
            "volume": "55", "open_interest": "1200",
            "date": "2024-01-04",
            "implied_volatility": "0.25113",
            "delta": "0.95", "gamma": "0.01", "theta": "-0.02",
            "vega": "0.05", "rho": "0.12",
        },
    ],
}

PREMIUM_PAYLOAD = {"Information": "This is a premium endpoint. Please subscribe "
                                  "to a premium plan to unlock this endpoint."}
RATE_PAYLOAD = {"Information": "API rate limit reached. 25 requests per day."}
BAD_KEY_PAYLOAD = {"Error Message": "the parameter apikey is invalid or missing."}


class ClassifyResponseTest(unittest.TestCase):
    def test_real_data(self):
        self.assertEqual(av.classify_response(GOOD_PAYLOAD), "real_data")

    def test_premium_required(self):
        self.assertEqual(av.classify_response(PREMIUM_PAYLOAD), "premium_required")

    def test_rate_limited(self):
        self.assertEqual(av.classify_response(RATE_PAYLOAD), "rate_limited")

    def test_bad_key(self):
        self.assertEqual(av.classify_response(BAD_KEY_PAYLOAD), "bad_key")

    def test_empty_day_is_no_data(self):
        self.assertEqual(av.classify_response({"data": []}), "no_data")


class ParseRowsTest(unittest.TestCase):
    def test_normalizes_to_archive_schema(self):
        rows = av.parse_rows(GOOD_PAYLOAD, symbol="IBM", date="2024-01-04")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["symbol"], "IBM")
        self.assertEqual(r["snap_date"], "2024-01-04")
        self.assertEqual(r["type"], "call")
        self.assertAlmostEqual(r["strike"], 150.0)
        self.assertAlmostEqual(r["iv"], 0.25113)
        self.assertAlmostEqual(r["bid"], 11.9)
        self.assertEqual(r["source"], "alphavantage")


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "chains.db")

    def test_backfill_respects_budget_and_resumes(self):
        fetched = []
        def fake_fetch(symbol, date, api_key):
            fetched.append(date)
            return "real_data", av.parse_rows(GOOD_PAYLOAD, symbol, date)
        # Mon 2024-01-01 .. Fri 2024-01-05 (5 weekdays), budget 2
        r1 = av.backfill("IBM", start="2024-01-01", end="2024-01-05",
                         db_path=self.db, budget=2, fetch_fn=fake_fetch,
                         api_key="k")
        self.assertEqual(r1["fetched"], 2)
        self.assertEqual(fetched, ["2024-01-01", "2024-01-02"])
        # resume: next run continues where it left off
        r2 = av.backfill("IBM", start="2024-01-01", end="2024-01-05",
                         db_path=self.db, budget=10, fetch_fn=fake_fetch,
                         api_key="k")
        self.assertEqual(fetched[-1], "2024-01-05")
        self.assertEqual(r2["fetched"], 3)

    def test_backfill_skips_weekends(self):
        fetched = []
        def fake_fetch(symbol, date, api_key):
            fetched.append(date)
            return "real_data", []
        av.backfill("IBM", start="2024-01-05", end="2024-01-08",  # Fri..Mon
                    db_path=self.db, budget=10, fetch_fn=fake_fetch, api_key="k")
        self.assertEqual(fetched, ["2024-01-05", "2024-01-08"])

    def test_backfill_stops_on_rate_limit(self):
        def fake_fetch(symbol, date, api_key):
            return "rate_limited", []
        r = av.backfill("IBM", start="2024-01-01", end="2024-01-05",
                        db_path=self.db, budget=10, fetch_fn=fake_fetch,
                        api_key="k")
        self.assertEqual(r["fetched"], 0)
        self.assertEqual(r["stopped"], "rate_limited")

    def test_rows_land_in_archive_db(self):
        def fake_fetch(symbol, date, api_key):
            return "real_data", av.parse_rows(GOOD_PAYLOAD, symbol, date)
        av.backfill("IBM", start="2024-01-04", end="2024-01-04",
                    db_path=self.db, budget=1, fetch_fn=fake_fetch, api_key="k")
        with sqlite3.connect(self.db) as conn:
            n, src = conn.execute(
                "SELECT COUNT(*), MAX(source) FROM chain_snapshots").fetchone()
        self.assertEqual(n, 1)
        self.assertEqual(src, "alphavantage")


if __name__ == "__main__":
    unittest.main()
