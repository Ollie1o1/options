"""Tests for src/dolt_options.py — DoltHub real-options client + cache.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_options -v
"""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from src import dolt_options as do


class QueryTest(unittest.TestCase):
    def _resp(self, status=200, payload=None):
        m = mock.Mock()
        m.status_code = status
        m.json.return_value = payload or {}
        return m

    def test_query_returns_rows_on_success(self):
        payload = {"query_execution_status": "Success", "rows": [{"n": "5"}]}
        with mock.patch("src.dolt_options.requests.get", return_value=self._resp(200, payload)) as g:
            rows = do._query("SELECT 1")
        self.assertEqual(rows, [{"n": "5"}])
        _, kwargs = g.call_args
        self.assertNotIn("authorization",
                         {k.lower(): v for k, v in (kwargs.get("headers") or {}).items()})

    def test_query_raises_on_api_error(self):
        payload = {"query_execution_status": "Error", "query_execution_message": "boom"}
        with mock.patch("src.dolt_options.requests.get", return_value=self._resp(200, payload)):
            with self.assertRaises(do.DoltQueryError):
                do._query("SELECT bad")


class RateLimitTest(unittest.TestCase):
    def _resp(self, status, payload=None, text=""):
        m = mock.Mock()
        m.status_code = status
        m.json.return_value = payload or {}
        m.text = text
        return m

    def test_403_retries_then_raises_rate_limited(self):
        # Always 403 → after backoff retries, raise DoltRateLimited (not crash).
        with mock.patch("src.dolt_options.requests.get", return_value=self._resp(403)), \
             mock.patch("src.dolt_options.time.sleep"):
            with self.assertRaises(do.DoltRateLimited):
                do._query("SELECT 1")

    def test_403_then_success_recovers(self):
        ok = self._resp(200, {"query_execution_status": "Success", "rows": [{"n": "1"}]})
        seq = [self._resp(403), ok]
        with mock.patch("src.dolt_options.requests.get", side_effect=seq), \
             mock.patch("src.dolt_options.time.sleep"):
            rows = do._query("SELECT 1")
        self.assertEqual(rows, [{"n": "1"}])


class NormalizeTest(unittest.TestCase):
    def test_normalize_row_types_and_mid(self):
        raw = {"date": "2026-06-12", "act_symbol": "AAPL", "expiration": "2026-07-17",
               "strike": "205.00", "call_put": "Call", "bid": "84.95", "ask": "88.05",
               "vol": "0.4274", "delta": "0.9905", "gamma": "0.0010", "theta": "-0.0405",
               "vega": "0.0123", "rho": "0.0456"}
        c = do._normalize(raw)
        self.assertEqual(c["symbol"], "AAPL")
        self.assertEqual(c["type"], "call")
        self.assertAlmostEqual(c["strike"], 205.0)
        self.assertAlmostEqual(c["mid"], (84.95 + 88.05) / 2)
        self.assertAlmostEqual(c["iv"], 0.4274)
        self.assertAlmostEqual(c["theta"], -0.0405)

    def test_clamp_and_snap_dates(self):
        self.assertEqual(do._clamp_date("2010-01-01"), do.COVERAGE_MIN)
        self.assertEqual(do._clamp_date("2099-01-01"), do.COVERAGE_MAX)
        self.assertEqual(do._clamp_date("2024-03-15"), "2024-03-15")


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "dolt.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_rows(self):
        return [{"date": "2024-03-15", "act_symbol": "AAPL", "expiration": "2024-04-19",
                 "strike": "170.00", "call_put": "Call", "bid": "5.00", "ask": "5.40",
                 "vol": "0.30", "delta": "0.55", "gamma": "0.01", "theta": "-0.05",
                 "vega": "0.10", "rho": "0.02"}]

    def test_get_chain_fetches_then_caches(self):
        with mock.patch("src.dolt_options._query", return_value=self._fake_rows()) as q:
            c1 = do.get_chain("AAPL", "2024-03-15", db_path=self.db)
            c2 = do.get_chain("AAPL", "2024-03-15", db_path=self.db)
        self.assertEqual(q.call_count, 1, "second call must hit cache, not the API")
        self.assertEqual(len(c1), 1)
        self.assertEqual(c1[0]["mid"], 5.2)
        self.assertEqual(c2[0]["symbol"], "AAPL")

    def test_empty_day_is_cached_as_miss(self):
        with mock.patch("src.dolt_options._query", return_value=[]) as q:
            do.get_chain("AAPL", "2024-03-16", db_path=self.db)
            do.get_chain("AAPL", "2024-03-16", db_path=self.db)
        self.assertEqual(q.call_count, 1)


class NearestContractTest(unittest.TestCase):
    def _chain(self):
        return [
            {"symbol": "X", "date": "2024-03-15", "expiration": "2024-04-19", "strike": 100.0,
             "type": "call", "bid": 2.0, "ask": 2.2, "mid": 2.1, "iv": 0.3,
             "delta": 0.5, "gamma": 0.01, "theta": -0.04, "vega": 0.1, "rho": 0.02},
            {"symbol": "X", "date": "2024-03-15", "expiration": "2024-04-19", "strike": 110.0,
             "type": "call", "bid": 0.8, "ask": 1.0, "mid": 0.9, "iv": 0.32,
             "delta": 0.3, "gamma": 0.01, "theta": -0.03, "vega": 0.08, "rho": 0.01},
            {"symbol": "X", "date": "2024-03-15", "expiration": "2024-06-21", "strike": 110.0,
             "type": "call", "bid": 2.0, "ask": 2.3, "mid": 2.15, "iv": 0.31,
             "delta": 0.35, "gamma": 0.01, "theta": -0.02, "vega": 0.2, "rho": 0.03},
        ]

    def test_picks_nearest_strike_and_dte(self):
        c = do.nearest_contract(self._chain(), opt_type="call",
                                target_strike=108.0, asof="2024-03-15", target_dte=35)
        self.assertEqual(c["strike"], 110.0)
        self.assertEqual(c["expiration"], "2024-04-19")

    def test_returns_none_when_no_type_match(self):
        self.assertIsNone(do.nearest_contract(self._chain(), opt_type="put",
                          target_strike=100.0, asof="2024-03-15", target_dte=30))


class DateRangeTest(unittest.TestCase):
    def test_weekly_from_non_friday_start_yields_fridays(self):
        # 2023-01-01 is a Sunday; weekly must still return Fridays in range.
        dates = do._date_range("2023-01-01", "2023-01-31", weekly=True)
        self.assertTrue(len(dates) >= 4)
        for d in dates:
            import datetime as _dt
            self.assertEqual(_dt.date.fromisoformat(d).weekday(), 4)

    def test_daily_range_inclusive(self):
        dates = do._date_range("2024-03-01", "2024-03-05", weekly=False)
        self.assertEqual(dates, ["2024-03-01", "2024-03-02", "2024-03-03",
                                 "2024-03-04", "2024-03-05"])


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "d.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_backfill_skips_already_fetched(self):
        rows = [{"date": "2024-03-15", "act_symbol": "AAPL", "expiration": "2024-04-19",
                 "strike": "170.00", "call_put": "Call", "bid": "5", "ask": "5.4",
                 "vol": "0.3", "delta": "0.5", "gamma": "0", "theta": "0", "vega": "0", "rho": "0"}]
        with mock.patch("src.dolt_options._query", return_value=rows) as q:
            n1 = do.backfill(["AAPL"], ["2024-03-15"], db_path=self.db)
            n2 = do.backfill(["AAPL"], ["2024-03-15"], db_path=self.db)
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 0)
        self.assertEqual(q.call_count, 1)


if __name__ == "__main__":
    unittest.main()
