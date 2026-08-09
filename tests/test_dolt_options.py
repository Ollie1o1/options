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

    def test_daily_range_skips_weekends(self):
        # 2024-03-02/03 are Sat/Sun. The options market never quotes then, so
        # fetching them burns a rate-limited API call to cache a guaranteed
        # miss. Range is inclusive of the weekday endpoints.
        dates = do._date_range("2024-03-01", "2024-03-05", weekly=False)
        self.assertEqual(dates, ["2024-03-01", "2024-03-04", "2024-03-05"])

    def test_daily_range_never_yields_a_weekend(self):
        for d in do._date_range("2020-01-27", "2020-03-31", weekly=False):
            import datetime as _dt
            self.assertLess(_dt.date.fromisoformat(d).weekday(), 5, d)


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


def _row(symbol="AAPL", date="2024-03-15", strike="170.00"):
    return {"date": date, "act_symbol": symbol, "expiration": "2024-04-19",
            "strike": strike, "call_put": "Call", "bid": "5", "ask": "5.4",
            "vol": "0.3", "delta": "0.5", "gamma": "0", "theta": "0",
            "vega": "0", "rho": "0"}


class MissingPairsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "d.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reports_every_pair_when_cache_is_empty(self):
        pairs = do.missing_pairs(["AAPL", "SPY"], ["2024-03-14", "2024-03-15"],
                                 db_path=self.db)
        self.assertEqual(len(pairs), 4)

    def test_excludes_pairs_already_fetched_including_cached_misses(self):
        with mock.patch("src.dolt_options._query", return_value=[]):
            do.backfill(["AAPL"], ["2024-03-14"], db_path=self.db)
        pairs = do.missing_pairs(["AAPL", "SPY"], ["2024-03-14", "2024-03-15"],
                                 db_path=self.db)
        # A cached MISS is still "fetched" — re-fetching it would burn the call
        # budget re-confirming a hole the dataset genuinely has.
        self.assertNotIn(("AAPL", "2024-03-14"), pairs)
        self.assertEqual(len(pairs), 3)

    def test_weekends_are_never_proposed(self):
        pairs = do.missing_pairs(["AAPL"], do._date_range("2024-03-01", "2024-03-31"),
                                 db_path=self.db)
        for _, d in pairs:
            import datetime as _dt
            self.assertLess(_dt.date.fromisoformat(d).weekday(), 5, d)


class ParallelBackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "d.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fetches_all_missing_pairs_and_is_resumable(self):
        dates = ["2024-03-14", "2024-03-15"]
        with mock.patch("src.dolt_options._query",
                        side_effect=lambda sql, **kw: [_row()]) as q:
            r1 = do.backfill_parallel(["AAPL", "SPY"], dates, db_path=self.db,
                                      workers=4, throttle=0.0)
            r2 = do.backfill_parallel(["AAPL", "SPY"], dates, db_path=self.db,
                                      workers=4, throttle=0.0)
        self.assertEqual(r1["fetched"], 4)
        self.assertEqual(r1["failed"], 0)
        self.assertEqual(r2["fetched"], 0)
        self.assertEqual(q.call_count, 4)

    def test_rows_land_in_the_cache_and_read_back(self):
        with mock.patch("src.dolt_options._query",
                        side_effect=lambda sql, **kw: [_row()]):
            do.backfill_parallel(["AAPL"], ["2024-03-15"], db_path=self.db,
                                 workers=2, throttle=0.0)
        chain = do.get_chain("AAPL", "2024-03-15", db_path=self.db)
        self.assertEqual(len(chain), 1)
        self.assertAlmostEqual(chain[0]["strike"], 170.0)

    def test_a_failed_fetch_is_never_cached_as_an_empty_day(self):
        # THE correctness property. Caching a network failure as "0 rows" would
        # mark a day the dataset really has as permanently absent, and every
        # later run would skip it. A failure must be reported and left missing.
        def flaky(sql, **kw):
            if "SPY" in sql:
                raise do.DoltQueryError("boom")
            return [_row()]

        with mock.patch("src.dolt_options._query", side_effect=flaky):
            res = do.backfill_parallel(["AAPL", "SPY"], ["2024-03-15"],
                                       db_path=self.db, workers=2, throttle=0.0)
        self.assertEqual(res["fetched"], 1)
        self.assertEqual(res["failed"], 1)
        with sqlite3.connect(self.db) as conn:
            marked = conn.execute(
                "SELECT 1 FROM dolt_fetched WHERE symbol='SPY'").fetchone()
        self.assertIsNone(marked)
        # ...and the pair is still offered on the next run.
        self.assertIn(("SPY", "2024-03-15"),
                      do.missing_pairs(["SPY"], ["2024-03-15"], db_path=self.db))

    def test_a_genuinely_empty_day_IS_cached_as_a_miss(self):
        # The counterpart: the API succeeding with zero rows is a real answer.
        with mock.patch("src.dolt_options._query", return_value=[]):
            res = do.backfill_parallel(["AAPL"], ["2024-03-15"], db_path=self.db,
                                       workers=2, throttle=0.0)
        self.assertEqual(res["fetched"], 1)
        self.assertEqual(res["failed"], 0)
        self.assertNotIn(("AAPL", "2024-03-15"),
                         do.missing_pairs(["AAPL"], ["2024-03-15"], db_path=self.db))

    def test_rate_limiting_stops_the_run_rather_than_burning_the_queue(self):
        # Hammering a rate-limited endpoint gets the IP blocked. Once DoltHub
        # says stop, the run must abort with work left, not grind every pair
        # into a failure.
        with mock.patch("src.dolt_options._query",
                        side_effect=do.DoltRateLimited("429")):
            res = do.backfill_parallel(["AAPL", "SPY", "MSFT"],
                                       ["2024-03-14", "2024-03-15"],
                                       db_path=self.db, workers=2, throttle=0.0)
        self.assertTrue(res["rate_limited"])
        self.assertEqual(res["fetched"], 0)
        self.assertGreater(res["remaining"], 0)


if __name__ == "__main__":
    unittest.main()
