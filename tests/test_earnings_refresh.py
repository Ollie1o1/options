"""The earnings cache has to be refreshable, or the gate quietly goes blind.

`earnings_dates` marks a symbol in `earnings_fetched` the first time it is
queried and never queries it again. Companies announce their next report about
three to four weeks ahead, so a cache written once in June holds every past
quarter and no future one. Measured 2026-08-20: 163 symbols cached, **18 with
any date at or after today**, and the oldest fetch marker was 2026-06-15.
Re-querying three of them moved AAPL 2026-04-30 -> 2026-07-30, MSFT
2026-04-29 -> 2026-07-29, and NVDA 2026-05-20 -> **2026-08-26**, a date six
days in the future that the gate needs and did not have.

That is the failure mode the earnings gate's UNKNOWN state was designed to make
visible (src/earnings_gate.py): a stale cache reports no event inside any
window, which is indistinguishable from safety. Making the cache refreshable is
what turns UNKNOWN back into an answer.

No network: the fetcher is injected.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dolt_earnings import earnings_dates, refresh_symbols


class _Fetcher:
    """Stands in for the DoltHub query; records what it was asked for."""

    def __init__(self, by_symbol):
        self.by_symbol = by_symbol
        self.calls = []

    def __call__(self, symbol):
        self.calls.append(symbol)
        return [{"date": d, "when": "amc"}
                for d in self.by_symbol.get(symbol.upper(), [])]


class RefreshOnAge(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "cache.db")

    def tearDown(self):
        self.dir.cleanup()

    def _age_marker(self, symbol, days):
        stamp = (dt.datetime.now() - dt.timedelta(days=days)).isoformat(
            timespec="seconds")
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT OR REPLACE INTO earnings_fetched "
                         "(symbol, fetched_at) VALUES (?,?)", (symbol, stamp))

    def test_a_first_query_fetches(self):
        f = _Fetcher({"NVDA": ["2026-05-20"]})
        self.assertEqual(earnings_dates("NVDA", self.db, fetcher=f),
                         ["2026-05-20"])
        self.assertEqual(f.calls, ["NVDA"])

    def test_a_fresh_cache_is_not_refetched(self):
        f = _Fetcher({"NVDA": ["2026-05-20"]})
        earnings_dates("NVDA", self.db, fetcher=f)
        earnings_dates("NVDA", self.db, max_age_days=7, fetcher=f)
        self.assertEqual(f.calls, ["NVDA"], "refetched a cache written today")

    def test_a_stale_cache_is_refetched_and_gains_the_new_date(self):
        # The whole point: NVDA's next report was announced after the first
        # fetch, and without this the gate never sees it.
        f = _Fetcher({"NVDA": ["2026-05-20"]})
        earnings_dates("NVDA", self.db, fetcher=f)
        f.by_symbol["NVDA"] = ["2026-05-20", "2026-08-26"]
        self._age_marker("NVDA", 30)
        out = earnings_dates("NVDA", self.db, max_age_days=7, fetcher=f)
        self.assertEqual(out, ["2026-05-20", "2026-08-26"])
        self.assertEqual(len(f.calls), 2)

    def test_without_max_age_the_old_never_refetch_behaviour_is_kept(self):
        # Every existing caller passes no max_age_days and must be unaffected.
        f = _Fetcher({"NVDA": ["2026-05-20"]})
        earnings_dates("NVDA", self.db, fetcher=f)
        self._age_marker("NVDA", 400)
        earnings_dates("NVDA", self.db, fetcher=f)
        self.assertEqual(len(f.calls), 1)

    def test_refetching_never_drops_a_date_already_cached(self):
        # A provider that returns less than last time must not erase history —
        # the IV-crush study reads the same table.
        f = _Fetcher({"NVDA": ["2026-02-25", "2026-05-20"]})
        earnings_dates("NVDA", self.db, fetcher=f)
        f.by_symbol["NVDA"] = ["2026-08-26"]
        self._age_marker("NVDA", 30)
        out = earnings_dates("NVDA", self.db, max_age_days=7, fetcher=f)
        self.assertEqual(out, ["2026-02-25", "2026-05-20", "2026-08-26"])

    def test_a_failed_fetch_leaves_the_cache_intact(self):
        def boom(symbol):
            raise RuntimeError("dolthub is down")
        f = _Fetcher({"NVDA": ["2026-05-20"]})
        earnings_dates("NVDA", self.db, fetcher=f)
        self._age_marker("NVDA", 30)
        out = earnings_dates("NVDA", self.db, max_age_days=7, fetcher=boom)
        self.assertEqual(out, ["2026-05-20"],
                         "an outage must not empty the calendar")


class RefreshMany(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "cache.db")

    def tearDown(self):
        self.dir.cleanup()

    def test_it_reports_what_each_symbol_gained(self):
        f = _Fetcher({"AAPL": ["2026-07-30"], "NVDA": ["2026-08-26"]})
        out = refresh_symbols(["AAPL", "NVDA"], db_path=self.db,
                              max_age_days=0, fetcher=f)
        self.assertEqual(out["AAPL"]["dates"], 1)
        self.assertEqual(out["NVDA"]["dates"], 1)

    def test_one_bad_symbol_does_not_stop_the_run(self):
        # 124 symbols per refresh; one failure must not cost the other 123.
        def flaky(symbol):
            if symbol == "BAD":
                raise RuntimeError("no such symbol")
            return [{"date": "2026-08-26", "when": "amc"}]
        out = refresh_symbols(["BAD", "NVDA"], db_path=self.db,
                              max_age_days=0, fetcher=flaky)
        self.assertIn("error", out["BAD"])
        self.assertEqual(out["NVDA"]["dates"], 1)

    def test_it_paces_itself_between_symbols(self):
        # 124 rapid-fire queries tripped DoltHub's capacity wall on 2026-08-20:
        # 68 of them came back empty, and the same symbols fetched fine one at
        # a time a minute later. Unpaced, this helper reports a refresh that
        # mostly did not happen.
        f = _Fetcher({s: ["2026-08-26"] for s in ("A", "B", "C")})
        waits = []
        refresh_symbols(["A", "B", "C"], db_path=self.db, max_age_days=0,
                        fetcher=f, pause=waits.append)
        self.assertEqual(len(waits), 3)
        self.assertTrue(all(w > 0 for w in waits), waits)

    def test_it_counts_symbols_with_a_future_date(self):
        # The number that decides whether the gate can act at all.
        today = dt.date.today()
        f = _Fetcher({"PAST": [(today - dt.timedelta(days=40)).isoformat()],
                      "FUTURE": [(today + dt.timedelta(days=6)).isoformat()]})
        out = refresh_symbols(["PAST", "FUTURE"], db_path=self.db,
                              max_age_days=0, fetcher=f)
        self.assertFalse(out["PAST"]["has_future"])
        self.assertTrue(out["FUTURE"]["has_future"])


if __name__ == "__main__":
    unittest.main()
