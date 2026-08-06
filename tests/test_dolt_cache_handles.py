"""Behaviour tests for the Dolt option-chain cache.

The cache had no tests, and it is now load-bearing: the allocation backtester
reads every chain through it, so a wrong answer here becomes wrong P&L
downstream with nothing to catch it.

The most important case is the empty day. A real trading day can legitimately
return zero contracts (2022-02-18 does). That must read back as `[]` — "fetched,
genuinely nothing there" — and never as `None`, which means "never fetched" and
would make a backfill re-request the same day forever while the backtest treats
the gap as a no-opportunity day.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_dolt_cache_handles -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

from src import dolt_options as do


class CacheRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "dolt.db")
        do._ensure_cache(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def _contract(self, strike=190.0, typ="call"):
        return {"expiration": "2024-02-16", "strike": strike, "type": typ,
                "bid": 1.0, "ask": 1.2, "mid": 1.1, "iv": 0.25,
                "delta": 0.5, "gamma": 0.01, "theta": -0.05,
                "vega": 0.1, "rho": 0.02}

    def test_write_then_read_roundtrip(self):
        do._cache_write(self.db, "AAPL", "2024-01-05", [self._contract()])
        got = do._cache_read(self.db, "AAPL", "2024-01-05")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["strike"], 190.0)
        self.assertEqual(got[0]["type"], "call")
        self.assertEqual(got[0]["mid"], 1.1)

    def test_empty_day_is_a_miss_not_a_gap(self):
        """[] means 'fetched, no contracts'. None means 'never fetched'."""
        do._cache_write(self.db, "AAPL", "2024-01-06", [])
        self.assertEqual(do._cache_read(self.db, "AAPL", "2024-01-06"), [])
        self.assertIsNone(do._cache_read(self.db, "AAPL", "2099-01-01"))

    def test_already_fetched_reflects_writes(self):
        self.assertFalse(do._already_fetched(self.db, "NVDA", "2024-01-05"))
        do._cache_write(self.db, "NVDA", "2024-01-05", [])
        self.assertTrue(do._already_fetched(self.db, "NVDA", "2024-01-05"))

    def test_rewrite_replaces_rather_than_duplicates(self):
        """A refetch of the same (symbol,date) must not double the chain."""
        do._cache_write(self.db, "AAPL", "2024-01-05", [self._contract()])
        do._cache_write(self.db, "AAPL", "2024-01-05", [self._contract()])
        self.assertEqual(len(do._cache_read(self.db, "AAPL", "2024-01-05")), 1)

    def test_calls_and_puts_at_one_strike_are_distinct(self):
        """Type is part of the key — a put must not overwrite its call."""
        do._cache_write(self.db, "AAPL", "2024-01-05",
                        [self._contract(typ="call"), self._contract(typ="put")])
        got = do._cache_read(self.db, "AAPL", "2024-01-05")
        self.assertEqual(len(got), 2)
        self.assertEqual({c["type"] for c in got}, {"call", "put"})

    def test_symbols_do_not_bleed_into_each_other(self):
        do._cache_write(self.db, "AAPL", "2024-01-05", [self._contract()])
        do._cache_write(self.db, "MSFT", "2024-01-05", [])
        self.assertEqual(len(do._cache_read(self.db, "AAPL", "2024-01-05")), 1)
        self.assertEqual(do._cache_read(self.db, "MSFT", "2024-01-05"), [])


class CachePathTest(unittest.TestCase):
    """DEFAULT_CACHE is a RELATIVE path.

    Running any tool that relies on the default from a directory other than the
    repo root silently builds a second, empty cache beside the caller — which is
    exactly how a universe backfill spent several minutes writing 43MB into a
    scratch directory while the real cache sat untouched. Callers outside the
    repo root must pass an absolute db_path.
    """

    def test_default_cache_is_relative_and_therefore_cwd_sensitive(self):
        self.assertFalse(
            os.path.isabs(do.DEFAULT_CACHE),
            "DEFAULT_CACHE became absolute — update the callers that were "
            "written to compensate for it being relative")


if __name__ == "__main__":
    unittest.main()
