"""Refusing to sell premium across a dated binary event.

2026-08-18 the feeder opened a WMT Bull Put two days before Walmart reported.
It was stopped out on the morning of the report for -$274.50, -254% of the
credit. The earnings date was already in this repo's own cache
(`src/dolt_earnings.py`, WMT 2026-08-20) and the correct predicate already
existed (`holds_through_earnings`) — but the live scan path never called
either. The only earnings logic it had, `earnings_buffer_days`, sets a DISPLAY
flag that tests `|expiration - earnings| <= 5` — proximity of the EXPIRY to the
event, not whether the event falls inside the holding period — and for WMT that
was 29 days, so nothing lit up. Worse, when that flag does fire it RAISES the
candidate's score in Premium Selling mode: "sellers: high crush = opportunity".

Three properties this module is built around, each pinned below.

  1. **Three states, not two.** THROUGH / CLEAR / UNKNOWN. A symbol with no
     cached earnings date, or whose cached dates all predate the trade, is
     UNKNOWN — never CLEAR. Measured 2026-08-20 over 543 credit trades, 72% of
     them are UNKNOWN: the cache covers about a quarter of the book's symbols,
     so a two-state gate would be silently inert on three trades in four. This
     is the partial-silence failure this repo has hit before.
  2. **Cache only, never the network.** `log_trade` is on the entry path;
     `dolt_earnings.earnings_dates` fetches on a cache miss and would put an
     HTTP call inside a ledger write.
  3. **Short premium only.** Selling a spread across an event is short a
     gap; buying one is long it. The book's evidence is about the former.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_earnings_gate -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.earnings_gate import (CLEAR, THROUGH, UNKNOWN, applies_to,
                               cached_earnings_dates, classify, horizon_end,
                               load_earnings_gate_config)


class Classify(unittest.TestCase):
    """The tri-state decision. Pure — dates in, verdict out."""

    def test_an_event_inside_the_window_is_through(self):
        self.assertEqual(
            classify(["2026-08-20"], "2026-08-18", "2026-09-18"), THROUGH)

    def test_an_event_after_the_window_is_clear(self):
        self.assertEqual(
            classify(["2026-09-20"], "2026-08-18", "2026-09-18"), CLEAR)

    def test_an_event_before_entry_is_clear_when_a_later_date_is_known(self):
        # The cache reaches past this trade, so its silence about the holding
        # period is real information.
        self.assertEqual(
            classify(["2026-08-01", "2026-11-20"], "2026-08-18", "2026-09-18"),
            CLEAR)

    def test_no_dates_at_all_is_unknown(self):
        self.assertEqual(classify([], "2026-08-18", "2026-09-18"), UNKNOWN)

    def test_a_cache_that_ends_before_the_trade_is_unknown_not_clear(self):
        # The property that keeps this gate honest. Every cached date predates
        # the entry, so the cache says nothing about this holding period —
        # calling that CLEAR would turn a stale cache into a silent all-clear.
        self.assertEqual(
            classify(["2025-11-20", "2026-02-19"], "2026-08-18", "2026-09-18"),
            UNKNOWN)

    def test_an_event_on_the_entry_date_is_not_held_through(self):
        # It is already public when the position is opened; the trade is priced
        # with it in the past.
        self.assertEqual(
            classify(["2026-08-18", "2026-11-20"], "2026-08-18", "2026-09-18"),
            CLEAR)

    def test_an_event_on_the_last_day_of_the_window_is_held_through(self):
        self.assertEqual(
            classify(["2026-09-18"], "2026-08-18", "2026-09-18"), THROUGH)

    def test_unknown_wins_over_clear_but_not_over_through(self):
        # A symbol can have a known event inside the window AND a cache that
        # stops there; the event is the stronger fact.
        self.assertEqual(
            classify(["2026-09-01"], "2026-08-18", "2026-09-18"), THROUGH)


class Horizon(unittest.TestCase):
    """How far ahead the gate looks."""

    def test_expiration_mode_looks_to_expiry(self):
        self.assertEqual(horizon_end("2026-09-18", 21, "expiration"),
                         "2026-09-18")

    def test_time_exit_mode_stops_where_the_time_exit_closes_the_position(self):
        # The DTE rule force-closes at 21 DTE and is the one exit that fires
        # without reading a mark, so it is a real ceiling on exposure.
        self.assertEqual(horizon_end("2026-09-18", 21, "time_exit"),
                         "2026-08-28")

    def test_an_unparseable_expiration_yields_none(self):
        self.assertIsNone(horizon_end("not-a-date", 21, "expiration"))

    def test_a_zero_time_exit_is_the_same_as_expiration(self):
        self.assertEqual(horizon_end("2026-09-18", 0, "time_exit"),
                         "2026-09-18")


class Applicability(unittest.TestCase):
    """Short premium only."""

    def test_credit_structures_are_gated(self):
        for name in ("Bull Put", "Bear Call", "Iron Condor",
                     "Bull Put Spread", "Short Put", "Cash-Secured Put"):
            self.assertTrue(applies_to(name), name)

    def test_long_premium_is_not_gated(self):
        # Buying premium across an event is being LONG the gap. Whatever is
        # wrong with that (IV crush) is a different trade and different
        # evidence; this gate does not claim it.
        for name in ("Long Call", "Long Put", "Lottery Long Call",
                     "Bull Call Spread"):
            self.assertFalse(applies_to(name), name)

    def test_an_empty_strategy_is_not_gated(self):
        self.assertFalse(applies_to(""))
        self.assertFalse(applies_to(None))


class CachedLookup(unittest.TestCase):
    """Reads the cache `dolt_earnings` fills. Never fetches."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "cache.db")

    def tearDown(self):
        self.dir.cleanup()

    def _seed(self, rows):
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE earnings_cal (symbol TEXT, date TEXT, "
                         "whn TEXT, PRIMARY KEY (symbol, date))")
            conn.executemany("INSERT INTO earnings_cal VALUES (?,?,?)", rows)

    def test_it_returns_the_cached_dates_sorted(self):
        self._seed([("WMT", "2026-08-20", "amc"), ("WMT", "2026-05-21", "bmo")])
        self.assertEqual(cached_earnings_dates("WMT", self.db),
                         ["2026-05-21", "2026-08-20"])

    def test_symbols_are_matched_case_insensitively(self):
        self._seed([("WMT", "2026-08-20", "amc")])
        self.assertEqual(cached_earnings_dates("wmt", self.db), ["2026-08-20"])

    def test_a_missing_database_is_empty_not_an_exception(self):
        # A ledger write must not fail because an analysis cache is absent.
        self.assertEqual(
            cached_earnings_dates("WMT", os.path.join(self.dir.name, "nope.db")),
            [])

    def test_a_database_without_the_table_is_empty(self):
        sqlite3.connect(self.db).close()
        self.assertEqual(cached_earnings_dates("WMT", self.db), [])

    def test_it_does_not_reach_the_network(self):
        # dolt_earnings.earnings_dates fetches on a cache miss. This one must
        # not: it runs inside log_trade.
        import src.dolt_earnings as de
        called = []
        original = de._fetch_live
        de._fetch_live = lambda symbol: called.append(symbol) or []
        try:
            self._seed([("WMT", "2026-08-20", "amc")])
            cached_earnings_dates("NOSUCH", self.db)
        finally:
            de._fetch_live = original
        self.assertEqual(called, [])


class ConfigLoading(unittest.TestCase):

    def test_a_missing_block_is_disabled(self):
        cfg = load_earnings_gate_config({})
        self.assertFalse(cfg["enabled"])

    def test_the_real_config_enables_it(self):
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.json")) as f:
            cfg = load_earnings_gate_config(json.load(f))
        self.assertTrue(cfg["enabled"])
        self.assertIn(cfg["horizon"], ("expiration", "time_exit"))

    def test_an_unrecognised_horizon_falls_back_to_the_conservative_one(self):
        cfg = load_earnings_gate_config(
            {"auto_log": {"refuse_through_earnings": True,
                          "earnings_horizon": "banana"}})
        self.assertEqual(cfg["horizon"], "expiration")


class TheWmtTradeThatCausedThis(unittest.TestCase):
    """Regression: the exact trade, against the exact cached dates."""

    WMT_DATES = ["2025-05-15", "2025-08-21", "2025-11-20",
                 "2026-02-19", "2026-05-21", "2026-08-20"]

    def test_it_is_refused_on_the_expiration_horizon(self):
        end = horizon_end("2026-09-18", 21, "expiration")
        self.assertEqual(classify(self.WMT_DATES, "2026-08-18", end), THROUGH)

    def test_it_is_refused_on_the_time_exit_horizon_too(self):
        # The event was 2 days after entry, so no exit rule could have dodged it.
        end = horizon_end("2026-09-18", 21, "time_exit")
        self.assertEqual(classify(self.WMT_DATES, "2026-08-18", end), THROUGH)

    def test_the_old_display_flag_would_not_have_seen_it(self):
        # |expiration - earnings| = 29 days, far outside earnings_buffer_days=5.
        # Pinned so nobody "simplifies" this gate back into that test.
        import datetime as dt
        gap = abs((dt.date(2026, 9, 18) - dt.date(2026, 8, 20)).days)
        self.assertGreater(gap, 5)


if __name__ == "__main__":
    unittest.main()
