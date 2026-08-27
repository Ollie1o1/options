"""Kalshi parsing and the archive. No network in tests."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.predmarkets import archive, kalshi

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "predmarkets", "kalshi_markets.json")


def payload():
    with open(FIXTURE) as f:
        return json.load(f)


# Real shapes observed 2026-08-26: a tight two-sided market, a wide one, and a
# zero-bid one-sided market (every KXCPI contract looked like the last).
SYNTHETIC = {"markets": [
    {"ticker": "KXFEDDECISION-28JAN-H0", "event_ticker": "E1",
     "title": "Hike 0bps at the January 2028 meeting?",
     "yes_bid_dollars": 0.63, "yes_ask_dollars": 0.64,
     "last_price_dollars": 0.64, "open_interest_fp": 400.71,
     "close_time": "2028-01-26T15:00:00Z"},
    {"ticker": "KXFED-27APR-T3.75", "event_ticker": "E2",
     "title": "Upper bound above 3.75%?",
     "yes_bid_dollars": 0.43, "yes_ask_dollars": 0.63,
     "last_price_dollars": 0.42, "open_interest_fp": 1737.10,
     "close_time": "2027-04-28T15:00:00Z"},
    {"ticker": "KXCPI-26AUG-T0.8", "event_ticker": "E3",
     "title": "Will CPI rise more than 0.8% in August 2026?",
     "yes_bid_dollars": 0.0, "yes_ask_dollars": 0.40,
     "last_price_dollars": 0.13, "open_interest_fp": 529.78,
     "close_time": "2026-09-11T15:00:00Z"},
]}


class TestParseMarkets(unittest.TestCase):
    def test_parses_the_recorded_fixture(self):
        quotes = kalshi.parse_markets(payload(), "KXFED")
        self.assertEqual(len(quotes), len(payload()["markets"]))

    def test_captures_both_sides_not_a_single_probability(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[0]
        self.assertEqual(q.yes_bid, 0.63)
        self.assertEqual(q.yes_ask, 0.64)
        self.assertEqual(q.last, 0.64)

    def test_series_is_recorded(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[0]
        self.assertEqual(q.series, "KXFED")

    def test_open_interest_is_kept(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[0]
        self.assertAlmostEqual(q.open_interest, 400.71, places=2)

    def test_a_market_with_no_ticker_is_skipped(self):
        self.assertEqual(kalshi.parse_markets({"markets": [{"title": "x"}]},
                                              "KXFED"), [])

    def test_missing_prices_are_none_not_zero(self):
        bad = {"markets": [{"ticker": "T", "yes_bid_dollars": None,
                            "yes_ask_dollars": None}]}
        q = kalshi.parse_markets(bad, "KXFED")[0]
        self.assertIsNone(q.yes_bid)
        self.assertIsNone(q.yes_ask)


class TestSpread(unittest.TestCase):
    def test_tight_two_sided_market(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[0]
        self.assertAlmostEqual(q.spread, 0.01, places=4)

    def test_wide_market(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[1]
        self.assertAlmostEqual(q.spread, 0.20, places=4)

    def test_spread_is_none_when_a_side_is_missing(self):
        bad = {"markets": [{"ticker": "T", "yes_bid_dollars": None,
                            "yes_ask_dollars": 0.5}]}
        self.assertIsNone(kalshi.parse_markets(bad, "KXFED")[0].spread)


class TestMid(unittest.TestCase):
    def test_mid_of_a_tight_market(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[0]
        self.assertAlmostEqual(q.mid(max_spread=0.05), 0.635, places=4)

    def test_wide_market_refuses_to_produce_a_mid(self):
        # 0.43/0.63 has no meaningful midpoint. Returning 0.53 would invent a
        # precision the market does not have.
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[1]
        self.assertIsNone(q.mid(max_spread=0.05))

    def test_zero_bid_one_sided_market_refuses(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[2]
        self.assertIsNone(q.mid(max_spread=0.05))

    def test_a_generous_threshold_still_admits_the_wide_one(self):
        q = kalshi.parse_markets(SYNTHETIC, "KXFED")[1]
        self.assertAlmostEqual(q.mid(max_spread=0.25), 0.53, places=4)


class TestArchive(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = archive.connect(os.path.join(self._dir.name, "pm.db"))
        self.addCleanup(self.conn.close)

    def test_records_one_row_per_market_per_day(self):
        quotes = kalshi.parse_markets(SYNTHETIC, "KXFED")
        archive.record(self.conn, quotes, "2026-08-26")
        n = self.conn.execute("SELECT COUNT(*) FROM pm_quotes").fetchone()[0]
        self.assertEqual(n, 3)

    def test_rerunning_the_same_day_is_idempotent(self):
        quotes = kalshi.parse_markets(SYNTHETIC, "KXFED")
        archive.record(self.conn, quotes, "2026-08-26")
        archive.record(self.conn, quotes, "2026-08-26")
        n = self.conn.execute("SELECT COUNT(*) FROM pm_quotes").fetchone()[0]
        self.assertEqual(n, 3)

    def test_a_new_day_adds_a_new_observation(self):
        quotes = kalshi.parse_markets(SYNTHETIC, "KXFED")
        archive.record(self.conn, quotes, "2026-08-26")
        archive.record(self.conn, quotes, "2026-08-27")
        n = self.conn.execute("SELECT COUNT(*) FROM pm_quotes").fetchone()[0]
        self.assertEqual(n, 6)

    def test_archived_at_is_stored_and_is_the_lookahead_guard(self):
        archive.record(self.conn, kalshi.parse_markets(SYNTHETIC, "KXFED"),
                       "2026-08-26")
        row = self.conn.execute(
            "SELECT archived_at FROM pm_quotes LIMIT 1").fetchone()
        self.assertEqual(row[0], "2026-08-26")

    def test_missing_prices_persist_as_null_not_zero(self):
        bad = {"markets": [{"ticker": "T", "yes_bid_dollars": None,
                            "yes_ask_dollars": None}]}
        archive.record(self.conn, kalshi.parse_markets(bad, "KXFED"),
                       "2026-08-26")
        row = self.conn.execute(
            "SELECT yes_bid, yes_ask FROM pm_quotes").fetchone()
        self.assertEqual(row, (None, None))


class TestFetchSeries(unittest.TestCase):
    def test_network_failure_returns_empty_not_raise(self):
        with mock.patch.object(kalshi, "_get_json", side_effect=OSError("x")):
            self.assertEqual(kalshi.fetch_series("KXFED"), [])

    def test_uses_the_public_elections_host(self):
        self.assertIn("api.elections.kalshi.com", kalshi.BASE)


if __name__ == "__main__":
    unittest.main()
