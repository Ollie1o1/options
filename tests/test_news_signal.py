"""News sentiment as a cross-sectional factor.

The hazards here are look-ahead and false power, in that order.

LOOK-AHEAD  A headline is usable on day D only if we ARCHIVED it by day D.
            Publication timestamps get revised and backfilled by feeds; the
            archive's `archived_at` is the only honest point-in-time stamp we
            control. Using `published` would let a story we first saw on the
            25th be traded on the 20th.

FALSE POWER 3,210 symbol-days sounds like a large sample and is not: they sit
            on ~30 distinct days, and every symbol on one day shares that day's
            market move. The unit of independence is the DAY, so the factor is
            measured as a per-day cross-sectional IC and t-tested across days.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_news_signal -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src.news_signal import (daily_sentiment, forward_returns,
                             cross_sectional_ic, days_for_power, validate)


def _news_db(path, rows):
    """rows: (symbol, archived_at, sentiment, relevance)"""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE news_archive (id INTEGER PRIMARY KEY, "
              "dedup_key TEXT, symbol TEXT, headline TEXT, source TEXT, "
              "published TEXT, sentiment REAL, relevance REAL, url TEXT, "
              "archived_at TEXT)")
    for i, (sym, arch, sent, rel) in enumerate(rows):
        c.execute("INSERT INTO news_archive (id,dedup_key,symbol,headline,"
                  "published,sentiment,relevance,archived_at) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (i, f"k{i}", sym, "h", arch, sent, rel, arch))
    c.commit(); c.close()


def _px_db(path, rows):
    """rows: (symbol, date, close)"""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL, "
              "volume REAL, PRIMARY KEY (symbol, date))")
    for sym, d, close in rows:
        c.execute("INSERT INTO px VALUES (?,?,?,NULL)", (d, sym, close))
    c.commit(); c.close()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.news = os.path.join(self.tmp, "n.db")
        self.px = os.path.join(self.tmp, "p.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class DailySentimentTest(_Base):
    def test_aggregates_headlines_per_symbol_day(self):
        _news_db(self.news, [("AAA", "2026-07-01T12:00:00", 0.5, 1.0),
                             ("AAA", "2026-07-01T15:00:00", 0.1, 1.0),
                             ("BBB", "2026-07-01T12:00:00", -0.4, 1.0)])
        got = daily_sentiment(self.news)
        self.assertAlmostEqual(got[("2026-07-01", "AAA")]["score"], 0.3)
        self.assertEqual(got[("2026-07-01", "AAA")]["n"], 2)
        self.assertAlmostEqual(got[("2026-07-01", "BBB")]["score"], -0.4)

    def test_relevance_weights_the_mean(self):
        _news_db(self.news, [("AAA", "2026-07-01T12:00:00", 1.0, 3.0),
                             ("AAA", "2026-07-01T15:00:00", 0.0, 1.0)])
        self.assertAlmostEqual(
            daily_sentiment(self.news)[("2026-07-01", "AAA")]["score"], 0.75)

    def test_the_day_is_the_archive_date_not_the_publish_date(self):
        # THE look-ahead test. Published on the 1st, but we did not see it
        # until the 5th, so it must not be tradeable before the 5th.
        c = sqlite3.connect(self.news)
        c.execute("CREATE TABLE news_archive (id INTEGER PRIMARY KEY, "
                  "dedup_key TEXT, symbol TEXT, headline TEXT, source TEXT, "
                  "published TEXT, sentiment REAL, relevance REAL, url TEXT, "
                  "archived_at TEXT)")
        c.execute("INSERT INTO news_archive (id,dedup_key,symbol,published,"
                  "sentiment,relevance,archived_at) VALUES "
                  "(1,'k','AAA','2026-07-01T09:00:00',0.9,1.0,"
                  "'2026-07-05T09:00:00')")
        c.commit(); c.close()
        got = daily_sentiment(self.news)
        self.assertIn(("2026-07-05", "AAA"), got)
        self.assertNotIn(("2026-07-01", "AAA"), got)

    def test_zero_relevance_rows_do_not_divide_by_zero(self):
        _news_db(self.news, [("AAA", "2026-07-01T12:00:00", 0.5, 0.0)])
        self.assertAlmostEqual(
            daily_sentiment(self.news)[("2026-07-01", "AAA")]["score"], 0.5)


class ForwardReturnTest(_Base):
    def test_return_starts_at_the_decision_date(self):
        _px_db(self.px, [("AAA", "2026-07-01", 100.0),
                         ("AAA", "2026-07-02", 110.0),
                         ("AAA", "2026-07-03", 121.0)])
        r = forward_returns(self.px, ["AAA"], horizon=2)
        # From the 1st, two trading days on, is +21% — never measured backwards.
        self.assertAlmostEqual(r[("2026-07-01", "AAA")], 0.21, places=6)

    def test_no_return_when_the_horizon_runs_past_the_data(self):
        _px_db(self.px, [("AAA", "2026-07-01", 100.0),
                         ("AAA", "2026-07-02", 110.0)])
        r = forward_returns(self.px, ["AAA"], horizon=5)
        self.assertNotIn(("2026-07-01", "AAA"), r)

    def test_horizon_counts_trading_rows_not_calendar_days(self):
        _px_db(self.px, [("AAA", "2026-07-01", 100.0),   # Wed
                         ("AAA", "2026-07-06", 120.0)])  # next Mon
        r = forward_returns(self.px, ["AAA"], horizon=1)
        self.assertAlmostEqual(r[("2026-07-01", "AAA")], 0.20, places=6)


class CrossSectionalIcTest(_Base):
    def test_a_day_with_too_few_names_is_skipped(self):
        sent = {("2026-07-01", "AAA"): {"score": 0.5, "n": 1}}
        rets = {("2026-07-01", "AAA"): 0.01}
        self.assertEqual(cross_sectional_ic(sent, rets, min_names=5), [])

    def test_perfect_agreement_scores_ic_one(self):
        day = "2026-07-01"
        sent = {(day, s): {"score": v, "n": 1}
                for s, v in zip("ABCDEFG", range(7))}
        rets = {(day, s): v / 100.0 for s, v in zip("ABCDEFG", range(7))}
        out = cross_sectional_ic(sent, rets, min_names=5)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["ic"], 1.0)

    def test_inverted_agreement_scores_ic_minus_one(self):
        day = "2026-07-01"
        sent = {(day, s): {"score": v, "n": 1}
                for s, v in zip("ABCDEFG", range(7))}
        rets = {(day, s): -v / 100.0 for s, v in zip("ABCDEFG", range(7))}
        self.assertAlmostEqual(
            cross_sectional_ic(sent, rets, min_names=5)[0]["ic"], -1.0)

    def test_a_day_where_every_score_is_identical_has_no_ic(self):
        day = "2026-07-01"
        sent = {(day, s): {"score": 0.2, "n": 1} for s in "ABCDEFG"}
        rets = {(day, s): i / 100.0 for i, s in enumerate("ABCDEFG")}
        self.assertEqual(cross_sectional_ic(sent, rets, min_names=5), [])


class PowerTest(unittest.TestCase):
    def test_detecting_a_smaller_effect_needs_more_days(self):
        self.assertGreater(days_for_power(0.02), days_for_power(0.10))

    def test_a_realistic_small_factor_needs_hundreds_of_days(self):
        # Guards the headline claim: a few weeks of archive cannot settle this.
        self.assertGreater(days_for_power(0.03, daily_sd=0.15), 150)


class ValidateTest(_Base):
    def test_reports_days_not_just_observations(self):
        # The number that matters for significance is the day count.
        _news_db(self.news, [(s, f"2026-07-{d:02d}T12:00:00", 0.1 * i, 1.0)
                             for d in range(1, 4)
                             for i, s in enumerate("ABCDEFG")])
        # Prices must differ ACROSS symbols, or there is no cross-section to
        # rank and every day is correctly dropped.
        _px_db(self.px, [(s, f"2026-07-{d:02d}", 100.0 + d * (1 + i))
                         for d in range(1, 12)
                         for i, s in enumerate("ABCDEFG")])
        out = validate(self.news, self.px, horizon=1, min_names=5)
        self.assertIn("n_days", out)
        self.assertIn("n_observations", out)
        self.assertGreater(out["n_observations"], out["n_days"])

    def test_an_empty_overlap_is_reported_not_crashed(self):
        _news_db(self.news, [("AAA", "2026-07-01T12:00:00", 0.5, 1.0)])
        _px_db(self.px, [("ZZZ", "2026-07-01", 100.0)])
        out = validate(self.news, self.px, horizon=1)
        self.assertEqual(out["n_days"], 0)
        self.assertIsNone(out["mean_ic"])


if __name__ == "__main__":
    unittest.main()
