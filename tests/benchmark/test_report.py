"""Tests for the run manifest (corpus fingerprint) and markdown rendering.

Every ledger here is a throwaway file built in `setUp`; no test names or
opens the real `paper_trades.db`.
"""
import os
import sqlite3
import tempfile
import unittest

from src.benchmark.excess import (
    AggregateResult, CumulativeComparison, DailyAggregateResult, LoadReport,
    MatchStats, NULL_CASH, NULL_SPY,
)
from src.benchmark.prices import StitchValidation
from src.benchmark.report import (
    corpus_fingerprint, manifests_comparable, render_markdown, run_manifest,
)

_TRADES_SCHEMA = """
CREATE TABLE trades (
    entry_id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, status TEXT,
    exit_date TEXT, pnl_usd REAL, capital_at_risk REAL
);
"""


def _make_ledger(rows):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(_TRADES_SCHEMA)
    con.executemany(
        "INSERT INTO trades (entry_id, date, ticker, status, exit_date, "
        "pnl_usd, capital_at_risk) VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


class TestCorpusFingerprint(unittest.TestCase):
    def setUp(self):
        self.path = _make_ledger([
            (1, "2026-01-01", "AAA", "CLOSED", "2026-01-05", 50.0, 500.0),
            (2, "2026-01-02", "BBB", "CLOSED", "2026-01-09", -10.0, 300.0),
            (3, "2026-01-03", "CCC", "OPEN", None, None, 300.0),
        ])

    def tearDown(self):
        os.remove(self.path)

    def test_counts_closed_rows_and_carries_loaded(self):
        fp = corpus_fingerprint(self.path, loaded=1)
        self.assertEqual(fp["terminal_count"], 2)
        self.assertEqual(fp["max_exit_date"], "2026-01-09")
        self.assertEqual(fp["loaded"], 1)


class TestManifestsComparable(unittest.TestCase):
    def _manifest(self, sha="abc", max_exit="2026-01-09", terminal=2):
        return {
            "git_sha": sha,
            "corpus_fingerprint": {
                "max_exit_date": max_exit, "terminal_count": terminal,
                "loaded": terminal,
            },
        }

    def test_same_everything_comparable(self):
        a = self._manifest()
        b = self._manifest()
        ok, _ = manifests_comparable(a, b)
        self.assertTrue(ok)

    def test_different_sha_not_comparable(self):
        a = self._manifest(sha="abc")
        b = self._manifest(sha="def")
        ok, reason = manifests_comparable(a, b)
        self.assertFalse(ok)
        self.assertIn("git sha", reason)

    def test_corpus_drift_not_comparable(self):
        a = self._manifest(terminal=2)
        b = self._manifest(terminal=5)
        ok, reason = manifests_comparable(a, b)
        self.assertFalse(ok)
        self.assertIn("terminal_count", reason)

    def test_terminal_count_zero_is_not_confused_with_missing(self):
        # A legitimate zero-count corpus must compare equal to itself, not
        # be treated as "missing" and silently pass/fail.
        a = self._manifest(terminal=0)
        b = self._manifest(terminal=0)
        ok, _ = manifests_comparable(a, b)
        self.assertTrue(ok)


class TestRunManifest(unittest.TestCase):
    def setUp(self):
        self.path = _make_ledger([
            (1, "2026-01-01", "AAA", "CLOSED", "2026-01-05", 50.0, 500.0),
        ])
        self.load_report = LoadReport(loaded=1, dropped={"exit_date_missing": 3})
        self.match = MatchStats(exact=1, tolerant=0, missing=0, tolerance_days=4)
        self.stitch = StitchValidation(
            n_common=28, median_diff=0.0, mean_diff=0.0005, max_abs_diff=0.012,
            median_abs_diff=0.0008, threshold=0.005, ok=True)

    def tearDown(self):
        os.remove(self.path)

    def test_manifest_has_all_required_keys(self):
        m = run_manifest(self.path, self.load_report, self.match, self.stitch,
                          seed=0)
        self.assertIn("git_sha", m)
        self.assertIn("generated_at", m)
        self.assertEqual(m["cluster_unit"], "(symbol, entry_date)")
        self.assertEqual(m["corpus_fingerprint"]["loaded"], 1)
        self.assertEqual(m["row_counts"]["dropped"]["exit_date_missing"], 3)
        self.assertEqual(m["match_stats"]["exact"], 1)
        self.assertTrue(m["stitch_validation"]["ok"])


class TestRenderMarkdown(unittest.TestCase):
    def setUp(self):
        self.path = _make_ledger([
            (1, "2026-01-01", "AAA", "CLOSED", "2026-01-05", 50.0, 500.0),
        ])
        load_report = LoadReport(loaded=1, dropped={})
        match = MatchStats(exact=1, tolerant=0, missing=0, tolerance_days=4)
        stitch = StitchValidation(28, 0.0, 0.0005, 0.012, 0.0008, 0.005, True)
        self.manifest = run_manifest(self.path, load_report, match, stitch, 0)
        self.cumulative = CumulativeComparison(
            n_trades=1, book_pnl_usd=50.0, total_capital_at_risk_usd=500.0,
            spy_equiv_primary_usd=25.0, mean_concurrent_usd=100.0,
            peak_concurrent_usd=500.0, spy_span_return=0.188,
            spy_equiv_secondary_usd=18.8)
        self.per_trade = {
            NULL_CASH: AggregateResult(
                n_rows=1, n_clusters=1, mean=0.10, sd=0.0, mean_book=0.10,
                mean_null=0.0, ci_lo=-0.02, ci_hi=0.22, ci_contains_zero=True,
                variance_ratio=1.0, corr_book_null=None,
                n_needed_80pct_power=107363.0),
            NULL_SPY: AggregateResult(
                n_rows=1, n_clusters=1, mean=-0.05, sd=0.0, mean_book=0.10,
                mean_null=0.15, ci_lo=-0.30, ci_hi=0.20, ci_contains_zero=True,
                variance_ratio=0.998, corr_book_null=0.07,
                n_needed_80pct_power=107363.0),
        }
        self.daily = DailyAggregateResult(
            n_days=92, mean_book=0.0178, sd_book=0.29, mean_spy=0.00098,
            sd_spy=0.0084, mean_excess=0.0168, sd_excess=0.29, t_stat=0.56,
            ci_lo=-0.04, ci_hi=0.08, ci_contains_zero=True, variance_ratio=0.996,
            n_needed_80pct_power=2319.0)

    def tearDown(self):
        os.remove(self.path)

    def test_renders_without_error_and_contains_key_sections(self):
        md = render_markdown(self.manifest, self.cumulative, self.per_trade,
                              self.daily)
        self.assertIn("# Benchmark Result", md)
        self.assertIn("Headline: cumulative dollar comparison", md)
        self.assertIn("Per-trade excess", md)
        self.assertIn("Daily portfolio series", md)
        self.assertIn("cannot establish statistically", md)

    def test_no_verdict_language(self):
        # "pass/fail" is not banned outright: the limitations section
        # legitimately says this is NOT a hypothesis test "with a pass/fail
        # bar" -- that negation is the point. What must never appear is an
        # actual promote/reject/verdict call on the numbers.
        md = render_markdown(self.manifest, self.cumulative, self.per_trade,
                              self.daily)
        lowered = md.lower()
        for banned in ("promote", "reject", "verdict"):
            self.assertNotIn(banned, lowered)

    def test_ci_contains_zero_rendered(self):
        md = render_markdown(self.manifest, self.cumulative, self.per_trade,
                              self.daily)
        self.assertIn("YES", md)  # ci_contains_zero=True surfaces visibly

    def test_daily_none_renders_not_computed(self):
        md = render_markdown(self.manifest, self.cumulative, self.per_trade,
                              None)
        self.assertIn("Not computed", md)

    def test_headline_has_no_ci_language_around_dollar_section(self):
        md = render_markdown(self.manifest, self.cumulative, self.per_trade,
                              self.daily)
        headline = md.split("Headline: cumulative dollar comparison")[1]
        headline = headline.split("## Per-trade")[0]
        self.assertIn("no confidence interval attaches", headline.lower())


if __name__ == "__main__":
    unittest.main()
