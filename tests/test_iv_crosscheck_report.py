"""One summary line per scan, not one per ticker.

`enrich_and_score` logged "IV corrected on N/M contracts" at INFO once per
ticker. On a 111-ticker DISCOVER scan that is 111 lines on stderr cutting
across a tqdm bar that renders on stdout — two streams, one terminal, so the
bar and the messages shred each other.

Silencing it to DEBUG would have been the easy fix and the wrong one: the rate
varies enormously by name (observed 4/113 on one ticker against 9/17 on
another), and "which names does Yahoo price badly" is exactly the kind of
data-quality signal this repo has learned not to throw away. So the counts are
accumulated and reported once, which both cleans the screen AND makes the rate
comparable across names for the first time.
"""
from __future__ import annotations

import unittest

from src import iv_crosscheck as ivc


class TestTheAccumulator(unittest.TestCase):

    def setUp(self):
        ivc.reset()

    def tearDown(self):
        ivc.reset()

    def test_nothing_recorded_says_nothing(self):
        """A scan with no corrections must print no summary at all."""
        self.assertIsNone(ivc.summary())

    def test_zero_corrections_still_says_nothing(self):
        """Recording clean tickers is not a reason to occupy a line."""
        ivc.record("AAPL", 0, 120)
        ivc.record("MSFT", 0, 90)
        self.assertIsNone(ivc.summary())

    def test_totals_are_summed_across_tickers(self):
        ivc.record("AAPL", 4, 100)
        ivc.record("MSFT", 6, 100)
        s = ivc.summary()
        self.assertIsNotNone(s)
        self.assertEqual(s.corrected, 10)
        self.assertEqual(s.total, 200)
        self.assertEqual(s.tickers, 2)
        self.assertAlmostEqual(s.pct, 5.0)

    def test_worst_offenders_are_ranked_by_rate_not_by_count(self):
        """A ticker correcting 9 of 17 is worse than one correcting 54 of 212,
        even though the second has the bigger raw count. Ranking by count
        would bury the genuinely broken names behind the merely large ones."""
        ivc.record("BIG", 54, 212)    # 25.5%
        ivc.record("SMALL", 9, 17)    # 52.9%
        ivc.record("MILD", 1, 100)    # 1.0%
        s = ivc.summary()
        self.assertEqual([w.symbol for w in s.worst][:2], ["SMALL", "BIG"])

    def test_worst_list_is_capped(self):
        for i in range(20):
            ivc.record(f"T{i:02d}", i + 1, 100)
        s = ivc.summary()
        self.assertLessEqual(len(s.worst), ivc.MAX_WORST)

    def test_tickers_with_no_corrections_are_excluded_from_worst(self):
        ivc.record("CLEAN", 0, 500)
        ivc.record("DIRTY", 3, 10)
        s = ivc.summary()
        self.assertEqual([w.symbol for w in s.worst], ["DIRTY"])
        self.assertEqual(s.tickers, 1, "tickers counts those with corrections")

    def test_reset_clears_state_between_scans(self):
        """Two scans in one process must not double-count — the interactive
        loop runs many scans without restarting."""
        ivc.record("AAPL", 5, 50)
        ivc.reset()
        self.assertIsNone(ivc.summary())

    def test_a_zero_contract_ticker_cannot_divide_by_zero(self):
        ivc.record("EMPTY", 0, 0)
        self.assertIsNone(ivc.summary())

    def test_repeated_symbol_accumulates_rather_than_replaces(self):
        """One ticker can be enriched more than once in a scan (multiple
        expirations / structure passes); the counts must add."""
        ivc.record("AAPL", 2, 50)
        ivc.record("AAPL", 3, 50)
        s = ivc.summary()
        self.assertEqual(s.corrected, 5)
        self.assertEqual(s.total, 100)
        self.assertEqual(s.tickers, 1)


class TestTheRenderedLines(unittest.TestCase):

    def setUp(self):
        ivc.reset()

    def tearDown(self):
        ivc.reset()

    def test_render_states_totals_and_percentage(self):
        ivc.record("AAPL", 4, 100)
        ivc.record("MSFT", 6, 100)
        text = "\n".join(ivc.render(ivc.summary()))
        self.assertIn("10", text)
        self.assertIn("200", text)
        self.assertIn("5.0%", text)

    def test_render_names_the_worst_offenders_with_their_rates(self):
        ivc.record("SMALL", 9, 17)
        text = "\n".join(ivc.render(ivc.summary()))
        self.assertIn("SMALL", text)
        self.assertIn("9/17", text)
        self.assertIn("53%", text)

    def test_render_mentions_where_the_detail_went(self):
        """The per-contract lines still exist at DEBUG. If the summary does not
        say so, the detail is effectively deleted."""
        ivc.record("AAPL", 4, 100)
        text = "\n".join(ivc.render(ivc.summary())).lower()
        self.assertIn("debug", text)

    def test_render_of_none_is_empty(self):
        self.assertEqual(ivc.render(None), [])


class TestBothScanEntryPointsAreWired(unittest.TestCase):
    """The bug this class exists for.

    The summary was first wired into `run_scan` only. `--top` goes through
    `run_top_scan`, which owns a separate scoring loop, so a live `--top` run
    printed nothing at all — the per-ticker lines were gone and no summary
    replaced them. The unit tests were green throughout: they proved the tally
    worked and said nothing about whether anyone called it.

    Asserted against the parsed AST rather than a text grep, so a call that
    merely appears in a comment or a neighbouring function cannot satisfy it.
    """

    ENTRY_POINTS = ("run_scan", "run_top_scan")

    def _calls_within(self, func_name):
        import ast
        import inspect
        from src import options_screener
        tree = ast.parse(inspect.getsource(options_screener))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return {
                    c.func.id
                    for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                }
        self.fail(f"{func_name} not found in options_screener")

    def test_every_entry_point_reports_the_tally(self):
        for fn in self.ENTRY_POINTS:
            with self.subTest(entry_point=fn):
                self.assertIn("_report_iv_crosscheck", self._calls_within(fn))

    def test_every_entry_point_resets_the_tally(self):
        """Without a reset a second scan in the same process reports the
        first one's contracts too."""
        for fn in self.ENTRY_POINTS:
            with self.subTest(entry_point=fn):
                self.assertIn("_reset_iv_crosscheck", self._calls_within(fn))


class TestTheReporterActuallyPrints(unittest.TestCase):
    """Render it and assert on the output — a passing tally test proved
    nothing about what reached the screen."""

    def setUp(self):
        ivc.reset()

    def tearDown(self):
        ivc.reset()

    def _capture(self, verbose=True):
        import contextlib
        import io
        from src.options_screener import _report_iv_crosscheck
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _report_iv_crosscheck(verbose)
        return buf.getvalue()

    def test_it_prints_the_summary_when_there_is_one(self):
        ivc.record("BA", 9, 17)
        out = self._capture()
        self.assertIn("IV cross-check", out)
        self.assertIn("BA", out)

    def test_it_prints_nothing_on_a_clean_scan(self):
        ivc.record("AAPL", 0, 300)
        self.assertEqual(self._capture().strip(), "")

    def test_verbose_false_prints_nothing(self):
        ivc.record("BA", 9, 17)
        self.assertEqual(self._capture(verbose=False).strip(), "")


if __name__ == "__main__":
    unittest.main()
