"""Tests for squeeze universe sourcing (href parsing + fallback, no network)."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bs4 import BeautifulSoup

from src.squeeze import universe as U


def _soup(rows):
    """Synthetic screener table mimicking finviz's two-anchor ticker cell."""
    trs = "".join(
        f'<tr><td>{i}</td><td><span>'
        f'<a class="company-ticker" href="stock?t={t}&ty=c">{t[0]}</a>'
        f'<a class="tab-link" href="stock?t={t}&ty=c&p=d&b=1">{t}</a>'
        f"</span></td></tr>"
        for i, t in enumerate(rows, 1)
    )
    return BeautifulSoup(
        f'<table class="screener_table"><tr><th>No.</th><th>Ticker</th></tr>{trs}</table>',
        "html.parser")


class TestExtractTickers(unittest.TestCase):
    def test_href_parse_immune_to_letter_icon_duplication(self):
        # cell .text would read "AABEO" — hrefs stay clean
        got = U._extract_tickers(_soup(["ABEO", "MARA", "BBAI"]))
        self.assertEqual(got, ["ABEO", "MARA", "BBAI"])

    def test_dedupes_repeated_anchors(self):
        soup = _soup(["MARA", "MARA"])
        self.assertEqual(U._extract_tickers(soup), ["MARA"])

    def test_missing_table_returns_empty(self):
        soup = BeautifulSoup("<html><body>rate limited</body></html>", "html.parser")
        self.assertEqual(U._extract_tickers(soup), [])


class TestSqueezeUniverse(unittest.TestCase):
    def test_fallback_on_fetch_failure(self):
        with mock.patch.dict(sys.modules, {"finvizfinance.util": None}):
            tickers = U.get_squeeze_universe(max_tickers=5)
        self.assertEqual(tickers, U.FALLBACK_TICKERS[:5])

    def test_fallback_on_empty_screen(self):
        with mock.patch.object(U, "finviz_tickers", return_value=[]):
            tickers = U.get_squeeze_universe(max_tickers=3)
        self.assertEqual(tickers, U.FALLBACK_TICKERS[:3])

    def test_filter_string_targets_high_short_float(self):
        self.assertIn("sh_short_o20", U.SQUEEZE_FILTERS_F)
        self.assertIn("sh_opt_option", U.SQUEEZE_FILTERS_F)

    def test_momentum_filter_adds_week_gain_to_the_base_screen(self):
        # docs/SQUEEZE_BACKTEST.md: top-5% SI *and* 5d return >= +10% is the
        # measured cohort (P(+20% in 42d) 50.5% vs 22.5% base).
        self.assertTrue(U.SQUEEZE_FILTERS_MOMENTUM_F.startswith(U.SQUEEZE_FILTERS_F))
        self.assertIn("ta_perf_1w10o", U.SQUEEZE_FILTERS_MOMENTUM_F)


def _screens(momentum, base):
    """Fake finviz_tickers dispatching on the filter string it is handed."""
    calls = []

    def fake(f_params, order="-averagevolume", limit=25):
        calls.append({"f": f_params, "order": order, "limit": limit})
        out = momentum if "ta_perf_1w10o" in f_params else base
        if isinstance(out, Exception):
            raise out
        return list(out)[:limit]

    return fake, calls


class TestSqueezeUniverseComposition(unittest.TestCase):
    def test_momentum_names_lead_and_base_screen_fills_the_rest(self):
        fake, _ = _screens(["AAA", "BBB"], ["CCC", "DDD", "EEE"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=4)
        self.assertEqual(uni.tickers, ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(uni.momentum, ["AAA", "BBB"])
        self.assertEqual(uni.source, "finviz")

    def test_overlap_between_the_screens_does_not_shrink_the_scan(self):
        # The momentum screen is a subset of the base filter set, so the base
        # screen returns those same names again. Asking it for only the
        # remaining slots leaves the scan short after dedup.
        fake, _ = _screens(["AAA", "BBB"], ["AAA", "BBB", "CCC", "DDD", "EEE"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=5)
        self.assertEqual(uni.tickers, ["AAA", "BBB", "CCC", "DDD", "EEE"])

    def test_a_name_on_both_screens_is_not_scanned_twice(self):
        fake, _ = _screens(["AAA"], ["AAA", "BBB"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=3)
        self.assertEqual(uni.tickers, ["AAA", "BBB"])

    def test_screens_are_ranked_by_short_interest_not_liquidity(self):
        # SI deciles are monotone in the study; average volume carries no
        # measured signal, so it must not decide which names survive the cut.
        fake, calls = _screens(["AAA"], ["BBB", "CCC"])
        with mock.patch.object(U, "finviz_tickers", fake):
            U.get_squeeze_universe_detailed(max_tickers=3)
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call["order"], U.SQUEEZE_ORDER)

    def test_base_screen_is_skipped_when_momentum_fills_the_quota(self):
        fake, calls = _screens(["AAA", "BBB", "CCC"], ["DDD"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=2)
        self.assertEqual(uni.tickers, ["AAA", "BBB"])
        self.assertEqual(len(calls), 1)
        self.assertIn("ta_perf_1w10o", calls[0]["f"])

    def test_quiet_week_with_no_momentum_names_still_scans_the_base_screen(self):
        fake, _ = _screens([], ["CCC", "DDD"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=3)
        self.assertEqual(uni.tickers, ["CCC", "DDD"])
        self.assertEqual(uni.momentum, [])
        self.assertEqual(uni.source, "finviz")

    def test_momentum_screen_failure_degrades_to_the_base_screen(self):
        # A momentum-screen error must not cost the whole universe — the base
        # screen is the same edge, minus one leg.
        fake, _ = _screens(RuntimeError("finviz 429"), ["CCC", "DDD"])
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=3)
        self.assertEqual(uni.tickers, ["CCC", "DDD"])
        self.assertEqual(uni.momentum, [])
        self.assertEqual(uni.source, "finviz")

    def test_both_screens_failing_degrades_to_the_hardcoded_list(self):
        fake, _ = _screens(RuntimeError("finviz 429"), RuntimeError("finviz 429"))
        with mock.patch.object(U, "finviz_tickers", fake):
            uni = U.get_squeeze_universe_detailed(max_tickers=3)
        self.assertEqual(uni.tickers, U.FALLBACK_TICKERS[:3])
        self.assertEqual(uni.momentum, [])
        self.assertEqual(uni.source, "fallback")

    def test_plain_helper_still_returns_a_bare_ticker_list(self):
        fake, _ = _screens(["AAA"], ["BBB"])
        with mock.patch.object(U, "finviz_tickers", fake):
            tickers = U.get_squeeze_universe(max_tickers=2)
        self.assertEqual(tickers, ["AAA", "BBB"])


class TestFinvizTickersPagination(unittest.TestCase):
    def test_stops_on_short_page_and_respects_limit(self):
        pages = {1: _soup([f"T{i}" for i in range(20)]),
                 21: _soup(["ZZA", "ZZB"])}
        calls = []

        def fake_scrap(url, params):
            calls.append(params["r"])
            return pages[params["r"]]

        fake_util = mock.MagicMock(web_scrap=fake_scrap)
        with mock.patch.dict(sys.modules, {"finvizfinance.util": fake_util}):
            got = U.finviz_tickers("f", limit=25)
        self.assertEqual(calls, [1, 21])
        self.assertEqual(len(got), 22)
        self.assertEqual(got[-1], "ZZB")


if __name__ == "__main__":
    unittest.main()
