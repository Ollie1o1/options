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
