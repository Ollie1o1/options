"""Tests for the DISCOVER candidate detail drill-down's on-demand fetch
layer (src/longterm/detail.py). Every source is independently mocked —
never touches the live network, matching the convention set by
tests/longterm/test_discover.py's TestDeepContext."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import detail as DTL
from src.longterm.discover import DeepRead
from src.news_fetcher import NewsData
from src.short_interest import ShortInterest


class TestFetchDetail(unittest.TestCase):
    def test_reuses_passed_in_deep_read_without_refetching(self):
        existing_deep = DeepRead(ticker="MU", insider={"label": "CLUSTER BUY"})
        fake_ticker = mock.MagicMock()
        fake_ticker.info = {"shortPercentOfFloat": 0.05, "shortRatio": 2.1}
        with mock.patch("src.longterm.discover.deep_context") as m_deep, \
             mock.patch("yfinance.Ticker", return_value=fake_ticker), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        return_value=NewsData(symbol="MU")):
            result = DTL.fetch_detail("MU", deep=existing_deep)
        m_deep.assert_not_called()
        self.assertIs(result.deep, existing_deep)

    def test_computes_deep_read_when_none_passed(self):
        fresh_deep = DeepRead(ticker="MU", earnings_days=5)
        fake_ticker = mock.MagicMock()
        fake_ticker.info = {}
        with mock.patch("src.longterm.discover.deep_context",
                        return_value=fresh_deep) as m_deep, \
             mock.patch("yfinance.Ticker", return_value=fake_ticker), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        return_value=NewsData(symbol="MU")):
            result = DTL.fetch_detail("MU")
        m_deep.assert_called_once_with("MU")
        self.assertIs(result.deep, fresh_deep)

    def test_short_interest_populated_from_info(self):
        fake_ticker = mock.MagicMock()
        fake_ticker.info = {"shortPercentOfFloat": 0.08, "shortRatio": 3.4}
        with mock.patch("src.longterm.discover.deep_context", return_value=DeepRead(ticker="MU")), \
             mock.patch("yfinance.Ticker", return_value=fake_ticker), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        return_value=NewsData(symbol="MU")):
            result = DTL.fetch_detail("MU")
        self.assertIsInstance(result.short_interest, ShortInterest)
        self.assertAlmostEqual(result.short_interest.pct_float, 0.08)

    def test_news_populated_and_passed_ticker_object(self):
        fake_ticker = mock.MagicMock()
        fake_ticker.info = {}
        fake_news = NewsData(symbol="MU", top_headlines=["MU beats estimates"])
        with mock.patch("src.longterm.discover.deep_context", return_value=DeepRead(ticker="MU")), \
             mock.patch("yfinance.Ticker", return_value=fake_ticker), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        return_value=fake_news) as m_news:
            result = DTL.fetch_detail("MU")
        self.assertIs(result.news, fake_news)
        m_news.assert_called_once()
        self.assertEqual(m_news.call_args.kwargs.get("ticker_obj"), fake_ticker)

    def test_short_interest_degrades_to_none_on_info_failure(self):
        with mock.patch("src.longterm.discover.deep_context", return_value=DeepRead(ticker="MU")), \
             mock.patch("yfinance.Ticker", side_effect=RuntimeError("timeout")), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        return_value=NewsData(symbol="MU")):
            result = DTL.fetch_detail("MU")
        self.assertIsNone(result.short_interest)

    def test_news_degrades_to_none_on_fetch_failure_without_losing_short_interest(self):
        fake_ticker = mock.MagicMock()
        fake_ticker.info = {"shortPercentOfFloat": 0.05, "shortRatio": 2.0}
        with mock.patch("src.longterm.discover.deep_context", return_value=DeepRead(ticker="MU")), \
             mock.patch("yfinance.Ticker", return_value=fake_ticker), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        side_effect=RuntimeError("boom")):
            result = DTL.fetch_detail("MU")
        self.assertIsNone(result.news)
        self.assertIsNotNone(result.short_interest)
        self.assertAlmostEqual(result.short_interest.pct_float, 0.05)

    def test_ticker_field_set(self):
        with mock.patch("src.longterm.discover.deep_context", return_value=DeepRead(ticker="MU")), \
             mock.patch("yfinance.Ticker", side_effect=RuntimeError("boom")), \
             mock.patch("src.news_fetcher.fetch_news_and_events",
                        side_effect=RuntimeError("boom")):
            result = DTL.fetch_detail("MU")
        self.assertEqual(result.ticker, "MU")


if __name__ == "__main__":
    unittest.main()
