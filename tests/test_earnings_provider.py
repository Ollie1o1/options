"""Tests for src.earnings_provider — Finnhub free-tier earnings & dividends.

The provider activates only when a free API key is present; otherwise every
entry point returns None so callers fall back to the existing yfinance path.
HTTP is injected as a function returning parsed JSON, so tests never touch the
network.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_earnings_provider -v
"""
from __future__ import annotations

import datetime as dt
import unittest

from src import earnings_provider as ep


def _cal_json():
    return {"earningsCalendar": [
        {"date": "2026-01-28", "symbol": "AAPL", "hour": "amc", "year": 2026, "quarter": 1},
        {"date": "2026-07-30", "symbol": "AAPL", "hour": "amc", "year": 2026, "quarter": 3},
        {"date": "2026-04-30", "symbol": "AAPL", "hour": "amc", "year": 2026, "quarter": 2},
    ]}


class NextEarningsTest(unittest.TestCase):
    def test_returns_earliest_future_date(self):
        today = dt.date(2026, 6, 25)
        got = ep.next_earnings_date("AAPL", api_key="k",
                                    fetcher=lambda url, params: _cal_json(),
                                    today=today)
        self.assertEqual(got.date(), dt.date(2026, 7, 30))

    def test_ignores_past_dates(self):
        today = dt.date(2026, 8, 1)  # all sample dates are in the past
        got = ep.next_earnings_date("AAPL", api_key="k",
                                    fetcher=lambda url, params: _cal_json(),
                                    today=today)
        self.assertIsNone(got)

    def test_none_without_api_key(self):
        self.assertIsNone(ep.next_earnings_date("AAPL", api_key=None,
                                                fetcher=lambda u, p: _cal_json()))

    def test_never_raises_on_bad_json(self):
        self.assertIsNone(ep.next_earnings_date(
            "AAPL", api_key="k", fetcher=lambda u, p: {"unexpected": 1}))
        self.assertIsNone(ep.next_earnings_date(
            "AAPL", api_key="k", fetcher=lambda u, p: (_ for _ in ()).throw(RuntimeError())))


class DividendYieldTest(unittest.TestCase):
    def test_extracts_yield_as_fraction(self):
        # Finnhub reports the indicated annual yield in percent (e.g. 0.45 == 0.45%).
        js = {"metric": {"dividendYieldIndicatedAnnual": 0.45}}
        got = ep.dividend_yield("AAPL", api_key="k", fetcher=lambda u, p: js)
        self.assertAlmostEqual(got, 0.0045, places=6)

    def test_none_without_api_key(self):
        self.assertIsNone(ep.dividend_yield("AAPL", api_key=None,
                                            fetcher=lambda u, p: {}))

    def test_zero_yield_for_non_payer(self):
        js = {"metric": {"dividendYieldIndicatedAnnual": None}}
        got = ep.dividend_yield("AAPL", api_key="k", fetcher=lambda u, p: js)
        self.assertIsNone(got)


class KeyResolutionTest(unittest.TestCase):
    def test_reads_key_from_env(self):
        import os
        old = os.environ.get("FINNHUB_API_KEY")
        os.environ["FINNHUB_API_KEY"] = "envkey123"
        try:
            self.assertEqual(ep.resolve_api_key(), "envkey123")
        finally:
            if old is None:
                del os.environ["FINNHUB_API_KEY"]
            else:
                os.environ["FINNHUB_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
