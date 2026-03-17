"""Unit tests for src/polygon_client.py.

All external HTTP calls are mocked via unittest.mock.patch("requests.get").
Tests are compatible with  python -m pytest tests/test_polygon_client.py -v
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


class TestPolygonClientNoKey(unittest.TestCase):
    """Tests that run without a POLYGON_API_KEY in the environment."""

    def _make_client(self):
        """Return a PolygonClient instantiated with no key and no env var."""
        import os
        # Ensure env var is absent for these tests
        os.environ.pop("POLYGON_API_KEY", None)
        from src.polygon_client import PolygonClient
        return PolygonClient()

    def test_get_news_returns_none_when_no_key(self):
        """get_news returns None immediately when the client is disabled."""
        client = self._make_client()
        result = client.get_news("AAPL")
        self.assertIsNone(result)

    def test_get_ticker_details_returns_none_when_no_key(self):
        """get_ticker_details returns None immediately when the client is disabled."""
        client = self._make_client()
        result = client.get_ticker_details("AAPL")
        self.assertIsNone(result)


class TestPolygonClientWithKey(unittest.TestCase):
    """Tests that run with a fake API key."""

    def setUp(self):
        import os
        os.environ["POLYGON_API_KEY"] = "test-key-1234"
        # Re-import after env var is set
        import importlib, src.polygon_client
        importlib.reload(src.polygon_client)
        from src.polygon_client import PolygonClient
        self.PolygonClient = PolygonClient
        self.client = PolygonClient(api_key="test-key-1234")

    def _mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        mock = MagicMock()
        mock.status_code = status_code
        mock.ok = (status_code == 200)
        mock.json.return_value = json_data
        return mock

    def test_get_news_parses_valid_response(self):
        """get_news correctly parses a mocked valid Polygon news response."""
        now_utc = datetime.now(timezone.utc)
        pub_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = {
            "results": [
                {
                    "title": "AAPL beats earnings estimates",
                    "publisher": {"name": "Reuters"},
                    "published_utc": pub_str,
                    "article_url": "https://example.com/news/1",
                    "tickers": ["AAPL"],
                    "insights": [
                        {"ticker": "AAPL", "sentiment": "positive"}
                    ],
                }
            ]
        }

        with patch("requests.get", return_value=self._mock_response(payload)):
            items = self.client.get_news("AAPL", limit=5)

        self.assertIsNotNone(items)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.headline, "AAPL beats earnings estimates")
        self.assertEqual(item.publisher, "Reuters")
        self.assertAlmostEqual(item.sentiment, 0.5)
        self.assertIn("AAPL", item.tickers)

    def test_get_news_returns_none_on_http_429(self):
        """get_news returns None (not raises) on a rate-limit response."""
        with patch("requests.get", return_value=self._mock_response({}, status_code=429)):
            result = self.client.get_news("TSLA")
        self.assertIsNone(result)

    def test_get_news_returns_none_on_timeout(self):
        """get_news returns None (not raises) on a requests.Timeout."""
        import requests as req
        with patch("requests.get", side_effect=req.exceptions.Timeout):
            result = self.client.get_news("MSFT")
        self.assertIsNone(result)

    def test_get_snapshot_extracts_vwap(self):
        """get_snapshot returns the raw dict; caller can read ticker.day.vw for VWAP."""
        payload = {
            "ticker": {
                "ticker": "AAPL",
                "day": {"vw": 172.45, "o": 170.0, "c": 174.0, "h": 175.0, "l": 169.5, "v": 1_234_567},
            }
        }
        with patch("requests.get", return_value=self._mock_response(payload)):
            result = self.client.get_snapshot("AAPL")

        self.assertIsNotNone(result)
        vwap = result["ticker"]["day"]["vw"]
        self.assertAlmostEqual(vwap, 172.45)

    def test_get_unusual_options_flow_filters_correctly(self):
        """get_unusual_options_flow applies vol/OI > 2 and dollar-volume >= min_premium."""
        # Contract A: passes (vol/oi=3.0, dollar_vol=150_000)
        # Contract B: fails vol/oi check (0.5)
        # Contract C: fails dollar volume check (tiny premium)
        snapshot_payload = {
            "results": [
                {
                    "details": {"strike_price": 150.0, "contract_type": "call"},
                    "day": {"volume": 300},
                    "open_interest": 100,
                },
                {
                    "details": {"strike_price": 100.0, "contract_type": "put"},
                    "day": {"volume": 50},
                    "open_interest": 100,
                },
                {
                    "details": {"strike_price": 1.0, "contract_type": "call"},
                    "day": {"volume": 400},
                    "open_interest": 100,
                    # vol/oi=4.0 passes ratio, but dollar_vol=400*1*100=$40K < min_premium=50_000 → filtered out
                },
            ]
        }

        # get_unusual_options_flow internally calls get_options_snapshot which calls _get
        with patch.object(self.client, "get_options_snapshot", return_value=snapshot_payload["results"]):
            result = self.client.get_unusual_options_flow("AAPL", min_premium=50_000)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["details"]["strike_price"], 150.0)
        self.assertAlmostEqual(result[0]["_dollar_volume"], 300 * 150.0 * 100)


if __name__ == "__main__":
    unittest.main()
