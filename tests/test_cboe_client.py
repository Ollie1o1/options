"""Tests for src/cboe_client.py — free CBOE delayed-quote chain client.

All offline: parsing operates on a fixture payload shaped exactly like
https://cdn.cboe.com/api/global/delayed_quotes/options/<SYMBOL>.json
(verified live 2026-06-10).
"""
from __future__ import annotations

import unittest

from src import cboe_client


FIXTURE = {
    "timestamp": "2026-06-10 13:45:40",
    "symbol": "AAPL",
    "data": {
        "symbol": "AAPL",
        "current_price": 290.89,
        "options": [
            {
                "option": "AAPL261016C00335000",
                "bid": 5.5, "bid_size": 5.0, "ask": 5.8, "ask_size": 7.0,
                "iv": 0.2597, "open_interest": 1131.0, "volume": 14.0,
                "delta": 0.2326, "gamma": 0.0068, "vega": 0.5315,
                "theta": -0.0517, "rho": 0.2201, "theo": 5.5429,
                "last_trade_price": 5.95, "last_trade_time": "2026-06-10T15:03:18",
            },
            {
                "option": "AAPL260619P00250000",
                "bid": 1.2, "bid_size": 10.0, "ask": 1.3, "ask_size": 12.0,
                "iv": 0.31, "open_interest": 500.0, "volume": 3.0,
                "delta": -0.12, "gamma": 0.004, "vega": 0.2,
                "theta": -0.03, "rho": -0.05, "theo": 1.25,
                "last_trade_price": 1.22, "last_trade_time": None,
            },
            {   # malformed OCC symbol -> skipped, never raises
                "option": "GARBAGE",
                "bid": 0, "ask": 0, "iv": None,
            },
        ],
    },
}


class ParseOccSymbolTest(unittest.TestCase):
    def test_call(self):
        r = cboe_client.parse_occ_symbol("AAPL261016C00335000")
        self.assertEqual(r, ("AAPL", "2026-10-16", "call", 335.0))

    def test_put_fractional_strike(self):
        r = cboe_client.parse_occ_symbol("BRKB260619P00007500")
        self.assertEqual(r, ("BRKB", "2026-06-19", "put", 7.5))

    def test_garbage_returns_none(self):
        self.assertIsNone(cboe_client.parse_occ_symbol("GARBAGE"))
        self.assertIsNone(cboe_client.parse_occ_symbol(""))


class ParseChainTest(unittest.TestCase):
    def test_normalizes_contracts(self):
        rows = cboe_client.parse_chain(FIXTURE)
        self.assertEqual(len(rows), 2)  # garbage row dropped
        call = rows[0]
        self.assertEqual(call["type"], "call")
        self.assertEqual(call["strike"], 335.0)
        self.assertEqual(call["expiration"], "2026-10-16")
        self.assertEqual(call["symbol"], "AAPL")
        self.assertAlmostEqual(call["iv"], 0.2597)
        self.assertAlmostEqual(call["bid"], 5.5)
        self.assertAlmostEqual(call["spot"], 290.89)
        self.assertEqual(call["source"], "cboe")
        put = rows[1]
        self.assertEqual(put["type"], "put")
        self.assertEqual(put["strike"], 250.0)
        self.assertIsNone(put["last_trade_time"])

    def test_empty_payload(self):
        self.assertEqual(cboe_client.parse_chain({}), [])
        self.assertEqual(cboe_client.parse_chain({"data": {}}), [])


class FetchChainCacheTest(unittest.TestCase):
    def test_fetch_uses_injected_getter_and_caches(self):
        calls = []
        def fake_get(url, timeout):
            calls.append(url)
            return FIXTURE
        cboe_client.clear_cache()
        rows1 = cboe_client.fetch_chain("AAPL", getter=fake_get)
        rows2 = cboe_client.fetch_chain("AAPL", getter=fake_get)
        self.assertEqual(len(calls), 1)        # second call served from cache
        self.assertEqual(rows1, rows2)
        self.assertIn("AAPL.json", calls[0])
        cboe_client.clear_cache()

    def test_fetch_failure_returns_empty(self):
        def boom(url, timeout):
            raise OSError("network down")
        cboe_client.clear_cache()
        self.assertEqual(cboe_client.fetch_chain("AAPL", getter=boom), [])
        cboe_client.clear_cache()


if __name__ == "__main__":
    unittest.main()
