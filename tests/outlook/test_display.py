"""Tests for the outlook narrative + cache (pure pieces, offline)."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG
from src.outlook.display import narrative, save_outlook_cache, load_outlook_cache


def _rows():
    return [
        {"ticker": "SMH", "direction": "BULLISH", "conviction": 95,
         "score": 1.2, "drivers": "12m momentum +, trend +"},
        {"ticker": "XLK", "direction": "BULLISH", "conviction": 80,
         "score": 0.8, "drivers": "trend +, rel-strength vs mkt +"},
        {"ticker": "XLV", "direction": "NEUTRAL", "conviction": 40,
         "score": -0.1, "drivers": "12m momentum −, trend −"},
        {"ticker": "TLT", "direction": "BEARISH", "conviction": 22,
         "score": -0.9, "drivers": "12m momentum −, trend −"},
    ]


class NarrativeTests(unittest.TestCase):
    def test_calls_out_top_leader_by_name(self):
        lines = narrative(_rows(), DEFAULT_OUTLOOK_CONFIG)
        text = " ".join(lines)
        self.assertIn("Semiconductors", text)  # SMH friendly name
        self.assertTrue(any(w in text.lower() for w in ("lead", "favor", "strength")))

    def test_calls_out_laggard(self):
        lines = narrative(_rows(), DEFAULT_OUTLOOK_CONFIG)
        text = " ".join(lines)
        self.assertIn("Treasuries", text)  # TLT
        self.assertTrue(any(w in text.lower() for w in ("lag", "weak", "underweight", "avoid")))

    def test_empty_rows_safe(self):
        self.assertEqual(narrative([], DEFAULT_OUTLOOK_CONFIG), [])


class CacheTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "outlook_cache.json")
            save_outlook_cache(_rows(), ["a leader line"], path=path)
            cached = load_outlook_cache(path=path)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["rows"][0]["ticker"], "SMH")
            self.assertIn("as_of", cached)
            self.assertEqual(cached["narrative"], ["a leader line"])

    def test_missing_cache_returns_none(self):
        self.assertIsNone(load_outlook_cache(path="/nonexistent/x.json"))


if __name__ == "__main__":
    unittest.main()
