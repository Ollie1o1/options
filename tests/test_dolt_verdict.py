"""Tests for src/dolt_verdict.py — per-segment verdict lookup (display-only).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_verdict -v
"""
import json
import os
import tempfile
import unittest

from src import dolt_verdict as dv


class SegmentLookupTest(unittest.TestCase):
    def test_known_symbols_map_to_segments(self):
        self.assertEqual(dv._segment_of("SPY"), "index")
        self.assertEqual(dv._segment_of("nvda"), "semi")
        self.assertEqual(dv._segment_of("AAPL"), "tech")

    def test_unknown_symbol_is_none(self):
        self.assertIsNone(dv._segment_of("TSLA"))

    def test_verdict_for_unknown_is_none(self):
        self.assertIsNone(dv.verdict_for("TSLA", cache_path="/nonexistent.json"))

    def test_verdict_falls_back_to_defaults(self):
        v = dv.verdict_for("SPY", cache_path="/nonexistent.json")
        self.assertEqual(v["segment"], "index")
        self.assertEqual(v["best"], "put_spread")
        self.assertEqual(v["source"], "default")

    def test_verdict_prefers_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.json")
            with open(path, "w") as fh:
                json.dump({"index": {"best": "long_call", "label": "LONG — buy calls"}}, fh)
            v = dv.verdict_for("SPY", cache_path=path)
        self.assertEqual(v["best"], "long_call")
        self.assertEqual(v["source"], "cache")

    def test_verdict_line_for_unknown_is_none(self):
        self.assertIsNone(dv.verdict_line("TSLA", cache_path="/nonexistent.json"))

    def test_verdict_line_contains_label(self):
        line = dv.verdict_line("AAPL", cache_path="/nonexistent.json")
        self.assertIn("STAND DOWN", line)
        self.assertIn("tech", line)


if __name__ == "__main__":
    unittest.main()
