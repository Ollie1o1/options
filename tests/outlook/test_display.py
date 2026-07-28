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


class NarrativeHonestyTests(unittest.TestCase):
    """display.py hardcoded 'these have been doing well' — a present-tense claim
    about a window that ended a month ago. When the leader is flagged, the
    narrative must say so rather than assert continuation unqualified."""

    def _rows(self, lagging):
        return [
            {"ticker": "SMH", "score": 1.2, "direction": "BULLISH", "conviction": 99,
             "drivers": "12m momentum +, trend +", "lagging": lagging,
             "ret_21d": -0.130, "excess_21d": -14.8},
            {"ticker": "XLK", "score": 0.8, "direction": "BULLISH", "conviction": 82,
             "drivers": "trend +, 12m momentum +", "lagging": False,
             "ret_21d": 0.012, "excess_21d": 1.1},
            {"ticker": "GLD", "score": -0.9, "direction": "NEUTRAL", "conviction": 12,
             "drivers": "trend −", "lagging": False,
             "ret_21d": -0.02, "excess_21d": -3.0},
        ]

    def test_flagged_leader_gets_an_explicit_caveat(self):
        text = " ".join(narrative(self._rows(True), DEFAULT_OUTLOOK_CONFIG))
        text = text.replace("−", "-")
        self.assertIn("-13.0%", text)
        self.assertIn("excludes the last month", text)

    def test_flagged_leader_drops_the_false_present_tense_claim(self):
        # Appending the caveat is not enough: leaving "have been doing well"
        # in place makes the narrative contradict itself in consecutive
        # sentences. The claim itself must go.
        text = " ".join(narrative(self._rows(True), DEFAULT_OUTLOOK_CONFIG))
        self.assertNotIn("doing well", text)
        self.assertNotIn("has outperformed", text)

    def test_single_flagged_leader_also_drops_the_claim(self):
        rows = [
            {"ticker": "SMH", "score": 1.2, "direction": "BULLISH", "conviction": 99,
             "drivers": "12m momentum +, trend +", "lagging": True,
             "ret_21d": -0.130, "excess_21d": -14.8},
            {"ticker": "GLD", "score": -0.9, "direction": "NEUTRAL", "conviction": 12,
             "drivers": "trend −", "lagging": False},
        ]
        text = " ".join(narrative(rows, DEFAULT_OUTLOOK_CONFIG))
        self.assertNotIn("has outperformed", text)
        self.assertIn("excludes the last month", text)

    def test_unflagged_leader_keeps_the_plain_reading(self):
        text = " ".join(narrative(self._rows(False), DEFAULT_OUTLOOK_CONFIG))
        self.assertIn("doing well", text)
        self.assertNotIn("excludes the last month", text)

    def test_rows_without_recent_fields_still_produce_narrative(self):
        rows = [{"ticker": "SMH", "score": 1.2, "direction": "BULLISH",
                 "conviction": 99, "drivers": "12m momentum +, trend +"},
                {"ticker": "XLK", "score": 0.8, "direction": "BULLISH",
                 "conviction": 82, "drivers": "trend +"}]
        text = " ".join(narrative(rows, DEFAULT_OUTLOOK_CONFIG))
        self.assertIn("Leading", text)


class LaggingNoteTests(unittest.TestCase):
    """A flagged row gets a second line stating what just happened. Rows from a
    pre-upgrade cache have no such fields and must render exactly as before."""

    def _cache(self, extra):
        row = {"ticker": "SMH", "score": 1.2, "direction": "BULLISH",
               "conviction": 99, "drivers": "12m momentum +, trend +"}
        row.update(extra)
        return {"as_of": "2026-07-28 12:00 UTC", "rows": [row], "narrative": []}

    def _render(self, cache):
        import contextlib
        import io
        import json
        from src.outlook.display import print_outlook_box
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cache.json")
            with open(p, "w") as fh:
                json.dump(cache, fh)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_outlook_box(cache_path=p, refresh_if_stale_hours=1e9)
            return buf.getvalue()

    def test_flagged_row_reports_the_recent_move(self):
        out = self._render(self._cache(
            {"lagging": True, "ret_21d": -0.130, "excess_21d": -14.8}))
        self.assertIn("-13.0%", out.replace("−", "-"))
        self.assertIn("14.8pp", out.replace("−", "-"))

    def test_unflagged_row_gets_no_note(self):
        out = self._render(self._cache(
            {"lagging": False, "ret_21d": 0.012, "excess_21d": 1.1}))
        self.assertNotIn("pp vs", out)

    def test_pre_upgrade_cache_renders_without_error(self):
        out = self._render(self._cache({}))   # no recent fields at all
        self.assertIn("SMH", out)
        self.assertNotIn("pp vs", out)


if __name__ == "__main__":
    unittest.main()
