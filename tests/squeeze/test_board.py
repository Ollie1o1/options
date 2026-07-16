"""Tests for squeeze display boards (render smoke + content)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src import formatting as fmt
from src.squeeze import board as B
from src.squeeze import detector as D


def _setup_nbis():
    return D.assess_squeeze({
        "short_interest": 0.2797,
        "short_interest_dtc": 3.5,
        "short_interest_trend": "rising",
        "iv_skew": -0.089,
        "ret_5d": -18.2,
    })


class TestBanner(unittest.TestCase):
    def setUp(self):
        # fmt.supports_color memoizes — pin the flag, never env vars
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_setup_banner_contains_evidence_and_caveat(self):
        text = B.banner(_setup_nbis(), "NBIS")
        self.assertIn("SHORT-SQUEEZE SETUP", text)
        self.assertIn("NBIS", text)
        self.assertIn("28.0%", text)
        self.assertIn("rising", text)
        self.assertIn("FINRA", text)  # staleness caveat present

    def test_none_grade_renders_nothing(self):
        self.assertIsNone(B.banner(D.assess_squeeze({}), "MU"))

    def test_watch_banner_says_watch(self):
        watch = D.assess_squeeze({"short_interest": 0.16,
                                  "short_interest_trend": "rising"})
        text = B.banner(watch, "XYZ")
        self.assertIn("SQUEEZE WATCH", text)


class TestCallBoard(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def _df(self):
        return pd.DataFrame([
            {"type": "put", "strike": 170.0, "expiration": "2026-08-14",
             "dte": 29, "delta": -0.38, "premium": 23.7, "spread_pct": 8.4,
             "ev_per_contract": -544.0, "quality_score": 0.40},
            {"type": "call", "strike": 185.0, "expiration": "2026-08-14",
             "dte": 29, "delta": 0.42, "premium": 14.2, "spread_pct": 6.0,
             "ev_per_contract": -120.0, "quality_score": 0.35},
            {"type": "call", "strike": 195.0, "expiration": "2026-08-14",
             "dte": 29, "delta": 0.30, "premium": 9.8, "spread_pct": 7.2,
             "ev_per_contract": -80.0, "quality_score": 0.31},
        ])

    def test_only_calls_shown_ranked(self):
        text = B.call_board(self._df(), "NBIS")
        self.assertIn("SQUEEZE CALLS", text)
        self.assertIn("185.0", text)
        self.assertIn("195.0", text)
        self.assertNotIn("170.0", text)  # the put stays out

    def test_no_calls_returns_none(self):
        df = self._df()
        self.assertIsNone(B.call_board(df[df["type"] == "put"], "NBIS"))
        self.assertIsNone(B.call_board(None, "NBIS"))


class TestScanBoard(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_setup_rows_sort_first(self):
        per = [
            {"ticker": "AAA", "setup": D.assess_squeeze({"short_interest": 0.16,
                                                         "short_interest_trend": "rising"}),
             "best_call": None},
            {"ticker": "NBIS", "setup": _setup_nbis(), "best_call": "$185C 08/14"},
        ]
        text = B.squeeze_scan_board(per)
        self.assertLess(text.index("NBIS"), text.index("AAA"))
        self.assertIn("SETUP", text)
        self.assertIn("WATCH", text)
        self.assertIn("$185C 08/14", text)


if __name__ == "__main__":
    unittest.main()
