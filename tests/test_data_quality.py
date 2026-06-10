"""Tests for src/data_quality.py — pure, network-free data-quality helpers."""

import math
import unittest

import numpy as np
import pandas as pd

from src.data_quality import classify_quote_freshness, check_market_hours


class TestClassifyQuoteFreshness(unittest.TestCase):
    # --- market open ---------------------------------------------------------
    def test_fresh_boundary(self):
        self.assertEqual(classify_quote_freshness(0, True), "fresh")
        self.assertEqual(classify_quote_freshness(20, True), "fresh")

    def test_delayed_boundary(self):
        self.assertEqual(classify_quote_freshness(20.1, True), "delayed")
        self.assertEqual(classify_quote_freshness(120, True), "delayed")

    def test_stale_when_old(self):
        self.assertEqual(classify_quote_freshness(120.1, True), "stale")
        self.assertEqual(classify_quote_freshness(10_000, True), "stale")

    def test_negative_age_counts_fresh(self):
        # slight clock skew (quote timestamp marginally in the future)
        self.assertEqual(classify_quote_freshness(-1, True), "fresh")

    # --- market closed -------------------------------------------------------
    def test_market_closed_always_stale_when_timestamped(self):
        self.assertEqual(classify_quote_freshness(0, False), "stale")
        self.assertEqual(classify_quote_freshness(5, False), "stale")
        self.assertEqual(classify_quote_freshness(10_000, False), "stale")

    # --- unknown -------------------------------------------------------------
    def test_none_is_unknown(self):
        self.assertEqual(classify_quote_freshness(None, True), "unknown")
        self.assertEqual(classify_quote_freshness(None, False), "unknown")

    def test_nan_is_unknown(self):
        self.assertEqual(classify_quote_freshness(float("nan"), True), "unknown")
        self.assertEqual(classify_quote_freshness(np.nan, False), "unknown")
        self.assertEqual(classify_quote_freshness(pd.NA, True), "unknown")


class TestCheckMarketHours(unittest.TestCase):
    def test_returns_bool_and_message(self):
        is_open, msg = check_market_hours()
        self.assertIsInstance(is_open, bool)
        self.assertIsInstance(msg, str)
        self.assertTrue(len(msg) > 0)


def _fastapi_available() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except Exception:
        return False


class TestProvenanceColumnsThroughSerialize(unittest.TestCase):
    @unittest.skipUnless(_fastapi_available(), "fastapi not installed (api.py import requires it)")
    def test_serialize_picks_carries_quality_columns(self):
        from src.api import _serialize_picks

        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "type": ["call", "put"],
                "strike": [150.0, 400.0],
                "expiration": ["2026-07-17", "2026-07-17"],
                "premium": [3.5, 5.0],
                "quality_score": [0.8, 0.6],
                "quote_source": ["yfinance", "yfinance+synthetic_spread"],
                "quote_as_of": ["2026-06-09T15:00:00+0000", "2026-06-09T11:00:00+0000"],
                "quote_age_min": [5.0, 240.0],
                "quote_freshness": ["fresh", "stale"],
            }
        )
        rows = _serialize_picks(df, n=10)
        self.assertEqual(len(rows), 2)
        # sorted by quality_score desc -> AAPL first
        self.assertEqual(rows[0]["symbol"], "AAPL")
        for r in rows:
            for col in ("quote_source", "quote_as_of", "quote_age_min", "quote_freshness"):
                self.assertIn(col, r)
        self.assertEqual(rows[0]["quote_freshness"], "fresh")
        self.assertEqual(rows[1]["quote_source"], "yfinance+synthetic_spread")


class TestDataQualitySummaryLine(unittest.TestCase):
    def test_summary_counts(self):
        from src.cli_display import format_data_quality_summary

        df = pd.DataFrame(
            {"quote_freshness": ["fresh", "fresh", "delayed", "stale", "unknown"]}
        )
        line = format_data_quality_summary(df)
        self.assertEqual(line, "Data quality: 2 fresh, 1 delayed, 1 stale, 1 unknown")

    def test_summary_none_without_column(self):
        from src.cli_display import format_data_quality_summary

        self.assertIsNone(format_data_quality_summary(pd.DataFrame({"x": [1]})))


class TestIVCrosscheckSummary(unittest.TestCase):
    def test_counts_verified_corrected_unsolvable(self):
        from src.cli_display import format_iv_crosscheck_summary

        # iv_verified: True (verified), False (corrected), None (solved, no yahoo
        # to compare — counts as neither), NaN solved (unsolvable).
        df = pd.DataFrame({
            "iv_solved": [0.30, 0.25, 0.40, np.nan],
            "iv_verified": [True, False, None, None],
        })
        line = format_iv_crosscheck_summary(df)
        # 1 verified, 1 corrected (only the False row), 1 unsolvable (NaN solved)
        self.assertEqual(line, "IV cross-check: 1 verified, 1 corrected, 1 unsolvable")

    def test_none_without_column(self):
        from src.cli_display import format_iv_crosscheck_summary

        self.assertIsNone(format_iv_crosscheck_summary(pd.DataFrame({"x": [1]})))


if __name__ == "__main__":
    unittest.main()
