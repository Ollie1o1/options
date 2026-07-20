"""Tests for the long-term discovery scan (src/longterm/discover.py).

Pure-function tests only in this file's first section (universe sourcing,
context math, ladder suggestion, narrative rendering) — network-touching
functions (universe's actual Finviz call inside finviz_tickers, deep_context,
scan) are thin wrappers tested by inspection/mocking only where noted, never
against the live network, matching the convention set by
tests/longterm/test_data.py for src/longterm/data.py's fetch_snapshots.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import discover as DSC


class TestSectorFilters(unittest.TestCase):
    def test_every_filter_value_is_nonempty_and_has_quality_gate(self):
        for keyword, f_params in DSC.SECTOR_FILTERS.items():
            self.assertTrue(f_params, f"{keyword} has an empty filter string")
            self.assertIn("cap_midover", f_params, f"{keyword} missing quality gate")

    def test_keywords_are_uppercase(self):
        for keyword in DSC.SECTOR_FILTERS:
            self.assertEqual(keyword, keyword.upper())


class TestUniverse(unittest.TestCase):
    def test_unknown_keyword_raises_with_valid_list(self):
        with self.assertRaises(ValueError) as ctx:
            DSC.universe("NOT_A_SECTOR")
        msg = str(ctx.exception)
        self.assertIn("SEMICONDUCTORS", msg)  # a real keyword should be listed

    def test_keyword_is_case_insensitive(self):
        with mock.patch("src.squeeze.universe.finviz_tickers",
                        return_value=["MU", "AMD"]) as mocked:
            result = DSC.universe("semiconductors", limit=10)
        self.assertEqual(result, ["MU", "AMD"])
        mocked.assert_called_once()
        called_f_params = mocked.call_args.args[0]
        self.assertEqual(called_f_params, DSC.SECTOR_FILTERS["SEMICONDUCTORS"])

    def test_passes_limit_through(self):
        with mock.patch("src.squeeze.universe.finviz_tickers",
                        return_value=[]) as mocked:
            DSC.universe("TECH", limit=7)
        self.assertEqual(mocked.call_args.kwargs.get("limit")
                         or mocked.call_args.args[2], 7)

    def test_empty_result_on_fetch_failure_does_not_raise(self):
        # finviz_tickers itself never raises (degrades to [] internally) —
        # confirm universe() doesn't add its own crash path on top.
        with mock.patch("src.squeeze.universe.finviz_tickers", return_value=[]):
            result = DSC.universe("BANKS")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
