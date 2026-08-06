"""`run_top_scan` must not hand a shapeless fetch result to the scorer.

The parallel scan path guards its fetch results (`if data_result is None or
"error" in data_result: ... continue`) before scoring. `run_top_scan` did not,
so any dict lacking "df" went straight into `_score_fetched_data`, raised
KeyError('df') inside its bare handler, and was appended to the operator's real
logs/scan_errors.log as `error: 'df'` — a message that names no cause.

That is how the suite's own fake fetch ended up writing what looked like a
recurring production SPY failure into the live error log, once per run.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_top_scan_fetch_guard -v
"""
from __future__ import annotations

import unittest
from unittest import mock

from src import options_screener as osc


class _Harness:
    """run_top_scan with the market-context calls stubbed out."""

    def __init__(self, fetch_result):
        self.fetch_result = fetch_result
        self.scored = []

    def run(self):
        def fake_score(symbol, data_result, *a, **k):
            self.scored.append(data_result)
            return {"success": False}

        with mock.patch.object(osc, "fetch_options_yfinance",
                               lambda *a, **k: self.fetch_result), \
             mock.patch.object(osc, "_score_fetched_data", fake_score), \
             mock.patch.object(osc, "get_risk_free_rate", lambda: 0.045), \
             mock.patch.object(osc, "get_vix_level", lambda: 15.0), \
             mock.patch.object(osc, "get_market_context",
                               lambda: ("Neutral", "Normal", False, 0.0)), \
             mock.patch("src.cli_display.print_top_n_table", lambda *a, **k: None):
            return osc.run_top_scan(["SPY"], top_n=1)


class TopScanFetchGuardTest(unittest.TestCase):
    def test_result_without_df_is_not_scored(self):
        h = _Harness({"success": False})
        h.run()
        self.assertEqual(h.scored, [], "a dict with no 'df' reached the scorer")

    def test_error_dict_is_not_scored(self):
        h = _Harness({"error": "Rate limited while fetching history for SPY"})
        h.run()
        self.assertEqual(h.scored, [])

    def test_none_is_not_scored(self):
        h = _Harness(None)
        h.run()
        self.assertEqual(h.scored, [])

    def test_wellformed_result_is_still_scored(self):
        """The guard must not swallow good data."""
        good = {"df": object(), "history_df": None, "context": {}}
        h = _Harness(good)
        h.run()
        self.assertEqual(h.scored, [good])

    def test_guard_does_not_write_to_the_error_log(self):
        """The whole point: no scan_errors.log entry for a guarded skip."""
        import os
        log = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(osc.__file__))),
            "logs", "scan_errors.log",
        )
        before = os.path.getsize(log) if os.path.exists(log) else 0
        _Harness({"success": False}).run()
        after = os.path.getsize(log) if os.path.exists(log) else 0
        self.assertEqual(before, after, "run_top_scan polluted scan_errors.log")


if __name__ == "__main__":
    unittest.main()
