"""--min-dte / --max-dte must reach the --top cross-ticker scan.

`run_top_scan` defaults to 7/45 and the CLI branch never forwarded the flags,
so `--min-dte 25 --max-dte 70` silently produced 7-23 DTE picks. It also never
passed the bounds to the fetch, which selects WHICH expiries to pull — the
fetch's own docstring warns that omitting them starves longer-DTE modes.
"""
import unittest
from unittest import mock

import pandas as pd

from src import options_screener as osc


class TopScanForwardsDteTest(unittest.TestCase):
    def test_the_fetch_receives_the_dte_bounds(self):
        """Without these the fetch takes the front `max_expiries` expiries,
        which on a longer-DTE request are the wrong ones entirely."""
        seen = {}

        def fake_fetch(symbol, max_expiries, min_dte=None, max_dte=None):
            seen["min_dte"], seen["max_dte"] = min_dte, max_dte
            return {"success": False}

        with mock.patch.object(osc, "fetch_options_yfinance", fake_fetch), \
             mock.patch.object(osc, "get_risk_free_rate", lambda: 0.045), \
             mock.patch.object(osc, "get_vix_level", lambda: 15.0), \
             mock.patch.object(osc, "get_market_context",
                               lambda: ("Neutral", "Normal", False, 0.0)), \
             mock.patch("src.cli_display.print_top_n_table", lambda *a, **k: None):
            osc.run_top_scan(["SPY"], top_n=1, min_dte=25, max_dte=70)

        self.assertEqual(seen["min_dte"], 25)
        self.assertEqual(seen["max_dte"], 70)

    def test_the_scoring_step_receives_the_same_bounds(self):
        seen = {}

        def fake_score(symbol, data_result, mode, min_dte, max_dte, *a, **k):
            seen["min_dte"], seen["max_dte"] = min_dte, max_dte
            return {"success": False}

        with mock.patch.object(osc, "fetch_options_yfinance",
                               lambda *a, **k: {"success": True}), \
             mock.patch.object(osc, "_score_fetched_data", fake_score), \
             mock.patch.object(osc, "get_risk_free_rate", lambda: 0.045), \
             mock.patch.object(osc, "get_vix_level", lambda: 15.0), \
             mock.patch.object(osc, "get_market_context",
                               lambda: ("Neutral", "Normal", False, 0.0)), \
             mock.patch("src.cli_display.print_top_n_table", lambda *a, **k: None):
            osc.run_top_scan(["SPY"], top_n=1, min_dte=25, max_dte=70)

        self.assertEqual((seen["min_dte"], seen["max_dte"]), (25, 70))


class CliForwardsDteTest(unittest.TestCase):
    """The actual reported bug: the flags parsed fine and were dropped on the
    way to run_top_scan."""

    def _run(self, argv):
        seen = {}

        def fake_top_scan(tickers, **kw):
            seen.update(kw)
            raise SystemExit(0)

        with mock.patch.object(osc, "run_top_scan", fake_top_scan), \
             mock.patch("sys.argv", argv):
            try:
                osc.main()
            except SystemExit:
                pass
        return seen

    def test_the_flags_reach_run_top_scan(self):
        seen = self._run(["prog", "--top", "5", "--min-dte", "25", "--max-dte", "70"])
        self.assertEqual(seen.get("min_dte"), 25)
        self.assertEqual(seen.get("max_dte"), 70)

    def test_omitting_the_flags_leaves_the_defaults_alone(self):
        seen = self._run(["prog", "--top", "5"])
        self.assertIsNone(seen.get("min_dte"))
        self.assertIsNone(seen.get("max_dte"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
