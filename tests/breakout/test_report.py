# tests/breakout/test_report.py
"""Tests for breakout rendering — smoke + color discipline."""
from __future__ import annotations
import unittest
from src.breakout.report import render_backtest, render_forecasts


class ReportTests(unittest.TestCase):
    def test_backtest_render_has_horizons_and_caveat(self):
        result = {"EOM": {"n": 100, "brier": 0.18, "ece": 0.04, "auc": 0.58,
                          "coverage": 0.79, "skill_vs_baseline": 0.06}}
        out = render_backtest(result)
        self.assertIn("EOM", out)
        self.assertIn("survivorship", out.lower())

    def test_forecasts_render_both_leaderboards(self):
        rows = [{"ticker": "AAA", "horizon": "EOM", "point": 0.03,
                 "band": (-0.04, 0.12), "up_prob": 0.30, "down_prob": 0.10},
                {"ticker": "BBB", "horizon": "EOM", "point": -0.02,
                 "band": (-0.15, 0.05), "up_prob": 0.08, "down_prob": 0.28}]
        out = render_forecasts(rows)
        self.assertIn("AAA", out)
        self.assertIn("BBB", out)

    def test_report_source_has_no_raw_color(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "breakout", "report.py")
        with open(os.path.abspath(path)) as f:
            src = f.read()
        # report.py must route all color through src/ui.py / fmt.style,
        # never hand-roll ANSI escapes or raw Colors.* constants.
        self.assertNotIn("\\033[", src)
        self.assertNotIn("Colors.", src)


if __name__ == "__main__":
    unittest.main()
