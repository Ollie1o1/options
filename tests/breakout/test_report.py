# tests/breakout/test_report.py
"""Tests for breakout rendering — smoke + color discipline."""
from __future__ import annotations
import re, unittest
from src.breakout.report import render_backtest, render_forecasts

_ANSI = re.compile(r"\033\[[0-9;]*m")


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

    def test_no_raw_ansi_escapes(self):
        out = render_backtest({"EOW": {"n": 1, "brier": 0.2, "ece": 0.0,
                                       "auc": None, "coverage": 0.8,
                                       "skill_vs_baseline": 0.0}})
        # color must come through fmt.style, not hand-rolled escapes in this module
        self.assertEqual(len(_ANSI.findall(out.replace("\033[0m", ""))), 0)


if __name__ == "__main__":
    unittest.main()
