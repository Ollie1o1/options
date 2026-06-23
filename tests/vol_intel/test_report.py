"""Tests for vol-intel rendering — smoke + color discipline (source scan)."""
from __future__ import annotations
import os, unittest
from src.vol_intel.report import render_iv_movers, render_vrp, render_report


class ReportTests(unittest.TestCase):
    def _movers(self):
        return [{"symbol": "NVDA", "iv": 0.48, "d_iv": 0.062, "rv_pctile": 0.81},
                {"symbol": "AAPL", "iv": 0.22, "d_iv": -0.01, "rv_pctile": 0.30}]

    def _vrp(self):
        return [{"symbol": "AAPL", "iv": 0.22, "rv": 0.15, "vrp": 0.07, "label": "RICH", "rv_pctile": 0.3},
                {"symbol": "TSLA", "iv": 0.40, "rv": 0.55, "vrp": -0.15, "label": "CHEAP", "rv_pctile": 0.9}]

    def test_movers_render_sorts_and_contains_symbols(self):
        out = render_iv_movers(self._movers())
        self.assertIn("NVDA", out)
        self.assertIn("IV MOVERS", out)

    def test_vrp_render_has_both_blocks(self):
        out = render_vrp(self._vrp())
        self.assertIn("AAPL", out)   # RICH
        self.assertIn("TSLA", out)   # CHEAP

    def test_report_has_coverage_and_caveat(self):
        out = render_report(self._movers(), self._vrp(), n_cov=13)
        self.assertIn("13", out)
        self.assertIn("Track-4", out)

    def test_report_source_has_no_raw_color(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "vol_intel", "report.py")
        with open(os.path.abspath(path)) as f:
            src = f.read()
        self.assertNotIn("\\033[", src)
        self.assertNotIn("Colors.", src)


if __name__ == "__main__":
    unittest.main()
