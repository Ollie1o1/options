"""Tests for equity-vol reporting — summary math + render + color discipline."""
from __future__ import annotations
import os, unittest
from src.equity_vol.engine import TradeResult
from src.equity_vol.report import summarize, render


def _tr(sym, date, ret, pnl=1.0):
    return TradeResult(sym, date, 30, 10.0, 0.0, 0.0, 0.0, pnl, ret)


class ReportTests(unittest.TestCase):
    def _results(self):
        return [_tr("AAPL", "2023-06-01", 0.10), _tr("AAPL", "2023-12-01", 0.05),
                _tr("MSFT", "2024-03-01", -0.08), _tr("MSFT", "2024-09-01", 0.02)]

    def test_summarize_keys_and_oos(self):
        s = summarize(self._results())
        for k in ("n", "mean_ret", "sharpe", "hit", "pf", "nw_t", "oos"):
            self.assertIn(k, s)
        self.assertEqual(s["oos"]["train"]["n"], 2)
        self.assertEqual(s["oos"]["test"]["n"], 2)

    def test_render_has_verdict_and_caveat(self):
        out = render(self._results())
        self.assertIn("OOS", out)
        self.assertIn("research", out.lower())

    def test_render_empty(self):
        self.assertIn("no", render([]).lower())

    def test_report_source_has_no_raw_color(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "equity_vol", "report.py")
        with open(os.path.abspath(path)) as f:
            src = f.read()
        self.assertNotIn("\\033[", src)
        self.assertNotIn("Colors.", src)


if __name__ == "__main__":
    unittest.main()
