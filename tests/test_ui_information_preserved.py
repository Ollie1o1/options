"""Guard: the restyle must not remove information.

Renders the full report for a fixture pick and asserts every load-bearing
data token is present in the ANSI-stripped output. Tokens are data values,
not styling, so this passes before and after the redesign.
"""
import contextlib
import io
import os
import re
import sys
import unittest

from src import formatting as fmt

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

REQUIRED_TOKENS = [
    # headline metrics
    "NVDA", "190", "2026-07-17", "0.78", "62", "2.1",
    # liquidity
    "1,240", "8,450", "2.1%",
    # greeks
    "0.45", "0.012", "0.31", "-0.08",
    # valuation / flow / context
    "41", "PCR", "0.82", "RSI", "58", "Bullish", "182.40",
    # breakeven block
    "194.20",
    # plan block
    "Entry", "Target", "Stop", "Breakeven", "Max Loss", "Confidence",
    # structural sections
    "Liquidity", "Greeks", "Flow", "Context", "Thesis",
    # second pick + buckets
    "AAPL", "210", "LOW", "MEDIUM",
    # comparison table
    "QUICK COMPARISON",
]


class InformationPreservedTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(False)
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _render_report(self):
        from ui_preview import df
        from src.cli_display import print_report
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(df(), 182.40, 0.043, 3, 14, 45, mode="Discovery scan",
                         market_trend="Bullish", volatility_regime="Normal",
                         config={}, compact=False)
        return ANSI_RE.sub("", buf.getvalue())

    def test_report_contains_all_data_tokens(self):
        out = self._render_report()
        missing = [t for t in REQUIRED_TOKENS if t not in out]
        self.assertFalse(missing, f"tokens missing from report output: {missing}")


if __name__ == "__main__":
    unittest.main()
