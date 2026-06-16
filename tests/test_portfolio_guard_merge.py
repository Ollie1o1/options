"""Merged concentration panel: guard absorbs correlation pairs."""
import re
import unittest

from src import formatting as fmt
from src.portfolio_guard import format_guard_lines

ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def strip(s):
    return ANSI_RE.sub("", s)


class GuardCorrelationTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _picks(self):
        return [
            {"symbol": "AAPL", "type": "call", "delta": 0.40, "vega": 0.30,
             "theta": -0.10, "quality_score": 0.6},
            {"symbol": "MSFT", "type": "call", "delta": 0.40, "vega": 0.30,
             "theta": -0.10, "quality_score": 0.6},
        ]

    def test_correlation_pairs_render_in_guard(self):
        pairs = [("AAPL", "MSFT", 0.93)]
        lines = [strip(l) for l in format_guard_lines(self._picks(), mode="Discovery", corr_pairs=pairs)]
        body = "\n".join(lines)
        self.assertIn("AAPL", body)
        self.assertIn("MSFT", body)
        self.assertIn("0.93", body)

    def test_no_pairs_no_correlation_rows(self):
        lines = [strip(l) for l in format_guard_lines(self._picks(), mode="Discovery", corr_pairs=[])]
        body = "\n".join(lines)
        self.assertNotIn("corr", body.lower())

    def test_backward_compatible_without_pairs(self):
        # existing callers pass no corr_pairs
        lines = format_guard_lines(self._picks(), mode="Discovery")
        self.assertTrue(any("net" in strip(l) for l in lines))


if __name__ == "__main__":
    unittest.main()
