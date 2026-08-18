"""The ASCII fallback header must not promise a column it never prints.

`print_report`'s plain-text table declares a trailing `Quality` column. No
revision of this file has ever printed a value under it — the row stops at
`Tag`. A header naming a column that renders nothing is the same defect as a
value rendering under no header, pointed the other way.

`quality_score` is also the metric `8e3c8ad` took off the boards: OOS IC -0.12,
"not distinguishable from zero". So the header goes, rather than the value
arriving.

This renders the table. `HAS_ENHANCED_CLI` is module-level, so the plain-text
branch is reached by pinning it — the same way the theme tests pin
`fmt._COLOR_ENABLED` rather than setting an env var.
"""
import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

import src.cli_display as cd


def _picks():
    return pd.DataFrame([{
        "symbol": "QQQ", "type": "call", "strike": 733.0,
        "expiration": "2026-09-18", "T_years": 40 / 365, "delta": 0.42,
        "abs_delta": 0.42, "premium": 20.0, "quality_score": 0.876,
        "friction_pct": 0.01, "prob_profit": 0.40, "impliedVolatility": 0.25,
        "openInterest": 1200, "volume": 340, "underlying": 700.0,
        "_rank": 1, "spread_pct": 0.02, "price_bucket": "MEDIUM",
    }])


class AsciiReportHeaderTest(unittest.TestCase):
    def _render(self):
        prior = cd.HAS_ENHANCED_CLI
        cd.HAS_ENHANCED_CLI = False
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cd.print_report(_picks(), 700.0, 0.04, 1, 30, 45)
            return buf.getvalue()
        finally:
            cd.HAS_ENHANCED_CLI = prior

    def test_the_table_actually_renders(self):
        """Guard: if the fixture stops reaching the table, the next assertion
        would pass for the wrong reason."""
        out = self._render()
        self.assertIn("MEDIUM PREMIUM", out)
        self.assertIn("QQQ", out)

    def test_the_header_does_not_promise_a_quality_column(self):
        self.assertNotIn("Quality", self._render(),
                         "the ASCII header names a column the row never fills")

    def test_the_columns_it_does_promise_are_still_there(self):
        out = self._render()
        for label in ("Rank", "Type", "Strike", "Expiry", "Delta", "Tag"):
            self.assertIn(label, out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
