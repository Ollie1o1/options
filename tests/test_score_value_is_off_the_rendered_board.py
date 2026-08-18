"""Score came off the HEADERS but not off the ROWS.

`8e3c8ad` dropped `quality_score` from the comparison and top-N boards because
its OOS IC measures -0.12 and a column called Score reads as a verdict. It
removed the `'Score'` header literal from both format strings — and left the
value in the row format strings.

The result on a live board is worse than the column it replaced: an unlabelled
number, sitting where the header says `Drivers`.

    1  AMZN  put  260.0  2026-09-11  24  -0.44  8%  32.9%  $6.70  +196  7%  61%  MARG  0.598  +VRP(0.14)

`TestScoreIsOffTheBoards` did not catch it because it greps the function source
for `'Score'`, which is the header literal only. That is the same mistake the
ranking guard's allowlist made: a claim about behaviour, asserted against text.

These tests RENDER the tables and look for the number.
"""
import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from src import formatting as fmt
from src.cli_display import print_comparison_table, print_top_n_table

# Distinctive, so a match cannot come from any other cell in the fixture.
SCORE = 0.876


def _row(**over):
    row = {
        "symbol": "QQQ", "type": "call", "strike": 733.0,
        "expiration": "2026-09-18", "T_years": 40 / 365, "delta": 0.42,
        "premium": 20.0, "quality_score": SCORE, "friction_pct": 0.01,
        "verdict_passed": True, "prob_profit": 0.40, "rr_ratio": 1.5,
        "iv_percentile_30": 0.33, "vega": 1.23, "ev_per_contract": 25.0,
        "spread_pct": 0.02, "score_drivers": "+Theta(0.08)",
    }
    row.update(over)
    return row


class TopNBoardTest(unittest.TestCase):
    """The board an operator reads first."""

    def _render(self):
        df = pd.DataFrame([_row()])
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_top_n_table(df, 1)
        return buf.getvalue()

    def test_the_row_does_not_print_the_score_value(self):
        self.assertNotIn(f"{SCORE:.3f}", self._render(),
                         "quality_score is still rendered in the row, now "
                         "with no header naming it")

    def test_the_row_still_renders_its_real_columns(self):
        """Guards the fix: removing the cell must not blank the line."""
        out = self._render()
        for expected in ("QQQ", "733.0", "$ 20.00"):
            self.assertIn(expected, out)

    def test_the_drivers_column_survives(self):
        self.assertIn("+Theta(0.08)", self._render())


class ComparisonBoardTest(unittest.TestCase):
    """The side-by-side board, same defect."""

    def _render(self):
        fmt.set_color_enabled(False)
        try:
            df = pd.DataFrame([_row()])
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_comparison_table(df)
            return buf.getvalue()
        finally:
            fmt._COLOR_ENABLED = None

    def test_the_row_does_not_print_the_score_value(self):
        out = self._render()
        if not out.strip():
            self.skipTest("comparison table requires the enhanced CLI")
        self.assertNotIn(f"{SCORE:.2f}", out,
                         "quality_score is still rendered in the row, now "
                         "with no header naming it")

    def test_the_row_still_renders_its_real_columns(self):
        out = self._render()
        if not out.strip():
            self.skipTest("comparison table requires the enhanced CLI")
        self.assertIn("QQQ", out)
        self.assertIn("$733C", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
