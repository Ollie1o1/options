"""Return-on-risk cell in the STRATEGY BREAKDOWN table.

The portfolio view is the surface where strategies get compared, and it
compared them on summed percentages alone. Risk per trade spans two orders of
magnitude, so that ranking is partly a position-size ranking. The cell must
degrade to n/a rather than raise when a row has no capital_at_risk yet — the
column is NULL on every trade logged before the backfill.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check_pnl import _ror_cell


class TestReturnOnRiskCell(unittest.TestCase):
    def test_positive_return_is_signed(self):
        self.assertEqual(_ror_cell(0.326).strip(), "+32.6%")

    def test_negative_return_is_signed(self):
        self.assertEqual(_ror_cell(-0.19).strip(), "-19.0%")

    def test_missing_value_reads_as_not_available(self):
        self.assertEqual(_ror_cell(None).strip(), "n/a")

    def test_cells_are_a_fixed_width_so_columns_line_up(self):
        widths = {len(_ror_cell(v)) for v in (0.326, -0.19, None, 1.5)}
        self.assertEqual(len(widths), 1)


if __name__ == "__main__":
    unittest.main()
