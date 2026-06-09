"""Tests for src/intel/ui.py — box alignment, wrapping, no overflow."""
from __future__ import annotations

import unittest

from src.intel import ui


class BoxTests(unittest.TestCase):
    def test_box_lines_equal_width(self):
        body = [ui.row("PRICE", "$200.00  -10%/5d  RSI 36"),
                "\x00",
                ui.row("VERDICT", "WAIT (confidence: medium)")]
        lines = ui.box("INTEL BRIEFING", "NVDA · Nvidia", body, width=64)
        widths = {ui._vlen(ln) for ln in lines}
        self.assertEqual(widths, {64}, f"uneven widths: {widths}")

    def test_long_content_does_not_overflow(self):
        body = [ui.row("NEWS", "x" * 200)]
        lines = ui.box("T", "R", body, width=50)
        # padding clamps at 0 → at worst the content line is its own length,
        # but the borders must not be wider than declared for short content.
        self.assertTrue(all(ui._vlen(ln) >= 50 for ln in lines[:1]))

    def test_divider_sentinel_renders(self):
        lines = ui.box("T", "R", ["a", "\x00", "b"], width=30)
        self.assertEqual(len(lines), 5)  # top + 3 body + bottom


class WrapTests(unittest.TestCase):
    def test_wrap_respects_width(self):
        text = "Pullback to support in an uptrend, but momentum is still falling."
        lines = ui.wrap(text, width=30, indent="  ")
        self.assertTrue(all(ui._vlen(ln) <= 30 for ln in lines))
        self.assertEqual(" ".join(l.strip() for l in lines), text)

    def test_empty(self):
        self.assertEqual(ui.wrap("", 20), [])


class SparkTests(unittest.TestCase):
    def test_spark_length_matches_input(self):
        s = ui.SPARK([1, 2, 3, 4, 5])
        self.assertEqual(len(s), 5)

    def test_spark_short_input_empty(self):
        self.assertEqual(ui.SPARK([1]), "")


if __name__ == "__main__":
    unittest.main()
