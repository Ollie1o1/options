"""Theme layer: semantic styles with truecolor + ANSI fallback + glyphs."""
import os
import re
import unittest

from src import formatting as fmt

ANSI_RE = re.compile(r'\033\[[0-9;]*m')


class ThemeTestCase(unittest.TestCase):
    def setUp(self):
        self._colorterm = os.environ.get("COLORTERM")
        os.environ["COLORTERM"] = ""

    def tearDown(self):
        fmt._COLOR_ENABLED = None
        if self._colorterm is None:
            os.environ.pop("COLORTERM", None)
        else:
            os.environ["COLORTERM"] = self._colorterm

    def test_style_plain_when_color_disabled(self):
        fmt.set_color_enabled(False)
        self.assertEqual(fmt.style("PoP 62%", "good"), "PoP 62%")

    def test_style_wraps_with_reset_when_color_enabled(self):
        fmt.set_color_enabled(True)
        out = fmt.style("PoP 62%", "good")
        self.assertTrue(out.endswith(fmt.Colors.RESET))
        self.assertIn("PoP 62%", out)
        self.assertEqual(ANSI_RE.sub("", out), "PoP 62%")

    def test_style_truecolor_uses_rgb(self):
        fmt.set_color_enabled(True)
        os.environ["COLORTERM"] = "truecolor"
        out = fmt.style("x", "good")
        self.assertIn("\033[38;2;", out)

    def test_style_ansi_fallback_without_truecolor(self):
        fmt.set_color_enabled(True)
        out = fmt.style("x", "good")
        self.assertIn(fmt.Colors.GREEN, out)
        self.assertNotIn("\033[38;2;", out)

    def test_heading_bold_by_default(self):
        fmt.set_color_enabled(True)
        self.assertIn(fmt.Colors.BOLD, fmt.style("T", "heading"))
        self.assertNotIn(fmt.Colors.BOLD, fmt.style("T", "good"))
        self.assertIn(fmt.Colors.BOLD, fmt.style("T", "good", bold=True))

    def test_value_style_is_passthrough_even_with_color(self):
        fmt.set_color_enabled(True)
        self.assertEqual(ANSI_RE.sub("", fmt.style("42", "value")), "42")

    def test_unknown_style_passthrough(self):
        fmt.set_color_enabled(True)
        self.assertEqual(fmt.style("x", "nope"), "x")

    def test_glyphs_single_width(self):
        for name, g in fmt.GLYPHS.items():
            self.assertEqual(len(g), 1, f"{name} glyph must be single char")


if __name__ == "__main__":
    unittest.main()
