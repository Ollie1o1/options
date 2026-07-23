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


class ThemeSwitchingTestCase(unittest.TestCase):
    def tearDown(self):
        fmt.set_theme("quant_desk")

    def test_default_theme_is_quant_desk(self):
        self.assertEqual(fmt.get_theme(), "quant_desk")

    def test_set_theme_switches_rgb(self):
        before = dict(fmt._THEME_RGB)
        self.assertTrue(fmt.set_theme("matrix_terminal"))
        self.assertEqual(fmt.get_theme(), "matrix_terminal")
        self.assertNotEqual(fmt._THEME_RGB["heading"], before["heading"])

    def test_set_theme_unknown_name_is_noop(self):
        fmt.set_theme("quant_desk")
        ok = fmt.set_theme("not_a_real_theme")
        self.assertFalse(ok)
        self.assertEqual(fmt.get_theme(), "quant_desk")

    def test_semantic_colors_fixed_across_all_themes(self):
        keys = ("good", "warn", "bad", "value")
        baseline_rgb = None
        baseline_ansi = None
        for name, _label in fmt.list_themes():
            palette = fmt.THEMES[name]
            rgb = {k: palette["rgb"][k] for k in keys}
            ansi = {k: palette["ansi"][k] for k in keys}
            if baseline_rgb is None:
                baseline_rgb, baseline_ansi = rgb, ansi
            else:
                self.assertEqual(rgb, baseline_rgb, f"{name} changed a semantic rgb color")
                self.assertEqual(ansi, baseline_ansi, f"{name} changed a semantic ansi color")

    def test_quant_desk_matches_original_values(self):
        rgb = fmt.THEMES["quant_desk"]["rgb"]
        self.assertEqual(rgb["heading"], (130, 170, 210))
        self.assertEqual(rgb["accent"], (130, 170, 210))
        self.assertEqual(rgb["label"], (130, 137, 145))
        self.assertIsNone(rgb["value"])
        self.assertEqual(rgb["good"], (94, 201, 141))
        self.assertEqual(rgb["warn"], (214, 164, 82))
        self.assertEqual(rgb["bad"], (224, 108, 117))
        self.assertEqual(rgb["muted"], (98, 104, 112))
        self.assertEqual(rgb["emph"], (240, 240, 240))
        ansi = fmt.THEMES["quant_desk"]["ansi"]
        self.assertEqual(ansi["heading"], fmt.Colors.BRIGHT_CYAN)
        self.assertEqual(ansi["good"], fmt.Colors.GREEN)

    def test_list_themes_includes_all_four_in_order(self):
        keys = [k for k, _ in fmt.list_themes()]
        self.assertEqual(
            keys, ["quant_desk", "cyberpunk_neon", "matrix_terminal", "amber_crt"]
        )
        labels = [label for _, label in fmt.list_themes()]
        self.assertEqual(len(labels), len(set(labels)), "theme labels must be unique")

    def test_style_reflects_active_theme(self):
        fmt.set_color_enabled(True)
        import os as _os
        prev = _os.environ.get("COLORTERM")
        _os.environ["COLORTERM"] = "truecolor"
        try:
            fmt.set_theme("cyberpunk_neon")
            out = fmt.style("x", "heading")
            r, g, b = fmt.THEMES["cyberpunk_neon"]["rgb"]["heading"]
            self.assertIn(fmt.rgb_fg(r, g, b), out)
        finally:
            if prev is None:
                _os.environ.pop("COLORTERM", None)
            else:
                _os.environ["COLORTERM"] = prev
            fmt.set_theme("quant_desk")


if __name__ == "__main__":
    unittest.main()
