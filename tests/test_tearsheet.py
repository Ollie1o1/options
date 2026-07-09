"""Ticker tearsheet — theme, charts, render, collect."""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tearsheet import theme  # noqa: E402


class TestTheme(unittest.TestCase):
    def test_light_and_dark_define_identical_keys(self):
        # A token present in light but missing in dark is an invisible-text bug
        # that no eyeball test reliably catches.
        self.assertEqual(set(theme.LIGHT), set(theme.DARK))

    def test_every_token_is_a_hex_colour(self):
        for name, table in (("LIGHT", theme.LIGHT), ("DARK", theme.DARK)):
            for k, v in table.items():
                self.assertRegex(v, r"^#[0-9a-fA-F]{6}$", f"{name}.{k}={v}")

    def test_dark_uses_the_terminal_palette(self):
        self.assertEqual(theme.DARK["good"], "#5ec98d")
        self.assertEqual(theme.DARK["bad"], "#e06c75")
        self.assertEqual(theme.DARK["warn"], "#d6a452")
        self.assertEqual(theme.DARK["muted"], "#626870")

    def test_heat_inks_returns_two_hexes(self):
        lo, hi = theme.heat_inks(500.0, 1000.0)
        self.assertRegex(lo, r"^#[0-9a-f]{6}$")
        self.assertRegex(hi, r"^#[0-9a-f]{6}$")
        self.assertNotEqual(lo, hi)

    def test_heat_sign_is_preserved_in_both_inks(self):
        # Loss must never render as the gain ink in either theme.
        gain_l, gain_d = theme.heat_inks(1000.0, 1000.0)
        loss_l, loss_d = theme.heat_inks(-1000.0, 1000.0)
        self.assertNotEqual(gain_l, loss_l)
        self.assertNotEqual(gain_d, loss_d)

    def test_zero_span_does_not_crash(self):
        self.assertEqual(len(theme.heat_inks(5.0, 0.0)), 2)

    def test_css_tokens_defines_both_themes(self):
        css = theme.css_tokens()
        self.assertIn(":root", css)
        self.assertIn('[data-theme="dark"]', css)
        self.assertIn("--good", css)


from src.tearsheet import charts  # noqa: E402


class TestCharts(unittest.TestCase):
    def test_line_chart_is_svg_and_uses_css_vars(self):
        svg = charts.line_chart([1.0, 3.0, 2.0, 5.0])
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("var(--ink)", svg)
        self.assertNotRegex(svg, r"stroke=\"#[0-9a-f]{6}\"")

    def test_line_chart_is_deterministic(self):
        s = [1.0, 2.0, 3.0]
        self.assertEqual(charts.line_chart(s), charts.line_chart(s))

    def test_line_chart_too_short_returns_empty(self):
        self.assertEqual(charts.line_chart([1.0]), "")
        self.assertEqual(charts.line_chart([]), "")

    def test_flat_series_does_not_divide_by_zero(self):
        svg = charts.line_chart([4.0, 4.0, 4.0])
        self.assertIn("<svg", svg)

    def test_price_with_bands_marks_levels(self):
        svg = charts.price_with_bands(
            [10.0, 11.0, 12.0], [{"label": "50d MA", "level": 9.5, "pct": -0.05}],
            [{"label": "swing", "level": 13.0, "pct": 0.08}])
        self.assertIn("var(--good)", svg)   # support band
        self.assertIn("var(--bad)", svg)    # resistance band

    def test_vol_cone_marks_current_iv(self):
        cone = [{"window": 30, "p25": .2, "median": .3, "p75": .4, "current": .32, "pctile": .78}]
        svg = charts.vol_cone(cone, current_iv=0.42)
        self.assertIn("<svg", svg)
        self.assertIn("var(--bad)", svg)

    def test_vol_cone_empty_returns_empty(self):
        self.assertEqual(charts.vol_cone([], None), "")

    def test_term_curve_labels_each_expiry(self):
        svg = charts.term_curve([[9, 0.30], [44, 0.37]])
        self.assertIn("9d", svg)
        self.assertIn("44d", svg)

    def test_waterfall_bars_totals(self):
        html = charts.waterfall_bars([["Gross edge", 6.0], ["Spread", -18.0]])
        self.assertIn("Gross edge", html)
        self.assertIn("var(--bad)", html)
        self.assertIn("var(--good)", html)


if __name__ == "__main__":
    unittest.main()
