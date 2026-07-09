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


if __name__ == "__main__":
    unittest.main()
