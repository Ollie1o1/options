import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from src import formatting as fmt
from src import ui


class TestErrorLine(unittest.TestCase):
    def test_plain_mode_no_ansi(self):
        saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False
        try:
            line = ui.error_line("Could not build briefing: boom")
            self.assertNotIn("\x1b[", line)
            self.assertIn("Could not build briefing: boom", line)
        finally:
            fmt._COLOR_ENABLED = saved

    def test_color_mode_styled(self):
        saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = True
        try:
            line = ui.error_line("boom")
            self.assertIn("\x1b[", line)
            self.assertIn("boom", line)
        finally:
            fmt._COLOR_ENABLED = saved


class TestIntelColorRouting(unittest.TestCase):
    def test_intel_color_routes_ansi_codes_to_semantic_tokens(self):
        saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = True
        try:
            from src.intel import ui as iui
            self.assertEqual(iui.color("hello", fmt.Colors.GREEN),
                             fmt.style("hello", "good", bold=False))
            self.assertEqual(iui.color("warn", fmt.Colors.YELLOW),
                             fmt.style("warn", "warn", bold=False))
            # Unmapped codes keep the legacy colorize path
            self.assertEqual(iui.color("x", fmt.Colors.MAGENTA),
                             fmt.colorize("x", fmt.Colors.MAGENTA, bold=False))
        finally:
            fmt._COLOR_ENABLED = saved


if __name__ == "__main__":
    unittest.main()
