"""Wiring assertions: SQUEEZE mode reachable from menu, CLI, and pipeline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "options_screener.py")


class TestSqueezeWiring(unittest.TestCase):
    """Source-level wiring checks (importing main() would run a scan)."""

    @classmethod
    def setUpClass(cls):
        with open(_SRC, encoding="utf-8") as f:
            cls.src = f.read()

    def test_cli_mode_choice_registered(self):
        self.assertIn('"squeeze"', self.src)
        self.assertIn('"squeeze": "SQUEEZE"', self.src)

    def test_menu_entries_present(self):
        self.assertIn('"11", "SQUEEZE"', self.src)
        self.assertIn('"11": "SQUEEZE"', self.src)

    def test_mode_string_flows_like_discovery(self):
        self.assertIn('mode in ("Discovery scan", "Squeeze Hunt")', self.src)

    def test_auto_log_excludes_squeeze_hunt(self):
        # display-only discipline: squeeze picks must never reach auto-log
        self.assertIn('mode not in ("Lottery Ticket", "Squeeze Hunt")', self.src)

    def test_universal_banner_hook_present(self):
        self.assertIn("assess_squeeze_row", self.src)
        self.assertIn("squeeze read skipped", self.src)


if __name__ == "__main__":
    unittest.main()
