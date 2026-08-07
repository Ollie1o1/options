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

    def test_squeeze_mode_reaches_past_the_dte_floor(self):
        # The board floors at SQUEEZE_MIN_DTE (60 calendar days). Discovery's
        # defaults fetch the nearest 4 expirations out to 45 DTE, which on
        # weekly-heavy squeeze names never reaches 60 — the floor would warn
        # on every run instead of ever selecting anything.
        self.assertTrue("max_days_to_expiration_squeeze" in self.src,
                        "squeeze mode has no max-DTE default of its own")
        self.assertTrue("max_expirations_squeeze" in self.src,
                        "squeeze mode has no expiration-count default of its own")

    def test_squeeze_fetch_window_covers_the_floor(self):
        from src.squeeze.board import SQUEEZE_MIN_DTE
        from src.options_screener import SQUEEZE_MAX_DTE, SQUEEZE_MAX_EXPIRIES
        self.assertGreater(SQUEEZE_MAX_DTE, SQUEEZE_MIN_DTE,
                           "fetch window ends at or before the floor")
        # Weeklies plus monthlies: 4 expirations is ~1 month out, not 2.
        self.assertGreaterEqual(SQUEEZE_MAX_EXPIRIES, 8)

    def test_window_reaches_the_january_leaps(self):
        # The intermediate monthlies are often too thin near the money: on
        # 2026-08-07 RH's 105d and 133d expiries had no in-band call under a
        # 15% spread, while 2027-01-15 (161d) had four passing every filter on
        # 1,288 open interest. That expiry sits at index 8-10 across these
        # names, so both bounds have to clear it or the fix misses.
        from src.options_screener import SQUEEZE_MAX_DTE, SQUEEZE_MAX_EXPIRIES
        self.assertGreaterEqual(SQUEEZE_MAX_DTE, 161)
        self.assertGreaterEqual(SQUEEZE_MAX_EXPIRIES, 11)


if __name__ == "__main__":
    unittest.main()
