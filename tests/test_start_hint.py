"""The one-line orienting hint above the mode menu.

Thirteen modes with no stated entry point is the first thing a stranger hits.
The hint has to render in BOTH menu paths — the enhanced one and the plain
fallback — because which one a user gets depends on their terminal, not on
anything they chose.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_screener import _START_HINT  # noqa: E402


class StartHintTest(unittest.TestCase):
    def test_it_names_the_flow_not_just_a_mode(self):
        # The value is the ORDER: context, then candidates, then drill in.
        self.assertIn("INTEL", _START_HINT)
        self.assertIn("DISCOVER", _START_HINT)
        self.assertLess(_START_HINT.index("INTEL"), _START_HINT.index("DISCOVER"))

    def test_the_mode_numbers_match_the_menu(self):
        self.assertIn("[10]", _START_HINT)   # INTEL
        self.assertIn("[3]", _START_HINT)    # DISCOVER

    def test_it_stays_one_line(self):
        # The menu is already tall; a hint that wraps defeats its own purpose.
        self.assertNotIn("\n", _START_HINT)
        self.assertLessEqual(len(_START_HINT), 100)

    def test_both_menu_paths_render_it(self):
        # Guards the failure where the hint is added to the pretty path only
        # and the fallback terminal — the one more likely to be a stranger's —
        # silently loses it.
        import re

        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "src", "options_screener.py")
        with open(path) as f:
            src = f.read()
        self.assertEqual(len(re.findall(r"_START_HINT", src)), 3,
                         "expected the definition plus one use in each menu path")


if __name__ == "__main__":
    unittest.main()
