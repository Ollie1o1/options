import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from unittest import mock


class TestIntelMenuBriefing(unittest.TestCase):
    def test_choice_d_builds_and_opens(self):
        from src import options_screener as osc
        calls = {}
        def fake_baw(out_dir="reports/briefings", slow=True):
            calls["built"] = True
            return "/tmp/x.html", "/tmp/x.json"
        with mock.patch("src.morning.build_and_write", fake_baw), \
             mock.patch.object(osc, "prompt_input", side_effect=["d", "x"]), \
             mock.patch.object(osc, "_open_briefing_file") as opener, \
             mock.patch.object(osc.sys.stdin, "isatty", return_value=True):
            osc._run_intel_menu()
        self.assertTrue(calls.get("built"))
        opener.assert_called_once_with("/tmp/x.html")


if __name__ == "__main__":
    unittest.main()
