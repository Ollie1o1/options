import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import tempfile
import unittest
from unittest import mock

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "desk.json")


class TestWriteDesk(unittest.TestCase):
    def test_offline_replay_writes_both_files(self):
        from src.research.__main__ import main
        with tempfile.TemporaryDirectory() as td:
            rc = main(["--json", FIX, "--no-open", "--out-dir", td])
            self.assertEqual(rc, 0)
            base = json.load(open(FIX))["meta"]["base"]
            html = os.path.join(td, base + ".html")
            self.assertTrue(os.path.exists(html))
            self.assertTrue(os.path.exists(os.path.join(td, base + ".json")))
            content = open(html).read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Research Desk", content)

    def test_missing_sidecar_returns_2(self):
        from src.research.__main__ import main
        rc = main(["--json", "/nonexistent.json", "--no-open"])
        self.assertEqual(rc, 2)

    def test_build_and_write_uses_collector(self):
        import src.research as R
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(R, "build",
                               return_value=json.load(open(FIX))) as b:
            html, sidecar = R.build_and_write(symbol="NVDA", out_dir=td)
            b.assert_called_once()
            self.assertTrue(html.endswith(".html"))
            self.assertTrue(os.path.exists(html))


class TestIntelMenuDesk(unittest.TestCase):
    def test_choice_e_builds_and_opens(self):
        from src import options_screener as osc
        calls = {}

        def fake_baw(symbol=None, out_dir="reports/research", slow=True,
                     budget_s=25.0):
            calls["symbol"] = symbol
            return "/tmp/desk.html", "/tmp/desk.json"

        with mock.patch("src.research.build_and_write", fake_baw), \
             mock.patch.object(osc, "prompt_input",
                               side_effect=["e", "nvda", "x"]), \
             mock.patch.object(osc, "_open_briefing_file") as opener, \
             mock.patch.object(osc.sys.stdin, "isatty", return_value=True):
            osc._run_intel_menu()
        self.assertEqual(calls.get("symbol"), "NVDA")
        opener.assert_called_once_with("/tmp/desk.html")

    def test_choice_e_blank_ticker_builds_market_desk(self):
        from src import options_screener as osc
        calls = {}

        def fake_baw(symbol=None, **kw):
            calls["symbol"] = symbol
            return "/tmp/desk.html", "/tmp/desk.json"

        with mock.patch("src.research.build_and_write", fake_baw), \
             mock.patch.object(osc, "prompt_input",
                               side_effect=["e", "", "x"]), \
             mock.patch.object(osc, "_open_briefing_file"), \
             mock.patch.object(osc.sys.stdin, "isatty", return_value=True):
            osc._run_intel_menu()
        self.assertIsNone(calls.get("symbol"))


if __name__ == "__main__":
    unittest.main()
