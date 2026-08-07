"""The launcher offers STRATEGIES, and the desk cannot trade."""
from __future__ import annotations

import unittest
from unittest import mock

from src import launcher


class MenuRowTest(unittest.TestCase):
    def test_strategies_row_is_rendered(self):
        self.assertIn("STRATEGIES", "\n".join(launcher._menu_rows()))

    def test_settings_moved_to_seven(self):
        self.assertRegex("\n".join(launcher._menu_rows()),
                         r"\[?7\]?\s+SETTINGS")


class DispatchTest(unittest.TestCase):
    """`main()` dispatches straight to the screener when argv carries flags, and
    under `python -m unittest` argv always does. Every test here pins it."""

    def setUp(self):
        patcher = mock.patch.object(launcher.sys, "argv", ["run.py"])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_choice_six_opens_strategies(self):
        called = {}
        with mock.patch("src.strategies.menu.run_menu",
                        lambda *a, **k: called.setdefault("yes", True)), \
             mock.patch("src.launcher._show_menu", side_effect=["6", "Q"]):
            launcher.main()
        self.assertTrue(called.get("yes"))

    def test_word_alias_opens_strategies(self):
        called = {}
        with mock.patch("src.strategies.menu.run_menu",
                        lambda *a, **k: called.setdefault("yes", True)), \
             mock.patch("src.launcher._show_menu",
                        side_effect=["STRATEGIES", "Q"]):
            launcher.main()
        self.assertTrue(called.get("yes"))

    def test_seven_still_opens_settings(self):
        called = {}
        with mock.patch("src.launcher._settings_menu",
                        lambda *a, **k: called.setdefault("yes", True)), \
             mock.patch("src.launcher._show_menu", side_effect=["7", "Q"]):
            launcher.main()
        self.assertTrue(called.get("yes"))


class DisplayOnlyTest(unittest.TestCase):
    def test_menu_cannot_reach_the_trade_logger(self):
        import inspect

        from src.strategies import menu
        src = inspect.getsource(menu)
        for forbidden in ("log_trade", "PaperManager", "enforce_exits"):
            self.assertNotIn(forbidden, src,
                             f"{forbidden} in a display-only desk")


class RunMenuTest(unittest.TestCase):
    def test_quitting_immediately_returns(self):
        with mock.patch("builtins.input", side_effect=["Q"]):
            from src.strategies.menu import run_menu
            run_menu()

    def test_default_arguments_branch(self):
        """No records passed: the desk loads its own library."""
        from src.strategies import menu
        with mock.patch("builtins.input", side_effect=[""]):
            menu.run_menu()

    def test_a_setup_id_opens_its_detail(self):
        from src.strategies import menu
        printed = []
        with mock.patch("builtins.input",
                        side_effect=["put_spread_ivr50", "", "Q"]), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(
                 " ".join(str(x) for x in a))):
            menu.run_menu()
        self.assertTrue(any("HYPOTHESIS" in p for p in printed))

    def test_the_tfsa_filter_toggles(self):
        from src.strategies import menu
        printed = []
        with mock.patch("builtins.input", side_effect=["T", "Q"]), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(
                 " ".join(str(x) for x in a))):
            menu.run_menu()
        self.assertFalse(any("naked_call_extended" in p for p in printed[-3:]))


if __name__ == "__main__":
    unittest.main()
