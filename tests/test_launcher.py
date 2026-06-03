import builtins
import sys
import unittest

import src.launcher as launcher
import src.leverage.__main__ as leverage_main


class TestLauncherRouting(unittest.TestCase):
    """The top-level menu must dispatch [3] to the leverage menu and never see
    the menu when invoked with argv (cron / power-user fast path)."""

    def _route(self, choice):
        calls = []
        orig_input = builtins.input
        orig_argv = sys.argv
        orig_lev = leverage_main.menu
        orig_crypto = sys.modules.get("src.crypto.screener")
        builtins.input = lambda *_a, **_k: choice
        sys.argv = ["prog"]  # no flags -> menu path
        leverage_main.menu = lambda: calls.append("leverage")
        try:
            launcher.main()
        finally:
            builtins.input = orig_input
            sys.argv = orig_argv
            leverage_main.menu = orig_lev
            if orig_crypto is not None:
                sys.modules["src.crypto.screener"] = orig_crypto
        return calls

    def test_choice_3_routes_to_leverage(self):
        self.assertEqual(self._route("3"), ["leverage"])

    def test_choice_L_routes_to_leverage(self):
        self.assertEqual(self._route("L"), ["leverage"])

    def test_quit_does_not_route(self):
        self.assertEqual(self._route("Q"), [])


if __name__ == "__main__":
    unittest.main()
