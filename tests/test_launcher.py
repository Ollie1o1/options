import builtins
import contextlib
import io
import sys
import types
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
        # Feed the choice once, then Q — the menu now loops back instead of
        # exiting, so a constant input would spin forever.
        _inputs = iter([choice, "Q"])
        builtins.input = lambda *_a, **_k: next(_inputs, "Q")
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


class TestLauncherLoadingFeedback(unittest.TestCase):
    """Picking a menu item must print immediate feedback BEFORE the heavy lazy
    import, so the screen never sits frozen while the sub-tool loads."""

    def test_stocks_prints_loading_before_dispatch(self):
        printed = []
        fake_mod = types.ModuleType("src.options_screener")
        # main() records what stdout already contained when it was reached.
        fake_mod.main = lambda: printed.append(buf.getvalue())
        orig_input = builtins.input
        orig_argv = sys.argv
        orig_mod = sys.modules.get("src.options_screener")
        _inputs = iter(["1", "Q"])  # pick STOCKS once, then quit the looped menu
        builtins.input = lambda *_a, **_k: next(_inputs, "Q")
        sys.argv = ["prog"]
        sys.modules["src.options_screener"] = fake_mod
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                launcher.main()
        finally:
            builtins.input = orig_input
            sys.argv = orig_argv
            if orig_mod is not None:
                sys.modules["src.options_screener"] = orig_mod
            else:
                sys.modules.pop("src.options_screener", None)
        # The loading line was on screen by the time dispatch happened.
        self.assertTrue(printed, "stocks dispatch never ran")
        self.assertIn("Loading equity options", printed[0])

    def test_menu_loops_back_until_quit(self):
        # Entering STOCKS must return to the top menu, not exit the app. Pick
        # STOCKS twice, then Q — main() should be dispatched twice.
        runs = []
        fake_mod = types.ModuleType("src.options_screener")
        fake_mod.main = lambda: runs.append(1)
        orig_input, orig_argv = builtins.input, sys.argv
        orig_mod = sys.modules.get("src.options_screener")
        _inputs = iter(["1", "1", "Q"])
        builtins.input = lambda *_a, **_k: next(_inputs, "Q")
        sys.argv = ["prog"]
        sys.modules["src.options_screener"] = fake_mod
        try:
            launcher.main()
        finally:
            builtins.input, sys.argv = orig_input, orig_argv
            if orig_mod is not None:
                sys.modules["src.options_screener"] = orig_mod
            else:
                sys.modules.pop("src.options_screener", None)
        self.assertEqual(len(runs), 2)


if __name__ == "__main__":
    unittest.main()
