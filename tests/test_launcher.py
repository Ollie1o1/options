import builtins
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest

import src.formatting as fmt
import src.launcher as launcher
import src.leverage.__main__ as leverage_main
import src.settings as settings_mod


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


class TestSettingsMenu(unittest.TestCase):
    """[6] must route to the settings menu, and picking a theme there must
    persist it via src.settings, not just mutate in-memory state."""

    def setUp(self):
        self._orig_path = settings_mod._SETTINGS_PATH
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(tmp_path)
        settings_mod._SETTINGS_PATH = tmp_path
        self._tmp_path = tmp_path

    def tearDown(self):
        settings_mod._SETTINGS_PATH = self._orig_path
        if os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        fmt.set_theme("quant_desk")

    def _route_to_settings(self, inputs):
        orig_input = builtins.input
        orig_argv = sys.argv
        _inputs = iter(inputs)
        builtins.input = lambda *_a, **_k: next(_inputs, "Q")
        sys.argv = ["prog"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                launcher.main()
        finally:
            builtins.input = orig_input
            sys.argv = orig_argv
        return buf.getvalue()

    def test_choice_6_opens_settings_menu(self):
        out = self._route_to_settings(["6", "B", "Q"])
        self.assertIn("SETTINGS", out)

    def test_theme_picker_lists_all_four_themes(self):
        out = self._route_to_settings(["6", "1", "B", "B", "Q"])
        self.assertIn("Quant Desk", out)
        self.assertIn("Cyberpunk Neon", out)
        self.assertIn("Matrix Terminal", out)
        self.assertIn("Amber CRT", out)

    def test_picking_a_theme_persists_it(self):
        self._route_to_settings(["6", "1", "3", "B", "B", "Q"])  # 3 = matrix_terminal
        self.assertEqual(settings_mod.get_theme(), "matrix_terminal")

    def test_unknown_top_level_choice_mentions_6(self):
        out = self._route_to_settings(["9", "Q"])
        self.assertIn("6", out)


if __name__ == "__main__":
    unittest.main()
