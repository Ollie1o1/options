"""[8] HELP on the launcher's first menu, and its crash isolation."""
import io
import unittest
from unittest import mock

from src import formatting as fmt
from src import launcher


class LauncherWiringTest(unittest.TestCase):
    def setUp(self):
        self._color = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)
        # launcher.main() calls settings.apply_saved_theme(), which mutates the
        # PROCESS-WIDE theme to whatever this operator has saved and never puts
        # it back. Every later test that asserts on a default-theme RGB then
        # fails — tests/test_desk_visuals.py did exactly that. Restore it.
        self._theme = fmt.get_theme()

    def tearDown(self):
        fmt._COLOR_ENABLED = self._color
        fmt.set_theme(self._theme)

    def test_help_row_is_present_at_eight(self):
        rows = launcher._menu_rows()
        numbered = [r for r in rows if "[8]" in r]
        self.assertEqual(len(numbered), 1)
        self.assertIn("HELP", numbered[0])

    def test_help_row_sits_last_before_quit(self):
        rows = launcher._menu_rows()
        self.assertIn("HELP", rows[-2])
        self.assertIn("QUIT", rows[-1])

    def _dispatch(self, token):
        # argv MUST be pinned. launcher.main() forwards straight to the equity
        # screener whenever any argv flag is present, and under the test runner
        # argv carries the module filter — so an unpinned test does not exercise
        # the menu at all, it launches a real scan and hangs.
        with mock.patch("sys.argv", ["run.py"]), \
             mock.patch.object(launcher, "_show_menu", side_effect=[token, "Q"]), \
             mock.patch("src.help_desk.run_menu") as run, \
             mock.patch("builtins.print"):
            launcher.main()
        return run

    def test_every_alias_dispatches_to_the_manual(self):
        for token in ("8", "HELP", "?"):
            with self.subTest(token=token):
                self.assertEqual(self._dispatch(token).call_count, 1)

    def test_h_still_belongs_to_holdings(self):
        """HOLDINGS had "H" first. Help must not have taken it."""
        with mock.patch("sys.argv", ["run.py"]), \
             mock.patch.object(launcher, "_show_menu", side_effect=["H", "Q"]), \
             mock.patch("src.longterm.board.menu") as holdings, \
             mock.patch("src.help_desk.run_menu") as helpmenu, \
             mock.patch("builtins.print"):
            launcher.main()
        self.assertEqual(holdings.call_count, 1)
        self.assertEqual(helpmenu.call_count, 0)

    def test_unknown_choice_message_names_eight(self):
        printed = []
        with mock.patch("sys.argv", ["run.py"]), \
             mock.patch.object(launcher, "_show_menu", side_effect=["zz", "Q"]), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: printed.append(
                            " ".join(str(x) for x in a))):
            launcher.main()
        self.assertTrue(any("Unknown choice" in p and "8" in p for p in printed),
                        printed)

    def test_a_broken_manual_does_not_kill_the_launcher(self):
        printed = []
        with mock.patch("sys.argv", ["run.py"]), \
             mock.patch.object(launcher, "_show_menu", side_effect=["8", "Q"]), \
             mock.patch("src.help_desk.run_menu",
                        side_effect=RuntimeError("boom")), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: printed.append(
                            " ".join(str(x) for x in a))):
            launcher.main()  # must return, not raise
        self.assertTrue(any("Help unavailable" in p for p in printed), printed)

    def test_front_door_hint_points_at_help(self):
        """The masthead animation must stay off here. _show_menu decides it is
        interactive from isatty(), and a MagicMock stdin answers truthy — which
        starts the motion thread and hangs the suite. Pin both to False."""
        printed = []
        with mock.patch("sys.stdin.isatty", return_value=False), \
             mock.patch("sys.stdout.isatty", return_value=False), \
             mock.patch("builtins.input", side_effect=EOFError), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: printed.append(
                            " ".join(str(x) for x in a))):
            launcher._show_menu()
        self.assertTrue(any("[8] HELP" in p for p in printed), printed)


class ThemeLeakTest(unittest.TestCase):
    """The launcher tests drive `launcher.main()`, which applies the operator's
    saved theme process-wide. Asserting the restore from inside those tests is
    impossible — tearDown has not run yet — so run the whole class and check the
    theme afterwards. Without the tearDown restore, this fails and so does
    tests/test_desk_visuals.py, which asserts on default-theme RGB values.
    """

    def test_running_the_launcher_tests_leaves_the_theme_untouched(self):
        from src import settings

        original = fmt.get_theme()
        self.addCleanup(fmt.set_theme, original)

        # Pin a sentinel that is definitely NOT the theme main() would apply,
        # so a leak always flips it. Reading the current theme instead would
        # silently pass: unittest runs LauncherWiringTest first (class names
        # sort L before T), so by now a leak has already happened and the
        # "before" value would be the leaked one.
        settings.apply_saved_theme()
        saved = fmt.get_theme()
        sentinel = next(k for k, _ in fmt.list_themes() if k != saved)
        fmt.set_theme(sentinel)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            LauncherWiringTest)
        result = unittest.TextTestRunner(
            stream=io.StringIO(), verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertEqual(fmt.get_theme(), sentinel)


if __name__ == "__main__":
    unittest.main()
