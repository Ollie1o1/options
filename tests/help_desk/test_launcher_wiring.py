"""[8] HELP on the launcher's first menu, and its crash isolation."""
import unittest
from unittest import mock

from src import formatting as fmt
from src import launcher


class LauncherWiringTest(unittest.TestCase):
    def setUp(self):
        self._color = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = self._color

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


if __name__ == "__main__":
    unittest.main()
