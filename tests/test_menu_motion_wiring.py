import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest import mock


class TestLauncherMasthead(unittest.TestCase):
    """The animated wordmark lives at the very top of the launcher menu
    (STOCKS / CRYPTO / LEVERAGE / RESEARCH) and nowhere else."""

    def test_interactive_launcher_prints_art_and_animates(self):
        from src import launcher
        motion = mock.MagicMock()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion) as hm, \
             mock.patch("builtins.input", return_value="q"), \
             mock.patch("builtins.print") as fake_print:
            choice = launcher._show_menu()
        self.assertEqual(choice, "Q")
        motion.start.assert_called_once()
        motion.stop.assert_called_once()
        # painter aimed above every line printed after the art
        self.assertGreater(hm.call_args.kwargs.get("offset"), 5)
        printed = "".join(str(c.args[0]) if c.args else "" for c in
                          fake_print.call_args_list)
        self.assertTrue("█" in printed or "OPTIONS DESK" in printed)

    def test_non_tty_launcher_has_no_art_no_motion(self):
        from src import launcher
        with mock.patch.object(sys.stdin, "isatty", return_value=False), \
             mock.patch.object(sys.stdout, "isatty", return_value=False), \
             mock.patch("src.ui_motion.HeaderMotion") as hm, \
             mock.patch("builtins.input", return_value="q"), \
             mock.patch("builtins.print") as fake_print:
            choice = launcher._show_menu()
        self.assertEqual(choice, "Q")
        hm.assert_not_called()
        printed = "".join(str(c.args[0]) if c.args else "" for c in
                          fake_print.call_args_list)
        self.assertNotIn("█", printed)

    def test_stop_called_even_when_input_interrupted(self):
        from src import launcher
        motion = mock.MagicMock()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion), \
             mock.patch("builtins.input", side_effect=KeyboardInterrupt), \
             mock.patch("builtins.print"):
            choice = launcher._show_menu()
        self.assertEqual(choice, "Q")   # interrupt maps to quit
        motion.stop.assert_called_once()

    def test_stocks_mode_menu_has_no_art(self):
        # The art must NOT come back to the stocks mode menu.
        import inspect
        from src import options_screener as osc
        src_text = inspect.getsource(osc)
        self.assertNotIn("art_frame", src_text)
        self.assertNotIn("_menu_prompt_with_motion", src_text)


if __name__ == "__main__":
    unittest.main()
