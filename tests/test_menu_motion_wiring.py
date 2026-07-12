import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest import mock


class TestMenuPrompt(unittest.TestCase):
    def test_non_interactive_skips_motion(self):
        from src import options_screener as osc
        with mock.patch.object(osc, "prompt_input", return_value="3") as pi, \
             mock.patch("src.ui_motion.HeaderMotion") as hm:
            out = osc._menu_prompt_with_motion("choose", "3", interactive=False,
                                               art_rows=6, art_offset=15)
        self.assertEqual(out, "3")
        hm.assert_not_called()
        pi.assert_called_once()

    def test_no_art_printed_means_no_motion(self):
        # Plain menu path prints no art; the prompt must not start a painter.
        from src import options_screener as osc
        with mock.patch.object(osc, "prompt_input", return_value="3"), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion") as hm:
            osc._menu_prompt_with_motion("choose", "3", interactive=True,
                                         art_rows=0, art_offset=0)
        hm.assert_not_called()

    def test_interactive_motion_started_and_stopped(self):
        from src import options_screener as osc
        motion = mock.MagicMock()
        with mock.patch.object(osc, "prompt_input", return_value="7"), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion) as hm:
            out = osc._menu_prompt_with_motion("choose", "3", interactive=True,
                                               art_rows=6, art_offset=15)
        self.assertEqual(out, "7")
        self.assertEqual(hm.call_args.kwargs.get("offset"), 15)
        motion.start.assert_called_once()
        motion.stop.assert_called_once()

    def test_stop_called_even_when_prompt_raises(self):
        from src import options_screener as osc
        motion = mock.MagicMock()
        with mock.patch.object(osc, "prompt_input", side_effect=KeyboardInterrupt), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion):
            with self.assertRaises(KeyboardInterrupt):
                osc._menu_prompt_with_motion("choose", "3", interactive=True,
                                             art_rows=6, art_offset=15)
        motion.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
