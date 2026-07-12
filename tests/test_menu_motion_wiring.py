import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest import mock


class TestSeedTape(unittest.TestCase):
    def test_seed_motion_tape_formats_segments(self):
        from src import regime_dashboard as rd
        from src import ui_motion
        rd._seed_motion_tape({"vix": 14.2, "posture": "RISK_ON",
                              "vix_term_structure": "CONTANGO",
                              "options_pcr": 0.92, "spy_ret_5d": 1.3})
        text = ui_motion.tape_text()
        self.assertIn("VIX 14.2", text)
        self.assertIn("RISK_ON", text)
        self.assertIn("SPY 5d +1.3%", text)

    def test_seed_motion_tape_never_raises_on_junk(self):
        from src import regime_dashboard as rd
        rd._seed_motion_tape({})            # all-None regime
        rd._seed_motion_tape(None)          # no regime at all


class TestMenuPrompt(unittest.TestCase):
    def test_non_interactive_skips_motion(self):
        from src import options_screener as osc
        with mock.patch.object(osc, "prompt_input", return_value="3") as pi, \
             mock.patch("src.ui_motion.HeaderMotion") as hm:
            out = osc._menu_prompt_with_motion("choose", "3", interactive=False)
        self.assertEqual(out, "3")
        hm.assert_not_called()
        pi.assert_called_once()

    def test_interactive_motion_started_and_stopped(self):
        from src import options_screener as osc
        motion = mock.MagicMock()
        with mock.patch.object(osc, "prompt_input", return_value="7"), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion):
            out = osc._menu_prompt_with_motion("choose", "3", interactive=True)
        self.assertEqual(out, "7")
        motion.start.assert_called_once()
        motion.stop.assert_called_once()

    def test_stop_called_even_when_prompt_raises(self):
        from src import options_screener as osc
        motion = mock.MagicMock()
        with mock.patch.object(osc, "prompt_input", side_effect=KeyboardInterrupt), \
             mock.patch("src.ui_motion.motion_allowed", return_value=True), \
             mock.patch("src.ui_motion.HeaderMotion", return_value=motion):
            with self.assertRaises(KeyboardInterrupt):
                osc._menu_prompt_with_motion("choose", "3", interactive=True)
        motion.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
