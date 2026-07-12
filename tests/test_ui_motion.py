import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest import mock

from src import formatting as fmt
from src import ui_motion


class TestTape(unittest.TestCase):
    def test_tape_frame_scrolls_and_wraps(self):
        ui_motion.set_tape(["SPY +0.4%", "VIX 14.2"])
        f0 = ui_motion.tape_frame(0, 12)
        f3 = ui_motion.tape_frame(3, 12)
        self.assertEqual(len(f0), 12)
        self.assertEqual(len(f3), 12)
        self.assertNotEqual(f0, f3)
        big = ui_motion.tape_frame(10_000, 12)   # offset far past text length wraps
        self.assertEqual(len(big), 12)

    def test_empty_tape_gives_tagline(self):
        ui_motion.set_tape([])
        f = ui_motion.tape_frame(0, 20)
        self.assertEqual(len(f), 20)


class TestMotionAllowed(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = True

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_denied_when_not_interactive(self):
        self.assertFalse(ui_motion.motion_allowed(False))

    def test_denied_when_no_tty(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertFalse(ui_motion.motion_allowed(True))

    def test_denied_when_plain_mode(self):
        fmt._COLOR_ENABLED = False
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertFalse(ui_motion.motion_allowed(True))

    def test_denied_when_dumb_term(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch.dict(os.environ, {"TERM": "dumb"}):
            self.assertFalse(ui_motion.motion_allowed(True))

    def test_allowed_when_fully_interactive(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch.dict(os.environ, {"TERM": "xterm-256color"}):
            self.assertTrue(ui_motion.motion_allowed(True))


class TestHeaderMotion(unittest.TestCase):
    def test_stop_without_start_is_noop(self):
        m = ui_motion.HeaderMotion(2, lambda w: ["a" * w, "b" * w])
        m.stop()   # must not raise

    def test_start_stop_lifecycle(self):
        m = ui_motion.HeaderMotion(1, lambda w: ["x" * w], fps=50)
        m.start()
        m.stop()
        self.assertFalse(m._thread and m._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
