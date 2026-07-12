import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest import mock

from src import formatting as fmt
from src import ui_motion


class TestArt(unittest.TestCase):
    def test_wide_terminal_gets_block_art(self):
        lines = ui_motion.art_lines(120)
        self.assertEqual(len(lines), 6)
        self.assertTrue(any("█" in l for l in lines))
        # centered: consistent leading pad on the widest row
        self.assertTrue(lines[0].startswith(" "))

    def test_narrow_terminal_gets_one_line_fallback(self):
        lines = ui_motion.art_lines(40)
        self.assertEqual(len(lines), 1)
        self.assertIn("OPTIONS DESK", lines[0])

    def test_art_rows_align(self):
        widths = {len(l) for l in ui_motion._ART}
        self.assertEqual(len(widths), 1)  # every row same width or art breaks

    def test_art_frame_plain_mode_is_static_unstyled(self):
        saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False
        try:
            frames = ui_motion.art_frame(120, tick=5)
            self.assertEqual(frames, ui_motion.art_lines(120))
        finally:
            fmt._COLOR_ENABLED = saved

    def test_art_frame_shimmer_moves(self):
        saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = True
        try:
            f1 = ui_motion.art_frame(120, tick=10)
            f2 = ui_motion.art_frame(120, tick=30)
            self.assertNotEqual(f1, f2)
            self.assertIn("\x1b[", f1[0])
        finally:
            fmt._COLOR_ENABLED = saved


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

    def test_offset_positions_paint_above_menu(self):
        m = ui_motion.HeaderMotion(2, lambda w: ["aa", "bb"], offset=15)
        writes = []
        with mock.patch.object(sys, "stdout") as out:
            out.write = writes.append
            m._paint(["aa", "bb"])
        joined = "".join(writes)
        self.assertIn("\033[17A", joined)  # offset 15 + 2 rows -> top row
        self.assertIn("\033[16A", joined)  # second row


if __name__ == "__main__":
    unittest.main()
