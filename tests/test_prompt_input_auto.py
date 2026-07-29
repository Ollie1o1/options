"""prompt_input must honour --auto, not just a non-tty stdin.

The 2026-07-28 fix gave spawned catch-up children stdin=DEVNULL, which covers
every automated path. An operator running `run.py -ds` by hand still stalls at
the ticker-source prompt and again at "Re-sort?", despite --auto being in the
expansion, because prompt_input only ever looked at isatty().
"""
import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import options_screener as S


class TestPromptInputAutoMode(unittest.TestCase):
    def setUp(self):
        self._previous = S.is_auto_mode()

    def tearDown(self):
        S.set_auto_mode(self._previous)

    def test_auto_mode_returns_the_default_without_reading_stdin(self):
        S.set_auto_mode(True)
        # A tty that would block forever if it were actually read.
        with mock.patch.object(sys, "stdin", mock.Mock(isatty=lambda: True)):
            self.assertEqual(S.prompt_input("Ticker source", "watchlist"), "watchlist")

    def test_auto_mode_with_no_default_still_does_not_block(self):
        S.set_auto_mode(True)
        with mock.patch.object(sys, "stdin", mock.Mock(isatty=lambda: True)):
            self.assertEqual(S.prompt_input("Something with no default"), "")

    def test_interactive_mode_is_unchanged(self):
        S.set_auto_mode(False)
        fake = io.StringIO("NVDA\n")
        fake.isatty = lambda: True  # type: ignore[method-assign]
        with mock.patch.object(sys, "stdin", fake):
            self.assertEqual(S.prompt_input("Ticker", "SPY"), "NVDA")

    def test_non_tty_still_returns_the_default(self):
        S.set_auto_mode(False)
        with mock.patch.object(sys, "stdin", mock.Mock(isatty=lambda: False)):
            self.assertEqual(S.prompt_input("Ticker", "SPY"), "SPY")

    def test_auto_mode_defaults_to_off(self):
        # A plain interactive run must keep prompting.
        S.set_auto_mode(False)
        self.assertFalse(S.is_auto_mode())


if __name__ == "__main__":
    unittest.main()
