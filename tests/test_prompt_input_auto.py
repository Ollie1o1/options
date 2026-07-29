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


class _Args:
    """Stand-in for the parsed argparse namespace main() builds."""

    def __init__(self, **kw):
        self.auto = False
        self.mode = None
        self.ticker = None
        self.auto_log = False
        self.__dict__.update(kw)


class TestSuppressPromptsFor(unittest.TestCase):
    """Which invocations may skip prompts.

    The first version of this keyed off the session-loop's `_interactive` flag,
    which is already false for --mode/--ticker/--auto-log. That silently ate the
    Save/Export menu on a hand-run `--ticker AAPL`: prompt_input returned the
    "" default, so the menu broke out before the operator could log the pick.
    """

    def test_auto_suppresses(self):
        self.assertTrue(S.suppress_prompts_for(_Args(auto=True)))

    def test_plain_run_does_not_suppress(self):
        self.assertFalse(S.suppress_prompts_for(_Args()))

    def test_scan_selectors_alone_do_not_suppress(self):
        # These say *what* to scan, not that the operator stopped wanting the
        # save menu. run.py's unattended shortcuts all carry --auto as well.
        for kw in ({"ticker": "AAPL"}, {"mode": "discover"}, {"auto_log": True}):
            with self.subTest(**kw):
                self.assertFalse(S.suppress_prompts_for(_Args(**kw)))

    def test_selectors_combined_with_auto_still_suppress(self):
        # `run.py -ds` → --mode discover --auto-log --log-top 5 --auto
        args = _Args(auto=True, mode="discover", auto_log=True)
        self.assertTrue(S.suppress_prompts_for(args))


if __name__ == "__main__":
    unittest.main()
