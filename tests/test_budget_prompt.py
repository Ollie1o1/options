"""The per-scan budget prompt.

Defaults to NO LIMIT and must never abort a scan. Bad input costs one
re-prompt and then falls back to no limit; a scan dying because someone typed
"5oo" would be worse than a missing constraint.

`prompt_input` already returns its default on --auto and on a non-TTY, so cron
and piped runs cannot block here. That is not what keeps the scheduler safe —
the scheduler never reaches this prompt at all (it runs --top N --auto-log
through run_top_scan) and therefore sets no `budget_at_entry` key.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import src.options_screener as S


class TestPromptForBudget(unittest.TestCase):

    def _with_inputs(self, *answers):
        it = iter(answers)
        return patch.object(S, "prompt_input", side_effect=lambda *a, **k: next(it))

    def test_empty_means_no_limit(self):
        with self._with_inputs(""):
            self.assertIsNone(S.prompt_for_budget())

    def test_the_word_none_means_no_limit(self):
        with self._with_inputs("none"):
            self.assertIsNone(S.prompt_for_budget())

    def test_a_number_is_returned_as_a_float(self):
        with self._with_inputs("2500"):
            self.assertEqual(S.prompt_for_budget(), 2500.0)

    def test_currency_formatting_is_tolerated(self):
        with self._with_inputs("$2,500"):
            self.assertEqual(S.prompt_for_budget(), 2500.0)

    def test_bad_input_reprompts_once_then_accepts(self):
        with self._with_inputs("abc", "1000"):
            self.assertEqual(S.prompt_for_budget(), 1000.0)

    def test_two_bad_inputs_fall_back_to_no_limit_rather_than_aborting(self):
        with self._with_inputs("abc", "def"):
            self.assertIsNone(S.prompt_for_budget())

    def test_zero_and_negative_mean_no_limit(self):
        for bad in ("0", "-100"):
            with self.subTest(answer=bad):
                with self._with_inputs(bad, bad):
                    self.assertIsNone(S.prompt_for_budget())

    def test_it_never_raises(self):
        with self._with_inputs(None, None):
            self.assertIsNone(S.prompt_for_budget())


if __name__ == "__main__":
    unittest.main()


class TestOnlyAnAnswerableRunCountsAsChosen(unittest.TestCase):
    """A prompt nobody could answer is not a choice.

    `prompt_input` returns its default without asking under `--auto` or on a
    non-TTY, so `prompt_for_budget()` yields None there. Recording that as
    "the operator chose NO LIMIT" is exactly the claim the key-presence design
    exists to prevent — it just arrives through the prompt instead of through
    the trade dict.
    """

    def tearDown(self):
        from src import options_screener as osc
        osc.set_auto_mode(False)

    def test_auto_mode_is_never_answerable(self):
        from src import options_screener as osc
        osc.set_auto_mode(True)
        self.assertFalse(osc._prompt_is_answerable())

    def test_a_non_tty_is_never_answerable(self):
        import io
        import sys
        from src import options_screener as osc
        osc.set_auto_mode(False)
        real = sys.stdin
        sys.stdin = io.StringIO("")          # a pipe: isatty() is False
        try:
            self.assertFalse(osc._prompt_is_answerable())
        finally:
            sys.stdin = real

    def test_a_tty_is_answerable(self):
        import sys
        from src import options_screener as osc
        osc.set_auto_mode(False)

        class _Tty:
            def isatty(self):
                return True

        real = sys.stdin
        sys.stdin = _Tty()
        try:
            self.assertTrue(osc._prompt_is_answerable())
        finally:
            sys.stdin = real

    def test_a_broken_stdin_is_not_answerable(self):
        """Fail closed: an unknown stdin must not claim a human answered."""
        import sys
        from src import options_screener as osc
        osc.set_auto_mode(False)

        class _Broken:
            def isatty(self):
                raise OSError("no stdin")

        real = sys.stdin
        sys.stdin = _Broken()
        try:
            self.assertFalse(osc._prompt_is_answerable())
        finally:
            sys.stdin = real
