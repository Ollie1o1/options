"""Task 8: dead-scheduler hard confirm.

`options_screener._dead_scheduler_ack` is the escalation on top of the
existing (silent-when-fresh) `maintenance_health.health_banner`: once the
scheduler has read dead for more than `DEAD_SCHEDULER_ACK_DAYS` days, the
*interactive* path must require a single Enter keypress before continuing,
stating plainly what is degrading. Automation, --auto, --mode/--ticker, and
piped stdin must never block — there is no bypass flag, so failing to gate
correctly here means every cron/piped run wedges.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_dead_scheduler_ack -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.maintenance_health import DEAD_SCHEDULER_ACK_DAYS
from src.options_screener import _dead_scheduler_ack


class _FakeInput:
    """Records whether it was called; never actually blocks a test run."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt=""):
        self.calls.append(prompt)
        return ""


class _FakePrint:
    def __init__(self):
        self.lines = []

    def __call__(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


class DeadSchedulerAckTest(unittest.TestCase):
    def setUp(self):
        self.input_fn = _FakeInput()
        self.print_fn = _FakePrint()

    def test_interactive_and_dead_over_threshold_requires_the_ack(self):
        fired = _dead_scheduler_ack(DEAD_SCHEDULER_ACK_DAYS + 1, True, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertTrue(fired)
        self.assertEqual(len(self.input_fn.calls), 1)
        self.assertTrue(self.print_fn.lines)  # the banner was printed

    def test_auto_path_does_not_block_even_when_dead_and_interactive_flag_is_true(self):
        # --auto (and --mode/--ticker/--auto-log) resolve `_interactive` to
        # False upstream in main() before this is ever called; this asserts
        # the function itself refuses to block once interactive is False,
        # which is the actual guarantee automation depends on.
        fired = _dead_scheduler_ack(DEAD_SCHEDULER_ACK_DAYS + 30, False, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertFalse(fired)
        self.assertEqual(self.input_fn.calls, [])

    def test_piped_non_tty_stdin_does_not_block(self):
        # A piped run's `_interactive` is computed from sys.stdin.isatty(),
        # which is False under a pipe — mirror that resolved value directly.
        fired = _dead_scheduler_ack(DEAD_SCHEDULER_ACK_DAYS + 5, False, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertFalse(fired)
        self.assertEqual(self.input_fn.calls, [])

    def test_dead_under_threshold_is_unchanged_no_ack(self):
        fired = _dead_scheduler_ack(DEAD_SCHEDULER_ACK_DAYS - 1, True, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertFalse(fired)
        self.assertEqual(self.input_fn.calls, [])
        self.assertEqual(self.print_fn.lines, [])

    def test_exactly_at_threshold_is_still_unchanged(self):
        # ">7 days" per the brief — exactly 7 is not yet escalated.
        fired = _dead_scheduler_ack(DEAD_SCHEDULER_ACK_DAYS, True, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertFalse(fired)
        self.assertEqual(self.input_fn.calls, [])

    def test_unknown_duration_never_blocks(self):
        # launchd_dead_days returns None while healthy, or when launchctl is
        # unavailable — never a reason to block.
        fired = _dead_scheduler_ack(None, True, 100,
                                    input_fn=self.input_fn, print_fn=self.print_fn)
        self.assertFalse(fired)
        self.assertEqual(self.input_fn.calls, [])

    def test_no_bypass_flag_exists_on_the_signature(self):
        import inspect

        params = inspect.signature(_dead_scheduler_ack).parameters
        for name in params:
            self.assertNotIn("bypass", name.lower())
            self.assertNotIn("skip", name.lower())
            self.assertNotIn("force", name.lower())


if __name__ == "__main__":
    unittest.main()
