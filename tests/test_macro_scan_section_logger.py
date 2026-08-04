"""The macro overlay's error handlers must not themselves raise.

`_macro_scan_section` wraps three optional steps — building macro context,
rendering the panel, and the opt-in AI ranking — in try/except so a failure in
a display-only overlay can never take a scan down. Each handler called
`logger.debug(...)`, but `options_screener` has no module-level `logger`: the
name is bound only inside two other functions. Reaching any of those handlers
raised NameError out of the except block, converting a swallowed overlay
failure into a crash at the end of a completed scan.

mypy found it (`Name "logger" is not defined`); it had never fired in testing
because the handlers only run when a macro import or render fails.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.options_screener as S  # noqa: E402


class HandlerDoesNotRaiseTest(unittest.TestCase):
    """Every except branch in the overlay, forced."""

    def setUp(self):
        # The function returns immediately when stdin is not a tty, which is
        # how it stays free in headless runs — so the handlers are only
        # reachable with this patched True.
        self._tty = mock.patch.object(S.sys.stdin, "isatty", lambda: True)
        self._tty.start()
        self.addCleanup(self._tty.stop)

    def _run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            S._macro_scan_section(["AAPL"], focus_symbol="AAPL")
        return buf.getvalue()

    def test_a_failing_context_build_does_not_raise(self):
        with mock.patch("src.macro_pulse.orchestrator.build_context",
                        side_effect=RuntimeError("no macro data")):
            self._run()  # must return quietly

    def test_a_failing_panel_render_does_not_raise(self):
        with mock.patch("src.macro_pulse.orchestrator.build_context",
                        return_value={"x": 1}), \
                mock.patch("src.macro_pulse.orchestrator._lookup_sector",
                           return_value="Tech"), \
                mock.patch("src.macro_pulse.ticker.render_ticker",
                           side_effect=RuntimeError("render blew up")):
            self._run()

    def test_the_failure_is_actually_logged_not_just_swallowed(self):
        # Not raising is half of it; the reason has to reach the log, or an
        # overlay that silently does nothing is indistinguishable from one
        # that worked.
        with mock.patch("src.macro_pulse.orchestrator.build_context",
                        side_effect=RuntimeError("no macro data")):
            with self.assertLogs("src.options_screener", level="DEBUG") as caught:
                self._run()
        self.assertTrue(
            any("macro scan section skipped" in m for m in caught.output),
            f"handler did not log the failure: {caught.output}")


if __name__ == "__main__":
    unittest.main()
