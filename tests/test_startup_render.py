"""Regression guard for the blank-UI bug: the regime dashboard used to render
in a daemon thread that reassigned the global sys.stdout and, when the fetch
overran the join timeout, left it pointing at a dead buffer — so the mode menu
printed afterwards vanished and the screen looked frozen. The render is now
synchronous in the calling thread, so sys.stdout is always restored."""
import contextlib
import io
import sys
import types
import unittest

import src.options_screener as osc
import src.regime_dashboard as rd


def _fake_pm():
    return types.SimpleNamespace(update_positions=lambda: None)


def _no_spinner(*_a, **_k):
    return contextlib.nullcontext()


class TestRegimeRender(unittest.TestCase):
    def _patch_dashboard(self, fn):
        orig = rd.print_regime_dashboard
        rd.print_regime_dashboard = fn
        self.addCleanup(lambda: setattr(rd, "print_regime_dashboard", orig))

    def test_dashboard_captured_and_stdout_restored(self):
        self._patch_dashboard(lambda width=90: print("REGIME_LINE"))
        before = sys.stdout
        out = osc._render_regime_with_exit_enforcement(
            _fake_pm(), 90, spinner_factory=_no_spinner)
        self.assertIn("REGIME_LINE", out)      # captured into the return value
        self.assertIs(sys.stdout, before)      # global stdout handed back

    def test_stdout_restored_even_when_dashboard_raises(self):
        def boom(width=90):
            print("PARTIAL")
            raise RuntimeError("fetch blew up")
        self._patch_dashboard(boom)
        before = sys.stdout
        osc._render_regime_with_exit_enforcement(
            _fake_pm(), 90, spinner_factory=_no_spinner)
        self.assertIs(sys.stdout, before)      # restored despite the exception

    def test_text_printed_after_render_reaches_real_stdout(self):
        # The core regression: output AFTER the render (the mode menu) must reach
        # the surrounding stdout, while the dashboard text is captured separately.
        self._patch_dashboard(lambda width=90: print("DASH"))
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            osc._render_regime_with_exit_enforcement(
                _fake_pm(), 90, spinner_factory=_no_spinner)
            print("MENU_VISIBLE")
        self.assertIn("MENU_VISIBLE", cap.getvalue())
        self.assertNotIn("DASH", cap.getvalue())


if __name__ == "__main__":
    unittest.main()
