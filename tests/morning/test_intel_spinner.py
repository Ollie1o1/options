import io
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from unittest import mock


class TestPrintViaSpinner(unittest.TestCase):
    def test_output_emitted_after_fn_runs(self):
        from src import options_screener as osc
        real_out = io.StringIO()
        with mock.patch.object(osc, "_spinner") as spin, \
             mock.patch("sys.stdout", real_out):
            osc._print_via_spinner("working…", lambda: print("hello box"))
        spin.assert_called_once_with("working…")
        self.assertIn("hello box", real_out.getvalue())

    def test_partial_output_survives_exception(self):
        from src import options_screener as osc
        real_out = io.StringIO()

        def boom():
            print("partial line")
            raise RuntimeError("feed died")

        with mock.patch.object(osc, "_spinner"), \
             mock.patch("sys.stdout", real_out):
            with self.assertRaises(RuntimeError):
                osc._print_via_spinner("working…", boom)
        self.assertIn("partial line", real_out.getvalue())


class TestIntelMenuSpinners(unittest.TestCase):
    def _run(self, choices, **patches):
        from src import options_screener as osc
        with mock.patch.object(osc, "prompt_input", side_effect=choices), \
             mock.patch.object(osc.sys.stdin, "isatty", return_value=True):
            osc._run_intel_menu()

    def test_market_overview_runs_under_spinner(self):
        from src import options_screener as osc
        from src.intel import market
        spinners = []

        def fake_spinner(label):
            import contextlib
            spinners.append(label)
            return contextlib.nullcontext()

        with mock.patch.object(osc, "_spinner", fake_spinner), \
             mock.patch.object(market, "print_market_overview") as pmo:
            self._run(["a", "x"])
        pmo.assert_called_once()
        self.assertTrue(spinners, "market overview should show a spinner")

    def test_ticker_briefing_runs_under_spinner(self):
        from src import options_screener as osc
        from src.intel import briefing
        spinners = []

        def fake_spinner(label):
            import contextlib
            spinners.append(label)
            return contextlib.nullcontext()

        with mock.patch.object(osc, "_spinner", fake_spinner), \
             mock.patch.object(briefing, "print_briefing") as pb:
            self._run(["b", "NVDA", "x"])
        pb.assert_called_once_with("NVDA")
        self.assertTrue(any("NVDA" in s for s in spinners),
                        "briefing spinner should name the ticker")

    def test_macro_pulse_runs_under_spinner(self):
        from src import options_screener as osc
        spinners = []

        def fake_spinner(label):
            import contextlib
            spinners.append(label)
            return contextlib.nullcontext()

        with mock.patch.object(osc, "_spinner", fake_spinner), \
             mock.patch("src.macro_pulse.run", return_value="PULSE OK") as mr:
            self._run(["c", "x"])
        mr.assert_called_once()
        self.assertTrue(spinners, "macro pulse should show a spinner")


if __name__ == "__main__":
    unittest.main()
