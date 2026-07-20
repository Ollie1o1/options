import io
import os
import subprocess
import sys
import threading
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tempfile
import unittest
from unittest import mock


class TestBuildReportBounded(unittest.TestCase):
    def test_returns_builder_result(self):
        from src import options_screener as osc
        out = osc._build_report_bounded("t", lambda: ("h.html", "h.json"))
        self.assertEqual(out, ("h.html", "h.json"))

    def test_propagates_builder_exception(self):
        from src import options_screener as osc

        def boom():
            raise ValueError("feed died")

        with self.assertRaises(ValueError):
            osc._build_report_bounded("t", boom)

    def test_overrun_returns_none_and_logs_stacks(self):
        from src import options_screener as osc
        release = threading.Event()
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(osc, "_INTEL_HANG_LOG",
                               os.path.join(d, "intel_hangs.log")):
            out = osc._build_report_bounded("research desk", release.wait,
                                            timeout_s=0.2)
            self.assertIsNone(out)
            with open(os.path.join(d, "intel_hangs.log")) as f:
                log = f.read()
        release.set()
        self.assertIn("research desk", log)
        self.assertIn("Thread", log)  # faulthandler stack dump present


class TestOpenBriefingBounded(unittest.TestCase):
    def test_open_timeout_prints_manual_path(self):
        from src import options_screener as osc
        buf = io.StringIO()

        def hang(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["open"], timeout=kw.get("timeout"))

        with mock.patch("subprocess.run", hang), \
             mock.patch.object(osc.sys, "platform", "darwin"), \
             mock.patch("sys.stdout", buf):
            osc._open_briefing_file("reports/research/x.html")  # must not raise
        self.assertIn("x.html", buf.getvalue())

    def test_open_failure_prints_manual_path(self):
        from src import options_screener as osc
        buf = io.StringIO()
        fail = mock.Mock(return_value=mock.Mock(returncode=1))
        with mock.patch("subprocess.run", fail), \
             mock.patch.object(osc.sys, "platform", "darwin"), \
             mock.patch("sys.stdout", buf):
            osc._open_briefing_file("reports/research/x.html")
        self.assertIn("x.html", buf.getvalue())
        self.assertIn("timeout", fail.call_args.kwargs)


class TestIntelMenuHangGuard(unittest.TestCase):
    def test_hung_research_build_does_not_hang_menu(self):
        from src import options_screener as osc
        release = threading.Event()

        def hung_build(symbol=None):
            release.wait()

        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(osc, "_INTEL_HANG_LOG",
                               os.path.join(d, "intel_hangs.log")), \
             mock.patch.object(osc, "_INTEL_BUILD_TIMEOUT_S", 0.2), \
             mock.patch("src.research.build_and_write", hung_build), \
             mock.patch.object(osc, "prompt_input", side_effect=["e", "", "x"]), \
             mock.patch.object(osc, "_open_briefing_file") as opener, \
             mock.patch.object(osc.sys.stdin, "isatty", return_value=True), \
             mock.patch("sys.stdout", buf):
            t0 = time.time()
            osc._run_intel_menu()
            elapsed = time.time() - t0
        release.set()
        self.assertLess(elapsed, 5.0, "menu must not hang on a stuck build")
        opener.assert_not_called()
        self.assertIn("intel_hangs.log", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
