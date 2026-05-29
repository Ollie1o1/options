"""Tests for the Phase 1 gate banner in src/regime_dashboard.py.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_regime_dashboard_gate -v
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest

from src.regime_dashboard import _print_gate_banner


class GateBannerTests(unittest.TestCase):

    def setUp(self):
        self._orig_dir = os.getcwd()
        self._tmp_dir = tempfile.mkdtemp()
        os.chdir(self._tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_dir)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_banner_prints_when_status_exists(self):
        """Banner should print when reports/GATE_STATUS.md is present."""
        os.makedirs("reports", exist_ok=True)
        with open("reports/GATE_STATUS.md", "w") as fh:
            fh.write(
                "GATE: **READY** as of 2026-07-10  (n=80, IC=+0.15, p=0.001, weeks=6)\n"
                "See `checkpoint_2026-07-10.md` for details.\n"
            )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_gate_banner()

        output = buf.getvalue()
        self.assertIn("GATE", output)
        self.assertIn("READY", output)

    # ------------------------------------------------------------------
    def test_no_banner_when_status_missing(self):
        """No output when reports/GATE_STATUS.md does not exist."""
        # Temp dir has no reports/ subdirectory at all.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_gate_banner()

        self.assertEqual("", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
