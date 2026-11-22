"""Startup and long waits must never look frozen, and a fresh clone must start
cleanly.

Measured 2026-11-22 by driving the real app in a pty and logging every silent
gap: the launcher, scans and menus all showed motion, but
  * startup maintenance ran chain-archive / estimates / candidate-marks inline
    with nothing on screen (minutes on a weekday afternoon),
  * the post-scan macro overlay sat silent ~5s on sector lookups,
  * a fresh clone opened on AUTOMATION CRITICAL, because the startup jobs had
    already stamped the state before the "empty state = first run" check,
  * the catch-up spawned ~/.venvs/options/bin/python, which a fresh clone does
    not have,
  * "Nothing cleared" was followed by an offer to open a tearsheet for the
    refused contract.
"""
from __future__ import annotations

import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from unittest import mock

from src import formatting as fmt
from src import maintenance as m
from src import ui

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _TTY(io.StringIO):
    def isatty(self):
        return True


class SpinnerFeedbackTest(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_a_long_wait_shows_its_elapsed_seconds(self):
        buf = _TTY()
        sp = ui.Spinner("loading", stream=buf, interval=0.01)
        sp.ELAPSED_AFTER_S = 0.05
        with sp:
            time.sleep(0.2)
        self.assertRegex(buf.getvalue(), r"loading  \ds")

    def test_a_short_wait_shows_no_clock(self):
        buf = _TTY()
        with ui.Spinner("loading", stream=buf, interval=0.01):
            time.sleep(0.05)
        self.assertNotRegex(buf.getvalue(), r"loading  \ds")

    def test_the_label_can_change_mid_spin_and_is_fully_cleared(self):
        buf = _TTY()
        sp = ui.Spinner("a much longer first label", stream=buf, interval=0.01)
        with sp:
            time.sleep(0.05)
            sp.set_label("short")
            time.sleep(0.05)
        out = buf.getvalue()
        self.assertIn("short", out)
        # The final clear covers the widest line drawn, not just the last label.
        tail = out.rsplit("\r", 2)[-2]
        self.assertGreaterEqual(len(tail), len("  ◐ a much longer first label"))
        self.assertEqual(tail.strip(), "")


class MaintenanceProgressTest(unittest.TestCase):
    """Each job that runs announces itself; skipped jobs don't; a broken
    progress callback can't take maintenance down."""

    NOW = datetime(2026, 11, 18, 15, 0)   # Wednesday, inside the archive window

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.d.name, "t.db")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, "
                     "status TEXT, paper_only INTEGER, quality_score REAL, "
                     "pnl_pct REAL)")
        conn.commit()
        conn.close()
        self.state = os.path.join(self.d.name, "state.json")
        self.patches = [
            mock.patch("src.estimates_archive.archive_symbols",
                       return_value={"symbols": 0, "with_estimates": 0,
                                     "complete": True}),
            mock.patch("src.candidate_marks.mark_candidates",
                       return_value={"marked": 0, "complete": True}),
            mock.patch("src.own_picks.record_pick_marks", return_value=0),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.d.cleanup()

    def _run(self, progress):
        return m.run_startup_maintenance(
            db_path=self.db, phase1_start=None, state_path=self.state,
            now=self.NOW, runner=lambda cmd: 0, background=True,
            spawn_fn=lambda: None,
            checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
            chain_archive_fn=lambda: {"rows": 1, "complete": True},
            morning_fn=lambda: 0, walk_forward_fn=lambda **k: None,
            watch_fn=lambda db: None, progress=progress)

    def test_running_jobs_are_announced(self):
        seen: list = []
        self._run(seen.append)
        joined = " ".join(seen).lower()
        self.assertIn("option chains", joined)
        self.assertIn("track record", joined)
        # The checkpoint is skipped without phase1_start, so it must not claim
        # to be running.
        self.assertNotIn("checkpoint", joined)

    def test_a_raising_callback_does_not_stop_the_jobs(self):
        def boom(_label):
            raise RuntimeError("display broke")
        out = self._run(boom)
        self.assertTrue(any(r.startswith("chain-archive") for r in out["ran"]))


class FreshCloneInterpreterTest(unittest.TestCase):
    def test_catchup_uses_the_running_interpreter_when_the_shared_venv_is_absent(self):
        home = tempfile.mkdtemp()
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; import src.maintenance as m; "
             "print(m.VENV_PY == sys.executable)"],
            cwd=REPO, env=dict(os.environ, HOME=home, PYTHONPATH=REPO),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip(), "True", out.stderr[-500:])


class TearsheetOnlyFromTheBoardTest(unittest.TestCase):
    def test_no_fallback_to_the_pre_gate_frame(self):
        with open(os.path.join(REPO, "src", "options_screener.py")) as f:
            src = f.read()
        self.assertNotIn("_display_df if _display_df is not None else picks", src)
        self.assertIn("if verbose and _display_df is not None and not _display_df.empty:",
                      src)


if __name__ == "__main__":
    unittest.main()
