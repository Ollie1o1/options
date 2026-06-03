"""Tests for the automation staleness/heartbeat checker (src/health.py).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_health -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta

from src.health import Check, stale_warnings, collect_checks


class StaleWarningsTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 6, 3, 12, 0, 0)

    def test_fresh_check_no_warning(self):
        c = Check("auto-log", self.now - timedelta(days=1), max_age_days=4)
        self.assertEqual(stale_warnings([c], self.now), [])

    def test_stale_check_warns(self):
        c = Check("auto-log", self.now - timedelta(days=14), max_age_days=4)
        warns = stale_warnings([c], self.now)
        self.assertEqual(len(warns), 1)
        self.assertIn("auto-log", warns[0])
        self.assertIn("14", warns[0])

    def test_missing_activity_warns(self):
        c = Check("checkpoint", None, max_age_days=9, hint="install the cron")
        warns = stale_warnings([c], self.now)
        self.assertEqual(len(warns), 1)
        self.assertIn("checkpoint", warns[0])
        self.assertIn("install the cron", warns[0])

    def test_exactly_at_threshold_no_warning(self):
        c = Check("x", self.now - timedelta(days=4), max_age_days=4)
        self.assertEqual(stale_warnings([c], self.now), [])

    def test_hint_included_when_stale(self):
        c = Check("auto-log", self.now - timedelta(days=30), max_age_days=4,
                  hint="check cron Full Disk Access")
        self.assertIn("check cron Full Disk Access", stale_warnings([c], self.now)[0])


class CollectChecksTest(unittest.TestCase):
    """collect_checks reads real artifacts and never raises on missing files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.now = datetime(2026, 6, 3, 12, 0, 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_artifacts_do_not_raise(self):
        checks = collect_checks(root=self.tmp, db_path=os.path.join(self.tmp, "nope.db"),
                                now=self.now)
        # Returns Check objects (with last=None for missing); never raises.
        self.assertTrue(all(isinstance(c, Check) for c in checks))
        self.assertTrue(len(checks) >= 1)

    def test_db_last_trade_date_drives_autolog_freshness(self):
        db = os.path.join(self.tmp, "t.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (date TEXT)")
        conn.execute("INSERT INTO trades (date) VALUES ('2026-06-02')")
        conn.execute("INSERT INTO trades (date) VALUES ('2026-05-01')")
        conn.commit(); conn.close()
        checks = collect_checks(root=self.tmp, db_path=db, now=self.now)
        autolog = next(c for c in checks if "auto-log" in c.label.lower())
        self.assertEqual(autolog.last.date(), datetime(2026, 6, 2).date())


if __name__ == "__main__":
    unittest.main()
