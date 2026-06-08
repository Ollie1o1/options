import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

from src import maintenance as m


class TestThrottle(unittest.TestCase):
    def test_exit_enforce_due_when_not_run_today(self):
        self.assertTrue(m.due_exit_enforce({"last_exit_enforce": "2026-06-06"}, "2026-06-07"))

    def test_exit_enforce_not_due_when_already_run_today(self):
        self.assertFalse(m.due_exit_enforce({"last_exit_enforce": "2026-06-07"}, "2026-06-07"))

    def test_exit_enforce_due_when_never_run(self):
        self.assertTrue(m.due_exit_enforce({}, "2026-06-07"))

    def test_checkpoint_due_after_7_days(self):
        self.assertTrue(m.due_checkpoint({"last_checkpoint": "2026-05-31"}, "2026-06-07"))

    def test_checkpoint_not_due_within_7_days(self):
        self.assertFalse(m.due_checkpoint({"last_checkpoint": "2026-06-02"}, "2026-06-07"))

    def test_checkpoint_due_when_never_run(self):
        self.assertTrue(m.due_checkpoint({}, "2026-06-07"))


class TestAutologWindow(unittest.TestCase):
    def test_ds_window(self):
        self.assertEqual(m.autolog_window(weekday=3, hhmm=1030), ("ds", "-ds"))

    def test_sps_window(self):
        self.assertEqual(m.autolog_window(weekday=3, hhmm=1300), ("sps", "-sps"))

    def test_ics_window(self):
        self.assertEqual(m.autolog_window(weekday=3, hhmm=1430), ("ics", "-ics"))

    def test_no_window_between(self):
        self.assertIsNone(m.autolog_window(weekday=3, hhmm=1145))

    def test_weekend_never(self):
        self.assertIsNone(m.autolog_window(weekday=6, hhmm=1030))

    def test_autolog_due_per_window_per_day(self):
        st = {"last_autolog": {"ds": "2026-06-07"}}
        self.assertFalse(m.due_autolog(st, "ds", "2026-06-07"))
        self.assertTrue(m.due_autolog(st, "ds", "2026-06-08"))
        self.assertTrue(m.due_autolog(st, "sps", "2026-06-07"))


class TestState(unittest.TestCase):
    def test_load_missing_returns_empty(self):
        self.assertEqual(m.load_state("/no/such/file.json"), {})

    def test_save_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "state.json")
            m.save_state(p, {"last_checkpoint": "2026-06-07"})
            self.assertEqual(m.load_state(p), {"last_checkpoint": "2026-06-07"})

    def test_load_corrupt_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "state.json")
            with open(p, "w") as f:
                f.write("{not json")
            self.assertEqual(m.load_state(p), {})


class TestCohortLine(unittest.TestCase):
    def _make_db(self, path):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                     "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
        rows = [
            ("2026-05-28", "Long Call", "CLOSED", 0, 70.0, 0.10),
            ("2026-05-29", "Long Call", "CLOSED", 0, 60.0, -0.05),
            ("2026-05-30", "Long Call", "OPEN", 0, 55.0, None),
            ("2026-05-30", "Long Call", "CLOSED", 1, 50.0, 0.20),
        ]
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?)", rows)
        conn.commit(); conn.close()

    def test_cohort_line_counts_and_decision(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._make_db(db)
            line = m.cohort_progress_line(db, "2026-05-27", today="2026-06-07")
            self.assertIn("2/50", line)
            self.assertIn("open: 1", line)
            self.assertIn("GATHERING", line)


class TestOrchestrator(unittest.TestCase):
    def _db(self, path):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                     "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
        conn.execute("INSERT INTO trades VALUES ('2026-05-28','Long Call','CLOSED',0,70.0,0.1)")
        conn.commit(); conn.close()

    def test_runs_due_steps_and_records_state(self):
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            state_path = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=state_path,
                now=datetime(2026, 6, 7, 14, 30), runner=fake_runner,
                checkpoint_fn=lambda **k: None)
            self.assertTrue(any("--enforce-exits" in " ".join(c) for c in calls))
            st = m.load_state(state_path)
            self.assertEqual(st["last_exit_enforce"], "2026-06-07")
            self.assertIn("cohort", summary)

    def test_second_run_same_day_skips_exit_enforce(self):
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            kw = dict(db_path=db, phase1_start="2026-05-27", state_path=sp,
                      now=datetime(2026, 6, 7, 16, 0), runner=fake_runner,
                      checkpoint_fn=lambda **k: None)
            m.run_startup_maintenance(**kw)
            calls.clear()
            m.run_startup_maintenance(**kw)
            self.assertFalse(any("--enforce-exits" in " ".join(c) for c in calls))

    def test_runner_exception_does_not_propagate(self):
        def boom(cmd):
            raise RuntimeError("subprocess blew up")
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 14, 30), runner=boom,
                checkpoint_fn=lambda **k: None)
            self.assertIn("cohort", summary)


if __name__ == "__main__":
    unittest.main()
