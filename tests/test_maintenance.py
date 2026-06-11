import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

from src import maintenance as m


class TestThrottle(unittest.TestCase):
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

    def test_autolog_fires_in_window_on_weekday_and_records_state(self):
        # 2026-06-04 is a Thursday; 14:30 is inside the 'ics' window.
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            state_path = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=state_path,
                now=datetime(2026, 6, 4, 14, 30), runner=fake_runner,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertTrue(any("-ics" in " ".join(c) for c in calls))
            st = m.load_state(state_path)
            self.assertEqual(st["last_autolog"]["ics"], "2026-06-04")
            self.assertIn("cohort", summary)

    def test_second_run_same_window_skips_autolog(self):
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            kw = dict(db_path=db, phase1_start="2026-05-27", state_path=sp,
                      now=datetime(2026, 6, 4, 14, 30), runner=fake_runner,
                      checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            m.run_startup_maintenance(**kw)
            calls.clear()
            m.run_startup_maintenance(**kw)
            self.assertFalse(calls)

    def test_no_autolog_on_weekend(self):
        # 2026-06-07 is a Sunday — auto-log must not fire.
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 14, 30), runner=fake_runner,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertFalse(calls)

    def test_checkpoint_runs_when_due_and_records_state(self):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 9, 0), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: ran.append(k),
                track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertEqual(len(ran), 1)
            self.assertEqual(m.load_state(sp)["last_checkpoint"], "2026-06-07")
            self.assertIn("checkpoint", summary["ran"])

    def test_track_record_runs_when_due_and_is_injectable(self):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 9, 0), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None,
                track_record_fn=lambda **k: ran.append(k), chain_archive_fn=lambda: 0)
            # injected fn was used (no real reports/TRACK_RECORD.md written)
            self.assertEqual(len(ran), 1)
            self.assertEqual(ran[0]["db_path"], db)
            self.assertEqual(m.load_state(sp)["last_track_record"], "2026-06-07")
            self.assertIn("track_record", summary["ran"])

    def test_runner_exception_does_not_propagate(self):
        def boom(cmd):
            raise RuntimeError("subprocess blew up")
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 4, 14, 30), runner=boom,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertIn("cohort", summary)


class TestChildGuard(unittest.TestCase):
    """The auto-log subprocess boots the screener, whose startup calls
    run_startup_maintenance again. Without a guard that recurses: each child
    sees the window still 'due' (the parent records state only after the child
    exits) and spawns another full scan — confirmed as a ~170-deep process
    bomb on 2026-06-10. The child env marker breaks the cycle."""

    def test_runner_env_marks_child(self):
        env = m._child_env()
        self.assertEqual(env.get(m.CHILD_ENV_MARKER), "1")

    def test_child_process_skips_all_maintenance(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                         "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
            conn.commit(); conn.close()
            old = os.environ.get(m.CHILD_ENV_MARKER)
            os.environ[m.CHILD_ENV_MARKER] = "1"
            try:
                summary = m.run_startup_maintenance(
                    db_path=db, phase1_start="2026-05-27",
                    state_path=os.path.join(d, "state.json"),
                    now=datetime(2026, 6, 4, 14, 30),  # in 'ics' window: would spawn
                    runner=lambda cmd: (calls.append(cmd), 0)[1],
                    checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            finally:
                if old is None:
                    os.environ.pop(m.CHILD_ENV_MARKER, None)
                else:
                    os.environ[m.CHILD_ENV_MARKER] = old
            self.assertEqual(calls, [])           # no recursive spawn
            self.assertEqual(summary["ran"], [])  # no jobs run in the child


class TestChainArchiveJob(unittest.TestCase):
    def _db(self, path):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                     "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
        conn.commit(); conn.close()

    def test_fires_weekday_afternoon_once_and_records_state(self):
        hits = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            kw = dict(db_path=db, phase1_start="2026-05-27", state_path=sp,
                      now=datetime(2026, 6, 4, 16, 0),  # Thu 16:00, no autolog window
                      runner=lambda cmd: 0,
                      checkpoint_fn=lambda **k: None,
                      track_record_fn=lambda **k: None,
                      chain_archive_fn=lambda: hits.append(1) or 7)
            summary = m.run_startup_maintenance(**kw)
            self.assertEqual(len(hits), 1)
            self.assertIn("chain-archive:7rows", summary["ran"])
            self.assertEqual(m.load_state(sp)["last_chain_archive"], "2026-06-04")
            m.run_startup_maintenance(**kw)     # same day: throttled
            self.assertEqual(len(hits), 1)

    def test_does_not_fire_in_the_morning(self):
        hits = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27",
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 9, 30), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: hits.append(1) or 0)
            self.assertEqual(hits, [])


class TestHeadless(unittest.TestCase):
    """run_headless: the LaunchAgent entry point. Same orchestrator, but it
    reads phase1_start from config.json itself, prints a summary, and never
    raises (exit code 0 always — failures are logged, not thrown)."""

    def _setup(self, d):
        db = os.path.join(d, "t.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                     "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
        conn.execute("INSERT INTO trades VALUES ('2026-05-28','Long Call','CLOSED',0,70.0,0.1)")
        conn.commit(); conn.close()
        cfg = os.path.join(d, "config.json")
        with open(cfg, "w") as f:
            json.dump({"auto_log": {"phase1_start_date": "2026-05-27"}}, f)
        return db, cfg

    def test_headless_runs_and_returns_summary(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            db, cfg = self._setup(d)
            summary = m.run_headless(
                db_path=db, config_path=cfg,
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertIn("cohort", summary)
            self.assertTrue(any("-ics" in " ".join(c) for c in calls))

    def test_headless_never_raises_on_bad_config(self):
        with tempfile.TemporaryDirectory() as d:
            summary = m.run_headless(
                db_path=os.path.join(d, "missing.db"),
                config_path=os.path.join(d, "no_such_config.json"),
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0)
            self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    unittest.main()
