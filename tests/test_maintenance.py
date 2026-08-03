import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from src import maintenance as m


class TestThrottle(unittest.TestCase):
    def test_checkpoint_due_after_7_days(self):
        self.assertTrue(m.due_checkpoint({"last_checkpoint": "2026-05-31"}, "2026-06-07"))

    def test_checkpoint_not_due_within_7_days(self):
        self.assertFalse(m.due_checkpoint({"last_checkpoint": "2026-06-02"}, "2026-06-07"))

    def test_checkpoint_due_when_never_run(self):
        self.assertTrue(m.due_checkpoint({}, "2026-06-07"))

    def test_walk_forward_due_after_30_days(self):
        self.assertTrue(m.due_walk_forward({"last_walk_forward": "2026-05-01"}, "2026-05-31"))

    def test_walk_forward_not_due_within_30_days(self):
        self.assertFalse(m.due_walk_forward({"last_walk_forward": "2026-05-10"}, "2026-05-31"))

    def test_walk_forward_due_when_never_run(self):
        self.assertTrue(m.due_walk_forward({}, "2026-06-07"))


class TestDueAutologWindows(unittest.TestCase):
    """Catch-up model: a single market-hours launch returns EVERY working
    strategy not yet logged today, decoupled from per-clock slots."""

    KEYS = {"ds", "sps", "ss", "ics"}

    def _keys(self, windows):
        return {w[0] for w in windows}

    def test_all_working_strategies_due_on_fresh_weekday(self):
        # 1030 Wednesday, empty state -> all four working strategies are due.
        got = m.due_autolog_windows({}, weekday=3, hhmm=1030, today="2026-06-24")
        self.assertEqual(self._keys(got), self.KEYS)

    def test_short_put_window_included(self):
        # The user-requested fix: short puts (-ss) must be a logged strategy.
        got = m.due_autolog_windows({}, weekday=3, hhmm=1400, today="2026-06-24")
        flags = {w[1] for w in got}
        self.assertIn("-ss", flags)

    def test_already_logged_windows_excluded(self):
        st = {"last_autolog": {"ds": "2026-06-24", "ics": "2026-06-24"}}
        got = m.due_autolog_windows(st, weekday=3, hhmm=1400, today="2026-06-24")
        self.assertEqual(self._keys(got), {"sps", "ss"})

    def test_empty_outside_rth_band(self):
        self.assertEqual(m.due_autolog_windows({}, weekday=3, hhmm=945, today="2026-06-24"), [])
        self.assertEqual(m.due_autolog_windows({}, weekday=3, hhmm=1630, today="2026-06-24"), [])

    def test_empty_on_weekend(self):
        self.assertEqual(m.due_autolog_windows({}, weekday=6, hhmm=1200, today="2026-06-27"), [])

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

    def test_autolog_catches_up_all_working_strategies(self):
        # 2026-06-04 is a Thursday, 14:30 (in RTH band): one launch logs ALL
        # four working strategies, not just the clock-matched one.
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            state_path = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=state_path,
                now=datetime(2026, 6, 4, 14, 30), runner=fake_runner,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            flags = {f for c in calls for f in c if f in ("-ds", "-sps", "-ss", "-ics")}
            self.assertEqual(flags, {"-ds", "-sps", "-ss", "-ics"})
            st = m.load_state(state_path)
            self.assertEqual(set(st["last_autolog"]), {"ds", "sps", "ss", "ics"})
            self.assertEqual(st["last_autolog"]["ics"], "2026-06-04")
            self.assertIn("cohort", summary)

    def test_background_spawns_detached_and_skips_inline_runner(self):
        # Interactive startup must NOT block on scans: it spawns a detached
        # catch-up instead of calling the runner inline.
        calls, spawned = [], []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                background=True, spawn_fn=lambda: spawned.append(True),
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            self.assertEqual(calls, [])           # no inline scans
            self.assertEqual(len(spawned), 1)     # detached catch-up fired once
            self.assertTrue(any("queued" in r for r in summary["ran"]))

    def test_run_catchup_runs_due_windows_and_records_state(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "state.json")
            # Isolated lock path: the default is shared with any catch-up
            # actually running on this machine, which would refuse this one.
            summary = m.run_catchup(
                state_path=sp, now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                swing_fn=lambda: None,
                lock_path=os.path.join(d, "catchup.lock"))
            self.assertEqual(set(summary["ran"]), {"ds", "sps", "ss", "ics"})
            self.assertEqual(set(m.load_state(sp)["last_autolog"]),
                             {"ds", "sps", "ss", "ics"})

    def test_catchup_takes_the_swing_hook_by_injection(self):
        """The crypto swing track calls Binance over the network.

        Every other side effect in this module goes through an injectable hook
        precisely so tests stay offline; this one was called directly, so any
        test of run_catchup made a live HTTP request. A socket spy over the
        suite found exactly these tests reaching fetch_klines_binance, which is
        both a flake source (the run is only as reliable as Binance) and why the
        suite occasionally took minutes instead of seconds.
        """
        seen = []
        with tempfile.TemporaryDirectory() as d:
            summary = m.run_catchup(
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: 0,
                swing_fn=lambda: (seen.append(1), {"opened": 1, "closed": 0})[1],
                lock_path=os.path.join(d, "catchup.lock"))
            self.assertEqual(seen, [1], "swing hook was not the injected one")
            self.assertIn("swing-paper", summary["ran"])

    def test_a_failing_swing_hook_cannot_touch_the_options_windows(self):
        # Crypto is a satellite: it must never jeopardise the options cohort.
        def _boom():
            raise RuntimeError("exchange down")

        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "state.json")
            summary = m.run_catchup(
                state_path=sp, now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: 0, swing_fn=_boom,
                lock_path=os.path.join(d, "catchup.lock"))
            self.assertEqual(set(summary["ran"]), {"ds", "sps", "ss", "ics"})

    def test_ds_window_feeds_cohort_with_dte_floor(self):
        # 10:30 Thursday = 'ds' window; the scan must carry --min-dte so its
        # Long Calls are gate-eligible (>=30 DTE), not paper_only data.
        calls = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27",
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 10, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: 0, walk_forward_fn=lambda **k: None)
            ds_cmds = [c for c in calls if "-ds" in c]
            self.assertEqual(len(ds_cmds), 1)
            joined = " ".join(ds_cmds[0])
            self.assertIn("--min-dte", joined)
            self.assertIn("30", joined.split("--min-dte")[1])

    def test_ics_window_has_no_dte_override(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27",
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: 0, walk_forward_fn=lambda **k: None)
            ics_cmds = [c for c in calls if "-ics" in c]
            self.assertEqual(len(ics_cmds), 1)
            self.assertNotIn("--min-dte", " ".join(ics_cmds[0]))

    def test_second_run_same_day_skips_autolog(self):
        calls = []
        def fake_runner(cmd):
            calls.append(cmd); return 0
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            kw = dict(db_path=db, phase1_start="2026-05-27", state_path=sp,
                      now=datetime(2026, 6, 4, 14, 30), runner=fake_runner,
                      checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                      walk_forward_fn=lambda **k: None)
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
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
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
                track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
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
                track_record_fn=lambda **k: ran.append(k), chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            # injected fn was used (no real reports/TRACK_RECORD.md written)
            self.assertEqual(len(ran), 1)
            self.assertEqual(ran[0]["db_path"], db)
            self.assertEqual(m.load_state(sp)["last_track_record"], "2026-06-07")
            self.assertIn("track_record", summary["ran"])

    def test_walk_forward_runs_when_due_and_is_injectable(self):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 9, 0), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: ran.append(k))
            # injected fn was used (no real walk-forward artifact written)
            self.assertEqual(len(ran), 1)
            self.assertEqual(ran[0]["db_path"], db)
            self.assertEqual(m.load_state(sp)["last_walk_forward"], "2026-06-07")
            self.assertIn("walk-forward", summary["ran"])

    def test_walk_forward_throttled_within_30_days(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            m.save_state(sp, {"last_walk_forward": "2026-06-01"})
            calls = []
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 9, 0), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: calls.append(k))
            self.assertEqual(calls, [])
            self.assertNotIn("walk-forward", summary["ran"])
            self.assertEqual(m.load_state(sp)["last_walk_forward"], "2026-06-01")

    def test_walk_forward_failure_does_not_propagate(self):
        def boom(**k):
            raise RuntimeError("scipy blew up")
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 7, 9, 0), runner=lambda cmd: 0,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: 0, walk_forward_fn=boom)
            self.assertIn("cohort", summary)
            self.assertNotIn("walk-forward", summary["ran"])

    def test_runner_exception_does_not_propagate(self):
        def boom(cmd):
            raise RuntimeError("subprocess blew up")
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); self._db(db)
            sp = os.path.join(d, "state.json")
            summary = m.run_startup_maintenance(
                db_path=db, phase1_start="2026-05-27", state_path=sp,
                now=datetime(2026, 6, 4, 14, 30), runner=boom,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
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

    def test_child_does_not_inherit_parent_stdin(self):
        """Catch-up children must see EOF on stdin, never the parent's terminal.

        Observed 2026-07-28: the detached catch-up inherited the tty the user
        was typing into, so `-ds`/`-sps` raced the interactive screener for
        keystrokes, ate a stray 'b' at the ticker-source prompt and died with
        `invalid literal for int() with base 10: 'b'`. prompt_input only falls
        back to its default when stdin is not a tty, so isolation has to happen
        here, at the spawn."""
        script = "import sys; sys.stdout.write('GOT:' + repr(sys.stdin.read()))"
        with tempfile.TemporaryDirectory() as d:
            feed = os.path.join(d, "feed.txt")
            with open(feed, "w") as f:
                f.write("b\n")
            old_cwd = os.getcwd()
            saved_stdin = os.dup(0)
            try:
                os.chdir(d)
                with open(feed) as f:
                    os.dup2(f.fileno(), 0)
                rc = m._default_runner([sys.executable, "-c", script])
            finally:
                os.dup2(saved_stdin, 0)
                os.close(saved_stdin)
                os.chdir(old_cwd)
            with open(os.path.join(d, "logs", "maintenance.log")) as f:
                log = f.read()
        self.assertEqual(rc, 0)
        self.assertIn("GOT:''", log)

    def test_detached_catchup_does_not_inherit_parent_stdin(self):
        """The detached catch-up must be isolated at its own spawn too.

        Isolating only _default_runner covers the grandchildren; the catch-up
        process itself is spawned by _spawn_catchup_detached, and without
        stdin=DEVNULL it keeps the user's tty on fd 0 for its whole (minutes
        long) life — confirmed with lsof on 2026-07-28. Anything it later reads
        comes straight out of what the user is typing into the screener."""
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen.update(kwargs)
            return None

        with mock.patch.object(m.subprocess, "Popen", fake_popen):
            m._spawn_catchup_detached()
        self.assertEqual(seen.get("stdin"), m.subprocess.DEVNULL)

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
                    checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            finally:
                if old is None:
                    os.environ.pop(m.CHILD_ENV_MARKER, None)
                else:
                    os.environ[m.CHILD_ENV_MARKER] = old
            self.assertEqual(calls, [])           # no recursive spawn
            self.assertEqual(summary["ran"], [])  # no jobs run in the child


class TestCatchupSingleInstance(unittest.TestCase):
    """Only one catch-up may run at a time.

    Observed 2026-07-28: a stalled window left state unmarked, so every new
    screener launch spawned another detached catch-up. Six piled up and five
    ran `-ics` concurrently; the overlapping `-sps` runs logged INTC 85p twice
    (entry_ids 825/829) because each re-fetch shifted the price enough to slip
    past the DB-layer dedup. Concurrency has to be refused at the source."""

    def test_second_catchup_skips_windows_while_one_is_running(self):
        import fcntl
        calls = []
        with tempfile.TemporaryDirectory() as d:
            lock_path = os.path.join(d, "catchup.lock")
            held = open(lock_path, "w")
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                res = m.run_catchup(
                    state_path=os.path.join(d, "state.json"),
                    now=datetime(2026, 6, 4, 14, 30),  # Thursday, in RTH band
                    runner=lambda cmd: (calls.append(cmd), 0)[1],
                    lock_path=lock_path)
            finally:
                fcntl.flock(held, fcntl.LOCK_UN)
                held.close()
        self.assertEqual(calls, [])      # ran no scans
        self.assertEqual(res["ran"], [])

    def test_catchup_runs_windows_when_lock_is_free(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            res = m.run_catchup(
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 14, 30),
                runner=lambda cmd: (calls.append(cmd), 0)[1],
                swing_fn=lambda: None,   # offline: the real hook calls Binance
                lock_path=os.path.join(d, "catchup.lock"))
        flags = {f for c in calls for f in c if f in ("-ds", "-sps", "-ss", "-ics")}
        self.assertEqual(flags, {"-ds", "-sps", "-ss", "-ics"})
        self.assertEqual(set(res["ran"]) & {"ds", "sps", "ss", "ics"},
                         {"ds", "sps", "ss", "ics"})


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
                      chain_archive_fn=lambda: hits.append(1) or 7,
                      walk_forward_fn=lambda **k: None)
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
                chain_archive_fn=lambda: hits.append(1) or 0,
                walk_forward_fn=lambda **k: None)
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
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
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
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            self.assertIsInstance(summary, dict)

    def test_headless_enforces_exits_every_run(self):
        """Exit enforcement must run directly on every headless invocation,
        not only transitively when an auto-log window happens to be open —
        otherwise trades that hit a stop/take-profit/time-exit on a day with
        no auto-log window (or while the cron/LaunchAgent skips auto-log)
        never close."""
        enforced = []
        with tempfile.TemporaryDirectory() as d:
            db, cfg = self._setup(d)
            # 09:00 weekday: NOT an auto-log window, so no child screener spawns.
            summary = m.run_headless(
                db_path=db, config_path=cfg,
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 9, 0),
                runner=lambda cmd: 0,
                enforce_exits_fn=lambda **k: enforced.append(k),
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            self.assertEqual(len(enforced), 1, "exits must be enforced exactly once per headless run")
            self.assertEqual(enforced[0]["db_path"], db)

    def test_headless_exit_enforcement_failure_does_not_propagate(self):
        with tempfile.TemporaryDirectory() as d:
            db, cfg = self._setup(d)
            def _boom(**k):
                raise RuntimeError("yfinance down")
            summary = m.run_headless(
                db_path=db, config_path=cfg,
                state_path=os.path.join(d, "state.json"),
                now=datetime(2026, 6, 4, 9, 0),
                runner=lambda cmd: 0,
                enforce_exits_fn=_boom,
                checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None, chain_archive_fn=lambda: 0,
                walk_forward_fn=lambda **k: None)
            self.assertIsInstance(summary, dict)


class TestCheckpointJobUsesTheBudget(unittest.TestCase):
    """The weekly checkpoint reports the affordable book, not the nominal one.

    Uncapped, the checkpoint's reporting sections describe positions the
    account could never have opened — the affordable subset disappears and the
    short-premium block is dominated by five-figure positions that flip its
    headline return negative. The scheduled job must pass the same budget the
    CLI entry point passes.
    """

    def _config(self, d, cap=4000):
        path = os.path.join(d, "config.json")
        auto_log = {"phase1_start_date": "2026-05-27"}
        if cap is not None:
            auto_log["max_capital_at_risk"] = cap
        with open(path, "w") as f:
            json.dump({"auto_log": auto_log}, f)
        return path

    def test_reads_the_budget_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(m._max_capital_at_risk(self._config(d, 4000)), 4000.0)

    def test_budget_is_none_when_unset_or_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(m._max_capital_at_risk(self._config(d, None)))
            self.assertIsNone(m._max_capital_at_risk(os.path.join(d, "nope.json")))

    def test_checkpoint_job_passes_the_budget_through(self):
        seen = {}

        def _compute(db_path, phase1_start, **kw):
            seen.update(db_path=db_path, phase1_start=phase1_start, **kw)
            return {"decision": "GATHERING"}

        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(d, 4000)
            with mock.patch.object(m.phase1_checkpoint, "compute_checkpoint", _compute), \
                    mock.patch.object(m.phase1_checkpoint, "write_checkpoint",
                                      lambda *a, **k: None):
                m._run_checkpoint(db_path=os.path.join(d, "t.db"),
                                  phase1_start="2026-05-27", config_path=cfg)
        self.assertEqual(seen["max_capital_at_risk"], 4000.0)

    def test_checkpoint_job_passes_none_when_no_budget_is_set(self):
        seen = {}

        def _compute(db_path, phase1_start, **kw):
            seen.update(kw)
            return {"decision": "GATHERING"}

        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(d, None)
            with mock.patch.object(m.phase1_checkpoint, "compute_checkpoint", _compute), \
                    mock.patch.object(m.phase1_checkpoint, "write_checkpoint",
                                      lambda *a, **k: None):
                m._run_checkpoint(db_path=os.path.join(d, "t.db"),
                                  phase1_start="2026-05-27", config_path=cfg)
        self.assertIsNone(seen["max_capital_at_risk"])


class TestLogRotation(unittest.TestCase):
    """`_default_runner` appends to logs/maintenance.log and nothing ever
    truncated it. Three fires per weekday for months took it to 11MB — a slow
    leak on a machine whose whole point is running unattended. Nothing reads
    the file, so the old bytes are kept compressed rather than discarded.
    """

    def _write(self, path, size):
        with open(path, "w") as f:
            f.write("x" * size)

    def test_an_oversized_log_is_rotated_before_the_next_write(self):
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            try:
                os.chdir(d)
                os.makedirs("logs", exist_ok=True)
                path = os.path.join("logs", "maintenance.log")
                self._write(path, m.MAX_LOG_BYTES + 1)
                m._rotate_log_if_large(path)
                self.assertTrue(os.path.getsize(path) == 0
                                or not os.path.exists(path))
            finally:
                os.chdir(old_cwd)

    def test_rotation_keeps_the_old_bytes_compressed(self):
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            try:
                os.chdir(d)
                os.makedirs("logs", exist_ok=True)
                path = os.path.join("logs", "maintenance.log")
                self._write(path, m.MAX_LOG_BYTES + 1)
                m._rotate_log_if_large(path)
                archives = [f for f in os.listdir("logs") if f.endswith(".gz")]
                self.assertEqual(len(archives), 1,
                                 "the rotated content must be kept, not dropped")
            finally:
                os.chdir(old_cwd)

    def test_a_small_log_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            try:
                os.chdir(d)
                os.makedirs("logs", exist_ok=True)
                path = os.path.join("logs", "maintenance.log")
                self._write(path, 128)
                m._rotate_log_if_large(path)
                self.assertEqual(os.path.getsize(path), 128)
                self.assertEqual([f for f in os.listdir("logs")
                                  if f.endswith(".gz")], [])
            finally:
                os.chdir(old_cwd)

    def test_a_missing_log_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            m._rotate_log_if_large(os.path.join(d, "nope.log"))  # must not raise

    def test_rotation_never_stops_the_run(self):
        # Logging hygiene must not be the reason maintenance fails. A path that
        # cannot be rotated (here: a directory) is skipped, not raised.
        with tempfile.TemporaryDirectory() as d:
            blocked = os.path.join(d, "a-directory")
            os.makedirs(blocked)
            m._rotate_log_if_large(blocked)  # must not raise


if __name__ == "__main__":
    unittest.main()
