"""Scheduler hardening, 2026-11-18.

A scan hung for 12 days after the disk filled; an offline run stamped the
chain archive done with 1 of 22 symbols; the start-up health check called a
working auto-log dead. Each test below pins one of those fixes.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

# Never let the orchestrator tests reach the real shadow corpus.
os.environ.setdefault(
    "OPTIONS_CANDIDATE_DB",
    os.path.join(tempfile.gettempdir(), "options_test_candidates.db"))

from src import chain_archive as ca  # noqa: E402
from src import health  # noqa: E402
from src import maintenance as m  # noqa: E402
from src import maintenance_health as mh  # noqa: E402


def _chain(spot=100.0):
    return [{"expiration": "2026-12-18", "spot": spot, "strike": 100.0,
             "open_interest": 10, "option_type": "call"}]


class TestArchiveStats(unittest.TestCase):
    def _run(self, fetcher, symbols, **kw):
        with tempfile.TemporaryDirectory() as d:
            st = {}
            rows = ca.archive_symbols(symbols, db_path=os.path.join(d, "a.db"),
                                      today="2026-11-18", fetcher=fetcher,
                                      stats=st, **kw)
            return rows, st

    def test_counts_archived_and_failed(self):
        def fetch(sym):
            if sym == "BAD":
                raise RuntimeError("dns")
            return _chain()
        rows, st = self._run(fetch, ["SPY", "BAD", "QQQ"])
        self.assertEqual(rows, 2)
        self.assertEqual(st["archived"], 2)
        self.assertEqual(st["failed"], ["BAD"])
        self.assertFalse(st["aborted"])

    def test_empty_chain_counts_as_failed(self):
        rows, st = self._run(lambda s: [], ["SPY"])
        self.assertEqual(st["failed"], ["SPY"])

    def test_dead_feed_stops_early(self):
        calls = []
        def fetch(sym):
            calls.append(sym)
            raise OSError("offline")
        _, st = self._run(fetch, [f"S{i}" for i in range(20)],
                          max_consecutive_failures=6)
        self.assertTrue(st["aborted"])
        self.assertEqual(len(calls), 6)

    def test_complete_rules(self):
        self.assertTrue(ca.archive_complete(
            {"symbols": 22, "already": 0, "archived": 21, "failed": ["X"]}))
        # 2026-11-18: one symbol archived out of 22 — not complete.
        self.assertFalse(ca.archive_complete(
            {"symbols": 22, "already": 0, "archived": 1,
             "failed": [str(i) for i in range(21)]}))
        self.assertFalse(ca.archive_complete(
            {"symbols": 22, "already": 0, "archived": 5, "failed": [],
             "aborted": True}))
        # A retry that finds everything already archived is complete.
        self.assertTrue(ca.archive_complete(
            {"symbols": 22, "already": 22, "archived": 0, "failed": []}))
        self.assertFalse(ca.archive_complete({"symbols": 0}))


class _Orchestrator(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.d.name, "t.db")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, "
                     "status TEXT, paper_only INTEGER, quality_score REAL, "
                     "pnl_pct REAL)")
        conn.commit(); conn.close()
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

    def run_at(self, now, **kw):
        base: dict = dict(db_path=self.db, phase1_start=None, state_path=self.state,
                    now=now, runner=lambda cmd: 0,
                    checkpoint_fn=lambda **k: None,
                    track_record_fn=lambda **k: None,
                    morning_fn=lambda: 0, walk_forward_fn=lambda **k: None,
                    watch_fn=lambda db: None)
        base.update(kw)
        return m.run_startup_maintenance(**base)


class TestArchiveStamping(_Orchestrator):
    # 2026-11-18 is a Wednesday; 15:00 is inside the archive window.
    NOW = datetime(2026, 11, 18, 15, 0)

    def test_partial_run_is_not_stamped(self):
        res = {"rows": 3569, "symbols": 22, "already": 0, "archived": 1,
               "failed": [str(i) for i in range(21)], "complete": False}
        out = self.run_at(self.NOW, chain_archive_fn=lambda: res)
        self.assertNotIn("last_chain_archive", m.load_state(self.state))
        self.assertTrue(any("chain-archive:partial" in r for r in out["ran"]))

    def test_complete_run_is_stamped(self):
        res = {"rows": 20000, "symbols": 22, "already": 0, "archived": 22,
               "failed": [], "complete": True}
        self.run_at(self.NOW, chain_archive_fn=lambda: res)
        self.assertEqual(m.load_state(self.state)["last_chain_archive"],
                         "2026-11-18")

    def test_int_result_keeps_old_rule(self):
        self.run_at(self.NOW, chain_archive_fn=lambda: 5)
        self.assertEqual(m.load_state(self.state)["last_chain_archive"],
                         "2026-11-18")


class TestOffline(_Orchestrator):
    def test_offline_skips_network_jobs_unstamped(self):
        calls, archived = [], []
        out = self.run_at(
            datetime(2026, 11, 18, 15, 0),
            runner=lambda cmd: (calls.append(cmd), 0)[1],
            chain_archive_fn=lambda: archived.append(1) or 5,
            online_fn=lambda: False)
        self.assertEqual(calls, [])          # no scans, no morning build
        self.assertEqual(archived, [])
        st = m.load_state(self.state)
        self.assertNotIn("last_chain_archive", st)
        self.assertNotIn("last_autolog", st)
        self.assertNotIn("last_candidate_marks", st)
        self.assertIn("offline:network-jobs-deferred", out["ran"])

    def test_default_is_online(self):
        calls = []
        self.run_at(datetime(2026, 11, 18, 11, 0),
                    runner=lambda cmd: (calls.append(cmd), 0)[1],
                    chain_archive_fn=lambda: 0)
        self.assertEqual(len([c for c in calls if "run.py" in c]), 3)


class TestRunnerTimeout(unittest.TestCase):
    def test_hung_child_is_killed_and_returns_124(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                with mock.patch.object(m, "CHILD_TIMEOUT_S", 1):
                    rc = m._default_runner([sys.executable, "-c",
                                            "import time; time.sleep(30)"])
                with open(os.path.join("logs", "maintenance.log")) as f:
                    log = f.read()
            finally:
                os.chdir(cwd)
        self.assertEqual(rc, 124)
        self.assertIn("TIMEOUT", log)


class _FakeDatetime(datetime):
    fixed = datetime(2026, 11, 18, 20, 30)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed


class TestExitsRthOnly(unittest.TestCase):
    def _headless(self, when, rth_only):
        hits = []
        _FakeDatetime.fixed = when
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(m, "datetime", _FakeDatetime), \
                mock.patch.object(m, "run_startup_maintenance",
                                  return_value={"cohort": "", "ran": []}):
            m.run_headless(db_path=os.path.join(d, "t.db"),
                           config_path=os.path.join(d, "missing.json"),
                           enforce_exits_fn=lambda **k: hits.append(1),
                           exits_rth_only=rth_only)
        return hits

    def test_evening_run_skips_exits(self):
        self.assertEqual(self._headless(datetime(2026, 11, 18, 20, 30), True), [])

    def test_session_run_enforces_exits(self):
        self.assertEqual(self._headless(datetime(2026, 11, 18, 11, 0), True), [1])

    def test_default_keeps_old_behaviour(self):
        self.assertEqual(self._headless(datetime(2026, 11, 18, 20, 30), False), [1])


class TestDiskAndStuckHealth(unittest.TestCase):
    def test_disk_thresholds(self):
        self.assertIn("[OK]", mh.disk_health_lines(free_bytes=int(100e9))[0])
        self.assertIn("[WARN]", mh.disk_health_lines(free_bytes=int(20e9))[0])
        self.assertIn("[CRITICAL]", mh.disk_health_lines(free_bytes=int(8e9))[0])

    def test_etime_parse(self):
        self.assertAlmostEqual(mh._etime_hours("12-08:54:55"), 296.9, places=1)
        self.assertAlmostEqual(mh._etime_hours("01:30:00"), 1.5)
        self.assertAlmostEqual(mh._etime_hours("05:00"), 5 / 60)

    def test_flags_the_2026_11_06_hang(self):
        ps = (" 4771 12-08:54:55 /bin/bash /Users/x/options/scripts/auto_log_equity.sh\n"
              " 4817 12-08:54:59 /opt/homebrew/Python run.py -ss --5 --no-ai\n"
              "  900 02:00:00 /usr/bin/python -m src.maintenance\n"
              "  901 30-00:00:00 /usr/bin/python someone_elses/run.py\n")
        lines = mh.stuck_job_lines(ps_output=ps, repo="/Users/x/options")
        self.assertIn("[CRITICAL]", lines[0])
        self.assertIn("2 process", lines[0])
        self.assertFalse(any("901" in l for l in lines))

    def test_quiet_when_nothing_stuck(self):
        lines = mh.stuck_job_lines(ps_output=" 1 00:05 run.py -ds --no-ai\n",
                                   repo="/r")
        self.assertIn("[OK]", lines[0])


class TestStartupAutologCheck(unittest.TestCase):
    def test_reads_maintenance_window_stamp(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "logs"))
            with open(os.path.join(d, "logs", ".maintenance_state.json"), "w") as f:
                json.dump({"last_autolog": {"ds": "2026-11-18", "sps": "2026-11-17"}}, f)
            db = os.path.join(d, "t.db")
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE trades (date TEXT)")
            conn.execute("INSERT INTO trades VALUES ('2026-11-05')")
            conn.commit(); conn.close()
            checks = health.collect_checks(root=d, db_path=db,
                                           now=datetime(2026, 11, 18, 21, 0))
        autolog = [c for c in checks if c.label == "auto-log"][0]
        self.assertEqual(autolog.last.date().isoformat(), "2026-11-18")

    def test_missing_state_is_none(self):
        self.assertIsNone(health._last_autolog_window("/nonexistent/state.json"))


if __name__ == "__main__":
    unittest.main()
