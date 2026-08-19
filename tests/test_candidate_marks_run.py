"""The daily marking run, end to end.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_marks_run -v

No network: the quote fetcher is injected. No real ledger, database or config.
"""
import os
import sqlite3
import tempfile
import unittest

from src import candidate_marks as cm
from tests.test_candidate_marks import _insert_candidate, _write_config


class TestDue(unittest.TestCase):
    def test_due_once_per_day(self):
        self.assertTrue(cm.due_candidate_marks({}, "2026-08-19"))
        self.assertFalse(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-19"}, "2026-08-19"))
        self.assertTrue(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-18"}, "2026-08-19"))

    def test_a_missing_state_is_due(self):
        self.assertTrue(cm.due_candidate_marks(None, "2026-08-19"))


class TestMarkCandidates(unittest.TestCase):
    def test_the_run_opens_marks_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            out = cm.mark_candidates(
                db_path=path, today="2026-08-19", cfg_path=cfg,
                fetch=lambda t, e: {(190.0, "call"): (11.0, 11.4)})
            self.assertEqual(out["opened"], 1)
            self.assertEqual(out["marked"], 1)
            self.assertEqual(out["closed"], 0)

    def test_a_position_opened_today_is_marked_today(self):
        # Opening must happen before marking, or a candidate waits a day for
        # its first mark and every entry-to-first-mark gap is silently wrong.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            cm.mark_candidates(db_path=path, today="2026-08-19", cfg_path=cfg,
                               fetch=lambda t, e: {(190.0, "call"): (11.0, 11.4)})
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_marks").fetchone()
            self.assertEqual(n, 1)

    def test_the_run_closes_a_position_that_hit_its_target(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            cm.mark_candidates(db_path=path, today="2026-08-01", cfg_path=cfg,
                               fetch=lambda t, e: {(190.0, "call"): (9.9, 10.1)})
            out = cm.mark_candidates(
                db_path=path, today="2026-08-10", cfg_path=cfg,
                fetch=lambda t, e: {(190.0, "call"): (24.9, 25.1)})
            self.assertEqual(out["closed"], 1)
            with sqlite3.connect(path) as conn:
                reason, = conn.execute(
                    "select exit_reason from candidate_positions").fetchone()
            self.assertEqual(reason, "take_profit")

    def test_a_broken_fetch_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)

            def boom(t, e):
                raise RuntimeError("boom")

            out = cm.mark_candidates(db_path=path, today="2026-08-19",
                                     cfg_path=cfg, fetch=boom)
            self.assertEqual(out["marked"], 0)
            self.assertEqual(out["opened"], 1)   # the rest of the run survives


if __name__ == "__main__":
    unittest.main()
