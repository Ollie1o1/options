"""Wiring tests — the recorder is actually called by the scan path.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_record_hooks -v

A source grep is not a rendering test. These call the real functions and
assert on what lands in a temp database. No test names the real ledger or the
real data/candidates.db.
"""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src import candidate_record as cr
from src import options_screener as osx


def _leg(**over):
    row = {"symbol": "AAPL", "strategy_name": "Long Call", "type": "call",
           "expiration": "2026-09-18", "strike": 190.0,
           "bid": 9.90, "ask": 10.10, "premium": 10.0, "theta": -0.05,
           "quality_score": 0.50, "ev_per_contract": 25.0}
    row.update(over)
    return row


class _TempDB(unittest.TestCase):
    """Points the recorder's default path at a temp file for the test."""

    def setUp(self):
        cr.reset_stats()
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "c.db")
        # Patch the env var, not DEFAULT_DB_PATH: the env var outranks it, so
        # patching the constant alone would leave these writes going to the
        # runner's shared temp database and this test would assert against an
        # empty file it created itself.
        patcher = mock.patch.dict(os.environ, {cr.DB_PATH_ENV: self.path})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._dir.cleanup)

    def rows(self, columns="*"):
        with sqlite3.connect(self.path) as conn:
            return conn.execute(f"select {columns} from candidates").fetchall()


class TestGateAndReportRecords(_TempDB):
    def test_a_gated_board_lands_in_the_database(self):
        df = pd.DataFrame([_leg(), _leg(strike=195.0)])
        with cr.scan("test"):
            osx.gate_and_report(df, "discover", verbose=False)
        self.assertEqual(len(self.rows()), 2)

    def test_refused_rows_are_recorded_with_their_reason(self):
        # The refused population is the entire point of this table.
        df = pd.DataFrame([_leg(ev_per_contract=-500.0)])
        with cr.scan("test"):
            osx.gate_and_report(df, "discover", verbose=False)
        got = self.rows("gate_passed, refused_by")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0], 0)
        self.assertIsNotNone(got[0][1])

    def test_the_board_name_is_recorded(self):
        with cr.scan("test"):
            osx.gate_and_report(pd.DataFrame([_leg()]), "condors", verbose=False)
        self.assertEqual(self.rows("board")[0][0], "condors")

    def test_a_recorder_failure_cannot_break_a_scan(self):
        df = pd.DataFrame([_leg()])
        with mock.patch.object(cr, "record_board_rows",
                               side_effect=RuntimeError("boom")):
            with cr.scan("test"):
                kept = osx.gate_and_report(df, "discover", verbose=False)
        self.assertIsNotNone(kept)          # the scan survives
        self.assertEqual(cr.STATS["errors"], 1)   # and says so

    def test_an_empty_board_records_nothing_and_does_not_error(self):
        with cr.scan("test"):
            osx.gate_and_report(pd.DataFrame(), "discover", verbose=False)
        self.assertEqual(cr.STATS["errors"], 0)


class TestRunScanOpensTheContext(_TempDB):
    """Drives the real run_scan far enough to prove it opens one scan_id.

    Uses run_scan, NOT main() — main() enforces exits against the real book.
    `_reset_scan_diagnostics` is the first call in run_scan's body, so probing
    there proves the context is open for the whole function.
    """

    def test_run_scan_opens_a_non_orphan_scan_id(self):
        seen = {}

        class _Stop(Exception):
            pass

        def _probe():
            seen["id"] = cr._SCAN_ID.get()
            raise _Stop

        with mock.patch.object(osx, "_reset_scan_diagnostics",
                               side_effect=_probe):
            with self.assertRaises(_Stop):
                osx.run_scan("Discovery", ["AAPL"], None, 1, 7, 45, "balanced",
                             mock.MagicMock(), "neutral", "normal")

        self.assertIsNotNone(seen["id"])
        self.assertIn("Discovery", seen["id"])
        self.assertNotIn("orphan", seen["id"])

    def test_the_context_does_not_leak_after_run_scan_returns(self):
        class _Stop(Exception):
            pass

        with mock.patch.object(osx, "_reset_scan_diagnostics",
                               side_effect=_Stop):
            with self.assertRaises(_Stop):
                osx.run_scan("Discovery", ["AAPL"], None, 1, 7, 45, "balanced",
                             mock.MagicMock(), "neutral", "normal")
        self.assertIsNone(cr._SCAN_ID.get())

    def test_run_scan_keeps_its_signature(self):
        import inspect
        params = list(inspect.signature(osx.run_scan).parameters)
        self.assertEqual(params[0], "mode")
        self.assertIn("session_budget", params)
        self.assertIn("custom_weights", params)


if __name__ == "__main__":
    unittest.main()
