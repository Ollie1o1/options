"""The exit-enforcement log belongs beside the ledger it describes.

`_render_regime_with_exit_enforcement` wrote to a hardcoded
`"logs/enforce_exits.log"` after every `pm.update_positions()`. `health.py`
reads that exact file and treats its MTIME as proof the exit enforcer ran on
the real book, with a 4-day staleness window and the hint "open positions may
not be closing".

The suite drives `main()` in 16 tests. Their PaperManagers are properly
sandboxed — checked 2026-08-17, all 12 `update_positions()` calls in a full
run went to tempfile ledgers and none touched the real book — but every one of
them stamped the PRODUCTION log on the way past. 180 entries were written on
2026-08-17 alone, in bursts of 16 that line up exactly with suite runs.

So the ledger was never at risk; the HEALTH SIGNAL was. A test run refreshes
the mtime the check reads, which can make a dead exit-enforcer look alive for
up to four days — the same class as the `scan_errors.log` debris already
recorded in [[project_scan_errors_test_debris]].

Fixed by deriving the log path from `pm` rather than hardcoding it: the log
follows the ledger. A run against the real book still writes `logs/`, and a
run against a tempfile writes beside that tempfile. No test-awareness in
production code — nothing here checks for PYTEST or an env var, because a
sandboxed ledger is already the thing that distinguishes the two cases.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from src.options_screener import _exit_log_path
from src.paths import repo_path


class _PM:
    def __init__(self, db_path):
        self.db_path = db_path


class TestTheLogFollowsTheLedger(unittest.TestCase):

    def test_the_real_ledger_still_writes_the_repo_log(self):
        pm = _PM(repo_path("paper_trades.db"))
        self.assertEqual(os.path.abspath(_exit_log_path(pm)),
                         os.path.abspath(repo_path(os.path.join(
                             "logs", "enforce_exits.log"))))

    def test_a_relative_path_to_the_real_ledger_also_counts(self):
        """`PaperManager("paper_trades.db")` is how the app constructs it."""
        pm = _PM("paper_trades.db")
        self.assertEqual(os.path.abspath(_exit_log_path(pm)),
                         os.path.abspath(repo_path(os.path.join(
                             "logs", "enforce_exits.log"))))

    def test_a_tempfile_ledger_writes_beside_itself(self):
        d = tempfile.mkdtemp()
        pm = _PM(os.path.join(d, "trades.db"))
        got = os.path.abspath(_exit_log_path(pm))
        self.assertEqual(os.path.dirname(got), os.path.abspath(d))
        self.assertEqual(os.path.basename(got), "enforce_exits.log")

    def test_a_tempfile_ledger_never_names_the_production_log(self):
        """The property that matters: the health signal stays untouched."""
        real = os.path.abspath(repo_path(os.path.join("logs", "enforce_exits.log")))
        for name in ("trades.db", "ledger.db", "t.db", "missing.db"):
            pm = _PM(os.path.join(tempfile.mkdtemp(), name))
            self.assertNotEqual(os.path.abspath(_exit_log_path(pm)), real)

    def test_a_manager_with_no_db_path_writes_nothing(self):
        """`PaperManager.__init__` ALWAYS sets `db_path`, so its absence means
        a test double — and a stub cannot have enforced exits on the real book.

        This assertion was the other way round on the first attempt, on the
        reasoning that a spurious freshness stamp beats a missing one.
        Measured, that was backwards: 17 of the 24 write-site calls in a suite
        run have no `db_path`, so treating "unknown" as "real" left the
        production log stamped 16 times per run and fixed nothing.
        """
        self.assertIsNone(_exit_log_path(object()))


class TestHealthStillReadsTheRepoLog(unittest.TestCase):
    """The reader must not drift from the writer."""

    def test_the_health_check_points_at_the_same_file(self):
        with open(repo_path("src/health.py")) as fh:
            src = fh.read()
        self.assertIn('_p("logs", "enforce_exits.log")', src)


class TestItIsWiredIn(unittest.TestCase):

    def test_the_writer_no_longer_hardcodes_the_path(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertNotIn('open("logs/enforce_exits.log"', src,
                         "the enforcement log is hardcoded again — a suite "
                         "run will stamp the production health signal")
        self.assertIn("_exit_log_path(pm)", src)


if __name__ == "__main__":
    unittest.main()
