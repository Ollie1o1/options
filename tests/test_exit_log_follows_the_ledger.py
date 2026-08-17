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


class TestAScanNeverOpensTheLedgerWritable(unittest.TestCase):
    """A scan reads the book; it must not hold a write handle on it.

    `RiskAggregator._load_open_trades` ran `sqlite3.connect(self.db_path)` — a
    READ-WRITE open — to run a single SELECT. SQLite rewrites the file header
    on a read-write open even when nothing changes, so a plain `run_scan`
    altered `paper_trades.db`'s checksum while leaving all 999 rows and every
    status identical. Traced 2026-08-17: one read-write open per scan against
    three read-only ones.

    Harmless in itself, and it defeats the property the whole session leaned
    on — that a scan cannot touch the book — and leaves a write handle open on
    production data during an operation that has no business holding one.
    """

    def test_the_open_trades_reader_is_read_only(self):
        with open(repo_path("src/portfolio_risk.py")) as fh:
            src = fh.read()
        i = src.index("def _load_open_trades(")
        body = src[i:src.index("\n    def ", i + 10)]
        self.assertIn("mode=ro", body,
                      "the scan path holds a WRITE handle on the real ledger")

    def test_it_still_reads_open_trades(self):
        import os
        import sqlite3
        import tempfile
        from src.portfolio_risk import RiskAggregator
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (ticker TEXT, status TEXT)")
        conn.executemany("INSERT INTO trades VALUES (?,?)",
                         [("SPY", "OPEN"), ("QQQ", "CLOSED"), ("F", "OPEN")])
        conn.commit(); conn.close()
        rows = RiskAggregator(db_path=path)._load_open_trades()
        self.assertEqual(sorted(r["ticker"] for r in rows), ["F", "SPY"])

    def test_a_missing_ledger_is_survivable(self):
        from src.portfolio_risk import RiskAggregator
        self.assertEqual(RiskAggregator(db_path="/nonexistent/none.db")._load_open_trades(), [])
