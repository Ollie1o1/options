"""Ledger paths resolve against the repo root, not the process CWD.

Worse than the config case, because these are WRITTEN. A relative
``"paper_trades.db"`` meant a run started from another directory would create a
second, empty ledger there and log real trades into it, while `check_pnl` from
the repo root kept showing the old book. Nothing would look broken from either
side.

One hazard is deliberate and load-bearing, so it is pinned below: anchoring
removes `os.chdir` as a way to sandbox a ledger write. `tests/leverage/test_cli.py`
relied on exactly that, and now patches `paper.DEFAULT_LEDGER_DB` instead. Any
future test that logs a trade must inject its path — a chdir will no longer
protect the real book.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.paths import PROJECT_ROOT, repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestProjectRoot(unittest.TestCase):

    def test_project_root_is_the_repo(self):
        self.assertEqual(PROJECT_ROOT, REPO_ROOT)
        self.assertTrue((PROJECT_ROOT / "config.json").exists())


class TestLedgerDefaultsAreAnchored(unittest.TestCase):
    """Each default is asserted directly, not via an injected path — the
    default is the branch that ships."""

    def test_paper_manager_anchors_a_relative_path(self):
        from src.paper_manager import PaperManager
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "x.db")
            self.assertEqual(PaperManager(db_path=db).db_path, db,
                             "an absolute path must pass through untouched")

    def test_paper_manager_resolves_through_repo_path(self):
        """Proves the wiring WITHOUT opening the real ledger.

        This assertion used to construct `PaperManager(db_path=REPO_ROOT /
        "paper_trades.db")`, and PaperManager runs schema migrations in
        __init__ — so the test silently migrated the live 947-trade book on
        every suite run. It was caught by a checksum, not by anything failing.
        A test must never name the real ledger, even to read it.
        """
        from unittest.mock import patch

        import src.paper_manager as pm_mod
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "t.db")
            with patch.object(pm_mod, "repo_path",
                              side_effect=pm_mod.repo_path) as spy:
                pm = pm_mod.PaperManager(db_path=db)
            spy.assert_any_call(db)
            self.assertEqual(pm.db_path, db)

    def test_module_level_defaults(self):
        from src.breakout.data import DEFAULT_DB as BREAKOUT_DB
        from src.check_pnl import DB_PATH
        from src.leverage.paper import DEFAULT_LEDGER_DB
        from src.longterm.fills import DEFAULT_DB as LONGTERM_DB
        from src.squeeze.backtest import DEFAULT_DB, PRICES_DB, SHARES_DB
        from src.structure.margins import DEFAULT_DB as MARGINS_DB

        for name, value in [
            ("check_pnl.DB_PATH", DB_PATH),
            ("leverage.paper.DEFAULT_LEDGER_DB", DEFAULT_LEDGER_DB),
            ("structure.margins.DEFAULT_DB", MARGINS_DB),
            ("breakout.data.DEFAULT_DB", BREAKOUT_DB),
            ("longterm.fills.DEFAULT_DB", LONGTERM_DB),
            ("squeeze.backtest.DEFAULT_DB", DEFAULT_DB),
            ("squeeze.backtest.PRICES_DB", PRICES_DB),
            ("squeeze.backtest.SHARES_DB", SHARES_DB),
        ]:
            with self.subTest(default=name):
                self.assertTrue(os.path.isabs(str(value)),
                                f"{name} is relative: {value!r}")
                self.assertTrue(str(value).startswith(str(REPO_ROOT)),
                                f"{name} points outside the repo: {value!r}")


class TestLeverageLedgerSandbox(unittest.TestCase):
    """Pins the regression that anchoring introduced.

    Before, a chdir into a temp directory was enough to keep a logged trade out
    of the real ledger. It is not any more, and the failure mode is writing to
    the user's actual book — so the patchable seam has to keep existing.
    """

    def test_default_ledger_db_is_patchable(self):
        from src.leverage import paper as P
        original = P.DEFAULT_LEDGER_DB
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = os.path.join(tmp, "sandbox.db")
                P.DEFAULT_LEDGER_DB = target
                ledger = P.PaperLedger()
                self.assertEqual(ledger.db_path, target)
                self.assertTrue(os.path.exists(target),
                                "the ledger did not create its sandbox file")
        finally:
            P.DEFAULT_LEDGER_DB = original

    def test_chdir_alone_no_longer_redirects_the_ledger(self):
        """The whole point: this is why the leverage test had to change."""
        from src.leverage import paper as P
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                resolved = P.PaperLedger().db_path
            finally:
                os.chdir(cwd)
        self.assertEqual(Path(resolved), REPO_ROOT / "paper_trades_leverage.db")
        self.assertNotIn(tmp, resolved)

    def test_an_explicit_relative_path_still_anchors(self):
        from src.leverage.paper import PaperLedger
        self.assertEqual(Path(PaperLedger(str(REPO_ROOT / "paper_trades_leverage.db")).db_path),
                         REPO_ROOT / "paper_trades_leverage.db")


class TestDiagnosticsNeverCreateTheLedger(unittest.TestCase):
    """A read must not bring the thing it is reading into existence.

    `sqlite3.connect` on a missing path CREATES an empty database. Every gate
    and banner read did that, so a fresh clone's first scan left behind a
    zero-row paper_trades.db — which then reads as a real book with no trades
    rather than as an absent one. Read-only opens fail instead.
    """

    def _probe(self, fn):
        p = os.path.join(tempfile.mkdtemp(), "no_such.db")
        try:
            fn(p)
        except Exception:
            pass
        return os.path.exists(p)

    def test_checkpoint_does_not_create_it(self):
        from src.phase1_checkpoint import compute_checkpoint
        self.assertFalse(self._probe(lambda p: compute_checkpoint(p, "2026-01-01")))

    def test_open_cohort_count_does_not_create_it(self):
        from src.maintenance import _open_cohort_count
        self.assertFalse(self._probe(lambda p: _open_cohort_count(p, "2026-01-01")))

    def test_preflight_does_not_create_it(self):
        from src.execution.preflight import run_preflight
        self.assertFalse(self._probe(
            lambda p: run_preflight(db_path=p, config_path="config.json")))

    def test_a_missing_ledger_still_reads_as_UNAVAILABLE_not_as_empty(self):
        """Load-bearing distinction. Swallowing the error and returning an
        empty cohort turns 'no ledger' into 'ledger with no evidence', and the
        gate then says GATHERING when it should say do-not-trade."""
        from src.execution.preflight import run_preflight
        p = os.path.join(tempfile.mkdtemp(), "no_such.db")
        result, checks = run_preflight(db_path=p, config_path="config.json")
        gate = next(c for c in checks if c.name == "gate")
        self.assertIn("unavailable", gate.detail)
        self.assertFalse(result["cleared"])


class TestRepoPathAcceptsBothShapes(unittest.TestCase):

    def test_str_and_path_relative(self):
        self.assertEqual(repo_path("a.db"), str(REPO_ROOT / "a.db"))
        self.assertEqual(repo_path(Path("a.db")), str(REPO_ROOT / "a.db"))

    def test_str_and_path_absolute(self):
        self.assertEqual(repo_path("/tmp/a.db"), "/tmp/a.db")
        self.assertEqual(repo_path(Path("/tmp/a.db")), "/tmp/a.db")


if __name__ == "__main__":
    unittest.main()
