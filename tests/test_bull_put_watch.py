"""The Bull Put cohort report: does the restored strategy clear its own bar?

Bull Put was switched off by accident on 2026-08-01 and restored 2026-08-18.
Its headline 66.4% over 131 closed trades was measured on the PRE-REPAIR EV
layer — before the spread-EV sign fix, the measured vol basis and the measured
error bar — so it is history, not a forecast. The trades entered from
2026-08-18 are the first ones measured by the repaired stack, and they are what
this report tracks.

The required rate is read from the ledger, never hardcoded. It is the MANAGED
figure — computed from realised payoffs under the exits actually used — and it
drifts as trades close. A report asserting a stale 50.9% would eventually be
comparing today's win rate against last month's bar.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import bull_put_watch as bpw

_COLS = ("entry_id, date, ticker, strategy_name, status, strike, long_strike, "
         "expiration, net_credit, capital_at_risk, pnl_usd, exit_date, exit_reason")


def _make_db(rows):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = sqlite3.connect(path)
    c.execute(f"CREATE TABLE trades ({_COLS.replace(', ', ' , ')})")
    for r in rows:
        c.execute(
            "INSERT INTO trades (entry_id,date,ticker,strategy_name,status,strike,"
            "long_strike,expiration,net_credit,capital_at_risk,pnl_usd,exit_date,"
            "exit_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.get("entry_id"), r.get("date"), r.get("ticker", "AAA"),
             r.get("strategy_name", "Bull Put"), r.get("status", "OPEN"),
             r.get("strike", 100.0), r.get("long_strike", 95.0),
             r.get("expiration", "2026-09-18"), r.get("net_credit", 1.0),
             r.get("capital_at_risk", 400.0), r.get("pnl_usd"),
             r.get("exit_date"), r.get("exit_reason")))
    c.commit()
    c.close()
    return path


class TestTheCohortIsScopedToTheRestore(unittest.TestCase):

    def setUp(self):
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _db(self, rows):
        p = _make_db(rows)
        self.paths.append(p)
        return p

    def test_trades_before_the_restore_are_excluded(self):
        """The pre-repair history is not evidence about the repaired stack."""
        db = self._db([
            {"entry_id": 1, "date": "2026-07-15", "status": "CLOSED",
             "pnl_usd": 500.0, "exit_date": "2026-07-30"},
            {"entry_id": 2, "date": "2026-08-19", "status": "CLOSED",
             "pnl_usd": -100.0, "exit_date": "2026-08-25"},
        ])
        rep = bpw.report(db_path=db, required=0.509)
        self.assertEqual(rep.n_closed, 1)
        self.assertEqual(rep.total_pnl, -100.0)

    def test_other_strategies_are_excluded(self):
        db = self._db([
            {"entry_id": 1, "date": "2026-08-19", "status": "CLOSED",
             "strategy_name": "Long Call", "pnl_usd": 900.0,
             "exit_date": "2026-08-20"},
            {"entry_id": 2, "date": "2026-08-19", "status": "CLOSED",
             "pnl_usd": 50.0, "exit_date": "2026-08-20"},
        ])
        rep = bpw.report(db_path=db, required=0.509)
        self.assertEqual(rep.n_closed, 1)
        self.assertEqual(rep.total_pnl, 50.0)

    def test_open_positions_are_counted_but_not_scored(self):
        """An open trade is not a win. Counting it as one is how a cohort
        flatters itself right up to the moment it closes."""
        db = self._db([
            {"entry_id": 1, "date": "2026-08-19", "status": "OPEN"},
            {"entry_id": 2, "date": "2026-08-19", "status": "CLOSED",
             "pnl_usd": 50.0, "exit_date": "2026-08-20"},
        ])
        rep = bpw.report(db_path=db, required=0.509)
        self.assertEqual(rep.n_open, 1)
        self.assertEqual(rep.n_closed, 1)


class TestTheVerdictArithmetic(unittest.TestCase):

    def setUp(self):
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _closed(self, pnls):
        rows = [{"entry_id": i, "date": "2026-08-19", "status": "CLOSED",
                 "pnl_usd": p, "exit_date": "2026-08-25"}
                for i, p in enumerate(pnls, 1)]
        p = _make_db(rows)
        self.paths.append(p)
        return p

    def test_win_rate_counts_positive_pnl(self):
        rep = bpw.report(db_path=self._closed([10, 20, -5, -5]), required=0.509)
        self.assertEqual(rep.n_closed, 4)
        self.assertEqual(rep.n_wins, 2)
        self.assertAlmostEqual(rep.win_rate, 0.5)

    def test_no_verdict_before_the_sample_is_big_enough(self):
        """n=3 at 100% says nothing. The report must not imply otherwise."""
        rep = bpw.report(db_path=self._closed([10, 10, 10]), required=0.509)
        self.assertTrue(rep.provisional)
        self.assertIn("provisional", " ".join(bpw.render(rep)).lower())

    def test_a_full_sample_is_not_provisional(self):
        rep = bpw.report(db_path=self._closed([10] * bpw.TARGET_N),
                         required=0.509)
        self.assertFalse(rep.provisional)

    def test_zero_closed_is_reported_not_divided_by(self):
        rep = bpw.report(db_path=self._closed([]), required=0.509)
        self.assertEqual(rep.n_closed, 0)
        self.assertIsNone(rep.win_rate)
        self.assertIsInstance(bpw.render(rep), list)

    def test_clears_flag_compares_against_the_required_rate(self):
        below = bpw.report(db_path=self._closed([10] + [-5] * 9), required=0.509)
        self.assertFalse(below.clears)
        above = bpw.report(db_path=self._closed([10] * 9 + [-5]), required=0.509)
        self.assertTrue(above.clears)


class TestTheRequiredRateIsNotHardcoded(unittest.TestCase):

    def test_it_reads_the_ledger_when_not_supplied(self):
        """The managed required rate drifts as trades close; a frozen 50.9%
        would silently become a comparison against a stale bar."""
        import inspect
        src = inspect.getsource(bull_put_watch_module := bpw)
        self.assertIn("required_win_rates_from_ledger", src)
        self.assertNotIn("0.509", src.replace("50.9%", ""))


class TestTheRender(unittest.TestCase):

    def setUp(self):
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def test_render_names_the_required_bar_and_the_sample_size(self):
        rows = [{"entry_id": 1, "date": "2026-08-19", "status": "CLOSED",
                 "pnl_usd": 25.0, "exit_date": "2026-08-25"}]
        p = _make_db(rows)
        self.paths.append(p)
        text = " ".join(bpw.render(bpw.report(db_path=p, required=0.509)))
        self.assertIn("50.9%", text)
        self.assertIn("n=1", text)

    def test_render_is_a_list_of_plain_strings(self):
        p = _make_db([])
        self.paths.append(p)
        for line in bpw.render(bpw.report(db_path=p, required=0.509)):
            self.assertIsInstance(line, str)
            self.assertNotIn("\x1b", line)


class TestTheSchedulerActuallyRunsIt(unittest.TestCase):
    """A watch nobody calls is not a watch.

    `watch_fn` is threaded from `run_catchup` down to `_run_catchup_locked` —
    it was not, on the first cut, so an injected watch would have been silently
    ignored and the default would have been the only path ever exercised. That
    is the default-branch blind spot this repo has hit before.
    """

    def test_run_catchup_passes_watch_fn_down(self):
        import ast
        import inspect
        from src import maintenance
        tree = ast.parse(inspect.getsource(maintenance))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_catchup":
                for call in ast.walk(node):
                    if (isinstance(call, ast.Call)
                            and getattr(call.func, "id", None) == "_run_catchup_locked"):
                        names = [getattr(a, "id", None) for a in call.args]
                        self.assertIn("watch_fn", names,
                                      "watch_fn never reaches the locked body")
                        return
        self.fail("run_catchup does not call _run_catchup_locked")

    def test_the_injected_watch_is_invoked(self):
        from src import maintenance
        seen = []
        maintenance._run_catchup_locked(
            db_path="nonexistent.db", state_path="/dev/null", now=None,
            runner=lambda *a, **k: 1,
            swing_fn=lambda: None,
            watch_fn=lambda db: seen.append(db) or {"ok": True})
        self.assertEqual(len(seen), 1, "the scheduler never called the watch")

    def test_every_catchup_test_injects_the_watch_hook(self):
        """Debris guard, and it caught a real one.

        `run_catchup` defaults `db_path` to the REAL ledger and the watch
        defaults to the REAL `logs/bull_put_watch.log`. On the first cut the
        maintenance tests injected `swing_fn` but not `watch_fn`, so one suite
        run read the live book and appended FOUR entries to the operator's log
        — the same shape as the `scan_errors.log` and `enforce_exits.log`
        debris, and the latter's mtime drives the health check.

        A statement about the code, so checked against the code.
        """
        import ast
        import pathlib
        src = pathlib.Path("tests/test_maintenance.py").read_text()
        missing = []
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "run_catchup"):
                continue
            kw = {k.arg for k in node.keywords}
            # A call that cannot reach the watch (lock contention returns
            # early) still may not name the real ledger.
            if "watch_fn" not in kw and "db_path" not in kw:
                missing.append(node.lineno)
        self.assertEqual(
            missing, [],
            f"run_catchup calls at these lines can reach the real ledger and "
            f"append to the operator's log: {missing}")

    def test_the_watch_writes_where_it_is_told(self):
        """No hardcoded path inside the writer — otherwise the injection above
        is the only thing standing between the suite and the operator's log."""
        import tempfile
        from src.maintenance import _run_bull_put_watch
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "sub", "watch.log")
            _run_bull_put_watch(db_path="nonexistent.db", log_path=log)
            self.assertTrue(os.path.exists(log))
            self.assertIn("BULL PUT WATCH", open(log).read())

    def test_a_failing_watch_cannot_break_the_catchup(self):
        """Isolated like the swing track: a report must never take down the
        job that produces the data it reports on."""
        from src import maintenance
        out = maintenance._run_catchup_locked(
            db_path="nonexistent.db", state_path="/dev/null", now=None,
            runner=lambda *a, **k: 1,
            swing_fn=lambda: None,
            watch_fn=lambda db: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertIn("ran", out)


if __name__ == "__main__":
    unittest.main()
