"""Rows ruled double-logs stay in the ledger and out of the evidence.

The audit (reports/duplicate_trades_audit.md) flags candidates; the ruling marks
`duplicate_of`. These tests pin the two halves of that: a marked row must vanish
from every cohort that carries evidential weight, and must survive in the ledger
itself, because the record of what happened is not the same object as the
evidence drawn from it.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import phase1_checkpoint
from src.short_premium_gate import load_cohort


def _db(path, with_column=True):
    conn = sqlite3.connect(path)
    cols = ("date TEXT, ticker TEXT, strategy_name TEXT, status TEXT, "
            "paper_only INTEGER, quality_score REAL, pnl_pct REAL, "
            "pnl_usd REAL, capital_at_risk REAL, net_credit REAL, "
            "entry_price REAL")
    if with_column:
        cols += ", duplicate_of INTEGER"
    conn.execute(f"CREATE TABLE trades (entry_id INTEGER PRIMARY KEY, {cols})")
    return conn


def _insert(conn, entry_id, strategy="Long Call", dup=None, has_col=True):
    base = ("INSERT INTO trades (entry_id, date, ticker, strategy_name, status, "
            "paper_only, quality_score, pnl_pct, pnl_usd, capital_at_risk, "
            "net_credit, entry_price")
    vals = [entry_id, "2026-06-01", "AAPL", strategy, "CLOSED", 0,
            50.0 + entry_id, 0.05, 50.0, 500.0, 1.0, 1.0]
    if has_col:
        base += ", duplicate_of"
        vals.append(dup)
    conn.execute(base + ") VALUES (" + ",".join("?" * len(vals)) + ")", vals)


class TestLongCallCohort(unittest.TestCase):
    def test_ruled_duplicate_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            for i in (1, 2, 3):
                _insert(conn, i)
            _insert(conn, 4, dup=1)
            conn.commit(); conn.close()
            scores, _returns, _dates = phase1_checkpoint._load_cohort(p, "2026-05-27")
            self.assertEqual(len(scores), 3, "the marked row must not be counted")

    def test_unmarked_rows_all_survive(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            for i in (1, 2, 3, 4):
                _insert(conn, i)
            conn.commit(); conn.close()
            scores, _r, _dt = phase1_checkpoint._load_cohort(p, "2026-05-27")
            self.assertEqual(len(scores), 4)


class TestShortPremiumCohort(unittest.TestCase):
    def test_ruled_duplicate_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            for i in (1, 2, 3):
                _insert(conn, i, strategy="Bull Put")
            _insert(conn, 4, strategy="Bull Put", dup=1)
            conn.commit(); conn.close()
            self.assertEqual(len(load_cohort(p, "2026-05-27")), 3)


class TestPooledICCohort(unittest.TestCase):
    """`run_paper_trade_ic` filters on status/quality_score/pnl_pct and nothing
    else — no strategy, no duplicate rule. The one row ruled a double-log on
    2026-08-01 satisfies all three, so it was still being counted as evidence
    here (821 rows against the ledger's 820) after the ruling removed it from
    the two gate cohorts. Same row, same ledger, two different answers to "is
    this evidence".
    """

    def test_ruled_duplicate_is_excluded(self):
        from src.backtester import run_paper_trade_ic

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            for i in (1, 2, 3):
                _insert(conn, i)
            _insert(conn, 4, dup=1)
            conn.commit(); conn.close()
            self.assertEqual(run_paper_trade_ic(p).get("n_trades"), 3)

    def test_it_loads_without_the_column(self):
        from src.backtester import run_paper_trade_ic

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p, with_column=False)
            for i in (1, 2):
                _insert(conn, i, has_col=False)
            conn.commit(); conn.close()
            self.assertEqual(run_paper_trade_ic(p).get("n_trades"), 2)


class TestCalibrationCount(unittest.TestCase):
    """`get_calibration_status` counts the same cohort, and the count is what
    tells the operator how much evidence exists."""

    def test_ruled_duplicate_is_not_counted(self):
        from src.backtester import get_calibration_status

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            for i in (1, 2, 3):
                _insert(conn, i)
            _insert(conn, 4, dup=1)
            conn.commit(); conn.close()
            n_closed, _label = get_calibration_status(p)
            self.assertEqual(n_closed, 3)


class TestWalkForwardCohort(unittest.TestCase):
    """The walk-forward OOS IC is what the evidence banner prints on every
    scan. It filters by strategy today, which is the only reason the ruled row
    does not reach it — an accident of that row's strategy, not a rule."""

    def _wf_db(self, path, with_column=True):
        """Walk-forward selects every scorer component, so its fixture needs
        the full component column set rather than the shared minimal one."""
        from src.walk_forward import _COMPONENT_COLS

        conn = sqlite3.connect(path)
        cols = ["date TEXT", "strategy_name TEXT", "status TEXT",
                "paper_only INTEGER", "pnl_pct REAL"]
        cols += [f"{c} REAL" for c in _COMPONENT_COLS]
        if with_column:
            cols.append("duplicate_of INTEGER")
        conn.execute(f"CREATE TABLE trades (entry_id INTEGER PRIMARY KEY, "
                     f"{', '.join(cols)})")
        return conn

    def _wf_insert(self, conn, entry_id, dup=None, has_col=True):
        from src.walk_forward import _COMPONENT_COLS

        names = ["entry_id", "date", "strategy_name", "status", "paper_only",
                 "pnl_pct"] + list(_COMPONENT_COLS)
        vals = [entry_id, "2026-06-01", "Long Call", "CLOSED", 0, 0.05]
        vals += [0.5] * len(_COMPONENT_COLS)
        if has_col:
            names.append("duplicate_of")
            vals.append(dup)
        conn.execute(f"INSERT INTO trades ({', '.join(names)}) VALUES "
                     f"({','.join('?' * len(vals))})", vals)

    def test_ruled_duplicate_is_excluded(self):
        from src.walk_forward import load_trades

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = self._wf_db(p)
            for i in (1, 2, 3):
                self._wf_insert(conn, i)
            self._wf_insert(conn, 4, dup=1)
            conn.commit(); conn.close()
            self.assertEqual(len(load_trades(p, "Long Call")), 3)

    def test_it_loads_without_the_column(self):
        from src.walk_forward import load_trades

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = self._wf_db(p, with_column=False)
            for i in (1, 2):
                self._wf_insert(conn, i, has_col=False)
            conn.commit(); conn.close()
            self.assertEqual(len(load_trades(p, "Long Call")), 2)


class TestBackwardCompatibility(unittest.TestCase):
    """Ledgers written before schema v17 have no such column and must still
    load — the filter probes rather than assumes."""

    def test_long_call_cohort_loads_without_the_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p, with_column=False)
            for i in (1, 2):
                _insert(conn, i, has_col=False)
            conn.commit(); conn.close()
            scores, _r, _dt = phase1_checkpoint._load_cohort(p, "2026-05-27")
            self.assertEqual(len(scores), 2)

    def test_short_premium_cohort_loads_without_the_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p, with_column=False)
            _insert(conn, 1, strategy="Bear Call", has_col=False)
            conn.commit(); conn.close()
            self.assertEqual(len(load_cohort(p, "2026-05-27")), 1)

    def test_the_filter_fragment_is_empty_without_the_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p, with_column=False)
            conn.commit()
            self.assertEqual(phase1_checkpoint.exclude_ruled_duplicates(conn), "")
            conn.close()

    def test_the_filter_fragment_survives_a_missing_table(self):
        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(os.path.join(d, "empty.db"))
            self.assertEqual(phase1_checkpoint.exclude_ruled_duplicates(conn), "")
            conn.close()


class TestTheRowSurvives(unittest.TestCase):
    def test_marking_does_not_delete(self):
        """The audit's own rule: rewriting the ledger silently is worse than
        the double-count it fixes."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            _insert(conn, 1)
            _insert(conn, 2, dup=1)
            conn.commit()
            n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            self.assertEqual(n, 2)
            row = conn.execute(
                "SELECT duplicate_of FROM trades WHERE entry_id=2").fetchone()
            self.assertEqual(row[0], 1)
            conn.close()


class TestTheRulingScript(unittest.TestCase):
    def test_ruling_is_reversible(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rule_dupes", os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "scripts",
                "rule_duplicate_trades.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            _insert(conn, 90, strategy="Short Put")
            _insert(conn, 91, strategy="Short Put")
            conn.commit(); conn.close()

            mod.apply(p, dry_run=False)
            conn = sqlite3.connect(p)
            self.assertEqual(conn.execute(
                "SELECT duplicate_of FROM trades WHERE entry_id=91").fetchone()[0], 90)
            conn.close()

            mod.apply(p, undo=True)
            conn = sqlite3.connect(p)
            self.assertIsNone(conn.execute(
                "SELECT duplicate_of FROM trades WHERE entry_id=91").fetchone()[0])
            conn.close()

    def test_dry_run_writes_nothing(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rule_dupes2", os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "scripts",
                "rule_duplicate_trades.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            conn = _db(p)
            _insert(conn, 90, strategy="Short Put")
            _insert(conn, 91, strategy="Short Put")
            conn.commit(); conn.close()

            res = mod.apply(p, dry_run=True)
            self.assertEqual(len(res["changed"]), 1)
            conn = sqlite3.connect(p)
            self.assertIsNone(conn.execute(
                "SELECT duplicate_of FROM trades WHERE entry_id=91").fetchone()[0])
            conn.close()


if __name__ == "__main__":
    unittest.main()
