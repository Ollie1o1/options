"""The gate reports the affordable cohort alongside the nominal one.

Roughly half the Long Call cohort ties up more than the whole account, so the
gate has been reading a population the account could not trade. The decision
rule is deliberately untouched here — DECISIONS.md 2026-06-07 forbids silent
gate changes — this only puts both numbers in front of the reader.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_checkpoint import compute_checkpoint


def _cohort_db(path, rows):
    """rows: (quality_score, pnl_pct, capital_at_risk)"""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades (entry_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " date TEXT, strategy_name TEXT, status TEXT, paper_only INTEGER,"
        " quality_score REAL, pnl_pct REAL, capital_at_risk REAL)"
    )
    conn.executemany(
        "INSERT INTO trades (date, strategy_name, status, paper_only,"
        " quality_score, pnl_pct, capital_at_risk)"
        " VALUES ('2026-06-01', 'Long Call', 'CLOSED', 0, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestAffordableCohortReporting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_the_affordable_subset_separately(self):
        _cohort_db(self.db, [
            (0.9, 0.10, 300.0), (0.8, 0.05, 400.0),
            (0.7, -0.20, 5000.0), (0.6, -0.30, 6000.0),
        ])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               max_capital_at_risk=750.0)
        self.assertEqual(r["n_trades"], 4)
        self.assertEqual(r["n_affordable"], 2)

    def test_reports_a_separate_ic_for_the_affordable_subset(self):
        # Nominal cohort: score tracks return. Affordable subset: it inverts.
        _cohort_db(self.db, [
            (0.9, -0.10, 300.0), (0.8, 0.05, 400.0), (0.7, 0.20, 500.0),
            (0.6, 0.50, 9000.0), (0.5, 0.60, 9000.0),
        ])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               max_capital_at_risk=750.0)
        self.assertIsNotNone(r["ic_pearson_affordable"])
        self.assertNotAlmostEqual(
            r["ic_pearson"], r["ic_pearson_affordable"], places=3
        )

    def test_the_gate_decision_still_reads_the_nominal_cohort(self):
        # Only 2 affordable rows, but the decision must not change on that basis.
        _cohort_db(self.db, [(0.5 + i * 0.01, 0.01 * i, 300.0 if i < 2 else 9000.0)
                             for i in range(60)])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               max_capital_at_risk=750.0)
        self.assertEqual(r["n_trades"], 60)
        self.assertNotEqual(r["decision"], "GATHERING")

    def test_without_a_cap_the_affordable_figures_are_absent(self):
        _cohort_db(self.db, [(0.9, 0.10, 300.0), (0.8, 0.05, 9000.0)])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28")
        self.assertIsNone(r["ic_pearson_affordable"])
        self.assertIsNone(r["n_affordable"])

    def test_rows_with_null_risk_are_excluded_from_the_affordable_subset(self):
        _cohort_db(self.db, [(0.9, 0.10, 300.0), (0.8, 0.05, None)])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               max_capital_at_risk=750.0)
        self.assertEqual(r["n_affordable"], 1)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_markdown_shows_both_cohorts_when_a_cap_is_set(self):
        from src.phase1_checkpoint import _format_markdown

        _cohort_db(self.db, [
            (0.9, 0.10, 300.0), (0.8, 0.05, 400.0),
            (0.7, -0.20, 5000.0), (0.6, -0.30, 6000.0),
        ])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               max_capital_at_risk=750.0)
        md = _format_markdown(r)
        self.assertIn("Affordable subset", md)
        self.assertIn("2 of 4", md)

    def test_markdown_omits_the_subset_when_no_cap_is_set(self):
        from src.phase1_checkpoint import _format_markdown

        _cohort_db(self.db, [(0.9, 0.10, 300.0), (0.8, 0.05, 9000.0)])
        md = _format_markdown(compute_checkpoint(self.db, "2026-05-27",
                                                 today="2026-07-28"))
        self.assertNotIn("Affordable subset", md)


if __name__ == "__main__":
    unittest.main()
