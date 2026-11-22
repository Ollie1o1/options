"""STRATEGY BREAKDOWN printed SUM(pnl_pct) as "Total P&L" — a sum of per-trade
returns on premium, not a quantity. On the 2026-11-22 ledger it showed Iron
Condor at -1193% while the line had made +$3,299. The breakdown now carries
dollars (`total_usd`) and the report prints those."""
import os
import sqlite3
import tempfile
import unittest

from src.paper_manager import PaperManager


class BreakdownDollarsTest(unittest.TestCase):
    def test_total_usd_is_the_dollar_sum_not_the_percent_sum(self):
        d = tempfile.mkdtemp()
        db = os.path.join(d, "ledger.db")
        pm = PaperManager(db_path=db)
        with sqlite3.connect(db) as c:
            # A credit-spread shape: a big loss on premium, a small dollar loss;
            # two small premium wins worth more dollars.
            for pct, usd in ((-3.0, -60.0), (0.5, 50.0), (0.5, 50.0)):
                c.execute("INSERT INTO trades (ticker, strategy_name, status, "
                          "pnl_pct, pnl_usd, capital_at_risk) "
                          "VALUES ('SPY', 'Iron Condor', 'CLOSED', ?, ?, 200)",
                          (pct, usd))
        row = {r["strategy"]: r for r in pm.get_strategy_breakdown()}["Iron Condor"]
        self.assertAlmostEqual(row["total_pnl"], -2.0)      # the old, meaningless sum
        self.assertAlmostEqual(row["total_usd"], 40.0)      # what actually happened
        self.assertAlmostEqual(row["return_on_risk"], 40.0 / 600.0)

    def test_the_report_no_longer_labels_a_percent_sum_total_pnl(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "src", "check_pnl.py")) as f:
            src = f.read()
        self.assertNotIn("{'Total P&L':>10}", src)
        self.assertIn("{'Total $':>10}", src)


class ClosedTypeLabelTest(unittest.TestCase):
    """A Bull Put and a Long Put both printed "PUT", with opposite signs."""

    def test_structures_are_told_apart(self):
        from src.check_pnl import _closed_type_label as L
        self.assertEqual(L({"type": "put", "strategy_name": "Bull Put",
                            "long_strike": 95}), "BPS")
        self.assertEqual(L({"type": "call", "strategy_name": "Bear Call",
                            "long_strike": 105}), "BCS")
        self.assertEqual(L({"type": "put", "strategy_name": "Iron Condor"}), "IC")
        self.assertEqual(L({"type": "put", "strategy_name": "Short Put"}), "S PUT")
        self.assertEqual(L({"type": "put", "strategy_name": "Long Put"}), "PUT")
        self.assertEqual(L({"type": "call", "strategy_name": "Long Call"}), "CALL")


if __name__ == "__main__":
    unittest.main()
