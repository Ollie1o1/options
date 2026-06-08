import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.execution import pipeline


def _seed_db(path, n_closed=2):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT, status TEXT, "
                 "paper_only INTEGER, quality_score REAL, pnl_pct REAL)")
    for i in range(n_closed):
        conn.execute("INSERT INTO trades VALUES ('2026-05-28','Long Call','CLOSED',0,?,?)",
                     (60.0 + i, 0.05 * (1 if i % 2 else -1)))
    conn.commit(); conn.close()


class TestPipeline(unittest.TestCase):
    def test_live_enabled_reads_config(self):
        self.assertFalse(pipeline.live_enabled({"live_execution": {"enabled": False}}))
        self.assertTrue(pipeline.live_enabled({"live_execution": {"enabled": True}}))
        self.assertFalse(pipeline.live_enabled({}))

    def test_current_gate_gathering_when_small(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            self.assertEqual(pipeline.current_gate(db, "2026-05-27"), "GATHERING")

    def test_build_ticket_is_dry_run_while_inert(self):
        # Gate GATHERING + flag off => must be a DRY RUN, never a live order.
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            pick = {"ticker": "AAPL", "strike": 180.0, "expiration": "2026-08-21",
                    "option_type": "call", "bid": 4.1, "ask": 4.3, "entry_price": 4.2}
            t = pipeline.build_ticket(pick, account_value=50_000, db_path=db,
                                      config={"live_execution": {"enabled": False}},
                                      phase1_start="2026-05-27")
            self.assertEqual(t["mode"], "DRY_RUN")

    def test_arm_status_reports_blockers(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            st = pipeline.arm_status(db_path=db,
                                     config={"live_execution": {"enabled": False}},
                                     phase1_start="2026-05-27")
            self.assertFalse(st["armed"])
            self.assertEqual(st["gate"], "GATHERING")
            self.assertFalse(st["live_enabled"])


if __name__ == "__main__":
    unittest.main()
