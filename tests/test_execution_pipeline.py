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
                 "paper_only INTEGER, quality_score REAL, pnl_pct REAL, "
                 "capital_at_risk REAL)")
    for i in range(n_closed):
        conn.execute("INSERT INTO trades VALUES ('2026-05-28','Long Call','CLOSED',0,?,?,?)",
                     (60.0 + i, 0.05 * (1 if i % 2 else -1), 500.0))
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


class TestWhichGateAuthorises(unittest.TestCase):
    """Two gates exist and they disagree. Which one arms must be explicit."""

    def test_default_is_the_long_call_gate(self):
        self.assertEqual(pipeline.authorising_gate({}), pipeline.LONG_CALL)
        self.assertEqual(pipeline.authorising_gate({"gate": {}}), pipeline.LONG_CALL)

    def test_config_can_select_the_short_premium_gate(self):
        cfg = {"gate": {"authorising_gate": "short_premium"}}
        self.assertEqual(pipeline.authorising_gate(cfg), pipeline.SHORT_PREMIUM)

    def test_arm_status_names_the_gate_it_read(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            st = pipeline.arm_status(db_path=db,
                                     config={"live_execution": {"enabled": False}},
                                     phase1_start="2026-05-27")
            self.assertEqual(st["authorising_gate"], pipeline.LONG_CALL)
            self.assertIn(pipeline.SHORT_PREMIUM, st["gate_readings"])

    def test_a_non_authorising_ready_cannot_arm(self):
        """The safety property: a READY on a gate that does not authorise is
        reported, never acted on."""
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            real = pipeline.gate_readings
            pipeline.gate_readings = lambda *a, **k: {
                pipeline.LONG_CALL: "STOP",
                pipeline.SHORT_PREMIUM: "READY",
                "short_premium_arm_b": "EXTEND"}
            try:
                st = pipeline.arm_status(
                    db_path=db, config={"live_execution": {"enabled": True}},
                    phase1_start="2026-05-27")
                self.assertFalse(st["armed"])
                self.assertEqual(st["gate"], "STOP")
                self.assertEqual(st["gate_readings"][pipeline.SHORT_PREMIUM], "READY")
            finally:
                pipeline.gate_readings = real

    def test_an_unknown_gate_name_never_reads_as_permission(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            real = pipeline.gate_readings
            pipeline.gate_readings = lambda *a, **k: {pipeline.LONG_CALL: "READY"}
            try:
                st = pipeline.arm_status(
                    db_path=db,
                    config={"live_execution": {"enabled": True},
                            "gate": {"authorising_gate": "typo_gate"}},
                    phase1_start="2026-05-27")
                self.assertFalse(st["armed"])
                self.assertEqual(st["gate"], "GATHERING")
            finally:
                pipeline.gate_readings = real

    def test_readings_respect_the_configured_capital_cap(self):
        """The short-premium gate is defined on a capped cohort. Reading it
        uncapped is a different cohort and a different verdict."""
        seen = {}
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db"); _seed_db(db)
            real = pipeline.phase1_checkpoint.compute_checkpoint

            def spy(*a, **k):
                seen.update(k)
                return real(*a, **k)

            pipeline.phase1_checkpoint.compute_checkpoint = spy
            try:
                pipeline.gate_readings(
                    db, "2026-05-27",
                    {"auto_log": {"max_capital_at_risk": 4000.0}})
            finally:
                pipeline.phase1_checkpoint.compute_checkpoint = real
        self.assertEqual(seen.get("max_capital_at_risk"), 4000.0)


if __name__ == "__main__":
    unittest.main()
