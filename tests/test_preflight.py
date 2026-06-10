"""Tests for src/execution/preflight.py — one-command go-live preflight.

Every check is a pure function over injected inputs; the CLI only gathers the
real inputs. The preflight must mirror docs/GO_LIVE_RUNBOOK.md and reuse the
existing sources of truth (gate, arm_status, health, sizing defaults) — it
must never introduce a second definition of any of them.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_preflight -v
"""
from __future__ import annotations

import unittest

from src.execution import preflight


class GateCheckTest(unittest.TestCase):
    def test_ready_passes(self):
        c = preflight.gate_check("READY")
        self.assertTrue(c.ok)

    def test_gathering_blocks(self):
        c = preflight.gate_check("GATHERING")
        self.assertFalse(c.ok)
        self.assertIn("GATHERING", c.detail)

    def test_stop_blocks(self):
        self.assertFalse(preflight.gate_check("STOP").ok)


class ArmingCheckTest(unittest.TestCase):
    def test_armed_passes(self):
        c = preflight.arming_check({"armed": True, "blockers": []})
        self.assertTrue(c.ok)

    def test_disarmed_blocks_and_lists_blockers(self):
        c = preflight.arming_check(
            {"armed": False, "blockers": ["gate=GATHERING (need READY)"]})
        self.assertFalse(c.ok)
        self.assertIn("gate=GATHERING", c.detail)


class RiskCapsCheckTest(unittest.TestCase):
    def test_default_caps_pass(self):
        # Reads the real sizing defaults — guards against accidental loosening.
        c = preflight.risk_caps_check()
        self.assertTrue(c.ok)
        self.assertIn("2", c.detail)  # 2% risk cap mentioned

    def test_loosened_caps_block(self):
        c = preflight.risk_caps_check(max_risk_pct=0.05, max_position_pct=0.10)
        self.assertFalse(c.ok)


class CheckpointFreshnessTest(unittest.TestCase):
    def test_fresh_checkpoint_passes(self):
        c = preflight.checkpoint_freshness_check("2026-06-08", today="2026-06-10")
        self.assertTrue(c.ok)

    def test_stale_checkpoint_blocks(self):
        c = preflight.checkpoint_freshness_check("2026-05-20", today="2026-06-10")
        self.assertFalse(c.ok)

    def test_missing_checkpoint_blocks(self):
        c = preflight.checkpoint_freshness_check(None, today="2026-06-10")
        self.assertFalse(c.ok)


class AutomationHealthCheckTest(unittest.TestCase):
    def test_no_warnings_passes(self):
        self.assertTrue(preflight.automation_health_check([]).ok)

    def test_warnings_block(self):
        c = preflight.automation_health_check(["exit-enforcer: last ran 21d ago"])
        self.assertFalse(c.ok)
        self.assertIn("exit-enforcer", c.detail)


class SlippageDbCheckTest(unittest.TestCase):
    def test_writable_dir_passes(self):
        import tempfile
        c = preflight.slippage_db_check(tempfile.mkdtemp())
        self.assertTrue(c.ok)

    def test_missing_dir_still_passes_if_creatable(self):
        import os, tempfile
        c = preflight.slippage_db_check(os.path.join(tempfile.mkdtemp(), "data"))
        self.assertTrue(c.ok)


class AggregateTest(unittest.TestCase):
    def test_all_ok_cleared(self):
        checks = [preflight.CheckResult("a", True, ""),
                  preflight.CheckResult("b", True, "")]
        r = preflight.aggregate(checks)
        self.assertTrue(r["cleared"])

    def test_any_block_not_cleared(self):
        checks = [preflight.CheckResult("a", True, ""),
                  preflight.CheckResult("b", False, "nope")]
        r = preflight.aggregate(checks)
        self.assertFalse(r["cleared"])
        self.assertEqual(r["failed"], ["b"])

    def test_render_mentions_verdict(self):
        checks = [preflight.CheckResult("gate", False, "gate=GATHERING")]
        text = preflight.render(preflight.aggregate(checks), checks)
        self.assertIn("NOT CLEARED", text)


class RunPreflightIntegrationTest(unittest.TestCase):
    """run_preflight against the real repo state must come back NOT CLEARED
    today (gate GATHERING, live flag off) — the honest answer."""

    def test_real_state_not_cleared_today(self):
        r, checks = preflight.run_preflight(db_path="paper_trades.db",
                                            config_path="config.json")
        self.assertFalse(r["cleared"])
        names = [c.name for c in checks]
        for expected in ("gate", "arming", "risk caps", "checkpoint freshness",
                         "automation health", "slippage db"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
