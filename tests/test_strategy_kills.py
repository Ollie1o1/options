"""What the measured cost wall retired, and what it deliberately did not.

Killing a setup is a claim about evidence, so each kill has to carry its reason
and each survivor has to have a reason to survive. Controls are not trades —
retiring them because they are expensive to trade would delete the yardstick
every future comparison is measured against.
"""
from __future__ import annotations

import unittest

from src.strategies import friction as fr
from src.strategies.seed import LIBRARY

# The single-name, $5-wide bull put deployments. 68% of credit round trip, and
# -6.76% RoC over 10,363 trades on the wide universe.
RETIRED = ("put_spread_ivr50", "put_spread_ivr50_hold",
           "bullish_trend_put_spread", "csp_single_names")

# Kept alive on purpose, all bull_put.
SURVIVING_BULL_PUTS = ("csp_index_only", "benchmark_unselected",
                       "null_random_days", "null_random_strikes")


def _rec(setup_id):
    return [r for r in LIBRARY if r.spec.id == setup_id][0]


class RetirementTest(unittest.TestCase):
    def test_the_condemned_deployments_are_dead(self):
        for sid in RETIRED:
            self.assertEqual(_rec(sid).status, "dead", sid)

    def test_each_kill_records_why(self):
        for sid in RETIRED:
            reasons = [a["reason"] for a in _rec(sid).amendments
                       if a["field"] == "status"]
            self.assertTrue(reasons, f"{sid} died with no reason recorded")
            self.assertRegex(reasons[-1].lower(), r"friction|credit|roc|-6\.76")

    def test_a_kill_keeps_the_previous_status(self):
        """Amend, never overwrite — the setup remembers it was once specified."""
        a = [x for x in _rec("put_spread_ivr50").amendments
             if x["field"] == "status"][-1]
        self.assertEqual(a["from"], "specified")


class SurvivorTest(unittest.TestCase):
    def test_the_control_tier_survives(self):
        """A control is measurement infrastructure, not a trade to be placed."""
        for sid in ("benchmark_unselected", "null_random_days",
                    "null_random_strikes"):
            self.assertNotEqual(_rec(sid).status, "dead", sid)

    def test_the_index_probe_survives(self):
        """The 68% toll was measured on single names; the index is where the
        only non-negative result in the study lives."""
        self.assertNotEqual(_rec("csp_index_only").status, "dead")

    def test_the_bearish_line_survives(self):
        self.assertNotEqual(_rec("bearish_trend_call_spread").status, "dead")

    def test_something_bullish_is_still_testable(self):
        """Killing every bullish expression would leave the desk unable to hold
        a bullish view at all."""
        bullish = [r for r in LIBRARY
                   if r.status != "dead"
                   and (r.signal.get("above_sma50")
                        or r.spec.structure in ("bull_put", "short_put"))]
        self.assertTrue(bullish)


class IndexFrictionTest(unittest.TestCase):
    """The structure-wide figure must not be quoted at a setup it never measured."""

    def test_the_index_probe_reports_unmeasured_friction(self):
        p = fr.profile_for(_rec("csp_index_only"), table=fr.RECORDED)
        self.assertFalse(p.measured)
        self.assertEqual(fr.format_cell(p), "—")

    def test_it_says_why_rather_than_going_silent(self):
        p = fr.profile_for(_rec("csp_index_only"), table=fr.RECORDED)
        self.assertIn("single name", fr.describe(p).lower())

    def test_an_unmeasured_marker_beats_a_wrong_number(self):
        """Guard the mechanism itself, not just this one setup."""
        r = _rec("benchmark_unselected").amend(
            "cost_profile", {"unmeasured": True, "why": "no quotes for this universe"},
            reason="test", date="2026-08-06")
        p = fr.profile_for(r, table=fr.RECORDED)
        self.assertIsNone(p.pct_of_credit)
        self.assertIn("no quotes", fr.describe(p))


if __name__ == "__main__":
    unittest.main()
