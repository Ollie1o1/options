import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.structure import express as E
from src.structure.types import StructureMargin, View


def _m(name, be, realized, state="ACTIVE"):
    return StructureMargin(strategy=name, n=40, wins=20, losses=20,
                           avg_win=100.0, avg_loss=100.0, breakeven_hit=be,
                           realized_hit=realized, margin=realized - be,
                           state=state, ci_lo=-0.05, ci_hi=0.3)


TABLE = {
    "Bull Put": _m("Bull Put", 0.375, 0.662),
    "Long Put": _m("Long Put", 0.229, 0.351),
    "Bear Call": _m("Bear Call", 0.540, 0.659),
    "Long Call": _m("Long Call", 0.455, 0.327, state="BENCHED"),
    "Iron Condor": _m("Iron Condor", 0.418, 0.435, state="UNPROVEN"),
}

CANDIDATES = {
    "Bull Put": {"capital_required": 178.0, "max_profit": 122.0},
    "Long Put": {"capital_required": 340.0, "max_profit": 900.0},
    "Bear Call": {"capital_required": 54.0, "max_profit": 47.5},
    "Long Call": {"capital_required": 710.0, "max_profit": 900.0},
    "Iron Condor": {"capital_required": 1380.0, "max_profit": 300.0},
}


class TestRoundTripCost(unittest.TestCase):
    def test_single_leg(self):
        self.assertAlmostEqual(E.round_trip_cost(1), 11.30, places=2)

    def test_two_leg_vertical(self):
        self.assertAlmostEqual(E.round_trip_cost(2), 22.60, places=2)

    def test_iron_condor(self):
        self.assertAlmostEqual(E.round_trip_cost(4), 45.20, places=2)


class TestExpress(unittest.TestCase):
    def test_benched_structure_is_rejected_with_reason(self):
        view = View("AAPL", "BULLISH", 0.9, [])
        _, rej = E.express(view, TABLE, 511.0, CANDIDATES)
        reasons = {r.strategy: r.reason for r in rej}
        self.assertIn("Long Call", reasons)
        self.assertIn("BENCHED", reasons["Long Call"])

    def test_unproven_structure_is_rejected(self):
        view = View("SPY", "NEUTRAL", 0.1, [])
        _, rej = E.express(view, TABLE, 511.0, CANDIDATES)
        reasons = {r.strategy: r.reason for r in rej}
        self.assertIn("Iron Condor", reasons)
        self.assertIn("UNPROVEN", reasons["Iron Condor"])

    def test_bullish_view_picks_bull_put(self):
        view = View("AAPL", "BULLISH", 0.9, [])
        exprs, _ = E.express(view, TABLE, 511.0, CANDIDATES)
        self.assertTrue(exprs)
        self.assertEqual(exprs[0].strategy, "Bull Put")

    def test_credit_structure_not_gated_on_implied_hit(self):
        # Bear Call B/E is 0.540. A bearish view has implied_hit <= 0.5, so a
        # naive gate would reject it - but credit structures win on direction
        # OR theta OR vol, so they must NOT be gated on directional accuracy.
        # Uses a wide-enough Bear Call so the cost-drag filter (tested
        # separately below) cannot be what lets it through or blocks it.
        cands = dict(CANDIDATES)
        cands["Bear Call"] = {"capital_required": 300.0, "max_profit": 200.0}
        view = View("NVDA", "BEARISH", 0.4, [])
        exprs, rej = E.express(view, TABLE, 511.0, cands)
        self.assertIn("Bear Call", [e.strategy for e in exprs])
        # and specifically: it was never rejected for lacking directional skill
        reasons = {r.strategy: r.reason for r in rej}
        self.assertNotIn("break even", reasons.get("Bear Call", ""))

    def test_narrow_bear_call_rejected_for_cost_drag_not_direction(self):
        # The real-world Bear Call: $47.50 median credit against a fixed
        # $22.60 round trip = 47.6% drag. Correctly rejected on cost, which is
        # the finding from docs/PROFITABILITY_FINDINGS.md section 7.
        view = View("NVDA", "BEARISH", 0.4, [])
        _, rej = E.express(view, TABLE, 511.0, CANDIDATES)
        reasons = {r.strategy: r.reason for r in rej}
        self.assertIn("cost drag", reasons.get("Bear Call", "").lower())

    def test_debit_structure_IS_gated_on_implied_hit(self):
        # Long Put B/E 0.229; a bearish view at confidence 0.4 implies
        # 0.5 + 0.4*(0.30-0.5) = 0.42 >= 0.229, so it passes.
        view = View("NVDA", "BEARISH", 0.4, [])
        exprs, _ = E.express(view, TABLE, 511.0, CANDIDATES)
        self.assertIn("Long Put", [e.strategy for e in exprs])

    def test_unaffordable_structure_rejected(self):
        view = View("AAPL", "BULLISH", 0.9, [])
        _, rej = E.express(view, TABLE, 100.0, CANDIDATES)
        reasons = {r.strategy: r.reason for r in rej}
        self.assertIn("Bull Put", reasons)
        self.assertIn("have", reasons["Bull Put"].lower())

    def test_nflx_regression_cost_drag_rejects_narrow_spread(self):
        # The 2026-07-27 NFLX trade: $1-wide spread, ~$35 max profit,
        # $22.60 round trip = 65% drag. Must be rejected.
        table = {"Bull Put": _m("Bull Put", 0.375, 0.662)}
        cands = {"Bull Put": {"capital_required": 65.0, "max_profit": 35.0}}
        view = View("NFLX", "BULLISH", 0.9, [])
        exprs, rej = E.express(view, table, 511.0, cands,
                               max_cost_drag_pct=25.0)
        self.assertEqual(exprs, [])
        self.assertIn("cost drag", rej[0].reason.lower())

    def test_neutral_view_suppresses_directional_structures(self):
        view = View("SPY", "NEUTRAL", 0.1, [])
        exprs, _ = E.express(view, TABLE, 511.0, CANDIDATES)
        self.assertNotIn("Long Put", [e.strategy for e in exprs])
        self.assertNotIn("Long Call", [e.strategy for e in exprs])

    def test_empty_table_yields_nothing(self):
        view = View("AAPL", "BULLISH", 0.9, [])
        exprs, rej = E.express(view, {}, 511.0, CANDIDATES)
        self.assertEqual(exprs, [])

    def test_results_ranked_by_margin(self):
        view = View("SPY", "NEUTRAL", 0.1, [])
        exprs, _ = E.express(view, TABLE, 511.0, CANDIDATES)
        margins = [e.margin for e in exprs]
        self.assertEqual(margins, sorted(margins, reverse=True))

    def test_costs_load_from_config_not_hardcoded(self):
        # Asserts the values come from the file, using values that are nobody's
        # default. Pinning the production numbers here made this test fail the
        # moment the real broker's fees were configured, which tests the config
        # rather than the code.
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"paper_trading": {"commission_per_contract": 1.23,
                                         "slippage_per_share": 0.42}}, fh)
            path = fh.name
        try:
            comm, slip = E.load_costs(path)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(comm, 1.23)
        self.assertAlmostEqual(slip, 0.42)

    def test_missing_config_falls_back_to_defaults(self):
        comm, slip = E.load_costs("/nonexistent/config.json")
        self.assertAlmostEqual(comm, 0.65)
        self.assertAlmostEqual(slip, 0.05)


class TestNegativeMarginOverride(unittest.TestCase):
    def test_debit_with_negative_margin_carries_a_warning(self):
        # Long Call B/E 0.374 vs realized 0.356 -> margin negative. A strong
        # bullish view still clears the forward-looking gate, but the user must
        # be told they are overriding the structure's own record.
        table = {"Long Call": _m("Long Call", 0.374, 0.356)}
        cands = {"Long Call": {"capital_required": 300.0,
                               "max_profit": 300.0}}
        view = View("SPY", "BULLISH", 0.8, [])
        exprs, _ = E.express(view, table, 511.0, cands)
        self.assertEqual(len(exprs), 1)
        self.assertIn("OVERRIDING", exprs[0].warning)

    def test_positive_margin_debit_has_no_warning(self):
        table = {"Long Put": _m("Long Put", 0.229, 0.351)}
        cands = {"Long Put": {"capital_required": 300.0,
                              "max_profit": 900.0}}
        view = View("NVDA", "BEARISH", 0.4, [])
        exprs, _ = E.express(view, table, 511.0, cands)
        self.assertEqual(exprs[0].warning, "")
