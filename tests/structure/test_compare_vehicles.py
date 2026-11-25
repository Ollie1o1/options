"""Offline unit tests for Phase 2B Shares-vs-Options Express Workbench & Walk-Away Ladder."""
from __future__ import annotations

import unittest

from src.structure.execution_plan import build_walk_away_ladder
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView


class TestExecutionPlanAndWorkbench(unittest.TestCase):
    def test_walk_away_ladder_debit_and_credit_bounds(self) -> None:
        # Debit: mid $2.00, cross $2.40, EV_gross $2.35, exit_half $0.20 => p_walk = $2.15
        lad = build_walk_away_ladder(
            structure="Debit Vertical",
            is_credit=False,
            p_mid=2.00,
            p_cross=2.40,
            ev_view_gross_per_share=2.35,
            exit_half_spread_per_share=0.20,
        )
        self.assertAlmostEqual(lad.p_walk, 2.15, places=2)
        self.assertAlmostEqual(lad.p_step25, 2.10, places=2)
        self.assertFalse(lad.never_chase_warning)

        # Credit: mid $1.80, cross $1.50, 25% friction ceiling => p_walk >= 1.80 / 1.125 = 1.60
        lad_c = build_walk_away_ladder(
            structure="Wide Credit Spread",
            is_credit=True,
            p_mid=1.80,
            p_cross=1.50,
            ev_view_gross_per_share=1.80,
            max_friction_to_credit=0.25,
        )
        self.assertAlmostEqual(lad_c.p_walk, 1.60, places=2)
        self.assertGreaterEqual(lad_c.p_step25, lad_c.p_walk)

    def test_compare_vehicles_bullish_tfsa_and_covered_call(self) -> None:
        view = DetailedView(
            symbol="AAPL",
            direction="BULLISH",
            target_move_pct=0.08,
            horizon_days=30,
            vol_mult=1.0,
            risk_budget_usd=500.0,
            account_type="tfsa",
            owns_100_shares=True,
        )
        rows = compare_vehicles(view, spot=200.0, iv=0.25)
        names = {r.vehicle for r in rows}
        self.assertIn("Shares", names)
        self.assertIn("Deep-ITM LEAPS", names)
        self.assertIn("Debit Vertical", names)
        self.assertIn("Wide Credit Spread", names)
        self.assertIn("Covered Call", names)
        self.assertIn("Single Option", names)
        # CRA TFSA rules: single-leg long options, covered calls, and long shares are legal;
        # multi-leg spreads (debit/credit verticals) require margin and are NOT TFSA-legal.
        by_name = {r.vehicle: r for r in rows}
        for legal_name in ("Shares", "Deep-ITM LEAPS", "Covered Call", "Single Option"):
            self.assertTrue(by_name[legal_name].tfsa_legal, f"{legal_name} should be TFSA-legal")
        for illegal_name in ("Debit Vertical", "Wide Credit Spread"):
            self.assertFalse(by_name[illegal_name].tfsa_legal, f"{illegal_name} is margin-only (not TFSA)")
            self.assertEqual(len(by_name[illegal_name].walk_away_steps), 3)
        # On a TFSA view, recommended vehicle must NEVER violate CRA TFSA legality
        recs = [r for r in rows if r.recommended]
        self.assertTrue(all(r.tfsa_legal for r in recs))
        # Shares friction should be tiny compared to options
        shares_row = by_name["Shares"]
        opt_row = by_name["Single Option"]
        self.assertLess(shares_row.friction_pct_of_reward, opt_row.friction_pct_of_reward)
        # Sorted descending by EV(view) net
        evs = [r.ev_view_net_usd for r in rows]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_compare_vehicles_bearish_tfsa_excludes_short_shares(self) -> None:
        view = DetailedView(
            symbol="SPY",
            direction="BEARISH",
            target_move_pct=-0.06,
            horizon_days=30,
            account_type="tfsa",
        )
        rows = compare_vehicles(view, spot=500.0, iv=0.18)
        self.assertNotIn("Shares", {r.vehicle for r in rows})
        self.assertIn("Debit Vertical", {r.vehicle for r in rows})
        self.assertIn("Wide Credit Spread", {r.vehicle for r in rows})


if __name__ == "__main__":
    unittest.main()
