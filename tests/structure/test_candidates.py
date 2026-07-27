import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

import pandas as pd

from src.structure import candidates as C


def _chain(spot=100.0):
    """Synthetic single-expiry chain, $1 strikes from 85 to 115.

    Time value decays with distance from spot, which is what makes a credit
    spread actually collect a credit. A flat time value would (correctly) be
    rejected by the builder as a zero-credit spread. No network.
    """
    rows = []
    for k in range(85, 116):
        tv = max(0.10, 3.0 - 0.20 * abs(k - spot))
        c_mid = max(0.0, spot - k) + tv
        p_mid = max(0.0, k - spot) + tv
        rows.append({"type": "call", "strike": float(k),
                     "bid": c_mid - 0.05, "ask": c_mid + 0.05})
        rows.append({"type": "put", "strike": float(k),
                     "bid": p_mid - 0.05, "ask": p_mid + 0.05})
    return pd.DataFrame(rows)


RULES = {"long_option": {"take_profit": 1.0},
         "spread": {"take_profit": 0.5},
         "short_premium": {"take_profit_ge_21_dte": 0.5}}


class TestBuildCandidates(unittest.TestCase):
    def test_builds_the_core_structures(self):
        out = C.build_candidates(_chain(), 100.0, RULES)
        for name in ("Long Call", "Long Put", "Bull Put", "Bear Call",
                     "Iron Condor", "Short Put"):
            self.assertIn(name, out, name)

    def test_all_candidates_have_positive_numbers(self):
        out = C.build_candidates(_chain(), 100.0, RULES)
        for name, v in out.items():
            self.assertGreater(v["capital_required"], 0, name)
            self.assertGreater(v["max_profit"], 0, name)

    def test_max_profit_is_take_profit_target_not_theoretical_max(self):
        half = C.build_candidates(_chain(), 100.0,
                                  {"spread": {"take_profit": 0.5}})
        full = C.build_candidates(_chain(), 100.0,
                                  {"spread": {"take_profit": 1.0}})
        # Same contract, same capital at risk - only the attainable target moves
        self.assertAlmostEqual(half["Bull Put"]["capital_required"],
                               full["Bull Put"]["capital_required"], places=2)
        self.assertAlmostEqual(half["Bull Put"]["max_profit"] * 2,
                               full["Bull Put"]["max_profit"], places=2)

    def test_long_call_capital_is_the_premium(self):
        out = C.build_candidates(_chain(), 100.0, RULES)
        # ATM call mid = intrinsic(0) + tv(3.00) -> $300 per contract
        self.assertAlmostEqual(out["Long Call"]["capital_required"], 300.0,
                               places=2)

    def test_short_put_is_cash_secured_and_expensive(self):
        out = C.build_candidates(_chain(), 100.0, RULES)
        # ~95 strike cash-secured -> thousands, far beyond a 700 CAD account
        self.assertGreater(out["Short Put"]["capital_required"], 8000.0)

    def test_empty_chain_yields_nothing(self):
        self.assertEqual(C.build_candidates(pd.DataFrame(), 100.0, RULES), {})

    def test_zero_spot_yields_nothing(self):
        self.assertEqual(C.build_candidates(_chain(), 0.0, RULES), {})

    def test_crossed_quotes_are_skipped(self):
        ch = _chain()
        ch.loc[ch["type"] == "call", "ask"] = 0.01   # ask < bid everywhere
        out = C.build_candidates(ch, 100.0, RULES)
        self.assertNotIn("Long Call", out)

    def test_missing_exit_rules_falls_back_to_defaults(self):
        out = C.build_candidates(_chain(), 100.0, None)
        self.assertIn("Bull Put", out)
        self.assertGreater(out["Bull Put"]["max_profit"], 0)


class TestEndToEndWithEngine(unittest.TestCase):
    def test_candidates_feed_the_expression_engine(self):
        from src.structure.express import express
        from src.structure.types import StructureMargin, View

        cands = C.build_candidates(_chain(), 100.0, RULES)
        table = {"Bull Put": StructureMargin(
            "Bull Put", 107, 68, 39, 116.0, 70.0, 0.376, 0.636, 0.259,
            "ACTIVE", 0.10, 0.40)}
        view = View("TEST", "BULLISH", 0.8, [])
        exprs, rej = express(view, table, 5000.0, cands,
                             max_cost_drag_pct=100.0)
        self.assertEqual([e.strategy for e in exprs], ["Bull Put"])
        self.assertGreater(exprs[0].cost_drag_pct, 0)


class TestCapitalAdaptiveWidth(unittest.TestCase):
    def test_small_account_gets_a_narrower_spread(self):
        small = C.build_candidates(_chain(), 100.0, RULES, capital_usd=150.0)
        big = C.build_candidates(_chain(), 100.0, RULES, capital_usd=5000.0)
        self.assertLessEqual(small["Bull Put"]["capital_required"],
                             big["Bull Put"]["capital_required"])

    def test_bigger_account_takes_more_credit(self):
        small = C.build_candidates(_chain(), 100.0, RULES, capital_usd=150.0)
        big = C.build_candidates(_chain(), 100.0, RULES, capital_usd=5000.0)
        # wider spread => more credit => the fixed round-trip is a smaller share
        self.assertGreaterEqual(big["Bull Put"]["max_profit"],
                                small["Bull Put"]["max_profit"])

    def test_spread_fits_the_budget_when_possible(self):
        out = C.build_candidates(_chain(), 100.0, RULES, capital_usd=400.0)
        self.assertLessEqual(out["Bull Put"]["capital_required"], 400.0)

    def test_long_option_walks_otm_to_fit_budget(self):
        # ATM call costs $300; a $150 budget must find a cheaper OTM strike
        out = C.build_candidates(_chain(), 100.0, RULES, capital_usd=150.0)
        self.assertLessEqual(out["Long Call"]["capital_required"], 150.0)
