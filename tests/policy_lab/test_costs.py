"""The cost model decides whether this lab tells the truth. Replaying an exit
at mid invents edge; these tests pin the direction of the crossing cost for
both credit and debit structures, and pin that an imputed (synthesized)
bid/ask is never trusted as if it were a real quote.
"""
import unittest

from src.policy_lab.costs import CostModel
from src.policy_lab.types import PathPoint


def _pt(bid, ask, spread_imputed=False):
    mid = (bid + ask) / 2.0
    return PathPoint(date="2026-08-20", bid=bid, ask=ask, mid=mid,
                     spot=100.0, dte=25, spread_imputed=spread_imputed)


class TestCostModel(unittest.TestCase):
    def test_mid_ignores_the_spread_both_ways(self):
        cm = CostModel.mid()
        pt = _pt(0.90, 1.10)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=True), 1.00)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=False), 1.00)

    def test_cross_pays_the_ask_to_close_a_credit_structure(self):
        # Closing a short spread means buying it back: you pay the ask.
        self.assertAlmostEqual(
            CostModel.cross().close_price(_pt(0.90, 1.10), is_credit=True), 1.10)

    def test_cross_takes_the_bid_to_close_a_debit_structure(self):
        # Closing a long option means selling it: you receive the bid.
        self.assertAlmostEqual(
            CostModel.cross().close_price(_pt(0.90, 1.10), is_credit=False), 0.90)

    def test_cross_is_never_better_than_mid_for_the_trader(self):
        pt = _pt(0.80, 1.20)
        for is_credit in (True, False):
            mid = CostModel.mid().close_price(pt, is_credit)
            cross = CostModel.cross().close_price(pt, is_credit)
            if is_credit:
                self.assertGreaterEqual(cross, mid)   # pay more
            else:
                self.assertLessEqual(cross, mid)      # receive less

    def test_zero_bid_is_a_price_not_a_gap(self):
        pt = _pt(0.0, 0.10)
        self.assertAlmostEqual(
            CostModel.cross().close_price(pt, is_credit=False), 0.0)

    def test_unknown_setting_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            CostModel("free")

    def test_settings_tuple_is_the_whole_menu(self):
        self.assertEqual(CostModel.SETTINGS, ("mid", "cross", "surface"))


class TestImputedSpread(unittest.TestCase):
    """MANDATORY AMENDMENT: bid/ask synthesized from mid (spread_imputed=True)
    are never crossed as if they were a real two-sided quote. `cross` and
    `surface` must widen off `mid` by `imputed_half_spread` instead, or a
    loader that sets bid=ask=mid would make cross() silently equal mid.
    """

    def test_imputed_point_under_cross_costs_credit_close_more_than_mid(self):
        # bid/ask on this point are junk (equal to mid, as a naive loader
        # would synthesize) — cross() must NOT just return them.
        pt = _pt(1.00, 1.00, spread_imputed=True)
        cm = CostModel.cross()
        mid = pt.mid
        cost = cm.close_price(pt, is_credit=True)
        self.assertGreater(cost, mid)
        self.assertAlmostEqual(cost, mid * (1 + cm.imputed_half_spread / 2))

    def test_imputed_point_under_cross_debit_close_less_than_mid(self):
        pt = _pt(1.00, 1.00, spread_imputed=True)
        cm = CostModel.cross()
        mid = pt.mid
        cost = cm.close_price(pt, is_credit=False)
        self.assertLess(cost, mid)
        self.assertAlmostEqual(cost, mid * (1 - cm.imputed_half_spread / 2))

    def test_imputed_point_under_mid_still_returns_exactly_mid(self):
        pt = _pt(1.00, 1.00, spread_imputed=True)
        cm = CostModel.mid()
        self.assertAlmostEqual(cm.close_price(pt, is_credit=True), pt.mid)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=False), pt.mid)

    def test_real_quote_point_is_unaffected_by_imputed_half_spread(self):
        pt = _pt(0.90, 1.10, spread_imputed=False)
        cm = CostModel("cross", imputed_half_spread=0.50)
        # A real quote ignores imputed_half_spread entirely: crossing still
        # takes the real ask/bid.
        self.assertAlmostEqual(cm.close_price(pt, is_credit=True), 1.10)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=False), 0.90)

    def test_zero_imputed_half_spread_behaves_like_mid_boundary(self):
        pt = _pt(1.00, 1.00, spread_imputed=True)
        cm = CostModel("cross", imputed_half_spread=0.0)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=True), pt.mid)
        self.assertAlmostEqual(cm.close_price(pt, is_credit=False), pt.mid)

    def test_costmodel_cross_still_constructs_positionally(self):
        cm = CostModel("cross")
        self.assertEqual(cm.setting, "cross")
        self.assertAlmostEqual(cm.imputed_half_spread, 0.095)


if __name__ == "__main__":
    unittest.main()
