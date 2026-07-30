"""Position sizing assumed every trade was long premium.

`get_position_sizing_recommendation` computed `max_loss = premium * 100` — the
debit — with no reference to the strategy, then divided the risk budget by it to
get a contract count. That is right for a long call and badly wrong for a short
put, where the capital tied up is collateral: a $318 strike at $0.50 was sized
as risking $50 when it actually ties up $31,825, ~600x more.

The number feeds the QUICK COMPARISON table, which renders in Premium Selling
mode too, so it was guidance the operator reads immediately before placing a
trade. Sizing now goes through the same `capital_at_risk` definition the ledger
and the auto-log budget gate use.
"""
import unittest

import pandas as pd

from src.trade_analysis import (get_position_sizing_recommendation,
                                strategy_label_for_mode)


def _row(**kw):
    base = {"symbol": "AAPL", "type": "put", "strike": 318.75,
            "premium": 0.50, "prob_profit": 0.7}
    base.update(kw)
    return pd.Series(base)


class ShortPremiumSizing(unittest.TestCase):
    ACCOUNT = 25_000.0

    def test_short_put_is_sized_off_collateral_not_the_premium(self):
        # (318.75 - 0.50) x 100 = $31,825 tied up — more than the account.
        res = get_position_sizing_recommendation(
            _row(), self.ACCOUNT, strategy_name="Short Put")
        self.assertEqual(res["contracts"], 0)

    def test_the_legacy_math_would_have_recommended_contracts(self):
        # Guards the regression itself: without the strategy the same row is
        # sized off a $50 "max loss" and comes back tradeable.
        legacy = get_position_sizing_recommendation(_row(), self.ACCOUNT)
        self.assertGreater(legacy["contracts"], 0)

    def test_an_affordable_short_put_is_still_sized(self):
        # A $30 underlying ties up $2,960 — that one fits.
        res = get_position_sizing_recommendation(
            _row(symbol="F", strike=30.0, premium=0.40),
            self.ACCOUNT, strategy_name="Short Put")
        self.assertGreaterEqual(res["contracts"], 1)

    def test_total_cost_reflects_collateral(self):
        res = get_position_sizing_recommendation(
            _row(symbol="F", strike=30.0, premium=0.40),
            self.ACCOUNT, strategy_name="Short Put")
        self.assertAlmostEqual(res["total_cost"],
                               2960.0 * res["contracts"], places=2)

    def test_naked_short_call_cannot_be_sized(self):
        # Unbounded loss has no denominator; recommending a count would be a lie.
        res = get_position_sizing_recommendation(
            _row(type="call", strike=500.0, premium=3.20),
            self.ACCOUNT, strategy_name="Short Call")
        self.assertEqual(res["contracts"], 0)


class LongPremiumSizingUnchanged(unittest.TestCase):
    ACCOUNT = 25_000.0

    def test_long_call_sizes_off_the_debit(self):
        row = _row(type="call", strike=465.0, premium=12.50)
        res = get_position_sizing_recommendation(
            row, self.ACCOUNT, strategy_name="Long Call")
        legacy = get_position_sizing_recommendation(row, self.ACCOUNT)
        self.assertEqual(res["contracts"], legacy["contracts"])

    def test_omitting_the_strategy_keeps_the_old_behaviour(self):
        # Callers that never learned about strategies must not change meaning.
        row = _row(type="call", strike=465.0, premium=12.50)
        res = get_position_sizing_recommendation(row, self.ACCOUNT)
        self.assertGreater(res["contracts"], 0)
        self.assertAlmostEqual(res["total_cost"], 1250.0 * res["contracts"])

    def test_a_zero_premium_row_is_untradeable(self):
        res = get_position_sizing_recommendation(
            _row(premium=0.0), self.ACCOUNT, strategy_name="Long Call")
        self.assertEqual(res["contracts"], 0)


class StrategyLabel(unittest.TestCase):
    def test_premium_selling_maps_to_short(self):
        self.assertEqual(strategy_label_for_mode("Premium Selling", "put"),
                         "Short Put")

    def test_other_modes_map_to_long(self):
        self.assertEqual(strategy_label_for_mode("Discovery", "call"),
                         "Long Call")

    def test_it_matches_the_screener_helper_it_replaces(self):
        # One rule, one place — the screener's copy must delegate, not diverge.
        from src.options_screener import _strategy_label_for_mode
        for mode in ("Premium Selling", "Discovery", "Budget"):
            for t in ("put", "call"):
                self.assertEqual(_strategy_label_for_mode(mode, t),
                                 strategy_label_for_mode(mode, t))


if __name__ == "__main__":
    unittest.main()
