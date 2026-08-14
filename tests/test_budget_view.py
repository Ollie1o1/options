"""Candidates expressed per dollar of capital at risk.

The point of the axis: a $127 spread and a $34,680 cash-secured put with the
same return on risk must produce the SAME reward_per_risk, so a small budget
and a large one can be compared at all. Measured on 877 closed trades, spending
more did not buy better outcomes per dollar — credit structures returned +16.0%
in the $250-500 bucket against -0.3% above $15,000.

These columns are DISPLAY-ONLY. They must not enter quality_score, any weight,
or any gate, and they must not re-sort the board: ranking was disproven at
Wilcoxon p=0.89, so the board keeps its cost-survival order.
"""
from __future__ import annotations

import unittest

import pandas as pd

from src import budget_view as bv


def _df():
    """Two Bull Puts with identical return on risk at very different sizes."""
    return pd.DataFrame([
        {"symbol": "CHEAP", "max_profit": 73.0, "max_loss": 127.0,
         "ev_per_contract": 5.0},
        {"symbol": "RICH", "max_profit": 19900.0, "max_loss": 34600.0,
         "ev_per_contract": 1362.2},
    ])


class TestPerRisk(unittest.TestCase):

    def test_divides_by_risk(self):
        self.assertAlmostEqual(bv.per_risk(50.0, 200.0), 0.25)

    def test_unknown_risk_is_none_not_zero(self):
        self.assertIsNone(bv.per_risk(50.0, None))

    def test_zero_or_negative_risk_is_none_not_infinity(self):
        self.assertIsNone(bv.per_risk(50.0, 0.0))
        self.assertIsNone(bv.per_risk(50.0, -1.0))

    def test_missing_value_is_none(self):
        self.assertIsNone(bv.per_risk(None, 200.0))


class TestAnnotate(unittest.TestCase):

    def test_cheap_and_expensive_land_on_the_same_axis(self):
        out = bv.annotate(_df(), "Bull Put")
        a, b = out["reward_per_risk"].tolist()
        self.assertAlmostEqual(a, b, places=3)

    def test_row_order_is_unchanged(self):
        out = bv.annotate(_df(), "Bull Put")
        self.assertEqual(out["symbol"].tolist(), ["CHEAP", "RICH"])

    def test_net_ev_is_per_dollar_not_per_contract(self):
        """ev_per_contract flatters big positions; that is the bug the book
        was made of."""
        out = bv.annotate(_df(), "Bull Put")
        cheap, rich = out["net_ev_per_risk"].tolist()
        self.assertAlmostEqual(cheap, rich, places=3)
        self.assertLess(out["net_ev_per_risk"].max(), 1.0)

    def test_unknown_risk_renders_none_rather_than_a_number(self):
        df = pd.DataFrame([{"symbol": "X", "max_profit": 10.0,
                            "ev_per_contract": 1.0}])
        out = bv.annotate(df, "Naked Call")
        self.assertIsNone(out["reward_per_risk"].iloc[0])

    def test_an_empty_frame_survives(self):
        out = bv.annotate(pd.DataFrame(), "Bull Put")
        self.assertTrue(out.empty)


class TestAffordable(unittest.TestCase):

    def test_no_budget_returns_everything(self):
        self.assertEqual(len(bv.affordable(_df(), None, "Bull Put")), 2)

    def test_a_budget_drops_what_does_not_fit(self):
        out = bv.affordable(_df(), 1000.0, "Bull Put")
        self.assertEqual(out["symbol"].tolist(), ["CHEAP"])

    def test_a_generous_budget_keeps_everything(self):
        self.assertEqual(len(bv.affordable(_df(), 100000.0, "Bull Put")), 2)

    def test_order_is_preserved(self):
        out = bv.affordable(_df(), 100000.0, "Bull Put")
        self.assertEqual(out["symbol"].tolist(), ["CHEAP", "RICH"])


class TestBudgetUseLine(unittest.TestCase):

    def test_reports_how_many_of_the_cheapest_fit(self):
        line = bv.budget_use_line(bv.annotate(_df(), "Bull Put"), 2000.0)
        self.assertIn("15", line)          # floor(2000 / 127)

    def test_omitted_entirely_when_no_budget_was_chosen(self):
        self.assertIsNone(bv.budget_use_line(bv.annotate(_df(), "Bull Put"), None))

    def test_omitted_when_there_are_no_rows(self):
        self.assertIsNone(bv.budget_use_line(pd.DataFrame(), 2000.0))


if __name__ == "__main__":
    unittest.main()
