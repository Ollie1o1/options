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

    def test_a_mixed_frame_keeps_none_rather_than_coercing_to_nan(self):
        """A single-row all-None frame stays object-dtype by accident, which
        masked this: pandas silently upcasts a column to float64 and turns
        None into NaN the moment the list backing it contains BOTH a None
        and a real float, and NaN is a third, undocumented state on top of
        the module's None-vs-0 contract."""
        df = pd.DataFrame([
            {"symbol": "GOOD", "max_profit": 73.0, "max_loss": 127.0,
             "ev_per_contract": 5.0},
            {"symbol": "GAPPY", "max_profit": 50.0, "ev_per_contract": 3.0},
        ])
        out = bv.annotate(df, "Bull Put")
        for col in ("capital_at_risk", "reward_per_risk", "net_ev_per_risk"):
            self.assertIsNone(out[col].iloc[1], msg=f"{col} on GAPPY")
        self.assertAlmostEqual(out["capital_at_risk"].iloc[0], 127.0)
        self.assertAlmostEqual(out["reward_per_risk"].iloc[0], 73.0 / 127.0)
        self.assertAlmostEqual(out["net_ev_per_risk"].iloc[0], 5.0 / 127.0)


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


class TestRewardOnSingleLegs(unittest.TestCase):
    """`Reward/$risk` has to answer on the board that motivated the budget.

    Only the spread and condor builders set a `max_profit` column, so reading
    it off the row left the reward blank on all four single-leg boards. For a
    long option that blank is correct — the upside is unbounded. For a SHORT
    put it was a hole: the credit over the collateral is the entire return,
    and the cash-secured put is the structure the $4,000 cap was flatlining.
    """

    def test_a_short_put_now_has_a_reward(self):
        df = pd.DataFrame([{"symbol": "AVGO", "strike": 350.0,
                            "premium": 2.00, "ev_per_contract": 60.0}])
        out = bv.annotate(df, "Short Put")
        self.assertAlmostEqual(out["capital_at_risk"].iloc[0], 34800.0)
        self.assertAlmostEqual(out["reward_per_risk"].iloc[0], 200.0 / 34800.0)

    def test_a_long_call_reward_stays_blank_not_zero(self):
        df = pd.DataFrame([{"symbol": "NVDA", "strike": 190.0,
                            "premium": 4.24, "ev_per_contract": 31.0}])
        out = bv.annotate(df, "Long Call")
        self.assertIsNone(out["reward_per_risk"].iloc[0],
                          "unbounded upside must read as unanswerable, not 0")
        self.assertIsNotNone(out["net_ev_per_risk"].iloc[0],
                             "EV per risk is still answerable for a long leg")

    def test_a_stale_max_profit_column_cannot_override_the_structure_rule(self):
        """A single-leg row that happens to carry a max_profit is not evidence.

        `max_loss` on a single-leg row is `entry_price * 100` and sizing a
        cash-secured put off it gave $50 instead of $31,850. The reward side
        must not repeat that: the strategy decides, not a stray column.
        """
        df = pd.DataFrame([{"symbol": "NVDA", "strike": 190.0, "premium": 4.24,
                            "max_profit": 99999.0, "ev_per_contract": 31.0}])
        out = bv.annotate(df, "Long Call")
        self.assertIsNone(out["reward_per_risk"].iloc[0])
