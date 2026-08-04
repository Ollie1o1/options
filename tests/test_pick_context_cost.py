"""The cost line on each pick: what you pay to trade it, and what you must beat.

The scan path priced every candidate at the mid and showed no friction at all,
which is how a 27%-of-credit crossing cost stayed invisible through 907 logged
trades. This puts it in front of the decision.
"""
import unittest

from src.pick_context import cost_line


class CostLineTest(unittest.TestCase):
    def test_a_tight_single_leg_reports_a_low_round_trip_cost(self):
        line = cost_line({"strategy_name": "Long Call", "bid": 9.90, "ask": 10.10})
        self.assertIn("2%", line)
        self.assertIn("Cost", line)

    def test_a_credit_spread_reports_both_friction_and_breakeven(self):
        line = cost_line({"strategy_name": "Bull Put", "net_credit": 1.00,
                          "spread_width": 2.50, "short_bid": 1.40, "short_ask": 1.60,
                          "long_bid": 0.40, "long_ask": 0.60})
        self.assertIn("40%", line)          # round-trip friction
        self.assertIn("breakeven", line.lower())

    def test_it_names_your_own_win_rate_when_one_is_supplied(self):
        line = cost_line({"strategy_name": "Bull Put", "net_credit": 1.00,
                          "spread_width": 2.50, "short_bid": 1.48, "short_ask": 1.52,
                          "long_bid": 0.48, "long_ask": 0.52},
                         win_rates={"Bull Put": 0.66})
        self.assertIn("66%", line)

    def test_a_refused_candidate_says_so(self):
        line = cost_line({"strategy_name": "Long Call", "bid": 5.0, "ask": 15.0})
        self.assertIn("REFUSED", line.upper())

    def test_an_unquotable_pick_returns_nothing_rather_than_guessing(self):
        self.assertEqual(cost_line({"strategy_name": "Long Call"}), "")

    def test_it_never_raises_on_a_malformed_row(self):
        for bad in ({}, {"bid": "x", "ask": None}, {"strategy_name": None}):
            self.assertIsInstance(cost_line(bad), str)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
