"""The cost line printed when positions auto-close.

This is the only place a non-developer sees what the system charged. It read
"$0.05/share slippage x2 + $0.65/contract commissions x2" — a commission that
does not exist on this broker, and no mention of the currency conversion that
turned out to be the largest cost in the book. If the disclosure is wrong, the
P&L looks unexplained.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import cost_disclosure


class TestCostDisclosure(unittest.TestCase):
    def test_names_every_cost_actually_charged(self):
        line = cost_disclosure(slippage=0.05, commission=0.0, fx_rate=0.015)
        self.assertIn("0.05", line)
        self.assertIn("1.5%", line)

    def test_a_zero_cost_is_not_listed_as_if_it_were_charged(self):
        line = cost_disclosure(slippage=0.05, commission=0.0, fx_rate=0.015)
        self.assertNotIn("commission", line.lower())

    def test_commission_appears_when_a_broker_actually_charges_one(self):
        line = cost_disclosure(slippage=0.05, commission=0.65, fx_rate=0.0)
        self.assertIn("commission", line.lower())
        self.assertIn("0.65", line)

    def test_conversion_is_omitted_with_a_usd_account(self):
        line = cost_disclosure(slippage=0.05, commission=0.0, fx_rate=0.0)
        self.assertNotIn("%", line)

    def test_points_the_reader_at_the_full_schedule(self):
        # A number with no source is not explainable to whoever reads it later.
        line = cost_disclosure(slippage=0.05, commission=0.0, fx_rate=0.015)
        self.assertIn("BROKER_COSTS", line)

    def test_never_returns_an_empty_disclosure(self):
        line = cost_disclosure(slippage=0.0, commission=0.0, fx_rate=0.0)
        self.assertIn("no", line.lower())


if __name__ == "__main__":
    unittest.main()
