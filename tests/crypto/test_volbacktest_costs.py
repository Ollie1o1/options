import unittest

from src.crypto.volbacktest.costs import CostModel


class TestCosts(unittest.TestCase):
    def test_zero_cost_model_charges_nothing(self):
        c = CostModel(option_spread_frac=0.0, hedge_slippage_bps=0.0)
        self.assertEqual(c.option_entry(premium_usd=1000.0), 0.0)
        self.assertEqual(c.hedge_trade(notional_usd=5000.0), 0.0)

    def test_option_entry_is_half_spread(self):
        c = CostModel(option_spread_frac=0.04, hedge_slippage_bps=0.0)
        self.assertAlmostEqual(c.option_entry(premium_usd=1000.0), 20.0)

    def test_hedge_cost_monotonic_in_slippage(self):
        lo = CostModel(0.0, 1.0).hedge_trade(10000.0)
        hi = CostModel(0.0, 5.0).hedge_trade(10000.0)
        self.assertGreater(hi, lo)
        self.assertAlmostEqual(CostModel(0.0, 10.0).hedge_trade(10000.0), 10.0)


if __name__ == "__main__":
    unittest.main()
