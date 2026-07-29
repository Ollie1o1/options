"""Capital-at-risk: the dollars a trade can actually lose, per structure.

Every strategy comparison in the repo sums raw USD across trades whose risk
differs by two orders of magnitude, and the auto-log feeder has no budget
limit. Both need one agreed definition of "how much does this position tie
up", so it lives in one place and is tested here.
"""
import unittest

from src.capital_risk import capital_at_risk


class TestLongPremium(unittest.TestCase):
    def test_long_call_risk_is_the_debit_paid(self):
        # $2.34 debit × 100 shares = $234 at risk, and that is the whole loss.
        self.assertAlmostEqual(
            capital_at_risk("Long Call", entry_price=2.34, strike=130.0), 234.0
        )

    def test_long_put_risk_is_the_debit_paid(self):
        self.assertAlmostEqual(
            capital_at_risk("Long Put", entry_price=0.53, strike=40.0), 53.0
        )

    def test_quantity_scales_the_debit(self):
        self.assertAlmostEqual(
            capital_at_risk("Long Call", entry_price=2.34, strike=130.0, quantity=3),
            702.0,
        )


class TestDefinedRiskCredit(unittest.TestCase):
    def test_credit_spread_uses_stored_max_loss(self):
        # Bull Put, $1 wide, $0.50 credit -> $50 max loss, as stored by log_spread.
        self.assertAlmostEqual(
            capital_at_risk(
                "Bull Put", entry_price=0.50, strike=80.0, max_loss_usd=50.0
            ),
            50.0,
        )

    def test_iron_condor_uses_stored_max_loss(self):
        self.assertAlmostEqual(
            capital_at_risk(
                "Iron Condor", entry_price=1.20, strike=400.0, max_loss_usd=4033.0
            ),
            4033.0,
        )

    def test_stored_max_loss_wins_over_the_debit_rule(self):
        # A defined-risk structure must never be costed as its premium.
        risk = capital_at_risk(
            "Bear Call", entry_price=0.45, strike=100.0, max_loss_usd=55.0
        )
        self.assertAlmostEqual(risk, 55.0)


class TestRealStrategyNameSpace(unittest.TestCase):
    """The ledger and the sleeve use qualified names, not just the six canonical
    ones. Exact-set matching silently classified 'Lottery Long Call' as
    unbounded, which the budget gate then refused to log."""

    def test_lottery_long_call_is_a_debit_like_any_other_long(self):
        self.assertAlmostEqual(
            capital_at_risk("Lottery Long Call", entry_price=1.20, strike=180.0), 120.0
        )

    def test_lottery_long_put_is_a_debit(self):
        self.assertAlmostEqual(
            capital_at_risk("Lottery Long Put", entry_price=0.40, strike=12.0), 40.0
        )

    def test_debit_spread_risk_is_the_net_debit_paid(self):
        self.assertAlmostEqual(
            capital_at_risk("Bull Call Spread", entry_price=1.75, strike=100.0), 175.0
        )

    def test_credit_spread_without_stored_max_loss_derives_it_from_width(self):
        # $1 wide, $0.50 credit -> $50 at risk. Costing it as the credit would
        # understate a credit spread as if it were a debit.
        self.assertAlmostEqual(
            capital_at_risk(
                "Bull Put", entry_price=0.50, strike=80.0,
                spread_width=1.0, net_credit=0.50,
            ),
            50.0,
        )

    def test_credit_spread_with_no_width_data_is_unknown_not_the_credit(self):
        self.assertIsNone(
            capital_at_risk("Bear Call", entry_price=0.45, strike=100.0)
        )

    def test_iron_condor_without_stored_max_loss_uses_the_widest_wing(self):
        self.assertAlmostEqual(
            capital_at_risk(
                "Iron Condor", entry_price=1.20, strike=400.0,
                spread_width=5.0, net_credit=1.20,
            ),
            380.0,
        )


class TestNakedShorts(unittest.TestCase):
    def test_short_put_is_collateral_not_credit(self):
        # The bug this guards: a WFC 77.5 short put collecting $1.52 ties up
        # $7,598 of collateral, not $152. Treating the credit as the risk made
        # 74 unaffordable short puts look affordable.
        self.assertAlmostEqual(
            capital_at_risk("Short Put", entry_price=1.52, strike=77.5), 7598.0
        )

    def test_short_call_risk_is_unbounded_and_reported_as_unknown(self):
        # Naked upside cannot be bounded, so it cannot be sized. None, not zero.
        self.assertIsNone(capital_at_risk("Short Call", entry_price=1.10, strike=95.0))


class TestCrypto(unittest.TestCase):
    def test_crypto_uses_a_multiplier_of_one(self):
        # BTC/ETH paper rows are whole-coin, not 100-share contracts.
        self.assertAlmostEqual(
            capital_at_risk("Long Call", entry_price=1200.0, strike=70000.0, ticker="BTC"),
            1200.0,
        )


class TestUnusableInputs(unittest.TestCase):
    def test_missing_entry_price_is_unknown(self):
        self.assertIsNone(capital_at_risk("Long Call", entry_price=None, strike=130.0))

    def test_unrecognised_debit_strategy_risks_what_was_paid(self):
        # A structure paid for with a debit cannot lose more than the debit, so
        # an unfamiliar long name is sizable. Refusing it was what stopped the
        # lottery sleeve from logging at all.
        self.assertAlmostEqual(
            capital_at_risk("Calendar", entry_price=1.0, strike=10.0), 100.0
        )

    def test_unrecognised_short_strategy_is_unknown(self):
        # The reverse does not hold: an unfamiliar short cannot be bounded.
        self.assertIsNone(
            capital_at_risk("Short Strangle", entry_price=2.0, strike=100.0)
        )

    def test_empty_strategy_name_is_unknown(self):
        self.assertIsNone(capital_at_risk("", entry_price=1.0, strike=10.0))

    def test_non_positive_stored_max_loss_falls_through_to_the_rule(self):
        # A zero/None max_loss is missing data, not a free trade.
        self.assertAlmostEqual(
            capital_at_risk(
                "Long Call", entry_price=2.00, strike=50.0, max_loss_usd=0.0
            ),
            200.0,
        )


class TestAffordability(unittest.TestCase):
    def test_trade_inside_the_cap_is_affordable(self):
        self.assertTrue(is_affordable("Long Call", 2.34, 130.0, cap=750.0))

    def test_trade_above_the_cap_is_not(self):
        self.assertFalse(is_affordable("Short Put", 1.52, 77.5, cap=750.0))

    def test_no_cap_means_everything_is_allowed(self):
        self.assertTrue(is_affordable("Short Put", 1.52, 77.5, cap=None))

    def test_unknown_risk_is_not_affordable_when_a_cap_is_set(self):
        # You cannot size what you cannot bound.
        self.assertFalse(is_affordable("Short Call", 1.10, 95.0, cap=750.0))


def is_affordable(strategy, entry_price, strike, cap, **kw):
    from src.capital_risk import within_budget

    return within_budget(
        capital_at_risk(strategy, entry_price=entry_price, strike=strike, **kw), cap
    )


if __name__ == "__main__":
    unittest.main()
