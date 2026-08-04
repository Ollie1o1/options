"""Tests for src/execution_restate.py — turning ledger rows into priced legs.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_execution_restate -v
"""
import unittest

from src import execution_restate as er


def row(**kw):
    """A ledger row with every column the restater reads, defaulted to NULL."""
    base = dict(entry_id=1, ticker="ORCL", date="2026-06-10", expiration="2026-07-17",
                strategy_name=None, type=None, strike=None, long_strike=None,
                short_call_strike=None, long_call_strike=None,
                short_put_strike=None, long_put_strike=None,
                spread_width=None, net_credit=None, entry_price=None)
    base.update(kw)
    return base


class LegsFromTradeTest(unittest.TestCase):
    def test_bull_put_sells_the_high_strike_and_buys_the_low(self):
        legs = er.legs_from_trade(row(strategy_name="Bull Put", type="put",
                                      strike=80.0, long_strike=79.0))
        self.assertEqual(legs, [
            {"strike": 80.0, "type": "put", "side": "sell"},
            {"strike": 79.0, "type": "put", "side": "buy"},
        ])

    def test_bear_call_sells_the_low_strike_and_buys_the_high(self):
        legs = er.legs_from_trade(row(strategy_name="Bear Call", type="call",
                                      strike=440.0, long_strike=442.5))
        self.assertEqual(legs, [
            {"strike": 440.0, "type": "call", "side": "sell"},
            {"strike": 442.5, "type": "call", "side": "buy"},
        ])

    def test_iron_condor_builds_all_four_legs(self):
        legs = er.legs_from_trade(row(
            strategy_name="Iron Condor", short_call_strike=420.0,
            long_call_strike=425.0, short_put_strike=400.0, long_put_strike=395.0))
        self.assertEqual(len(legs), 4)
        self.assertIn({"strike": 420.0, "type": "call", "side": "sell"}, legs)
        self.assertIn({"strike": 425.0, "type": "call", "side": "buy"}, legs)
        self.assertIn({"strike": 400.0, "type": "put", "side": "sell"}, legs)
        self.assertIn({"strike": 395.0, "type": "put", "side": "buy"}, legs)

    def test_an_iron_condor_missing_its_call_side_is_refused(self):
        """13 of 187 logged condors stored only the put legs. A two-legged
        'condor' is not a condor, and pricing it as one would understate its
        friction by half."""
        legs = er.legs_from_trade(row(
            strategy_name="Iron Condor", short_put_strike=415.0, long_put_strike=390.0))
        self.assertIsNone(legs)

    def test_long_call_is_a_single_bought_leg(self):
        legs = er.legs_from_trade(row(strategy_name="Long Call", type="call", strike=130.0))
        self.assertEqual(legs, [{"strike": 130.0, "type": "call", "side": "buy"}])

    def test_long_put_is_a_single_bought_leg(self):
        legs = er.legs_from_trade(row(strategy_name="Long Put", type="put", strike=80.0))
        self.assertEqual(legs, [{"strike": 80.0, "type": "put", "side": "buy"}])

    def test_short_put_is_a_single_sold_leg(self):
        legs = er.legs_from_trade(row(strategy_name="Short Put", type="put", strike=77.5))
        self.assertEqual(legs, [{"strike": 77.5, "type": "put", "side": "sell"}])

    def test_a_spread_missing_its_long_strike_is_refused(self):
        self.assertIsNone(er.legs_from_trade(
            row(strategy_name="Bull Put", type="put", strike=80.0)))

    def test_an_unknown_strategy_is_refused_rather_than_guessed(self):
        self.assertIsNone(er.legs_from_trade(
            row(strategy_name="Calendar Diagonal", type="put", strike=80.0)))


class RestateTest(unittest.TestCase):
    """Restating one row against a quote lookup."""

    BULL_PUT = row(strategy_name="Bull Put", type="put", strike=80.0,
                   long_strike=79.0, spread_width=1.0, net_credit=0.50)

    QUOTES = {(80.0, "put"): (1.40, 1.60), (79.0, "put"): (0.40, 0.60)}

    def test_restating_prices_all_three_policies(self):
        out = er.restate(self.BULL_PUT, lambda s, t: self.QUOTES.get((s, t)))
        self.assertAlmostEqual(out["entry_price_mid"], 1.00)
        self.assertAlmostEqual(out["entry_price_cross"], 0.80)
        self.assertAlmostEqual(out["entry_price_fill"], 1.00 - 0.35 * 0.20)

    def test_restating_records_the_policy_and_a_live_quote_source(self):
        out = er.restate(self.BULL_PUT, lambda s, t: self.QUOTES.get((s, t)))
        self.assertEqual(out["fill_policy"], "limit")
        self.assertEqual(out["fill_source"], "live_quote")

    def test_a_row_with_no_quotes_is_marked_unknown_not_invented(self):
        out = er.restate(self.BULL_PUT, lambda s, t: None)
        self.assertEqual(out["fill_source"], "unknown")
        self.assertIsNone(out["entry_price_mid"])
        self.assertIsNone(out["entry_price_fill"])
        self.assertIsNone(out["entry_price_cross"])

    def test_a_row_with_one_missing_leg_is_unknown_not_half_priced(self):
        partial = {(80.0, "put"): (1.40, 1.60)}
        out = er.restate(self.BULL_PUT, lambda s, t: partial.get((s, t)))
        self.assertEqual(out["fill_source"], "unknown")
        self.assertIsNone(out["entry_price_fill"])

    def test_an_unpriceable_structure_is_unknown(self):
        bad = row(strategy_name="Iron Condor", short_put_strike=415.0, long_put_strike=390.0)
        out = er.restate(bad, lambda s, t: (1.0, 1.2))
        self.assertEqual(out["fill_source"], "unknown")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
