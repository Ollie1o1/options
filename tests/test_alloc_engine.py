"""Replay must use only information available on the day it acts.

The leakage test is the most valuable one here: it deliberately looks for a
future peek and fails if the engine ever makes one.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_engine -v
"""
from __future__ import annotations

import unittest

from src.alloc.engine import (Trade, capital_at_risk, replay, select_legs)
from src.alloc.fills import Leg
from src.strategies.spec import StrategySpec


class FakeSource:
    """Chain source that records every (symbol, date) it is asked for."""

    def __init__(self, data):
        self.data = data
        self.requested = []

    def chain(self, symbol, date):
        self.requested.append((symbol, date))
        return self.data.get((symbol, date), [])


def _c(strike, typ, bid, ask, expiration="2024-03-15", delta=-0.25):
    return {"expiration": expiration, "strike": float(strike), "type": typ,
            "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.2,
            "delta": delta, "gamma": 0.01, "theta": -0.05,
            "vega": 0.1, "rho": 0.01}


def _spec(**kw):
    base = dict(id="bp", version=1, structure="bull_put",
                universe={"strata": ["liquid"]},
                entry={"dte": [20, 90], "short_delta": 0.25, "width": 5.0},
                exit={"profit_target": 0.5, "stop": 2.0,
                      "hold_to_expiry": False},
                sizing={"max_capital_at_risk": 4000, "max_concurrent": 5},
                created="2026-08-06", trial_count=1)
    base.update(kw)
    return StrategySpec(**base)


def _bull_put_chain(short_bid=2.0, short_ask=2.3, long_bid=0.8, long_ask=1.0):
    return [_c(100.0, "put", short_bid, short_ask, delta=-0.25),
            _c(95.0, "put", long_bid, long_ask, delta=-0.12)]


class LeakageTest(unittest.TestCase):
    def test_engine_never_requests_a_future_date(self):
        """THE test. Any request beyond the acting date is look-ahead."""
        dates = ["2024-01-05", "2024-01-12", "2024-01-19"]
        data = {("AAA", d): _bull_put_chain() for d in dates}
        src = FakeSource(data)
        replay(_spec(), ["AAA"], dates, src)

        seen_by_order = []
        for _sym, asked in src.requested:
            seen_by_order.append(asked)
        # every request must be one of the dates we walked, in order
        self.assertEqual(sorted(set(seen_by_order)), dates)
        for i, d in enumerate(seen_by_order):
            earlier = seen_by_order[:i]
            self.assertFalse([e for e in earlier if e > d],
                             "engine went backwards after reading a later date")

    def test_no_request_beyond_the_final_date(self):
        dates = ["2024-01-05", "2024-01-12"]
        src = FakeSource({("AAA", d): _bull_put_chain() for d in dates})
        replay(_spec(), ["AAA"], dates, src)
        self.assertLessEqual(max(d for _s, d in src.requested), max(dates))


class LegSelectionTest(unittest.TestCase):
    def test_bull_put_sells_high_buys_low(self):
        legs = select_legs(_spec(), _bull_put_chain(), "2024-01-05")
        self.assertEqual([(l.strike, l.action) for l in legs],
                         [(100.0, "sell"), (95.0, "buy")])

    def test_bear_call_sells_low_buys_high(self):
        chain = [_c(100.0, "call", 2.0, 2.3, delta=0.25),
                 _c(105.0, "call", 0.8, 1.0, delta=0.12)]
        legs = select_legs(_spec(structure="bear_call"), chain, "2024-01-05")
        self.assertEqual([(l.strike, l.action) for l in legs],
                         [(100.0, "sell"), (105.0, "buy")])

    def test_iron_condor_has_four_legs(self):
        chain = _bull_put_chain() + [_c(120.0, "call", 1.5, 1.7, delta=0.16),
                                     _c(125.0, "call", 0.5, 0.65, delta=0.08)]
        legs = select_legs(_spec(structure="iron_condor",
                                 entry={"dte": [20, 90], "short_delta": 0.20,
                                        "width": 5.0}),
                           chain, "2024-01-05")
        self.assertEqual(len(legs), 4)

    def test_long_call_is_a_single_bought_leg(self):
        chain = [_c(100.0, "call", 2.0, 2.3, delta=0.40)]
        legs = select_legs(_spec(structure="long_call",
                                 entry={"dte": [20, 90], "target_delta": 0.40}),
                           chain, "2024-01-05")
        self.assertEqual([(l.action) for l in legs], ["buy"])

    def test_expiry_outside_the_dte_window_is_rejected(self):
        chain = _bull_put_chain()          # expires 2024-03-15
        self.assertIsNone(select_legs(
            _spec(entry={"dte": [1, 5], "short_delta": 0.25, "width": 5.0}),
            chain, "2024-01-05"))

    def test_empty_chain_selects_nothing(self):
        self.assertIsNone(select_legs(_spec(), [], "2024-01-05"))


class FillDisciplineTest(unittest.TestCase):
    def test_entry_credit_is_the_crossed_price_not_the_mid(self):
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], src)
        self.assertTrue(trades)
        self.assertAlmostEqual(trades[0].entry_price, 1.00)   # 2.00 - 1.00
        self.assertNotAlmostEqual(trades[0].entry_price, 1.25)  # the mid

    def test_missing_wing_is_skipped_and_counted(self):
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25)]   # no 95 leg
        trades, stats = replay(_spec(), ["AAA"], ["2024-01-05"],
                               FakeSource({("AAA", "2024-01-05"): chain}))
        self.assertEqual(trades, [])
        self.assertGreaterEqual(stats["skipped_missing"], 1)

    def test_crossed_quote_is_skipped_and_counted(self):
        chain = [_c(100.0, "put", 2.5, 2.0, delta=-0.25),
                 _c(95.0, "put", 0.8, 1.0, delta=-0.12)]
        trades, stats = replay(_spec(), ["AAA"], ["2024-01-05"],
                               FakeSource({("AAA", "2024-01-05"): chain}))
        self.assertEqual(trades, [])
        self.assertGreaterEqual(stats["skipped_crossed"], 1)

    def test_empty_chain_produces_no_trade_and_no_crash(self):
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], FakeSource({}))
        self.assertEqual(trades, [])

    def test_a_credit_structure_filling_at_a_debit_is_rejected(self):
        """Paying to open a credit spread is a broken quote, not a trade."""
        chain = [_c(100.0, "put", 0.10, 0.20, delta=-0.25),
                 _c(95.0, "put", 2.00, 2.50, delta=-0.12)]
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"],
                           FakeSource({("AAA", "2024-01-05"): chain}))
        self.assertEqual(trades, [])


class CapitalTest(unittest.TestCase):
    def test_defined_risk_is_width_minus_credit(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell"), Leg("2024-03-15", 95.0, "put", "buy")]
        self.assertAlmostEqual(capital_at_risk(_spec(), legs, 1.00), 400.0)

    def test_cash_secured_put_ties_up_the_whole_strike(self):
        legs = [Leg("2024-03-15", 100.0, "put", "sell")]
        car = capital_at_risk(_spec(structure="short_put"), legs, 2.00)
        self.assertAlmostEqual(car, 9800.0)

    def test_position_over_the_cap_is_not_opened(self):
        spec = _spec(sizing={"max_capital_at_risk": 10, "max_concurrent": 5})
        trades, stats = replay(spec, ["AAA"], ["2024-01-05"],
                               FakeSource({("AAA", "2024-01-05"):
                                           _bull_put_chain()}))
        self.assertEqual(trades, [])
        self.assertGreaterEqual(stats["skipped_capital"], 1)

    def test_concurrency_cap_is_respected(self):
        dates = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
        src = FakeSource({("AAA", d): _bull_put_chain() for d in dates})
        spec = _spec(sizing={"max_capital_at_risk": 4000, "max_concurrent": 1},
                     exit={"hold_to_expiry": True})
        trades, _ = replay(spec, ["AAA"], dates, src)
        self.assertEqual(len(trades), 1)


class ExitTest(unittest.TestCase):
    def _dates(self):
        return ["2024-01-05", "2024-01-12"]

    def test_profit_target_closes_the_position(self):
        data = {("AAA", "2024-01-05"): _bull_put_chain(),
                ("AAA", "2024-01-12"): _bull_put_chain(0.30, 0.40, 0.05, 0.10)}
        trades, _ = replay(_spec(), ["AAA"], self._dates(), FakeSource(data))
        closed = [t for t in trades if t.exit_date]
        self.assertTrue(closed)
        self.assertEqual(closed[0].exit_reason, "profit_target")

    def test_closing_pays_the_ask_and_sells_the_bid(self):
        data = {("AAA", "2024-01-05"): _bull_put_chain(),
                ("AAA", "2024-01-12"): _bull_put_chain(0.30, 0.40, 0.05, 0.10)}
        trades, _ = replay(_spec(), ["AAA"], self._dates(), FakeSource(data))
        closed = [t for t in trades if t.exit_date][0]
        # buy back short at 0.40, sell long at 0.05 -> -0.35
        self.assertAlmostEqual(closed.exit_price, -0.35)

    def test_pnl_is_entry_plus_exit_times_100(self):
        data = {("AAA", "2024-01-05"): _bull_put_chain(),
                ("AAA", "2024-01-12"): _bull_put_chain(0.30, 0.40, 0.05, 0.10)}
        trades, _ = replay(_spec(), ["AAA"], self._dates(), FakeSource(data))
        closed = [t for t in trades if t.exit_date][0]
        self.assertAlmostEqual(closed.pnl, (1.00 - 0.35) * 100)

    def test_hold_to_expiry_ignores_the_profit_target(self):
        data = {("AAA", "2024-01-05"): _bull_put_chain(),
                ("AAA", "2024-01-12"): _bull_put_chain(0.30, 0.40, 0.05, 0.10)}
        trades, _ = replay(_spec(exit={"hold_to_expiry": True}), ["AAA"],
                           self._dates(), FakeSource(data))
        self.assertFalse([t for t in trades
                          if t.exit_reason == "profit_target"])


class TickerEndedTest(unittest.TestCase):
    """FB stopped existing on 2022-06-03. Open positions must not vanish."""

    def test_position_is_closed_when_the_ticker_ends(self):
        dates = ["2024-01-05", "2024-01-12"]
        data = {("AAA", "2024-01-05"): _bull_put_chain()}   # nothing on the 12th
        trades, stats = replay(_spec(exit={"hold_to_expiry": True}),
                               ["AAA"], dates, FakeSource(data),
                               terminal={"AAA": "2024-01-05"})
        self.assertTrue(trades)
        self.assertEqual(trades[0].exit_reason, "ticker_ended")
        self.assertEqual(stats["ticker_ended"], 1)

    def test_a_live_ticker_is_not_force_closed(self):
        dates = ["2024-01-05", "2024-01-12"]
        data = {("AAA", d): _bull_put_chain() for d in dates}
        trades, stats = replay(_spec(exit={"hold_to_expiry": True}),
                               ["AAA"], dates, FakeSource(data),
                               terminal={"AAA": "2024-06-01"})
        self.assertEqual(stats["ticker_ended"], 0)


class StratumTest(unittest.TestCase):
    def test_trades_carry_their_stratum(self):
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], src,
                           stratum_of={"AAA": "broad"})
        self.assertEqual(trades[0].stratum, "broad")

    def test_default_arguments_all_omitted(self):
        """terminal, stratum_of and seed all defaulted."""
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, stats = replay(_spec(), ["AAA"], ["2024-01-05"], src)
        self.assertTrue(trades)
        self.assertIn("opened", stats)


if __name__ == "__main__":
    unittest.main()
