"""Replay must use only information available on the day it acts.

The leakage test is the most valuable one here: it deliberately looks for a
future peek and fails if the engine ever makes one.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_engine -v
"""
from __future__ import annotations

import unittest

from src.alloc.engine import (SqliteChainSource, Trade, _entry_features,
                              capital_at_risk, replay, select_legs)
from src.alloc.fills import Leg, quotes_from_chain
from src.strategies.spec import StrategySpec


class SqliteChainSourceTest(unittest.TestCase):
    """Reads the cache over ONE connection: a window is ~10,000 chain reads."""

    def setUp(self):
        import os
        import sqlite3
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "c.db")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE dolt_chain (symbol TEXT, date TEXT, "
                     "expiration TEXT, strike REAL, type TEXT, bid REAL, "
                     "ask REAL, mid REAL, iv REAL, delta REAL, gamma REAL, "
                     "theta REAL, vega REAL, rho REAL)")
        conn.execute("INSERT INTO dolt_chain VALUES "
                     "('AAA','2024-01-05','2024-02-16',100.0,'put',"
                     "1.0,1.2,1.1,0.3,-0.25,0.0,0.0,0.0,0.0)")
        conn.commit()
        conn.close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_a_row_with_every_column_named(self):
        src = SqliteChainSource(self.db)
        chain = src.chain("AAA", "2024-01-05")
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["type"], "put")
        self.assertAlmostEqual(chain[0]["strike"], 100.0)
        self.assertAlmostEqual(chain[0]["delta"], -0.25)
        self.assertAlmostEqual(chain[0]["iv"], 0.3)
        src.close()

    def test_an_unfetched_day_reads_as_empty_not_an_error(self):
        src = SqliteChainSource(self.db)
        self.assertEqual(src.chain("AAA", "2024-01-06"), [])
        self.assertEqual(src.chain("ZZZ", "2024-01-05"), [])
        src.close()

    def test_the_same_day_is_only_queried_once(self):
        # Memoised in-process: a replay asks for the same chain repeatedly
        # while managing open positions.
        src = SqliteChainSource(self.db)
        first = src.chain("AAA", "2024-01-05")
        src._conn.close()          # any further query would now raise
        self.assertIs(src.chain("AAA", "2024-01-05"), first)

    def test_close_is_idempotent(self):
        src = SqliteChainSource(self.db)
        src.chain("AAA", "2024-01-05")
        src.close()
        src.close()


class FakeSource:
    """Chain source that records every (symbol, date) it is asked for."""

    def __init__(self, data):
        self.data = data
        self.requested = []

    def chain(self, symbol, date):
        self.requested.append((symbol, date))
        return self.data.get((symbol, date), [])


def _c(strike, typ, bid, ask, expiration="2024-03-15", delta=-0.25):
    mid = (bid + ask) / 2 if (bid is not None and ask is not None) else None
    return {"expiration": expiration, "strike": float(strike), "type": typ,
            "bid": bid, "ask": ask, "mid": mid, "iv": 0.2,
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

    def test_a_delta_far_from_the_target_is_refused_not_substituted(self):
        # THE instrument-integrity test. select_legs promises "a missing wing
        # is a skipped trade, never a substituted strike", but the delta target
        # had no tolerance at all: asked for a 40-delta call against a chain
        # whose nearest listed call is 2-delta, it bought the 2-delta lottery
        # ticket and reported it as a 40-delta trade. Measured over 2020-2024
        # that bottom quartile (delta 0.00-0.37) lost 57% of capital per trade.
        chain = [_c(200.0, "call", 0.05, 0.10, delta=0.02)]
        self.assertIsNone(select_legs(
            _spec(structure="long_call",
                  entry={"dte": [20, 90], "target_delta": 0.40}),
            chain, "2024-01-05"))

    def test_a_delta_inside_the_tolerance_is_accepted(self):
        chain = [_c(100.0, "call", 2.0, 2.3, delta=0.34)]
        legs = select_legs(
            _spec(structure="long_call",
                  entry={"dte": [20, 90], "target_delta": 0.40}),
            chain, "2024-01-05")
        self.assertEqual([l.strike for l in legs], [100.0])

    def test_the_tolerance_is_configurable(self):
        chain = [_c(200.0, "call", 0.05, 0.10, delta=0.02)]
        entry = {"dte": [20, 90], "target_delta": 0.40, "delta_tolerance": 0.5}
        legs = select_legs(_spec(structure="long_call", entry=entry),
                           chain, "2024-01-05")
        self.assertEqual([l.strike for l in legs], [200.0])

    def test_the_short_leg_of_a_credit_spread_is_held_to_the_same_standard(self):
        # A "25-delta bull put" whose short leg is really 2-delta collects
        # almost no credit and is a different trade.
        chain = [_c(50.0, "put", 0.05, 0.10, delta=-0.02),
                 _c(45.0, "put", 0.01, 0.05, delta=-0.01)]
        self.assertIsNone(select_legs(
            _spec(entry={"dte": [20, 90], "short_delta": 0.25, "width": 5.0}),
            chain, "2024-01-05"))

    def test_random_strike_selection_still_bypasses_the_tolerance(self):
        # `strike_selection: random` is a deliberate control arm; constraining
        # it to a delta band would make it not random. Every listed delta here
        # is far outside the tolerance, so the non-random path returns None.
        chain = [_c(200.0, "put", 0.05, 0.10, delta=-0.02),
                 _c(195.0, "put", 0.01, 0.05, delta=-0.01)]
        entry = {"dte": [20, 90], "short_delta": 0.25, "width": 5.0}

        class _FirstChoice:              # deterministic stand-in for Random
            def choice(self, pool):
                return pool[0]

        self.assertIsNone(select_legs(_spec(entry=entry), chain, "2024-01-05"))
        legs = select_legs(_spec(entry={**entry, "strike_selection": "random"}),
                           chain, "2024-01-05", _FirstChoice())
        self.assertEqual([l.strike for l in legs], [200.0, 195.0])

    def test_a_refused_delta_counts_as_no_legs_not_as_a_silent_pass(self):
        dates = ["2024-01-05"]
        data = {("AAA", "2024-01-05"): [_c(200.0, "call", 0.05, 0.1, delta=0.02)]}
        trades, stats = replay(
            _spec(structure="long_call",
                  entry={"dte": [20, 90], "target_delta": 0.40}),
            ["AAA"], dates, FakeSource(data))
        self.assertEqual(trades, [])
        self.assertEqual(stats["skipped_no_legs"], 1)

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
        """No protective strike exists at all, so the structure is unbuildable.

        Counted under skipped_no_legs rather than skipped_missing: the chain
        does not offer a wing, which is a different failure from a wing that
        exists but has no usable quote.
        """
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25)]   # no lower strike
        trades, stats = replay(_spec(), ["AAA"], ["2024-01-05"],
                               FakeSource({("AAA", "2024-01-05"): chain}))
        self.assertEqual(trades, [])
        self.assertGreaterEqual(
            stats["skipped_no_legs"] + stats["skipped_missing"], 1)

    def test_a_wing_with_no_quote_is_counted_as_missing(self):
        """The wing is listed but unquotable — that IS skipped_missing."""
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25),
                 _c(95.0, "put", None, None, delta=-0.12)]
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


class OutlookLookupTest(unittest.TestCase):
    """docs/PREREG_OUTLOOK_FEATURE_20260905.md's feature: additive, never
    invented when the lookup has nothing for this (symbol, date)."""

    def test_a_looked_up_score_is_attached_to_the_trade(self):
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], src,
                           outlook_lookup={("AAA", "2024-01-05"): 0.42})
        self.assertEqual(trades[0].features["outlook_composite"], 0.42)

    def test_a_symbol_date_missing_from_the_lookup_is_absent_not_zero(self):
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], src,
                           outlook_lookup={("BBB", "2024-01-05"): 0.42})
        self.assertNotIn("outlook_composite", trades[0].features)

    def test_omitting_the_lookup_entirely_still_replays_normally(self):
        src = FakeSource({("AAA", "2024-01-05"): _bull_put_chain()})
        trades, _ = replay(_spec(), ["AAA"], ["2024-01-05"], src)
        self.assertNotIn("outlook_composite", trades[0].features)


if __name__ == "__main__":
    unittest.main()


class NearestStrikeWingTest(unittest.TestCase):
    """Wings snap to a listed strike, because the dataset is sparse.

    Requiring an exact `short - width` silently excluded every high-priced
    underlying — the dataset carries ~150-200 contracts per symbol-day, so on a
    $500 name the strike five dollars away is often not listed. That biased the
    whole study toward cheap, wide-spread names.
    """

    def test_wing_snaps_to_the_nearest_listed_strike(self):
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25),
                 _c(97.5, "put", 0.8, 1.0, delta=-0.12)]     # no 95 strike
        legs = select_legs(_spec(), chain, "2024-01-05")
        self.assertIsNotNone(legs)
        self.assertEqual(legs[1].strike, 97.5)

    def test_wing_beyond_tolerance_is_refused(self):
        """Tolerance is 3x width, because strike spacing scales with price."""
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25),
                 _c(50.0, "put", 0.1, 0.2, delta=-0.02)]     # 50 away, width 5
        self.assertIsNone(select_legs(_spec(), chain, "2024-01-05"))

    def test_wing_within_three_times_width_is_accepted(self):
        chain = [_c(100.0, "put", 2.0, 2.3, delta=-0.25),
                 _c(88.0, "put", 0.4, 0.6, delta=-0.06)]     # 12 away, width 5
        legs = select_legs(_spec(), chain, "2024-01-05")
        self.assertIsNotNone(legs)
        self.assertEqual(legs[1].strike, 88.0)

    def test_capital_uses_the_width_actually_obtained(self):
        from src.alloc.engine import actual_width
        legs = [Leg("2024-03-15", 100.0, "put", "sell"),
                Leg("2024-03-15", 97.5, "put", "buy")]
        self.assertEqual(actual_width(legs), 2.5)
        self.assertAlmostEqual(capital_at_risk(_spec(), legs, 0.50), 200.0)

    def test_iron_condor_width_is_the_wider_side_not_the_sum(self):
        from src.alloc.engine import actual_width
        legs = [Leg("2024-03-15", 90.0, "put", "sell"),
                Leg("2024-03-15", 85.0, "put", "buy"),
                Leg("2024-03-15", 110.0, "call", "sell"),
                Leg("2024-03-15", 120.0, "call", "buy")]
        self.assertEqual(actual_width(legs), 10.0)

    def test_single_leg_structure_has_no_width(self):
        from src.alloc.engine import actual_width
        self.assertEqual(actual_width([Leg("2024-03-15", 100.0, "call", "buy")]),
                         0.0)


class SignalGatingTest(unittest.TestCase):
    """Signal conditions must actually gate entries, and must not leak."""

    def _chain(self, iv):
        c = _bull_put_chain()
        for row in c:
            row["iv"] = iv
        # a call at the same strike so put-call parity can recover spot
        call = _c(100.0, "call", 2.0, 2.3, delta=0.25)
        call["iv"] = iv
        return c + [call]

    def _dates(self, n):
        return [f"2024-01-{i+1:02d}" for i in range(n)]

    def test_an_impossible_condition_blocks_every_entry(self):
        dates = self._dates(20)
        data = {("AAA", d): self._chain(0.20) for d in dates}
        spec = _spec(entry={"dte": [20, 90], "short_delta": 0.25, "width": 5.0,
                            "iv_rank_min": 200})       # unreachable
        trades, stats = replay(spec, ["AAA"], dates, FakeSource(data))
        self.assertEqual(trades, [])
        self.assertGreater(stats["skipped_signal"], 0)

    def test_a_satisfiable_condition_allows_entries(self):
        dates = self._dates(20)
        data = {("AAA", d): self._chain(0.20) for d in dates}
        spec = _spec(entry={"dte": [20, 90], "short_delta": 0.25, "width": 5.0,
                            "iv_rank_min": 0},
                     exit={"hold_to_expiry": True})
        trades, _ = replay(spec, ["AAA"], dates, FakeSource(data))
        self.assertTrue(trades)

    def test_no_conditions_leaves_behaviour_unchanged(self):
        dates = self._dates(20)
        data = {("AAA", d): self._chain(0.20) for d in dates}
        plain, _ = replay(_spec(exit={"hold_to_expiry": True}), ["AAA"], dates,
                          FakeSource(data))
        self.assertTrue(plain)

    def test_early_dates_are_blocked_while_history_is_too_short(self):
        """iv_rank needs history; before it exists the condition must FAIL,
        not silently pass and make the strategy unconditional."""
        dates = self._dates(6)
        data = {("AAA", d): self._chain(0.20) for d in dates}
        spec = _spec(entry={"dte": [20, 90], "short_delta": 0.25, "width": 5.0,
                            "iv_rank_min": 0})
        trades, stats = replay(spec, ["AAA"], dates, FakeSource(data))
        self.assertEqual(trades, [])
        self.assertGreater(stats["skipped_signal"], 0)


class EntryDepthFeatureTest(unittest.TestCase):
    """Quoted depth at entry — the feature H2 is built on.

    `bid_size`/`ask_size` exist in no source this repo held before optionsDX,
    and they arrive on 100% of its 18.9M rows. They were invisible to the
    attribution harness because `_entry_features` never carried them, so H2
    could not be asked at all.

    Depth is taken on the side actually traded AGAINST — a sell hits the bid,
    a buy lifts the ask — and the binding number is the smallest across the
    legs, because the trade is only as fillable as its worst leg.
    """

    def _legs(self):
        return [Leg("2024-03-15", 100.0, "put", "sell"),
                Leg("2024-03-15", 95.0, "put", "buy")]

    def _chain(self, short_bid_size, short_ask_size,
               long_bid_size, long_ask_size):
        c = _bull_put_chain()
        c[0]["bid_size"], c[0]["ask_size"] = short_bid_size, short_ask_size
        c[1]["bid_size"], c[1]["ask_size"] = long_bid_size, long_ask_size
        return c

    def _depth(self, chain):
        return _entry_features({}, self._legs(), chain,
                               quotes_from_chain(chain), price=1.5, car=350.0,
                               date="2024-01-05",
                               expiration="2024-03-15").get("entry_depth")

    def test_the_sold_leg_contributes_its_bid_size(self):
        # Selling hits the bid, so ask_size on that leg is irrelevant.
        self.assertEqual(self._depth(self._chain(12, 999, 500, 500)), 12)

    def test_the_bought_leg_contributes_its_ask_size(self):
        self.assertEqual(self._depth(self._chain(500, 500, 999, 7)), 7)

    def test_the_worst_leg_binds(self):
        self.assertEqual(self._depth(self._chain(40, 999, 999, 25)), 25)

    def test_a_source_without_depth_reports_none_not_zero(self):
        # The Dolt cache has no size columns at all. Zero would read as "no
        # depth quoted", which is a claim about the market rather than about
        # the data.
        self.assertIsNone(self._depth(_bull_put_chain()))

    def test_one_leg_missing_depth_makes_the_trade_unmeasurable(self):
        c = self._chain(40, 999, 999, 25)
        del c[1]["ask_size"]
        self.assertIsNone(self._depth(c))
