"""Tests for src/execution_truth.py — what a fill actually costs.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_execution_truth -v
"""
import json
import os
import statistics
import unittest

from src import execution_truth as et


class LegFillTest(unittest.TestCase):
    """A single leg, priced under each policy. Quotes are literal — no DB."""

    def test_selling_at_cross_gets_the_bid(self):
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "sell", "cross"), 1.00)

    def test_buying_at_cross_pays_the_ask(self):
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "buy", "cross"), 1.20)

    def test_selling_at_mid_gets_the_mid(self):
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "sell", "mid"), 1.10)

    def test_buying_at_mid_pays_the_mid(self):
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "buy", "mid"), 1.10)

    def test_limit_concedes_a_fraction_of_the_half_spread(self):
        # half-spread is 0.10; k=0.35 concedes 0.035 from the mid, in the
        # direction that costs the trader.
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "sell", "limit", k=0.35), 1.065)
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "buy", "limit", k=0.35), 1.135)

    def test_limit_at_k_zero_is_mid_and_k_one_is_cross(self):
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "sell", "limit", k=0.0), 1.10)
        self.assertAlmostEqual(et.leg_fill(1.00, 1.20, "sell", "limit", k=1.0), 1.00)


class StructureFillTest(unittest.TestCase):
    """A whole structure. Legs are (bid, ask, side) tuples."""

    # A $2.50-wide bull put: sell the 100 put, buy the 97.50 put.
    BULL_PUT = [
        {"bid": 1.40, "ask": 1.60, "side": "sell"},
        {"bid": 0.40, "ask": 0.60, "side": "buy"},
    ]

    def test_credit_at_mid_is_the_difference_of_mids(self):
        r = et.structure_fill(self.BULL_PUT, "mid")
        self.assertAlmostEqual(r.price, 1.00)      # 1.50 - 0.50

    def test_credit_when_crossed_sells_the_bid_and_buys_the_ask(self):
        r = et.structure_fill(self.BULL_PUT, "cross")
        self.assertAlmostEqual(r.price, 0.80)      # 1.40 - 0.60

    def test_slip_vs_mid_is_the_sum_of_both_half_spreads(self):
        r = et.structure_fill(self.BULL_PUT, "cross")
        self.assertAlmostEqual(r.slip_vs_mid, 0.20)   # 0.10 + 0.10

    def test_a_debit_structure_reports_a_negative_price(self):
        long_call = [{"bid": 2.00, "ask": 2.40, "side": "buy"}]
        r = et.structure_fill(long_call, "cross")
        self.assertAlmostEqual(r.price, -2.40)
        self.assertAlmostEqual(r.slip_vs_mid, 0.20)

    def test_slip_is_never_negative_regardless_of_policy(self):
        for policy in et.POLICIES:
            self.assertGreaterEqual(et.structure_fill(self.BULL_PUT, policy).slip_vs_mid, 0.0)

    def test_a_missing_quote_refuses_to_price_rather_than_guessing(self):
        legs = [{"bid": 1.40, "ask": 1.60, "side": "sell"},
                {"bid": None, "ask": None, "side": "buy"}]
        self.assertIsNone(et.structure_fill(legs, "cross"))

    def test_a_zero_ask_refuses_to_price(self):
        legs = [{"bid": 1.40, "ask": 1.60, "side": "sell"},
                {"bid": 0.0, "ask": 0.0, "side": "buy"}]
        self.assertIsNone(et.structure_fill(legs, "cross"))

    def test_a_crossed_quote_refuses_to_price(self):
        # ask < bid is a corrupt quote, not a gift.
        legs = [{"bid": 1.60, "ask": 1.40, "side": "sell"}]
        self.assertIsNone(et.structure_fill(legs, "cross"))


class BreakevenTest(unittest.TestCase):
    """p* = 1 - C/W: the win rate a credit spread must beat to break even."""

    def test_breakeven_falls_as_the_credit_rises(self):
        self.assertAlmostEqual(et.breakeven_win_rate(1.00, 2.50), 0.60)
        self.assertAlmostEqual(et.breakeven_win_rate(1.25, 2.50), 0.50)

    def test_the_measured_bull_put_case(self):
        # The headline numbers this module exists to surface: median $2.50-wide
        # bull put, credit $1.05 at mid vs $0.60 once crossed.
        self.assertAlmostEqual(et.breakeven_win_rate(1.05, 2.50), 0.58)
        self.assertAlmostEqual(et.breakeven_win_rate(0.60, 2.50), 0.76)

    def test_a_credit_at_or_above_the_width_cannot_lose(self):
        self.assertAlmostEqual(et.breakeven_win_rate(2.50, 2.50), 0.0)
        self.assertAlmostEqual(et.breakeven_win_rate(3.00, 2.50), 0.0)

    def test_a_non_positive_credit_can_never_break_even(self):
        self.assertEqual(et.breakeven_win_rate(0.0, 2.50), 1.0)
        self.assertEqual(et.breakeven_win_rate(-0.10, 2.50), 1.0)

    def test_zero_width_is_undefined_rather_than_a_divide_by_zero(self):
        self.assertIsNone(et.breakeven_win_rate(1.00, 0.0))


class EdgeReportTest(unittest.TestCase):
    """What the pre-trade gate consumes: p* under every policy at once."""

    BULL_PUT = StructureFillTest.BULL_PUT

    def test_reports_a_breakeven_for_each_policy(self):
        rep = et.edge_report(self.BULL_PUT, width=2.50)
        self.assertEqual(set(rep.breakeven), set(et.POLICIES))

    def test_crossing_demands_a_higher_win_rate_than_filling_at_mid(self):
        rep = et.edge_report(self.BULL_PUT, width=2.50)
        self.assertLess(rep.breakeven["mid"], rep.breakeven["limit"])
        self.assertLess(rep.breakeven["limit"], rep.breakeven["cross"])

    def test_unpriceable_legs_produce_no_report(self):
        legs = [{"bid": None, "ask": None, "side": "sell"}]
        self.assertIsNone(et.edge_report(legs, width=2.50))


class GateTest(unittest.TestCase):
    """The pre-trade refusal. A candidate that cannot clear its own breakeven
    once the spread is paid is not a trade, however well it scores."""

    TIGHT = [{"bid": 1.45, "ask": 1.50, "side": "sell"},
             {"bid": 0.45, "ask": 0.50, "side": "buy"}]
    WIDE = [{"bid": 1.00, "ask": 2.00, "side": "sell"},
            {"bid": 0.10, "ask": 1.10, "side": "buy"}]

    def test_a_tight_spread_with_a_fat_credit_passes(self):
        v = et.gate(self.TIGHT, width=2.50, max_breakeven=0.65)
        self.assertTrue(v.passed)

    def test_a_wide_spread_is_refused_for_its_breakeven(self):
        v = et.gate(self.WIDE, width=2.50, max_breakeven=0.65)
        self.assertFalse(v.passed)
        self.assertIn("breakeven", v.reason)

    def test_the_verdict_carries_the_number_that_caused_it(self):
        v = et.gate(self.WIDE, width=2.50, max_breakeven=0.65)
        self.assertIsNotNone(v.breakeven)
        self.assertGreater(v.breakeven, 0.65)

    def test_an_unquotable_candidate_is_refused_not_assumed(self):
        legs = [{"bid": None, "ask": None, "side": "sell"}]
        v = et.gate(legs, width=2.50, max_breakeven=0.65)
        self.assertFalse(v.passed)
        self.assertIn("quote", v.reason)

    def test_the_gate_judges_the_limit_fill_not_the_mid(self):
        """A candidate that clears at the mid but not at a worked limit must
        be refused — pricing at the mid is the defect this exists to stop."""
        legs = [{"bid": 1.20, "ask": 1.80, "side": "sell"},
                {"bid": 0.20, "ask": 0.80, "side": "buy"}]
        mid = et.breakeven_win_rate(et.structure_fill(legs, "mid").price, 2.50)
        lim = et.breakeven_win_rate(et.structure_fill(legs, "limit").price, 2.50)
        threshold = (mid + lim) / 2         # between the two
        self.assertFalse(et.gate(legs, 2.50, max_breakeven=threshold).passed)

    def test_a_disabled_threshold_passes_anything_priceable(self):
        self.assertTrue(et.gate(self.WIDE, width=2.50, max_breakeven=None).passed)


class ArchivedQuotesTest(unittest.TestCase):
    """Golden test against real quotes.

    `tests/fixtures/bull_put_archived_quotes.json` is 30 Bull Puts from the
    live ledger paired with the bid/ask that `data/chain_archive.db` recorded
    on their entry day. It is real market data, not a construction, and it is
    what pins this module to reality: if the fill model drifts, these numbers
    move and the test fails."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "fixtures",
                            "bull_put_archived_quotes.json")
        with open(path) as fh:
            cls.cases = json.load(fh)
        cls.legs = [
            [{"bid": c["short"]["bid"], "ask": c["short"]["ask"], "side": "sell"},
             {"bid": c["long"]["bid"], "ask": c["long"]["ask"], "side": "buy"}]
            for c in cls.cases
        ]

    def test_the_fixture_is_the_sample_that_was_measured(self):
        self.assertEqual(len(self.cases), 30)

    def test_the_ledger_has_never_booked_a_crossed_fill(self):
        """The defect, asserted conservatively. 30 of 30 logged credits are at
        or above what the archived book would have paid to cross, and 23 of 30
        are at or above its mid.

        Deliberately NOT asserting that the logged credit equals the archived
        mid: it doesn't (median +$0.10, ratio 1.11). The archive is CBOE
        snapshotted anywhere from pre-market to mid-session, the scanner reads
        yfinance intraday, so that comparison mixes source and timing
        differences and cannot speak to fill policy. The proof that entries are
        booked at the mid is the code — `options_screener.py:2160` sets
        `premium = mid` — not this fixture."""
        crossed = sum(
            1 for c, legs in zip(self.cases, self.legs)
            if c["logged_net_credit"] >= et.structure_fill(legs, "cross").price
        )
        self.assertEqual(crossed, len(self.cases))

    def test_crossing_costs_thirty_five_cents_of_credit(self):
        slips = sorted(et.structure_fill(l, "cross").slip_vs_mid for l in self.legs)
        self.assertAlmostEqual(statistics.median(slips), 0.35, places=2)

    def test_crossing_costs_about_a_quarter_of_the_credit_collected(self):
        fracs = [
            et.structure_fill(l, "cross").slip_vs_mid / et.structure_fill(l, "mid").price
            for l in self.legs if et.structure_fill(l, "mid").price > 0
        ]
        self.assertGreater(statistics.median(fracs), 0.20)
        self.assertLess(statistics.median(fracs), 0.35)

    def test_crossing_raises_the_required_win_rate_by_over_ten_points(self):
        """The headline: these spreads need 58% to break even on paper and
        over 70% once you pay to get in. The book wins 70.4%."""
        gaps = []
        for c, legs in zip(self.cases, self.legs):
            rep = et.edge_report(legs, width=c["width"])
            if rep is None or rep.breakeven["mid"] is None:
                continue
            gaps.append(rep.breakeven["cross"] - rep.breakeven["mid"])
        self.assertGreater(statistics.median(gaps), 0.10)

    def test_a_worked_limit_recovers_most_of_the_crossing_cost(self):
        """The reason the default policy is `limit` and not `cross`."""
        for legs in self.legs:
            crossed = et.structure_fill(legs, "cross").slip_vs_mid
            limit = et.structure_fill(legs, "limit").slip_vs_mid
            self.assertLess(limit, crossed)
            self.assertAlmostEqual(limit, crossed * et.DEFAULT_LIMIT_K, places=6)

    def test_some_of_these_spreads_have_no_credit_left_once_crossed(self):
        """3 of 30. A structure the screener showed as a $1+ credit is not a
        trade at all once you pay the spread."""
        dead = [l for l in self.legs if et.structure_fill(l, "cross").price <= 0]
        self.assertGreaterEqual(len(dead), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
