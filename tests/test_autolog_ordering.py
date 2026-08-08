"""The auto-logger chose what to log by quality_score.

rank_by_verdict replaced that ordering on the display paths in the
execution-truth work, but the auto-log path kept sorting by the composite —
so every row in the ledger was selected by the score measured at rank IC
-0.10 against friction-adjusted return (docs/ADJUSTMENT_STACK_20260807.md).

These pin the ordering helper the auto-log path now uses.
"""
import unittest

import pandas as pd

from src.options_screener import (
    rank_single_legs_by_verdict,
    rank_structures_by_verdict,
    structure_strategy_name,
)


def _leg(symbol, qs, bid, ask, ev, opt_type="call"):
    return dict(symbol=symbol, type=opt_type, quality_score=qs,
                bid=bid, ask=ask, ev_per_contract=ev,
                strike=100.0, expiration="2026-09-18")


class TestOrderSingleLegsForLogging(unittest.TestCase):
    def test_net_ev_outranks_a_higher_quality_score(self):
        """The whole point: the composite no longer decides what gets logged."""
        df = pd.DataFrame([
            _leg("LOSER", qs=0.95, bid=1.00, ask=1.04, ev=-50.0),
            _leg("WINNER", qs=0.20, bid=1.00, ask=1.04, ev=+250.0),
        ])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(list(out["symbol"]), ["WINNER", "LOSER"])

    def test_quality_score_still_breaks_ties(self):
        df = pd.DataFrame([
            _leg("LOW", qs=0.10, bid=1.00, ask=1.04, ev=100.0),
            _leg("HIGH", qs=0.90, bid=1.00, ask=1.04, ev=100.0),
        ])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(list(out["symbol"]), ["HIGH", "LOW"])

    def test_premium_selling_rows_are_labelled_short(self):
        """_legs_of reads the side off strategy_name. Without it a short put
        is priced as a debit buy, which flips is_credit and skips both the
        credit-disappears and breakeven checks."""
        df = pd.DataFrame([_leg("XYZ", qs=0.5, bid=1.00, ask=1.04, ev=10.0,
                                opt_type="put")])
        out = rank_single_legs_by_verdict(df, "Premium Selling")
        self.assertEqual(out["strategy_name"].iloc[0], "Short Put")

    def test_buyer_modes_are_labelled_long(self):
        df = pd.DataFrame([_leg("XYZ", qs=0.5, bid=1.00, ask=1.04, ev=10.0)])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(out["strategy_name"].iloc[0], "Long Call")

    def test_unquotable_rows_sink_below_quotable_ones(self):
        """A row with no two-sided quote is refused, not assumed tradeable."""
        df = pd.DataFrame([
            _leg("NOQUOTE", qs=0.99, bid=0.0, ask=0.0, ev=999.0),
            _leg("QUOTED", qs=0.10, bid=1.00, ask=1.04, ev=1.0),
        ])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(out["symbol"].iloc[0], "QUOTED")

    def test_every_input_row_survives(self):
        """Ordering only. Dropping candidates here would starve the cohort;
        the allowlist and budget filters downstream do the dropping."""
        df = pd.DataFrame([
            _leg("A", qs=0.5, bid=1.00, ask=1.04, ev=5.0),
            _leg("B", qs=0.6, bid=0.0, ask=0.0, ev=None),
            _leg("C", qs=0.7, bid=2.00, ask=2.10, ev=-5.0),
        ])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out["symbol"]), {"A", "B", "C"})

    def test_empty_frame_is_returned_unchanged(self):
        out = rank_single_legs_by_verdict(pd.DataFrame(), "Discovery scan")
        self.assertTrue(out.empty)

    def test_a_frame_without_type_is_left_alone_rather_than_raising(self):
        """The auto-log path already guards on `type`; this must not be the
        thing that breaks a scan."""
        df = pd.DataFrame([{"symbol": "X", "quality_score": 0.5}])
        out = rank_single_legs_by_verdict(df, "Discovery scan")
        self.assertEqual(len(out), 1)


def _condor(symbol, qs, half, credit=1.00):
    """A four-leg condor whose every leg is `half` off the mid."""
    row = dict(symbol=symbol, quality_score=qs, total_credit=credit,
               spread_width=2.50, expiration="2026-09-18")
    for prefix, mid in (("short_put", 1.00), ("long_put", 0.50),
                        ("short_call", 1.00), ("long_call", 0.50)):
        row[f"{prefix}_bid"] = mid - half
        row[f"{prefix}_ask"] = mid + half
    return row


class TestStructureStrategyName(unittest.TestCase):
    def test_a_total_credit_marks_a_condor(self):
        row = pd.Series({"total_credit": 1.0, "type": "call"})
        self.assertEqual(structure_strategy_name(row), "Iron Condor")

    def test_a_vertical_is_named_by_its_short_leg(self):
        self.assertEqual(
            structure_strategy_name(pd.Series({"type": "call"})), "Bear Call")
        self.assertEqual(
            structure_strategy_name(pd.Series({"type": "put"})), "Bull Put")

    def test_a_nan_total_credit_is_not_a_condor(self):
        row = pd.Series({"total_credit": float("nan"), "type": "put"})
        self.assertEqual(structure_strategy_name(row), "Bull Put")


class TestRankStructuresByVerdict(unittest.TestCase):
    def test_a_cheap_condor_outranks_an_expensive_higher_scored_one(self):
        """Four crossings against one credit. A condor whose friction eats its
        credit was being logged ahead of a tradeable one on score alone."""
        df = pd.DataFrame([_condor("EXPENSIVE", qs=0.99, half=0.30),
                           _condor("CHEAP", qs=0.10, half=0.02)])
        out = rank_structures_by_verdict(df)
        self.assertEqual(list(out["symbol"]), ["CHEAP", "EXPENSIVE"])

    def test_condors_are_labelled_so_they_can_be_priced(self):
        out = rank_structures_by_verdict(pd.DataFrame([_condor("X", 0.5, 0.02)]))
        self.assertEqual(out["strategy_name"].iloc[0], "Iron Condor")
        self.assertTrue(bool(out["verdict_passed"].iloc[0]))

    def test_a_condor_eating_its_credit_in_friction_is_refused(self):
        out = rank_structures_by_verdict(pd.DataFrame([_condor("X", 0.5, 0.30)]))
        self.assertFalse(bool(out["verdict_passed"].iloc[0]))

    def test_verticals_still_rank(self):
        df = pd.DataFrame([
            {"symbol": "V", "type": "put", "quality_score": 0.5,
             "net_credit": 1.00, "spread_width": 2.50, "short_bid": 1.40,
             "short_ask": 1.60, "long_bid": 0.40, "long_ask": 0.60},
        ])
        out = rank_structures_by_verdict(df)
        self.assertEqual(out["strategy_name"].iloc[0], "Bull Put")
        self.assertIn("friction_pct", out.columns)

    def test_every_input_row_survives(self):
        df = pd.DataFrame([_condor("A", 0.5, 0.02), _condor("B", 0.6, 0.30)])
        self.assertEqual(len(rank_structures_by_verdict(df)), 2)

    def test_empty_frame_is_returned_unchanged(self):
        self.assertTrue(rank_structures_by_verdict(pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
