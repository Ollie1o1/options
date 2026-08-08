"""The auto-logger chose what to log by quality_score.

rank_by_verdict replaced that ordering on the display paths in the
execution-truth work, but the auto-log path kept sorting by the composite —
so every row in the ledger was selected by the score measured at rank IC
-0.10 against friction-adjusted return (docs/ADJUSTMENT_STACK_20260807.md).

These pin the ordering helper the auto-log path now uses.
"""
import unittest

import pandas as pd

from src.options_screener import rank_single_legs_by_verdict


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


if __name__ == "__main__":
    unittest.main()
