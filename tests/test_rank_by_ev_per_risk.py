"""The board ranks by EV per dollar AT RISK, not by EV in dollars.

Found on a live scan, 2026-08-24. MU trades at $910, so one MU contract costs
$4,367; SLB trades at $54, so one SLB contract costs $139. Dollar EV scales
with contract price, so ranking on it compared positions 30x apart in size:

    SLB $55C    risk $139     EV/$risk +0.070    dollar EV    +$10
    WMT $105P   risk $172     EV/$risk +0.430    dollar EV    +$74
    MU  $900P   risk $4,367   EV/$risk +0.288    dollar EV  +$1,288

MU took the top slot on a per-dollar edge that was only THIRD best, and all
12 of its fetched contracts swept the top 15 — the scan's own concentration
warning fired. Meanwhile the position was unenterable: $4,475 of capital at
risk against a $4,000 cap, on volume 178 against SLB's 1,427.

This is the same defect class this repo keeps paying for: a ratio compared
across positions whose denominators differ. Normalising is a denominator
correction, not a new ranking hypothesis — the ordering question (does EV
predict outcome at all?) remains frozen until 2026-11-19.
"""
from __future__ import annotations

import unittest

from src import candidate_verdict as cv


def _row(symbol, strike, premium, ev, opt_type="call", **over):
    row = {
        "symbol": symbol, "type": opt_type, "strike": strike,
        "expiration": "2026-12-18", "premium": premium,
        "bid": (premium - 0.02) if premium is not None else None,
        "ask": (premium + 0.02) if premium is not None else None,
        "ev_per_contract": ev, "quality_score": 0.5,
        "strategy_name": "Long Call" if opt_type == "call" else "Long Put",
        "underlying": strike, "volume": 1000, "openInterest": 5000,
    }
    row.update(over)
    return row


class TestSizeNoLongerBuysTheTopSlot(unittest.TestCase):

    def test_the_better_edge_per_dollar_outranks_the_bigger_ticket(self):
        """The live case, reduced. Cheap contract with a strong per-dollar
        edge must beat an expensive one with a weaker edge and a bigger
        absolute number."""
        cheap = _row("WMT", 105.0, 1.72, ev=74.0)      # +0.430 per $ at risk
        pricey = _row("MU", 900.0, 43.67, ev=1288.0)   # +0.288 per $ at risk
        out = cv.rank([pricey, cheap])
        self.assertEqual(out[0]["symbol"], "WMT",
                         "the larger dollar EV still won — the board is still "
                         "ranking by position size")

    def test_dollar_ev_alone_would_have_ordered_it_the_other_way(self):
        """Guards the premise: without normalising, MU wins. If this ever
        stops being true the test above proves nothing."""
        cheap = _row("WMT", 105.0, 1.72, ev=74.0)
        pricey = _row("MU", 900.0, 43.67, ev=1288.0)
        self.assertGreater(pricey["ev_per_contract"], cheap["ev_per_contract"])

    def test_equal_edges_per_dollar_do_not_reorder_on_size(self):
        """A 10x bigger position with the same per-dollar edge must land at
        the same rank, not above it. `delta` rather than `places` because
        capital at risk is priced off the ask, not the nominal premium, so
        two 'equal' fixtures differ in the third decimal — the point is that
        neither is an order of magnitude apart, as dollar EV made them."""
        small = _row("AAA", 50.0, 1.00, ev=20.0)     # ~+0.20 per $
        large = _row("BBB", 500.0, 10.00, ev=200.0)  # ~+0.20 per $
        out = cv.rank([small, large])
        self.assertAlmostEqual(
            cv.ev_per_risk(out[0]), cv.ev_per_risk(out[1]), delta=0.01)
        # Dollar EV would have put them 10x apart.
        self.assertAlmostEqual(large["ev_per_contract"] /
                               small["ev_per_contract"], 10.0, places=6)

    def test_a_negative_edge_still_ranks_below_a_positive_one(self):
        good = _row("AAA", 50.0, 1.00, ev=20.0)
        bad = _row("BBB", 50.0, 1.00, ev=-20.0)
        self.assertEqual(cv.rank([bad, good])[0]["symbol"], "AAA")


class TestUnknownsNeverOutrankMeasured(unittest.TestCase):
    """Same principle the dollar-EV sort already held: 'unknown' must not
    float above a candidate that demonstrably clears its costs."""

    def test_a_row_with_no_ev_sinks(self):
        good = _row("AAA", 50.0, 1.00, ev=20.0)
        blank = _row("BBB", 50.0, 1.00, ev=None)
        self.assertEqual(cv.rank([blank, good])[0]["symbol"], "AAA")

    def test_a_row_whose_risk_cannot_be_computed_sinks(self):
        good = _row("AAA", 50.0, 1.00, ev=20.0)
        no_risk = _row("BBB", 50.0, None, ev=999.0)
        no_risk["premium"] = None
        self.assertEqual(cv.rank([no_risk, good])[0]["symbol"], "AAA")

    def test_zero_risk_is_not_treated_as_infinite_edge(self):
        """Dividing by a zero denominator must not mint the best pick on the
        board out of a missing number."""
        good = _row("AAA", 50.0, 1.00, ev=20.0)
        zero = _row("BBB", 50.0, 0.0, ev=999.0)
        out = cv.rank([zero, good])
        self.assertEqual(out[0]["symbol"], "AAA")

    def test_ev_per_risk_is_none_when_it_cannot_be_computed(self):
        blank = _row("BBB", 50.0, 1.00, ev=None)
        self.assertEqual(cv.ev_per_risk(blank), float("-inf"))


class TestTheGateStillOutranksEverything(unittest.TestCase):
    """Normalising changes the ORDER WITHIN survivors. It must not promote a
    candidate that failed its gate above one that passed."""

    def test_a_refused_candidate_stays_below_a_survivor(self):
        # Friction far above the ceiling: refused whatever its edge per dollar.
        refused = _row("BBB", 50.0, 1.00, ev=500.0)
        refused["bid"], refused["ask"] = 0.10, 2.00
        passed = _row("AAA", 50.0, 1.00, ev=5.0)
        out = cv.rank([refused, passed])
        self.assertTrue(out[0]["verdict"].passed)
        self.assertEqual(out[0]["symbol"], "AAA")


if __name__ == "__main__":
    unittest.main()
