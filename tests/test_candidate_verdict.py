"""Tests for src/candidate_verdict.py — what a candidate is worth after costs.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_verdict -v
"""
import unittest

from src import candidate_verdict as cv


class SingleLegVerdictTest(unittest.TestCase):
    """A bought option. Friction is one crossing against the premium paid —
    measured at 0.7-1.7% on real archived quotes."""

    def test_a_tight_market_reports_low_friction(self):
        v = cv.verdict_for({"strategy_name": "Long Call", "bid": 9.90, "ask": 10.10})
        self.assertAlmostEqual(v.friction_pct, 0.01, places=3)
        self.assertTrue(v.priced)

    def test_a_wide_market_reports_high_friction(self):
        v = cv.verdict_for({"strategy_name": "Long Call", "bid": 8.00, "ask": 12.00})
        self.assertAlmostEqual(v.friction_pct, 0.20, places=3)

    def test_friction_is_charged_both_ways_for_a_round_trip(self):
        v = cv.verdict_for({"strategy_name": "Long Call", "bid": 9.90, "ask": 10.10})
        self.assertAlmostEqual(v.round_trip_pct, 2 * v.friction_pct, places=6)

    def test_a_missing_quote_is_unpriced_not_free(self):
        v = cv.verdict_for({"strategy_name": "Long Call", "bid": None, "ask": None})
        self.assertFalse(v.priced)
        self.assertFalse(v.passed)


class SpreadVerdictTest(unittest.TestCase):
    """A two-leg credit spread. Friction is BOTH crossings against the credit —
    measured at 33% on the logged Bull Puts, ~30x the single-leg burden."""

    LEGS = {"strategy_name": "Bull Put", "net_credit": 1.00, "spread_width": 2.50,
            "short_bid": 1.40, "short_ask": 1.60, "long_bid": 0.40, "long_ask": 0.60}

    def test_friction_is_the_sum_of_both_half_spreads_over_the_credit(self):
        v = cv.verdict_for(self.LEGS)
        self.assertAlmostEqual(v.friction_pct, 0.20, places=3)   # 0.20 / 1.00

    def test_the_breakeven_win_rate_is_reported(self):
        v = cv.verdict_for(self.LEGS)
        self.assertIsNotNone(v.breakeven)
        self.assertGreater(v.breakeven, 0.5)

    def test_the_breakeven_uses_the_filled_credit_not_the_mid(self):
        """Pricing at the mid is the defect; a candidate must be judged on what
        it would actually fill for."""
        v = cv.verdict_for(self.LEGS)
        mid_breakeven = 1.0 - 1.00 / 2.50
        self.assertGreater(v.breakeven, mid_breakeven)

    def test_a_spread_whose_credit_vanishes_when_crossed_is_refused(self):
        v = cv.verdict_for({"strategy_name": "Bull Put", "net_credit": 0.30,
                            "spread_width": 2.50, "short_bid": 1.00, "short_ask": 2.00,
                            "long_bid": 0.10, "long_ask": 1.10})
        self.assertFalse(v.passed)


class GateTest(unittest.TestCase):
    # A spread tight enough to clear the friction ceiling, so the win-rate
    # gate is what is actually under test. Note how tight it has to be: the
    # realistic fixture above (a $0.10 half-spread on each $2.50-wide leg)
    # carries 40% round-trip friction and never reaches this gate at all.
    TIGHT = {"strategy_name": "Bull Put", "net_credit": 1.00, "spread_width": 2.50,
             "short_bid": 1.48, "short_ask": 1.52, "long_bid": 0.48, "long_ask": 0.52}

    def test_a_candidate_needing_a_higher_win_rate_than_history_is_refused(self):
        v = cv.verdict_for(self.TIGHT, historical_win_rate=0.50)
        self.assertFalse(v.passed)
        self.assertIn("win rate", v.reason)

    def test_the_same_candidate_passes_against_a_high_enough_history(self):
        v = cv.verdict_for(self.TIGHT, historical_win_rate=0.95)
        self.assertTrue(v.passed)

    def test_a_realistic_spread_is_refused_on_friction_before_anything_else(self):
        """The measured case. Both gates would refuse it; friction gets there
        first, and that ordering is the finding: the spread eats the trade
        before the win rate ever becomes the question."""
        v = cv.verdict_for(SpreadVerdictTest.LEGS, historical_win_rate=0.95)
        self.assertFalse(v.passed)
        self.assertIn("friction", v.reason)

    def test_friction_above_the_ceiling_is_refused(self):
        v = cv.verdict_for({"strategy_name": "Long Call", "bid": 5.0, "ask": 15.0},
                           max_friction_pct=0.10)
        self.assertFalse(v.passed)
        self.assertIn("friction", v.reason)


class RankingTest(unittest.TestCase):
    """The change of philosophy: rank by what survives costs, not by the
    composite. quality_score correlates -0.13 with return on the long-premium
    book and cannot rank; it is retained only to break ties."""

    ROWS = [
        {"strategy_name": "Long Call", "bid": 5.00, "ask": 15.00, "quality_score": 0.95},
        {"strategy_name": "Long Call", "bid": 9.90, "ask": 10.10, "quality_score": 0.60},
        {"strategy_name": "Long Call", "bid": 9.50, "ask": 10.50, "quality_score": 0.80},
    ]

    def test_the_cheapest_to_trade_ranks_first_regardless_of_score(self):
        ranked = cv.rank(self.ROWS)
        self.assertAlmostEqual(ranked[0]["bid"], 9.90)

    def test_the_highest_scoring_candidate_can_rank_last(self):
        ranked = cv.rank(self.ROWS)
        self.assertAlmostEqual(ranked[-1]["quality_score"], 0.95)

    def test_refused_candidates_sort_below_every_passing_one(self):
        ranked = cv.rank(self.ROWS, max_friction_pct=0.10)
        passed = [r["verdict"].passed for r in ranked]
        self.assertEqual(passed, sorted(passed, reverse=True))

    def test_every_row_carries_its_verdict(self):
        for r in cv.rank(self.ROWS):
            self.assertIsInstance(r["verdict"], cv.Verdict)

    def test_ranking_an_empty_list_is_not_an_error(self):
        self.assertEqual(cv.rank([]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
