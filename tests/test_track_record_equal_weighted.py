"""The equal-weighted section of the published track record.

The document led with dollar figures from a book in which position size was
never chosen: every ledger row carried `quantity = 1.0`, so bet size was the
option's premium. A headline built on that is partly a statement about which
trades happened to be large, and the reader cannot tell how much of it is the
picks and how much is the sizing.

This section answers that directly: give every closed trade the SAME capital at
risk and see what the book does. Three properties are pinned here.

  1. The normalisation basis is capital at risk, not entry premium. Premium is
     a debit on long structures and a credit on short ones, so equalising it
     compares two different quantities — and the two bases disagree on the SIGN
     of the book's result, which is exactly why the basis has to be stated.
  2. The interval is published beside the point estimate, and the document says
     in words whether it contains 1. A profit factor of 1.06 over 897 trades
     with a 95% interval of [0.88, 1.26] is not an edge, and a bare "1.06"
     reads like one.
  3. The bootstrap is SEEDED. This file is committed to git and regenerated
     often; an unseeded interval would produce a different diff every run and
     nobody would read it.

Pure computation against seeded rows — no db file, no network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.publish_track_record import (  # noqa: E402
    EQUAL_WEIGHT_RISK,
    equal_weighted,
    profit_factor,
    render_track_record,
    summarize_equal_weighted_strategies,
)

_EVIDENCE = {
    "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
    "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
}


def _row(pnl, risk, strategy="Bull Put", pct=None):
    return {"strategy_name": strategy, "status": "CLOSED", "pnl_usd": pnl,
            "capital_at_risk": risk,
            "pnl_pct": pct if pct is not None else (pnl / risk if risk else None),
            "entry_price": 1.0, "net_credit": 1.0, "quantity": 1.0,
            "date": "2026-08-01", "exit_date": "2026-08-10", "ticker": "AAA"}


class ProfitFactor(unittest.TestCase):

    def test_wins_over_losses(self):
        self.assertAlmostEqual(profit_factor([2.0, -1.0, 1.0, -1.0]), 1.5)

    def test_no_losses_is_not_a_number_this_can_report(self):
        # An infinite profit factor is a sample-size statement, not a result.
        self.assertIsNone(profit_factor([1.0, 2.0]))

    def test_empty_is_none_not_zero(self):
        self.assertIsNone(profit_factor([]))


class EqualWeighting(unittest.TestCase):

    def test_identical_risk_makes_the_two_weightings_agree(self):
        # When every trade already risks the same dollars there is nothing for
        # equal weighting to change — the clearest check that it is measuring
        # size and not something else.
        rows = [_row(200.0, 1000.0), _row(-100.0, 1000.0), _row(50.0, 1000.0)]
        eq = equal_weighted(rows)
        dollars = profit_factor([r["pnl_usd"] for r in rows])
        self.assertAlmostEqual(eq["profit_factor"], dollars)

    def test_one_large_winner_cannot_carry_the_equal_weighted_book(self):
        # The artifact this section exists to expose: as sized, a single big
        # position makes the book profitable; per trade, it loses.
        rows = [_row(5_000.0, 20_000.0)] + [_row(-100.0, 400.0) for _ in range(9)]
        self.assertGreater(profit_factor([r["pnl_usd"] for r in rows]), 1.0)
        self.assertLess(equal_weighted(rows)["profit_factor"], 1.0)

    def test_the_dollar_figure_states_the_size_it_assumes(self):
        # +25% and -10% on the same risk: $250 - $100 at $1,000 a trade.
        rows = [_row(250.0, 1000.0), _row(-100.0, 1000.0)]
        eq = equal_weighted(rows)
        self.assertAlmostEqual(eq["net_pnl"], 0.15 * EQUAL_WEIGHT_RISK)
        self.assertEqual(eq["risk_per_trade"], EQUAL_WEIGHT_RISK)

    def test_the_interval_brackets_the_point_estimate(self):
        rows = [_row(200.0, 1000.0) for _ in range(30)] + \
               [_row(-150.0, 1000.0) for _ in range(30)]
        eq = equal_weighted(rows)
        self.assertLessEqual(eq["ci_low"], eq["profit_factor"])
        self.assertGreaterEqual(eq["ci_high"], eq["profit_factor"])

    def test_the_bootstrap_is_seeded(self):
        # Same rows, two calls, identical interval — or the committed report
        # churns on every regeneration.
        rows = [_row(200.0, 1000.0), _row(-100.0, 800.0), _row(30.0, 500.0)] * 20
        first, second = equal_weighted(rows), equal_weighted(rows)
        self.assertEqual(first["ci_low"], second["ci_low"])
        self.assertEqual(first["ci_high"], second["ci_high"])

    def test_rows_without_recorded_risk_are_excluded_not_zeroed(self):
        rows = [_row(200.0, 1000.0), _row(-500.0, None)]
        self.assertEqual(equal_weighted(rows)["n"], 1)

    def test_an_empty_book_reports_nothing_rather_than_raising(self):
        self.assertIsNone(equal_weighted([])["profit_factor"])


class PerStrategy(unittest.TestCase):

    def test_each_strategy_gets_its_own_interval(self):
        rows = ([_row(200.0, 1000.0, "Bull Put") for _ in range(20)] +
                [_row(-200.0, 1000.0, "Iron Condor") for _ in range(20)] +
                [_row(100.0, 1000.0, "Iron Condor") for _ in range(10)])
        out = {s["strategy"]: s for s in summarize_equal_weighted_strategies(rows)}
        self.assertEqual(set(out), {"Bull Put", "Iron Condor"})
        self.assertLess(out["Iron Condor"]["profit_factor"], 1.0)

    def test_a_line_too_short_to_bootstrap_is_still_listed(self):
        # Dropping it would silently remove a strategy from the comparison.
        rows = [_row(200.0, 1000.0, "Bull Put"), _row(-100.0, 1000.0, "Bear Call")]
        out = {s["strategy"]: s for s in summarize_equal_weighted_strategies(rows)}
        self.assertIn("Bear Call", out)


class Rendered(unittest.TestCase):

    def _doc(self, rows):
        return render_track_record(rows, _EVIDENCE)

    def test_the_section_is_published(self):
        rows = [_row(200.0, 1000.0), _row(-100.0, 800.0)] * 30
        self.assertIn("Equal-weighted", self._doc(rows))

    def test_it_names_the_basis_and_the_size_it_assumes(self):
        doc = self._doc([_row(200.0, 1000.0), _row(-100.0, 800.0)] * 30)
        self.assertIn("capital at risk", doc)
        self.assertIn("$1,000", doc)

    def test_an_interval_containing_one_is_called_out_in_words(self):
        # 60 trades, +20% / -15%, a profit factor near 1.1 with a wide interval.
        rows = ([_row(200.0, 1000.0) for _ in range(30)] +
                [_row(-150.0, 1000.0) for _ in range(30)])
        doc = self._doc(rows)
        eq = equal_weighted(rows)
        self.assertLess(eq["ci_low"], 1.0)
        self.assertIn("contains 1", doc)

    def test_a_book_that_clears_its_interval_is_not_told_it_contains_one(self):
        rows = [_row(500.0, 1000.0) for _ in range(40)] + \
               [_row(-20.0, 1000.0) for _ in range(10)]
        doc = self._doc(rows)
        self.assertGreater(equal_weighted(rows)["ci_low"], 1.0)
        self.assertNotIn("contains 1", doc)

    def test_the_as_sized_headline_is_still_there(self):
        # This section qualifies the headline; it does not replace it.
        doc = self._doc([_row(200.0, 1000.0), _row(-100.0, 800.0)] * 30)
        self.assertIn("## Headline", doc)
        self.assertIn("Net P&L", doc)


if __name__ == "__main__":
    unittest.main()
