"""Short-interest-only baseline for the squeeze grader.

The original study's sharpest finding was that ranking by short-interest level
alone matched or beat the full multi-factor grader (+4.36pp asymmetry vs
+3.74pp). The 2026-07-28 rescoring lifted the grader, but the baseline was never
re-measured under the same conditions — it had been computed ad hoc and never
committed, so it could not be re-run. This is that baseline, as code.

Cohort size is matched PER DATE, not globally. Inference bootstraps over
settlement dates, so a baseline that took its names from different dates than
the grader would not be comparable however well the totals lined up.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.squeeze.backtest.study import cohort_overlap, select_top_by_si


def _row(date, symbol, si, grade=None):
    return {"date": date, "symbol": symbol, "si_ratio": si, "grade": grade}


class TestSelection(unittest.TestCase):
    def test_takes_the_highest_short_interest_on_each_date(self):
        panel = [_row("d1", "A", 0.10), _row("d1", "B", 0.30), _row("d1", "C", 0.20)]
        picked = select_top_by_si(panel, {"d1": 2})
        self.assertEqual({r["symbol"] for r in picked}, {"B", "C"})

    def test_matches_the_graders_count_date_by_date(self):
        panel = [_row("d1", "A", 0.5), _row("d1", "B", 0.4),
                 _row("d2", "C", 0.3), _row("d2", "D", 0.2)]
        picked = select_top_by_si(panel, {"d1": 2, "d2": 1})
        by_date = {}
        for r in picked:
            by_date[r["date"]] = by_date.get(r["date"], 0) + 1
        self.assertEqual(by_date, {"d1": 2, "d2": 1})

    def test_a_date_the_grader_selected_nothing_on_contributes_nothing(self):
        panel = [_row("d1", "A", 0.5), _row("d2", "B", 0.9)]
        picked = select_top_by_si(panel, {"d1": 1})
        self.assertEqual([r["symbol"] for r in picked], ["A"])

    def test_rows_without_short_interest_are_not_selectable(self):
        # Ungradeable rows are excluded from the grader cohort, so including
        # them here would compare different populations.
        panel = [_row("d1", "A", None), _row("d1", "B", 0.01)]
        picked = select_top_by_si(panel, {"d1": 2})
        self.assertEqual([r["symbol"] for r in picked], ["B"])

    def test_asking_for_more_names_than_exist_returns_what_there_is(self):
        panel = [_row("d1", "A", 0.5)]
        self.assertEqual(len(select_top_by_si(panel, {"d1": 10})), 1)

    def test_selection_is_deterministic_under_ties(self):
        panel = [_row("d1", "B", 0.2), _row("d1", "A", 0.2), _row("d1", "C", 0.2)]
        first = [r["symbol"] for r in select_top_by_si(panel, {"d1": 2})]
        second = [r["symbol"] for r in select_top_by_si(list(reversed(panel)), {"d1": 2})]
        self.assertEqual(first, second)


class TestOverlap(unittest.TestCase):
    def test_identical_cohorts_overlap_completely(self):
        rows = [_row("d1", "A", 0.5), _row("d1", "B", 0.4)]
        self.assertAlmostEqual(cohort_overlap(rows, rows), 1.0)

    def test_disjoint_cohorts_do_not_overlap(self):
        self.assertAlmostEqual(
            cohort_overlap([_row("d1", "A", 0.5)], [_row("d1", "B", 0.4)]), 0.0
        )

    def test_overlap_is_measured_on_date_and_symbol_together(self):
        # The same ticker on a different settlement date is a different pick.
        a = [_row("d1", "A", 0.5)]
        b = [_row("d2", "A", 0.5)]
        self.assertAlmostEqual(cohort_overlap(a, b), 0.0)

    def test_half_shared_reads_as_half(self):
        a = [_row("d1", "A", 0.5), _row("d1", "B", 0.4)]
        b = [_row("d1", "A", 0.5), _row("d1", "C", 0.3)]
        self.assertAlmostEqual(cohort_overlap(a, b), 0.5)

    def test_empty_cohorts_do_not_divide_by_zero(self):
        self.assertEqual(cohort_overlap([], []), 0.0)


class TestPairedComparison(unittest.TestCase):
    """The difference between the two cohorts' asymmetry, bootstrapped over
    dates. Paired, because both cohorts are drawn from the same dates — an
    unpaired interval would carry the between-date variance twice and hide a
    real difference."""

    def _panel(self, dates=6):
        # Grader SETUP names never reach the up-tail; the high-SI names always
        # do. A working comparison must report the baseline ahead.
        rows = []
        for d in range(dates):
            date = f"2020-01-{d + 1:02d}"
            rows.append({"date": date, "symbol": "GOOD", "si_ratio": 0.9,
                         "grade": "NONE", "z_21": 3.0, "zdn_21": -0.5})
            rows.append({"date": date, "symbol": "PICKED", "si_ratio": 0.1,
                         "grade": "SETUP", "z_21": 0.1, "zdn_21": -3.0})
        return rows

    def test_reports_both_cohorts_and_their_difference(self):
        from src.squeeze.backtest.study import si_only_comparison

        out = si_only_comparison(self._panel(), horizon=21, n_boot=200, block=1)
        self.assertIn("grader_asymmetry", out)
        self.assertIn("si_only_asymmetry", out)
        self.assertAlmostEqual(
            out["difference"], out["si_only_asymmetry"] - out["grader_asymmetry"]
        )

    def test_detects_a_baseline_that_beats_the_grader(self):
        from src.squeeze.backtest.study import si_only_comparison

        out = si_only_comparison(self._panel(), horizon=21, n_boot=200, block=1)
        self.assertGreater(out["difference"], 0)

    def test_matches_the_cohort_sizes(self):
        from src.squeeze.backtest.study import si_only_comparison

        out = si_only_comparison(self._panel(), horizon=21, n_boot=200, block=1)
        self.assertEqual(out["grader_n"], out["si_only_n"])

    def test_reports_overlap_between_the_two_selections(self):
        from src.squeeze.backtest.study import si_only_comparison

        # Disjoint here: the grader picks the low-SI name every time.
        out = si_only_comparison(self._panel(), horizon=21, n_boot=200, block=1)
        self.assertAlmostEqual(out["overlap"], 0.0)

    def test_a_confidence_interval_is_returned(self):
        from src.squeeze.backtest.study import si_only_comparison

        out = si_only_comparison(self._panel(), horizon=21, n_boot=200, block=1)
        self.assertLessEqual(out["ci_lo"], out["difference"])
        self.assertGreaterEqual(out["ci_hi"], out["difference"])

    def test_an_empty_panel_reports_nothing_rather_than_raising(self):
        from src.squeeze.backtest.study import si_only_comparison

        out = si_only_comparison([], horizon=21, n_boot=10, block=1)
        self.assertEqual(out["n_dates"], 0)


if __name__ == "__main__":
    unittest.main()
