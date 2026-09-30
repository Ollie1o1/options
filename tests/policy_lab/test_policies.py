"""The grid is preregistered: its cardinality is the multiple-testing bar every
result must clear, so these tests pin the grid's size and the presence of the
null policies a knob has to beat.
"""
import unittest

from src.policy_lab.policies import (
    LIVE_BASELINE_LONG, LIVE_BASELINE_SHORT, LONG_PREMIUM_GRID, NOSTOP_GRID,
    NULL_POLICIES, SHORT_PREMIUM_GRID, ExitPolicy, grid_cardinality,
)

# Baseline the prereg compares NOSTOP_GRID against — not itself a grid
# member. Mirrors `CORPUS_A_RECORDER_SHORT` in report.py.
_NOSTOP_BASELINE = ExitPolicy("corpusA_recorder", 0.50, 2.0, None, None)


class TestPolicies(unittest.TestCase):
    def test_policy_is_frozen(self):
        with self.assertRaises(Exception):
            NULL_POLICIES[0].name = "x"

    def test_nulls_include_hold_to_expiry_and_never_stop(self):
        names = {p.name for p in NULL_POLICIES}
        self.assertIn("null_hold_to_expiry", names)
        self.assertIn("null_never_stop", names)

    def test_hold_to_expiry_has_no_exit_knobs_set(self):
        p = next(p for p in NULL_POLICIES if p.name == "null_hold_to_expiry")
        self.assertIsNone(p.take_profit_frac)
        self.assertIsNone(p.stop_mult)
        self.assertIsNone(p.time_exit_dte)
        self.assertIsNone(p.max_hold_days)

    def test_short_grid_spans_the_preregistered_knobs(self):
        tps = {p.take_profit_frac for p in SHORT_PREMIUM_GRID}
        self.assertEqual(tps, {0.25, 0.35, 0.50, 0.65, 0.75, None})

    def test_live_baselines_are_in_their_grids(self):
        self.assertIn(LIVE_BASELINE_SHORT, SHORT_PREMIUM_GRID)
        self.assertIn(LIVE_BASELINE_LONG, LONG_PREMIUM_GRID)

    def test_live_short_baseline_matches_config(self):
        # Live: take profit at 50% of credit, stop at 2.0x credit, 21 DTE.
        self.assertEqual(LIVE_BASELINE_SHORT.take_profit_frac, 0.50)
        self.assertEqual(LIVE_BASELINE_SHORT.stop_mult, 2.0)
        self.assertEqual(LIVE_BASELINE_SHORT.time_exit_dte, 21)

    def test_grid_cardinality_counts_every_trial_including_nulls(self):
        n = grid_cardinality(SHORT_PREMIUM_GRID)
        self.assertEqual(n, len(SHORT_PREMIUM_GRID))
        self.assertGreater(n, 20)

    def test_policy_names_are_unique(self):
        for grid in (SHORT_PREMIUM_GRID, LONG_PREMIUM_GRID, NULL_POLICIES):
            names = [p.name for p in grid]
            self.assertEqual(len(names), len(set(names)))

    def test_grids_are_never_pooled(self):
        # Short and long premium sweep different knobs and must stay separate.
        self.assertEqual(
            set(p.name for p in SHORT_PREMIUM_GRID)
            & set(p.name for p in LONG_PREMIUM_GRID), set())


class TestNostopGrid(unittest.TestCase):
    """Pins the preregistered 6-cell grid in
    docs/PREREG_POLICY_LAB_NOSTOP_20260928.md exactly — expanding it after
    seeing results would invalidate the confirmatory run.
    """

    def test_exactly_six_cells(self):
        self.assertEqual(len(NOSTOP_GRID), 6)

    def test_cells_match_the_preregistered_table(self):
        cells = {(p.take_profit_frac, p.stop_mult) for p in NOSTOP_GRID}
        self.assertEqual(cells, {
            (0.50, None),
            (0.50, 3.0),
            (0.50, 1.5),
            (0.65, None),
            (0.65, 2.0),
            (0.35, None),
        })

    def test_no_dte_or_max_hold_knobs_are_armed(self):
        for p in NOSTOP_GRID:
            self.assertIsNone(p.time_exit_dte)
            self.assertIsNone(p.max_hold_days)

    def test_baseline_is_not_a_grid_member(self):
        # The recorder baseline is the comparison point, not a candidate.
        self.assertNotIn(_NOSTOP_BASELINE, NOSTOP_GRID)

    def test_policy_names_are_unique(self):
        names = [p.name for p in NOSTOP_GRID]
        self.assertEqual(len(names), len(set(names)))

    def test_grid_cardinality_is_six(self):
        self.assertEqual(grid_cardinality(NOSTOP_GRID), 6)

    def test_manifest_n_trials_is_six_not_seven_fifty(self):
        # This is what reaches the family-wise correction as `n_trials` —
        # see `run_manifest` in report.py.
        from src.policy_lab.report import run_manifest
        m = run_manifest(corpus="A", grid=NOSTOP_GRID, costs="cross",
                         seed=0, row_counts={})
        self.assertEqual(m["n_trials"], 6)
        self.assertNotEqual(m["n_trials"], 750)
        self.assertNotEqual(m["n_trials"], len(SHORT_PREMIUM_GRID))


if __name__ == "__main__":
    unittest.main()
