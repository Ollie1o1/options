"""Tests for src/gate_separation.py — the pre-registered gate separation test.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_gate_separation -v

Every statistic is checked against a frame with a KNOWN answer, and the two
properties the registration leans on hardest are asserted directly:

  * the primary statistic is rank-based, so the unbounded credit-spread
    denominator (`pnl_pct` reaches -168 on Bull Put) cannot move it;
  * the NaN-masked family-wise permutation reduces EXACTLY to the unmasked
    reference when nothing is masked.

See docs/PREREG_GATE_SEPARATION_20260929.md.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import gate_separation as gs


def _cell(symbol, expiration, scan_date, passed, refused, strategy="Long Put"):
    """One cell's rows, given the two arms' outcomes as plain lists."""
    rows = []
    for v in passed:
        rows.append({"symbol": symbol, "expiration": expiration,
                     "scan_date": scan_date, "strategy_name": strategy,
                     "gate_passed": 1, "pnl_pct": v})
    for v in refused:
        rows.append({"symbol": symbol, "expiration": expiration,
                     "scan_date": scan_date, "strategy_name": strategy,
                     "gate_passed": 0, "pnl_pct": v})
    return rows


def _frame(*cells):
    return pd.DataFrame([r for c in cells for r in c])


def _planted(delta, n_symbols=40, cells_per_symbol=4, per_arm=5, seed=1):
    """A frame where the passed arm is shifted by `delta` against the refused."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_symbols):
        for c in range(cells_per_symbol):
            passed = rng.normal(delta, 1.0, size=per_arm)
            refused = rng.normal(0.0, 1.0, size=per_arm)
            rows += _cell(f"SYM{s}", "2026-10-30", f"2026-09-{1 + c:02d}",
                          list(passed), list(refused))
    return pd.DataFrame(rows)


class TestCellSuperiority(unittest.TestCase):
    def test_passed_strictly_better_reads_plus_half(self):
        df = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [3, 4], [1, 2]))
        cells = gs.cell_superiority(df)
        self.assertEqual(len(cells), 1)
        self.assertAlmostEqual(float(cells["d_auc"].iloc[0]), 0.5)

    def test_passed_strictly_worse_reads_minus_half(self):
        df = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [1, 2], [3, 4]))
        cells = gs.cell_superiority(df)
        self.assertAlmostEqual(float(cells["d_auc"].iloc[0]), -0.5)

    def test_interleaved_arms_read_zero(self):
        # passed straddles refused: (2,1) win, (2,4) loss, (3,1) win,
        # (3,4) loss -> 2 of 4 pairs -> AUC 0.5 -> d_auc 0.
        df = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [2, 3], [1, 4]))
        cells = gs.cell_superiority(df)
        self.assertAlmostEqual(float(cells["d_auc"].iloc[0]), 0.0)

    def test_a_partial_ordering_reads_between_the_extremes(self):
        # passed [1,3] vs refused [2,4] wins only (3,2) of four pairs.
        df = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [1, 3], [2, 4]))
        cells = gs.cell_superiority(df)
        self.assertAlmostEqual(float(cells["d_auc"].iloc[0]), -0.25)

    def test_ties_count_as_half(self):
        df = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [1.0], [1.0]))
        cells = gs.cell_superiority(df)
        self.assertAlmostEqual(float(cells["d_auc"].iloc[0]), 0.0)

    def test_a_cell_with_one_arm_is_dropped(self):
        df = _frame(
            _cell("AAA", "2026-10-30", "2026-09-01", [1, 2], []),
            _cell("BBB", "2026-10-30", "2026-09-01", [], [1, 2]),
            _cell("CCC", "2026-10-30", "2026-09-01", [2], [1]),
        )
        cells = gs.cell_superiority(df)
        self.assertEqual(list(cells["symbol"]), ["CCC"])

    def test_cells_are_split_by_expiration_and_date(self):
        df = _frame(
            _cell("AAA", "2026-10-30", "2026-09-01", [2], [1]),
            _cell("AAA", "2026-11-20", "2026-09-01", [2], [1]),
            _cell("AAA", "2026-10-30", "2026-09-02", [2], [1]),
        )
        self.assertEqual(len(gs.cell_superiority(df)), 3)

    def test_rank_statistic_ignores_an_unbounded_outlier(self):
        """The property the whole design rests on.

        Bull Put's `pnl_pct` reaches -168 because return-on-premium explodes as
        the entry credit approaches zero. A mean would be dominated by it; the
        rank statistic must not move at all.
        """
        mild = _frame(_cell("AAA", "2026-10-30", "2026-09-01", [1, 2], [3, 4]))
        wild = _frame(_cell("AAA", "2026-10-30", "2026-09-01",
                            [1, 2], [3, 659.0]))
        self.assertEqual(float(gs.cell_superiority(mild)["d_auc"].iloc[0]),
                         float(gs.cell_superiority(wild)["d_auc"].iloc[0]))


class TestClusterStatistic(unittest.TestCase):
    def test_a_planted_shift_is_recovered_with_the_right_sign(self):
        cells = gs.cell_superiority(_planted(1.0))
        means = gs.cluster_means(cells, "d_auc")
        self.assertGreater(gs.strategy_statistic(means), 0.15)

    def test_a_planted_inversion_keeps_its_sign(self):
        cells = gs.cell_superiority(_planted(-1.0))
        means = gs.cluster_means(cells, "d_auc")
        self.assertLess(gs.strategy_statistic(means), -0.15)

    def test_no_shift_reads_near_zero(self):
        cells = gs.cell_superiority(_planted(0.0))
        means = gs.cluster_means(cells, "d_auc")
        self.assertAlmostEqual(gs.strategy_statistic(means), 0.0, delta=0.05)

    def test_one_heavily_scanned_symbol_cannot_carry_the_result(self):
        """Cluster weighting: 200 cells on one ticker must not outvote 30 others."""
        rows = []
        for i in range(200):
            rows += _cell("LOUD", "2026-10-30", f"2026-09-{1 + i % 28:02d}",
                          [5.0], [1.0])
        for s in range(30):
            rows += _cell(f"SYM{s}", "2026-10-30", "2026-09-01", [1.0], [5.0])
        cells = gs.cell_superiority(pd.DataFrame(rows))
        means = gs.cluster_means(cells, "d_auc")
        # 30 symbols say -0.5, one says +0.5 -> the mean must be negative.
        self.assertLess(gs.strategy_statistic(means), 0.0)

    def test_bootstrap_ci_brackets_a_planted_effect(self):
        cells = gs.cell_superiority(_planted(1.0))
        lo, hi = gs.cluster_bootstrap_ci(cells, "d_auc", n_boot=2000)
        stat = gs.strategy_statistic(gs.cluster_means(cells, "d_auc"))
        self.assertLess(lo, stat)
        self.assertGreater(hi, stat)
        self.assertGreater(lo, 0.0)

    def test_bootstrap_ci_contains_zero_when_there_is_no_effect(self):
        cells = gs.cell_superiority(_planted(0.0))
        lo, hi = gs.cluster_bootstrap_ci(cells, "d_auc", n_boot=2000)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)


class TestWinsorizedSecondary(unittest.TestCase):
    def test_winsorizing_caps_the_denominator_blowup(self):
        s = pd.Series([-168.0] + [0.1] * 98 + [659.0])
        w = gs.winsorize(s, 0.01, 0.99)
        self.assertGreater(w.min(), -168.0)
        self.assertLess(w.max(), 659.0)

    def test_mean_difference_keeps_the_sign_of_a_planted_shift(self):
        cells = gs.cell_mean_difference(_planted(1.0))
        means = gs.cluster_means(cells, "d_mean")
        self.assertGreater(gs.strategy_statistic(means), 0.0)

    def test_unwinsorized_mean_is_dominated_by_one_outlier(self):
        """Why the secondary is winsorized — demonstrated, not asserted in prose."""
        rows = []
        for s in range(30):
            rows += _cell(f"SYM{s}", "2026-10-30", "2026-09-01", [1.0], [0.0])
        rows += _cell("BOOM", "2026-10-30", "2026-09-01", [-168.0], [0.0])
        df = pd.DataFrame(rows)
        raw = gs.strategy_statistic(gs.cluster_means(
            gs.cell_mean_difference(df, winsor=None), "d_mean"))
        win = gs.strategy_statistic(gs.cluster_means(
            gs.cell_mean_difference(df, winsor=(0.01, 0.99)), "d_mean"))
        self.assertLess(raw, 0.0)       # one row flips the whole statistic
        self.assertGreater(win, raw)    # winsorizing pulls it back


def _reference_family_wise_p(cluster_means, n_perm=2000, seed=0):
    """Verbatim reference: policy_lab.stats.family_wise_p (feat/policy-lab).

    Copied rather than imported — that module lives on another branch. The
    equivalence test below is what keeps this copy honest.
    """
    arr = np.asarray(cluster_means, dtype="float64")
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return 1.0
    n_clusters = arr.shape[1]
    obs = float(np.abs(arr.mean(axis=1)).max())
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype="float64")
    for i in range(int(n_perm)):
        s = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        null[i] = np.abs((arr * s).mean(axis=1)).max()
    return float((null >= obs).mean())


class TestFamilyWisePermutation(unittest.TestCase):
    def test_masked_version_reduces_exactly_to_the_reference(self):
        """Registration guard 2. Must be exact, not approximate."""
        rng = np.random.default_rng(7)
        arr = rng.normal(size=(6, 40))
        self.assertEqual(
            gs.family_wise_p_masked(arr, n_perm=500, seed=3),
            _reference_family_wise_p(arr, n_perm=500, seed=3))

    def test_masking_preserves_each_strategys_own_denominator(self):
        """A strategy absent from a symbol must not be diluted toward zero."""
        arr = np.full((2, 40), np.nan)
        arr[0, :] = 0.3            # present everywhere
        arr[1, :10] = 0.3          # present on 10 symbols only
        masked = gs.masked_row_means(arr)
        self.assertAlmostEqual(masked[0], 0.3)
        self.assertAlmostEqual(masked[1], 0.3)

    def test_a_strong_common_effect_beats_the_permutation_null(self):
        arr = np.full((6, 40), 0.3)
        self.assertLess(gs.family_wise_p_masked(arr, n_perm=500, seed=1), 0.05)

    def test_pure_noise_does_not_beat_the_null(self):
        rng = np.random.default_rng(11)
        arr = rng.normal(scale=0.3, size=(6, 40))
        self.assertGreater(gs.family_wise_p_masked(arr, n_perm=500, seed=1),
                           0.05)

    def test_six_looks_cost_more_than_one(self):
        """The whole reason this correction exists.

        The same single strategy's vector must be harder to call significant
        when it is the best of six noisy siblings than when it stands alone.
        """
        rng = np.random.default_rng(5)
        target = rng.normal(0.18, 0.3, size=40)
        alone = gs.family_wise_p_masked(target.reshape(1, -1),
                                        n_perm=2000, seed=2)
        siblings = np.vstack([target] + [rng.normal(0.0, 0.3, size=40)
                                         for _ in range(5)])
        family = gs.family_wise_p_masked(siblings, n_perm=2000, seed=2)
        self.assertGreater(family, alone)


class TestNegativeControl(unittest.TestCase):
    def test_shuffling_the_arm_label_destroys_a_real_effect(self):
        res = gs.negative_control(_planted(1.0), n_shuffles=60, seed=4)
        self.assertGreater(abs(res["observed"]), res["p95_abs"])
        self.assertLess(abs(res["null_mean"]), 0.01)

    def test_the_null_is_centred_on_zero_when_there_is_no_effect(self):
        res = gs.negative_control(_planted(0.0), n_shuffles=60, seed=4)
        self.assertLess(abs(res["null_mean"]), 0.01)


class TestBothArmFilter(unittest.TestCase):
    def test_single_arm_cells_are_removed(self):
        df = _frame(
            _cell("AAA", "2026-10-30", "2026-09-01", [1, 2], []),
            _cell("BBB", "2026-10-30", "2026-09-01", [2], [1]),
        )
        self.assertEqual(sorted(set(gs.both_arm_rows(df)["symbol"])), ["BBB"])

    def test_filtering_does_not_change_the_statistic(self):
        df = _frame(
            _cell("AAA", "2026-10-30", "2026-09-01", [1, 2], []),
            _cell("BBB", "2026-10-30", "2026-09-01", [3, 4], [1, 2]),
            _cell("CCC", "2026-10-30", "2026-09-01", [], [1, 2]),
        )
        full = gs.strategy_statistic(
            gs.cluster_means(gs.cell_superiority(df), "d_auc"))
        filtered = gs.strategy_statistic(
            gs.cluster_means(gs.cell_superiority(gs.both_arm_rows(df)),
                             "d_auc"))
        self.assertEqual(full, filtered)

    def test_a_permutation_cannot_move_a_cell_across_the_boundary(self):
        """Why the filter is exact rather than approximate."""
        df = gs.both_arm_rows(_planted(0.5, n_symbols=5, seed=3))
        rng = np.random.default_rng(0)
        shuffled = df.copy()
        shuffled["gate_passed"] = shuffled.groupby(
            list(gs.CELL_COLS))["gate_passed"].transform(
                lambda s: s.to_numpy()[rng.permutation(len(s))])
        self.assertEqual(len(gs.both_arm_rows(shuffled)), len(df))


class TestLeaveOneSymbolOut(unittest.TestCase):
    def test_a_uniform_effect_survives_dropping_any_symbol(self):
        means = pd.Series({f"S{i}": 0.2 for i in range(25)})
        self.assertTrue(gs.loso_sign_stable(means))

    def test_an_effect_carried_by_one_symbol_does_not_survive(self):
        means = pd.Series({f"S{i}": -0.01 for i in range(20)})
        means["HERO"] = 5.0
        self.assertGreater(gs.strategy_statistic(means), 0.0)
        self.assertFalse(gs.loso_sign_stable(means))

    def test_a_single_cluster_is_never_stable(self):
        self.assertFalse(gs.loso_sign_stable(pd.Series({"S0": 0.4})))


class TestVerdict(unittest.TestCase):
    def _result(self, **kw):
        base = dict(strategy="Long Put", n_rows=500, n_cells=93, n_clusters=38,
                    stat=-0.2, ci_lo=-0.3, ci_hi=-0.1, secondary=-0.05,
                    fwer_p=0.01, loso_stable=True, half_a=-0.2, half_b=-0.18,
                    skew=0.1)
        base.update(kw)
        return gs.StrategyResult(**base)

    def test_too_few_clusters_is_insufficient_not_null(self):
        self.assertEqual(gs.verdict(self._result(n_clusters=19)), "insufficient")

    def test_a_clean_inversion_reads_inverted(self):
        self.assertEqual(gs.verdict(self._result()), "inverted")

    def test_a_clean_separation_reads_separates(self):
        self.assertEqual(gs.verdict(self._result(
            stat=0.2, ci_lo=0.1, ci_hi=0.3, secondary=0.05,
            half_a=0.2, half_b=0.18)), "separates")

    def test_a_ci_containing_zero_is_null(self):
        self.assertEqual(gs.verdict(self._result(ci_hi=0.05)), "null")

    def test_failing_the_family_wise_correction_is_null(self):
        self.assertEqual(gs.verdict(self._result(fwer_p=0.06)), "null")

    def test_a_disagreeing_secondary_is_null(self):
        self.assertEqual(gs.verdict(self._result(secondary=0.05)), "null")

    def test_an_unstable_result_is_null(self):
        self.assertEqual(gs.verdict(self._result(loso_stable=False)), "null")

    def test_halves_disagreeing_in_sign_is_null(self):
        self.assertEqual(gs.verdict(self._result(half_b=0.18)), "null")

    def test_a_missing_half_does_not_crash_and_is_null(self):
        self.assertEqual(gs.verdict(self._result(half_b=None)), "null")


class TestStrategyLabel(unittest.TestCase):
    def test_a_recorded_name_wins(self):
        self.assertEqual(
            gs._derive_strategy("Bull Put", "Premium Selling", "put"),
            "Bull Put")

    def test_premium_selling_puts_derive_to_short_put(self):
        self.assertEqual(gs._derive_strategy(None, "Premium Selling", "put"),
                         "Short Put")

    def test_discovery_calls_derive_to_long_call(self):
        self.assertEqual(gs._derive_strategy("", "Discovery scan", "call"),
                         "Long Call")

    def test_a_row_with_no_mode_stays_unlabelled(self):
        self.assertEqual(gs._derive_strategy("", "", "put"), gs.UNLABELLED)
        self.assertEqual(gs._derive_strategy(None, None, None), gs.UNLABELLED)

    def test_puts_and_calls_never_collapse_together(self):
        """A put sold and a call bought are not exchangeable."""
        self.assertNotEqual(
            gs._derive_strategy(None, "Discovery scan", "put"),
            gs._derive_strategy(None, "Discovery scan", "call"))


class TestHalfSplit(unittest.TestCase):
    def test_the_split_is_at_the_median_scan_date(self):
        df = pd.DataFrame({"scan_date": [f"2026-09-{d:02d}" for d in
                                         range(1, 11)]})
        a, b = gs.half_split(df)
        self.assertTrue(a["scan_date"].max() < b["scan_date"].min())
        self.assertEqual(len(a) + len(b), 10)

    def test_a_single_date_puts_everything_in_one_half(self):
        df = pd.DataFrame({"scan_date": ["2026-09-01"] * 5})
        a, b = gs.half_split(df)
        self.assertEqual(len(a) + len(b), 5)


if __name__ == "__main__":
    unittest.main()
