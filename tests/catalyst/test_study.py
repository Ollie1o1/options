"""Effect sizes and CIs. A CI containing zero must read NO EVIDENCE."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst.backtest import study


class TestBootstrapCI(unittest.TestCase):
    def test_is_deterministic_for_a_fixed_seed(self):
        a, b = [0.1] * 30, [0.0] * 30
        self.assertEqual(study.bootstrap_ci(a, b, seed=7),
                         study.bootstrap_ci(a, b, seed=7))

    def test_a_large_clean_separation_excludes_zero(self):
        a = [0.5 + i * 0.001 for i in range(80)]
        b = [-0.5 + i * 0.001 for i in range(80)]
        lo, hi = study.bootstrap_ci(a, b, seed=1)
        self.assertGreater(lo, 0)

    def test_identical_groups_straddle_zero(self):
        a = [0.05, -0.02, 0.11, -0.07] * 20
        lo, hi = study.bootstrap_ci(a, list(a), seed=1)
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)


class TestClusterBootstrapCI(unittest.TestCase):
    """The outcome is a property of the TICKER, not of the trial.

    `outcomes_for(ticker, vintage, ...)` never sees an nct_id, so every trial
    on one ticker at one vintage appends a byte-identical value to its arm.
    Measured 2026-08-27: 832 cached trials resolve to 270 tickers, mean 3.08
    each, VNDA alone 17. Resampling ROWS treats those copies as independent
    evidence and returns an interval that is too narrow — which is the one
    thing a "large effects are ruled out" claim depends on.
    """

    def _obs(self, values, clusters, arms):
        return list(zip(values, clusters, arms))

    def test_duplicating_rows_within_a_cluster_does_not_shrink_the_interval(self):
        # THE defining property. Copying each observation five times adds no
        # information, so the interval must not narrow. A row bootstrap would
        # shrink it by about sqrt(5).
        vals_a = [0.10 + i * 0.01 for i in range(20)]
        vals_b = [0.00 + i * 0.01 for i in range(20)]
        single = ([(v, f"A{i}", True) for i, v in enumerate(vals_a)]
                  + [(v, f"B{i}", False) for i, v in enumerate(vals_b)])
        copied = [(v, c, s) for (v, c, s) in single for _ in range(5)]

        lo1, hi1 = study.cluster_bootstrap_ci(single, seed=3)
        lo5, hi5 = study.cluster_bootstrap_ci(copied, seed=3)
        self.assertAlmostEqual(hi1 - lo1, hi5 - lo5, places=6)

    def test_a_row_bootstrap_would_shrink_it_which_is_the_bug(self):
        # Guards the premise: if duplication did not narrow the row bootstrap,
        # this whole change would be pointless and the test above would pass
        # for the wrong reason.
        a = [0.10 + i * 0.01 for i in range(20)]
        b = [0.00 + i * 0.01 for i in range(20)]
        lo1, hi1 = study.bootstrap_ci(a, b, seed=3)
        lo5, hi5 = study.bootstrap_ci(a * 5, b * 5, seed=3)
        self.assertLess(hi5 - lo5, (hi1 - lo1) * 0.75)

    def test_one_row_per_cluster_is_never_narrower_than_the_row_bootstrap(self):
        # With no duplication to exploit the two must broadly agree, and the
        # clustered one must never be the TIGHTER of the pair — a correction
        # that buys back precision is not a correction.
        a = [0.10 + i * 0.01 for i in range(40)]
        b = [0.00 + i * 0.01 for i in range(40)]
        obs = ([(v, f"A{i}", True) for i, v in enumerate(a)]
               + [(v, f"B{i}", False) for i, v in enumerate(b)])
        lo_c, hi_c = study.cluster_bootstrap_ci(obs, seed=11)
        lo_r, hi_r = study.bootstrap_ci(a, b, seed=11)
        self.assertGreaterEqual(hi_c - lo_c, (hi_r - lo_r) * 0.9)
        self.assertLess(hi_c - lo_c, (hi_r - lo_r) * 2.5)

    def test_is_deterministic_for_a_fixed_seed(self):
        obs = [(0.1, "T1", True), (0.2, "T1", True), (0.0, "T2", False),
               (0.05, "T3", False)] * 10
        self.assertEqual(study.cluster_bootstrap_ci(obs, seed=7),
                         study.cluster_bootstrap_ci(obs, seed=7))

    def test_a_cluster_spanning_both_arms_is_drawn_once(self):
        # A ticker whose trials land in both arms is ONE unit. Drawing it
        # independently per arm would break the correlation the cluster
        # bootstrap exists to preserve.
        obs = []
        for i in range(30):
            obs.append((0.2, f"T{i}", True))
            obs.append((0.1, f"T{i}", False))   # same ticker, other arm
        lo, hi = study.cluster_bootstrap_ci(obs, seed=5)
        # Every ticker carries the same within-ticker difference of +0.1, so
        # resampling tickers cannot move the difference at all.
        self.assertAlmostEqual(lo, 0.1, places=6)
        self.assertAlmostEqual(hi, 0.1, places=6)

    def test_identical_arms_straddle_zero(self):
        obs = ([(v, f"A{i}", True) for i, v in enumerate([0.05, -0.02, 0.11] * 10)]
               + [(v, f"B{i}", False) for i, v in enumerate([0.05, -0.02, 0.11] * 10)])
        lo, hi = study.cluster_bootstrap_ci(obs, seed=1)
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)


class TestCompareClustered(unittest.TestCase):
    def test_compare_clustered_widens_the_interval_on_duplicated_rows(self):
        vals_a = [0.10 + i * 0.01 for i in range(20)]
        vals_b = [0.00 + i * 0.01 for i in range(20)]
        obs = [(v, f"A{i}", True) for i, v in enumerate(vals_a) for _ in range(5)]
        obs += [(v, f"B{i}", False) for i, v in enumerate(vals_b) for _ in range(5)]
        clustered = study.compare_clustered(obs, key="H1", label="x")
        flat = study.compare(vals_a * 5, vals_b * 5, key="H1", label="x")
        self.assertGreater(clustered.ci_hi - clustered.ci_lo,
                           flat.ci_hi - flat.ci_lo)

    def test_it_reports_clusters_not_just_rows(self):
        obs = [(0.2, f"T{i}", True) for i in range(20) for _ in range(3)]
        obs += [(0.0, f"U{i}", False) for i in range(20) for _ in range(3)]
        r = study.compare_clustered(obs, key="H1", label="x")
        self.assertEqual(r.n_true, 60)
        self.assertEqual(r.n_false, 60)
        self.assertEqual(r.k_true, 20)
        self.assertEqual(r.k_false, 20)

    def test_underpowered_counts_CLUSTERS_not_rows(self):
        # 60 rows from 3 tickers is 3 observations, not 60. Counting rows here
        # is exactly the error that made 2,137 rows look like 150% of target.
        obs = [(0.2, f"T{i}", True) for i in range(3) for _ in range(20)]
        obs += [(0.0, f"U{i}", False) for i in range(3) for _ in range(20)]
        r = study.compare_clustered(obs, key="H1", label="x")
        self.assertEqual(r.verdict, "UNDERPOWERED")


class TestCompare(unittest.TestCase):
    def test_no_evidence_when_the_ci_contains_zero(self):
        a = [0.05, -0.02, 0.11, -0.07] * 10
        r = study.compare(a, list(a), key="H1", label="funded", seed=1)
        self.assertEqual(r.verdict, "NO EVIDENCE")

    def test_reports_n_for_both_arms(self):
        r = study.compare([0.1] * 12, [0.0] * 9, key="H1", label="f", seed=1)
        self.assertEqual((r.n_true, r.n_false), (12, 9))

    def test_underpowered_when_either_arm_is_tiny(self):
        r = study.compare([0.1] * 3, [0.0] * 40, key="H1", label="f", seed=1)
        self.assertEqual(r.verdict, "UNDERPOWERED")

    def test_empty_arm_is_underpowered_not_a_crash(self):
        r = study.compare([], [0.0] * 40, key="H1", label="f", seed=1)
        self.assertEqual(r.verdict, "UNDERPOWERED")

    def test_diff_is_true_minus_false(self):
        r = study.compare([0.2] * 30, [0.1] * 30, key="H1", label="f", seed=1)
        self.assertAlmostEqual(r.diff, 0.1, places=6)


class TestNotComputable(unittest.TestCase):
    """A declared hypothesis that cannot be run must SAY SO.

    Silently omitting it would leave a reader thinking it came back empty,
    which is a different claim from "the data to test this does not exist".
    """

    def test_verdict_and_reason_are_carried(self):
        r = study.not_computable("H4", "implied vs realised",
                                 "no historical option chains for biotech")
        self.assertEqual(r.verdict, "NOT COMPUTABLE")
        self.assertIn("chains", r.label)

    def test_arms_are_zero_not_fabricated(self):
        r = study.not_computable("H4", "x", "y")
        self.assertEqual((r.n_true, r.n_false), (0, 0))


if __name__ == "__main__":
    unittest.main()
