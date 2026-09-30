"""The paired difference is why this lab exists: both arms replay the same
entry on the same path, so the underlying's move cancels. These tests pin that
the bootstrap resamples whole clusters and that the variance reduction is
measured rather than assumed.
"""
import unittest

import numpy as np
import pandas as pd

from src.policy_lab.stats import (
    PAIRING_INEFFECTIVE_RATIO, cluster_bootstrap_mean_ci,
    cluster_bootstrap_mean_ci_many, paired_frame, variance_reduction,
)
from src.policy_lab.types import Outcome, PathPoint, PricePath


def _path(pid, symbol, entry):
    pt = PathPoint(entry, 0.95, 1.05, 1.00, 100.0, 30)
    pt2 = PathPoint("2026-08-25", 0.35, 0.45, 0.40, 100.0, 23)
    return PricePath(pid, symbol, "Bull Put", entry, 1.00, 400.0, True,
                     (pt, pt2), "2026-08-25", 0.15, "A")


def _out(pid, r):
    return Outcome(pid, "2026-08-25", "take_profit", r, 3)


class TestPairedStats(unittest.TestCase):
    def setUp(self):
        self.paths = [_path("p1", "AMD", "2026-08-18"),
                      _path("p2", "AMD", "2026-08-19"),
                      _path("p3", "MU", "2026-08-18")]
        self.base = {"p1": _out("p1", 0.10), "p2": _out("p2", 0.20),
                     "p3": _out("p3", 0.30)}
        self.alt = {"p1": _out("p1", 0.15), "p2": _out("p2", 0.28),
                    "p3": _out("p3", 0.33)}

    def test_frame_has_one_row_per_paired_position(self):
        df = paired_frame(self.paths, self.base, self.alt)
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df.columns),
                         {"position_id", "symbol", "entry_date", "cluster",
                          "r_base", "r_alt", "d"})

    def test_cluster_is_symbol_and_entry_date(self):
        df = paired_frame(self.paths, self.base, self.alt)
        self.assertEqual(sorted(df["cluster"].unique()),
                         ["AMD|2026-08-18", "AMD|2026-08-19", "MU|2026-08-18"])

    def test_difference_is_alt_minus_base(self):
        df = paired_frame(self.paths, self.base, self.alt)
        row = df[df["position_id"] == "p1"].iloc[0]
        self.assertAlmostEqual(row["d"], 0.05)

    def test_positions_missing_from_either_arm_are_dropped(self):
        alt = dict(self.alt); alt.pop("p3")
        df = paired_frame(self.paths, self.base, alt)
        self.assertEqual(len(df), 2)

    def test_bootstrap_ci_brackets_a_clear_positive_effect(self):
        df = pd.DataFrame({
            "cluster": [f"c{i}" for i in range(60)],
            "d": [0.10] * 60,
        })
        lo, hi = cluster_bootstrap_mean_ci(df, "d", "cluster", n_boot=500, seed=1)
        self.assertGreater(lo, 0.0)
        self.assertAlmostEqual(hi, 0.10, places=6)

    def test_bootstrap_ci_contains_zero_for_noise(self):
        import numpy as np
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "cluster": [f"c{i}" for i in range(80)],
            "d": rng.normal(0.0, 0.4, 80),
        })
        lo, hi = cluster_bootstrap_mean_ci(df, "d", "cluster", n_boot=500, seed=1)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_bootstrap_resamples_clusters_not_rows(self):
        # One cluster holding 100 rows must not act like 100 clusters.
        df = pd.DataFrame({"cluster": ["only"] * 100, "d": [0.1] * 100})
        lo, hi = cluster_bootstrap_mean_ci(df, "d", "cluster", n_boot=200, seed=1)
        self.assertAlmostEqual(lo, 0.1, places=6)
        self.assertAlmostEqual(hi, 0.1, places=6)

    def test_cluster_bootstrap_is_wider_than_a_row_bootstrap(self):
        """A row bootstrap would report a far tighter interval — and be wrong.

        Five clusters of twenty identical rows each. Resampling CLUSTERS draws
        five values, so the interval reflects n=5. Resampling ROWS would draw a
        hundred, reporting an interval about sqrt(20) times too narrow. This is
        the error this repo has made three times; the earlier bootstrap tests
        cannot detect it because their clusters hold one row each, where the two
        implementations coincide exactly.
        """
        import numpy as np
        rows = []
        for i, mean in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
            rows.extend({"cluster": f"c{i}", "d": mean} for _ in range(20))
        df = pd.DataFrame(rows)

        lo, hi = cluster_bootstrap_mean_ci(df, "d", "cluster", n_boot=4000, seed=3)
        cluster_width = hi - lo

        # What a row-level bootstrap would have produced on the same frame.
        rng = np.random.default_rng(3)
        vals = df["d"].to_numpy(dtype="float64")
        row_means = [float(rng.choice(vals, size=vals.size, replace=True).mean())
                     for _ in range(4000)]
        row_width = float(np.percentile(row_means, 97.5)
                          - np.percentile(row_means, 2.5))

        self.assertGreater(cluster_width, 2.0 * row_width)

    def test_empty_frame_returns_none(self):
        df = pd.DataFrame({"cluster": [], "d": []})
        self.assertEqual(cluster_bootstrap_mean_ci(df, "d", "cluster"),
                         (None, None))

    def test_variance_reduction_is_sd_d_over_sd_base(self):
        df = pd.DataFrame({"r_base": [0.0, 1.0, 2.0, 3.0],
                           "d": [0.0, 0.1, 0.0, 0.1]})
        self.assertLess(variance_reduction(df), PAIRING_INEFFECTIVE_RATIO)

    def test_pairing_that_does_not_help_is_visible(self):
        df = pd.DataFrame({"r_base": [0.0, 1.0, 2.0, 3.0],
                           "d": [0.0, 1.0, 2.0, 3.0]})
        self.assertGreaterEqual(variance_reduction(df),
                                PAIRING_INEFFECTIVE_RATIO)


class TestClusterBootstrapMany(unittest.TestCase):
    """`cluster_bootstrap_mean_ci_many` vectorizes the bootstrap across
    policies (report.py's sweep otherwise pays ~4s per policy, in a Python
    loop, per survivor of the family-wise screen — 750 policies x 4s = 48
    minutes on the real grid).

    EQUIVALENCE CLAIM: statistical, not bit-exact. The single-policy function
    draws `n_clusters` cluster-indices per boot via `rng.integers(...)`; the
    vectorized function draws one multinomial per boot via
    `rng.multinomial(...)`. Both are the same distribution (n_clusters iid
    uniform draws with replacement) but consume the underlying PRNG bit
    stream differently, so the same seed does not reproduce the same
    resample sequence between the two implementations — confirmed empirically
    below (`test_same_seed_is_not_bit_identical_because_rng_order_differs`).
    What IS asserted instead: for the same data, the two implementations'
    intervals agree with each other to a small tolerance across several
    seeds, and both consistently bracket a planted, known mean.
    """

    def _clustered_data(self, seed, n_clusters=50, rows_per_cluster=8,
                        n_policies=5, base_mean=0.02, step=0.01, noise=0.03):
        rng = np.random.default_rng(seed)
        cluster_ids = np.repeat(
            [f"c{i}" for i in range(n_clusters)], rows_per_cluster)
        n = cluster_ids.shape[0]
        base = rng.normal(base_mean, 0.1, n)
        per_policy_d = {}
        planted_means = {}
        for k in range(n_policies):
            planted = base_mean + step * k
            per_policy_d[f"pol{k}"] = base - base_mean + planted + \
                rng.normal(0.0, noise, n)
            planted_means[f"pol{k}"] = planted
        return cluster_ids, per_policy_d, planted_means

    def test_keys_match_input_policies(self):
        cluster_ids, per_policy_d, _ = self._clustered_data(seed=1)
        out = cluster_bootstrap_mean_ci_many(
            per_policy_d, cluster_ids, n_boot=200, seed=0)
        self.assertEqual(set(out), set(per_policy_d))

    def test_empty_policies_returns_empty_dict(self):
        cluster_ids, _, _ = self._clustered_data(seed=1)
        self.assertEqual(
            cluster_bootstrap_mean_ci_many({}, cluster_ids, n_boot=100), {})

    def test_mismatched_length_raises(self):
        cluster_ids, per_policy_d, _ = self._clustered_data(seed=1)
        bad = dict(per_policy_d)
        bad["short"] = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            cluster_bootstrap_mean_ci_many(bad, cluster_ids, n_boot=100)

    def test_single_cluster_is_deterministic_like_the_single_policy_version(
            self):
        cluster_ids = np.array(["only"] * 40)
        per_policy_d = {"a": np.full(40, 0.07), "b": np.full(40, -0.02)}
        out = cluster_bootstrap_mean_ci_many(
            per_policy_d, cluster_ids, n_boot=200, seed=1)
        self.assertAlmostEqual(out["a"][0], 0.07, places=6)
        self.assertAlmostEqual(out["a"][1], 0.07, places=6)
        self.assertAlmostEqual(out["b"][0], -0.02, places=6)
        self.assertAlmostEqual(out["b"][1], -0.02, places=6)

    def test_chunking_across_a_boot_count_bigger_than_one_chunk(self):
        # Internal chunk size is 1000; 2,500 forces three chunks to be
        # concatenated and must still produce a sane, ordered interval.
        cluster_ids, per_policy_d, _ = self._clustered_data(seed=2)
        out = cluster_bootstrap_mean_ci_many(
            per_policy_d, cluster_ids, n_boot=2500, seed=4)
        for name, (lo, hi) in out.items():
            self.assertIsNotNone(lo)
            self.assertLessEqual(lo, hi)

    def test_same_seed_is_not_bit_identical_because_rng_order_differs(self):
        # Documents WHY the equivalence test below is statistical rather
        # than exact — see the class docstring.
        cluster_ids, per_policy_d, _ = self._clustered_data(seed=3)
        name = "pol0"
        df = pd.DataFrame({"cluster": cluster_ids, "d": per_policy_d[name]})
        single = cluster_bootstrap_mean_ci(df, "d", n_boot=1000, seed=9)
        many = cluster_bootstrap_mean_ci_many(
            {name: per_policy_d[name]}, cluster_ids, n_boot=1000, seed=9)
        self.assertNotEqual(single, many[name])

    def test_statistical_equivalence_to_single_policy_across_seeds(self):
        """For several seeds and several policies at once: `_many`'s interval
        matches the single-policy function's interval on the SAME data to a
        small tolerance.

        This is the equivalence claim this fix stands on: a correctly
        calibrated 95% CI is EXPECTED to miss its own true mean ~5% of the
        time by construction, so "does this one draw's CI contain the true
        value" is not a property either implementation should be graded on
        per-draw (that's `test_both_bracket_a_clear_planted_effect` below,
        on one verified scenario) — but "do the two implementations land on
        the same interval from the same data" has no such flakiness, and is
        the actual correctness bar for a drop-in-faster replacement.
        """
        tol = 0.015
        for data_seed in (1, 2, 3):
            cluster_ids, per_policy_d, _ = self._clustered_data(
                data_seed, n_clusters=60, rows_per_cluster=10)
            for boot_seed in (5, 6):
                many = cluster_bootstrap_mean_ci_many(
                    per_policy_d, cluster_ids, n_boot=4000, seed=boot_seed)
                for name, d in per_policy_d.items():
                    df = pd.DataFrame({"cluster": cluster_ids, "d": d})
                    lo_s, hi_s = cluster_bootstrap_mean_ci(
                        df, "d", n_boot=4000, seed=boot_seed)
                    lo_m, hi_m = many[name]
                    self.assertLess(
                        abs(lo_s - lo_m), tol,
                        f"{name} lo diverges: single={lo_s} many={lo_m}")
                    self.assertLess(
                        abs(hi_s - hi_m), tol,
                        f"{name} hi diverges: single={hi_s} many={hi_m}")

    def test_both_bracket_a_clear_planted_effect(self):
        """One verified scenario (fixed seeds): several policies with
        clearly separated planted means, well away from zero and from each
        other, and both the single-policy and the vectorized function
        bracket the RIGHT planted mean for every policy — not just any
        interval, the one that corresponds to that policy's own effect.
        """
        cluster_ids, per_policy_d, planted = self._clustered_data(
            seed=11, n_clusters=80, rows_per_cluster=12,
            base_mean=0.10, step=0.08, noise=0.02)
        many = cluster_bootstrap_mean_ci_many(
            per_policy_d, cluster_ids, n_boot=4000, seed=8)
        for name, d in per_policy_d.items():
            df = pd.DataFrame({"cluster": cluster_ids, "d": d})
            lo_s, hi_s = cluster_bootstrap_mean_ci(
                df, "d", n_boot=4000, seed=8)
            lo_m, hi_m = many[name]
            self.assertLess(lo_s, planted[name])
            self.assertGreater(hi_s, planted[name])
            self.assertLess(lo_m, planted[name])
            self.assertGreater(hi_m, planted[name])
            # Also confirm the interval is specific, not just "wide enough
            # to catch anything": it must exclude neighboring policies'
            # planted means, 0.08 apart.
            for other_name, other_mean in planted.items():
                if other_name == name:
                    continue
                if abs(other_mean - planted[name]) > 0.06:
                    self.assertFalse(lo_m <= other_mean <= hi_m)


if __name__ == "__main__":
    unittest.main()
