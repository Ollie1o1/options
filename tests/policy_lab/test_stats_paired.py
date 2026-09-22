"""The paired difference is why this lab exists: both arms replay the same
entry on the same path, so the underlying's move cancels. These tests pin that
the bootstrap resamples whole clusters and that the variance reduction is
measured rather than assumed.
"""
import unittest

import pandas as pd

from src.policy_lab.stats import (
    PAIRING_INEFFECTIVE_RATIO, cluster_bootstrap_mean_ci, paired_frame,
    variance_reduction,
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


if __name__ == "__main__":
    unittest.main()
