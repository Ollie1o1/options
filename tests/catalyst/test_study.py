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


if __name__ == "__main__":
    unittest.main()
