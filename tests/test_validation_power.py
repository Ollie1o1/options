import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import validation_power_analysis as vpa


class TestPowerMath(unittest.TestCase):
    def test_required_n_for_small_effect_is_large(self):
        # For r=0.10 at alpha=0.05 (two-sided), power=0.80, the Fisher-z
        # sample-size formula gives ~782.
        n = vpa.required_n_pearson(0.10, alpha=0.05, power=0.80)
        self.assertTrue(770 <= n <= 800, f"got {n}")

    def test_required_n_for_large_effect_is_small(self):
        n = vpa.required_n_pearson(0.50, alpha=0.05, power=0.80)
        self.assertTrue(20 <= n <= 35, f"got {n}")

    def test_min_detectable_r_at_n50(self):
        # Smallest |r| that is significant at p<0.05 (two-sided) with n=50
        # solves t = r*sqrt((n-2)/(1-r^2)) = 1.96 -> r ~ 0.279.
        r = vpa.min_detectable_r(50, alpha=0.05)
        self.assertAlmostEqual(r, 0.279, places=2)

    def test_min_detectable_r_decreases_with_n(self):
        self.assertGreater(vpa.min_detectable_r(50), vpa.min_detectable_r(200))

    def test_p_value_for_correlation_matches_known(self):
        # r=0.279, n=50 -> p just under 0.05.
        p = vpa.pearson_p_value(0.279, 50)
        self.assertTrue(0.03 <= p <= 0.06, f"got {p}")


class TestBayesian(unittest.TestCase):
    def test_posterior_prob_above_threshold_in_unit_range(self):
        prob = vpa.posterior_prob_ic_above(observed_r=0.30, n=50, threshold=0.08)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_strong_positive_gives_high_posterior(self):
        # A large observed IC on a decent n should make P(true IC>0.08) high.
        prob = vpa.posterior_prob_ic_above(observed_r=0.40, n=80, threshold=0.08)
        self.assertGreater(prob, 0.90)

    def test_near_threshold_gives_middling_posterior(self):
        prob = vpa.posterior_prob_ic_above(observed_r=0.08, n=50, threshold=0.08)
        self.assertTrue(0.4 <= prob <= 0.6, f"got {prob}")


if __name__ == "__main__":
    unittest.main()
