"""Matched low-SI controls: calipers, k, diagnostics, and the validity gate."""
import unittest

from src.squeeze.sleeve import matching


def _u(key, rv=0.9, mcap=20.0, price=3.0):
    return matching.Unit(key=key, rv=rv, log_mcap=mcap, log_price=price)


class MatchingTest(unittest.TestCase):
    def test_each_treated_unit_gets_k_controls(self):
        treated = [_u("T1"), _u("T2")]
        controls = [_u(f"C{i}", rv=0.9 + 0.01 * i) for i in range(6)]
        got = matching.match(treated, controls, k=3)
        self.assertEqual(len(got.pairs["T1"]), 3)
        self.assertEqual(len(got.pairs["T2"]), 3)
        self.assertEqual(got.drop_rate, 0.0)

    def test_the_nearest_controls_are_the_ones_chosen(self):
        treated = [_u("T1", rv=1.00)]
        controls = [_u("C_near", rv=1.01), _u("C_mid", rv=1.08),
                    _u("C_far", rv=1.15), _u("C_farther", rv=1.19)]
        got = matching.match(treated, controls, k=2)
        self.assertEqual(got.pairs["T1"], ["C_near", "C_mid"])

    def test_a_treated_unit_with_no_in_caliper_control_is_dropped_and_counted(self):
        treated = [_u("T1", rv=1.0), _u("T2", rv=1.0)]
        # every control is >20% away in rv, so both treated units drop
        controls = [_u(f"C{i}", rv=5.0) for i in range(5)]
        got = matching.match(treated, controls, k=3)
        self.assertEqual(sorted(got.dropped), ["T1", "T2"])
        self.assertAlmostEqual(got.drop_rate, 1.0)
        self.assertFalse(matching.is_valid(got))

    def test_the_market_cap_caliper_binds(self):
        treated = [_u("T1", mcap=20.0)]
        controls = [_u("C_far", mcap=22.0), _u("C_ok", mcap=20.5)]
        got = matching.match(treated, controls, k=2)
        self.assertEqual(got.pairs["T1"], ["C_ok"])

    def test_controls_may_be_reused_and_reuse_is_logged(self):
        treated = [_u("T1"), _u("T2"), _u("T3")]
        controls = [_u("C1"), _u("C2")]
        got = matching.match(treated, controls, k=2)
        self.assertEqual(got.reuse["C1"], 3)
        self.assertEqual(got.reuse["C2"], 3)

    def test_a_well_matched_cohort_is_valid(self):
        treated = [_u(f"T{i}", rv=0.9 + 0.001 * i) for i in range(10)]
        controls = [_u(f"C{i}", rv=0.9 + 0.001 * i) for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertTrue(matching.is_valid(got))
        for value in got.smd.values():
            self.assertLessEqual(value, matching.MAX_SMD)

    def test_a_constant_covariate_is_not_spurious_imbalance(self):
        # Both arms share exactly identical covariates. Float noise in the
        # variance of a repeated constant once produced pooled_sd ~1e-16,
        # which cleared the zero guard and turned noise/noise into an SMD of
        # sqrt(2) — rejecting every observation date downstream.
        treated = [_u(f"T{i}") for i in range(6)]
        controls = [_u(f"C{i}") for i in range(12)]
        got = matching.match(treated, controls, k=3)
        for value in got.smd.values():
            self.assertEqual(value, 0.0)
        self.assertTrue(matching.is_valid(got))

    def test_a_biased_cohort_fails_the_standardised_difference_check(self):
        treated = [_u(f"T{i}", rv=1.10 + 0.001 * i) for i in range(10)]
        # inside the 20% rv caliper, but systematically lower
        controls = [_u(f"C{i}", rv=0.92 + 0.001 * i) for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertFalse(matching.is_valid(got))

    def test_no_controls_at_all_drops_everything(self):
        got = matching.match([_u("T1")], [], k=3)
        self.assertEqual(got.dropped, ["T1"])
        self.assertFalse(matching.is_valid(got))
