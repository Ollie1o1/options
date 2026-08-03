"""Matched low-SI controls: calipers, k, diagnostics, and the validity gate."""
import unittest

from src.squeeze.sleeve import matching


def _u(key, rv=0.9, mcap=20.0, price=3.0, ret5d=0.12):
    return matching.Unit(key=key, rv=rv, log_mcap=mcap, log_price=price,
                         ret_5d=ret5d)


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


class MomentumMatchingTest(unittest.TestCase):
    def test_a_flat_control_cannot_match_a_name_that_ran(self):
        treated = [_u("T1", ret5d=0.12)]
        controls = [_u("C_flat", ret5d=0.00), _u("C_ran", ret5d=0.10)]
        got = matching.match(treated, controls, k=2)
        self.assertEqual(got.pairs["T1"], ["C_ran"])

    def test_no_control_inside_the_momentum_caliper_drops_the_treated_unit(self):
        treated = [_u("T1", ret5d=0.12)]
        controls = [_u(f"C{i}", ret5d=-0.20) for i in range(5)]
        got = matching.match(treated, controls, k=3)
        self.assertEqual(got.dropped, ["T1"])

    def test_smd_now_reports_the_momentum_covariate(self):
        treated = [_u(f"T{i}") for i in range(10)]
        controls = [_u(f"C{i}") for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertIn("ret_5d", got.smd)
        self.assertTrue(matching.is_valid(got))

    def test_a_momentum_imbalance_inside_the_caliper_still_fails_validity(self):
        treated = [_u(f"T{i}", ret5d=0.14 + 0.0001 * i) for i in range(10)]
        controls = [_u(f"C{i}", ret5d=0.10 + 0.0001 * i) for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertFalse(matching.is_valid(got))


class BalanceTest(unittest.TestCase):
    """`is_balanced` is `is_valid` minus the drop-rate arm.

    D_hist's estimand is the matchable subsample, so a treated unit with no
    in-caliper control is a SELECTION to be documented, not a defect in the
    comparison. Balance between the units that did match is what still has to
    hold: it is the only thing making treated-minus-control a fair difference.
    """

    def test_balance_holds_even_when_most_treated_units_are_unmatchable(self):
        # Ten well-matched units, ten with no control within any caliper.
        treated = ([_u(f"T{i}", rv=0.9 + 0.001 * i) for i in range(10)]
                   + [_u(f"X{i}", rv=9.0) for i in range(10)])
        controls = [_u(f"C{i}", rv=0.9 + 0.001 * i) for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertGreater(got.drop_rate, matching.MAX_DROP_RATE)
        self.assertFalse(matching.is_valid(got))
        self.assertTrue(matching.is_balanced(got))

    def test_an_imbalanced_cohort_is_still_not_balanced(self):
        treated = [_u(f"T{i}", rv=1.10 + 0.001 * i) for i in range(10)]
        controls = [_u(f"C{i}", rv=0.92 + 0.001 * i) for i in range(30)]
        got = matching.match(treated, controls, k=3)
        self.assertFalse(matching.is_balanced(got))

    def test_a_cohort_with_no_pairs_at_all_is_not_balanced(self):
        # Nothing matched means there is no comparison to be balanced, and an
        # empty SMD dict must never read as "no imbalance detected".
        got = matching.match([_u("T1")], [], k=3)
        self.assertFalse(matching.is_balanced(got))
