"""The five verdicts, and proof that STOP is reachable."""
import unittest

import numpy as np

from src.squeeze.sleeve import gate


def _posteriors(con30, cen30, con60, cen60):
    return {30: {"conservative": con30, "central": cen30},
            60: {"conservative": con60, "central": cen60}}


class CombineTest(unittest.TestCase):
    def test_the_pricing_terms_are_subtracted_elementwise(self):
        got = gate.combine(np.array([1.0, 2.0]), np.array([0.5, 0.5]),
                           np.array([0.1, 0.1]))
        np.testing.assert_allclose(got, [0.4, 1.4])

    def test_unequal_lengths_truncate_to_the_shortest(self):
        got = gate.combine(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]),
                           np.array([0.1, 0.1, 0.1]))
        self.assertEqual(len(got), 2)

    def test_the_posterior_is_the_share_of_draws_above_zero(self):
        self.assertAlmostEqual(
            gate.posterior_above_zero(np.array([-1.0, 1.0, 1.0, 1.0])), 0.75)

    def test_an_empty_draw_array_has_no_posterior(self):
        self.assertIsNone(gate.posterior_above_zero(np.array([])))


class DecideTest(unittest.TestCase):
    def test_go_needs_a_conservative_tenor_and_a_sign_agreeing_partner(self):
        got = gate.decide(_posteriors(0.95, 0.99, 0.40, 0.60),
                          n_cycles=6, covered_of_first_six=6, match_valid=True)
        self.assertEqual(got, "GO")

    def test_go_is_refused_when_the_other_tenor_disagrees_on_sign(self):
        got = gate.decide(_posteriors(0.95, 0.99, 0.10, 0.20),
                          n_cycles=6, covered_of_first_six=6, match_valid=True)
        self.assertNotEqual(got, "GO")

    def test_stop_fires_when_both_tenors_are_dead_on_the_central_variant(self):
        got = gate.decide(_posteriors(0.01, 0.05, 0.01, 0.08),
                          n_cycles=6, covered_of_first_six=6, match_valid=True)
        self.assertEqual(got, "STOP")

    def test_stop_is_judged_centrally_so_conservatism_alone_cannot_kill(self):
        # conservative says dead, central says alive -> must not be STOP
        got = gate.decide(_posteriors(0.02, 0.60, 0.02, 0.55),
                          n_cycles=6, covered_of_first_six=6, match_valid=True)
        self.assertNotEqual(got, "STOP")

    def test_an_unresolved_result_extends_within_the_budget(self):
        got = gate.decide(_posteriors(0.50, 0.60, 0.50, 0.60),
                          n_cycles=6, covered_of_first_six=6, match_valid=True,
                          extensions_used=0)
        self.assertEqual(got, "EXTEND")

    def test_the_extension_budget_is_bounded_and_then_it_is_no_go(self):
        got = gate.decide(_posteriors(0.50, 0.60, 0.50, 0.60),
                          n_cycles=10, covered_of_first_six=6, match_valid=True,
                          extensions_used=gate.MAX_EXTENSIONS)
        self.assertEqual(got, "NO-GO")

    def test_the_hard_stop_forces_no_go_regardless_of_extensions_left(self):
        got = gate.decide(_posteriors(0.50, 0.60, 0.50, 0.60),
                          n_cycles=gate.HARD_STOP_CYCLES,
                          covered_of_first_six=6, match_valid=True,
                          extensions_used=0)
        self.assertEqual(got, "NO-GO")

    def test_too_few_cycles_cannot_produce_any_verdict_yet(self):
        got = gate.decide(_posteriors(0.99, 0.99, 0.99, 0.99),
                          n_cycles=3, covered_of_first_six=3, match_valid=True)
        self.assertEqual(got, "EXTEND")

    def test_thin_coverage_invalidates_rather_than_deciding(self):
        got = gate.decide(_posteriors(0.99, 0.99, 0.99, 0.99),
                          n_cycles=6, covered_of_first_six=2, match_valid=True)
        self.assertEqual(got, "INVALID")

    def test_a_failed_match_invalidates_rather_than_deciding(self):
        got = gate.decide(_posteriors(0.01, 0.02, 0.01, 0.02),
                          n_cycles=6, covered_of_first_six=6, match_valid=False)
        self.assertEqual(got, "INVALID")

    def test_invalid_beats_stop_so_no_verdict_is_quotable_either_way(self):
        got = gate.decide(_posteriors(0.00, 0.00, 0.00, 0.00),
                          n_cycles=6, covered_of_first_six=1, match_valid=True)
        self.assertEqual(got, "INVALID")

    def test_a_missing_tenor_posterior_does_not_crash(self):
        got = gate.decide({30: {"conservative": 0.95, "central": 0.99}},
                          n_cycles=6, covered_of_first_six=6, match_valid=True)
        self.assertIn(got, ("EXTEND", "NO-GO"))
