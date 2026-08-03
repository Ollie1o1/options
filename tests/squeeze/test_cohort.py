"""Cohort policy: who is treated, who is an eligible control, who is neither."""
import unittest

from src.squeeze.sleeve import cohort


def _r(si, ret5d=0.20):
    return {"si_ratio": si, "ret_5d": ret5d}


class CohortTest(unittest.TestCase):
    def _spread(self, n=100, ret5d=0.20):
        # si_ratio 0.01 .. 1.00, ascending
        return [_r(0.01 * (i + 1), ret5d) for i in range(n)]

    def test_the_top_five_percent_with_momentum_are_treated(self):
        got = cohort.label(self._spread())
        self.assertEqual(got[-1], "treated")
        self.assertEqual(got[-5], "treated")

    def test_the_sixth_percentile_from_the_top_is_not_treated(self):
        got = cohort.label(self._spread())
        self.assertNotEqual(got[-6], "treated")

    def test_the_bottom_half_are_controls(self):
        got = cohort.label(self._spread())
        self.assertEqual(got[0], "control")
        self.assertEqual(got[49], "control")

    def test_the_middle_band_is_excluded_from_both_arms(self):
        got = cohort.label(self._spread())
        self.assertIsNone(got[60])
        self.assertIsNone(got[80])

    def test_high_short_interest_without_momentum_is_not_treated(self):
        got = cohort.label(self._spread(ret5d=0.02))
        self.assertNotIn("treated", got)

    def test_the_momentum_threshold_is_a_fraction_not_a_percent(self):
        # ret_5d = 0.10 is exactly +10% and must qualify; 10.0 would be +1000%
        rows = self._spread(ret5d=0.10)
        self.assertEqual(cohort.label(rows)[-1], "treated")

    def test_a_missing_five_day_return_is_never_treated_but_may_control(self):
        rows = self._spread()
        rows[-1]["ret_5d"] = None
        rows[0]["ret_5d"] = None
        got = cohort.label(rows)
        self.assertNotEqual(got[-1], "treated")
        self.assertEqual(got[0], "control")

    def test_ranking_is_within_the_batch_not_against_absolute_levels(self):
        # every name lightly shorted: the top of THIS date is still treated
        got = cohort.label([_r(0.01), _r(0.02), _r(0.03)] * 10 + [_r(0.04)])
        self.assertEqual(got[-1], "treated")

    def test_rows_without_short_interest_are_excluded(self):
        got = cohort.label([_r(None), _r(0.5), _r(0.9)])
        self.assertIsNone(got[0])

    def test_an_empty_batch_returns_an_empty_labelling(self):
        self.assertEqual(cohort.label([]), [])
