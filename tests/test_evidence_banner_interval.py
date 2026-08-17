"""The banner must not report a point estimate as if it were the answer.

`format_evidence_banner` printed `OOS IC -0.12 (p=0.11, n=232)` and stopped.
The walk-forward artifact behind it also carries:

    fold_ic_mean      +0.0665
    fold_ic_ci_95     [-0.0989, +0.2389]
    folds_ic_positive 11 of 18

so the interval CONTAINS ZERO and the fold-level estimate has the OPPOSITE
SIGN to the pooled one. A reader sees "-0.12" and reasonably concludes the
ranking model is mildly anti-predictive. The evidence says it is not
distinguishable from zero, which is a different claim and the one the data
supports.

This is the same defect this repo has been clearing all of 2026-08-17: a
displayed number implying more than its evidence carries. It matters more here
than anywhere, because this is the line that tells the operator how much to
trust every other number on the board.

The "not distinguishable" verdict is read off the CI containing zero, not from
a threshold anybody chose.
"""
from __future__ import annotations

import unittest
from datetime import date

from src.evidence import format_evidence_banner


def _ev(**kw):
    base = {
        "pooled_ic": -0.1189, "p_value": 0.1119, "n_oos": 232,
        "cohort_n": 120, "gate_decision": "STOP",
        "wf_as_of": "2026-08-17T16:28:07",
        "fold_ic_mean": 0.0665, "fold_ic_ci_95": [-0.0989, 0.2389],
        "folds_ic_positive": 11, "n_folds": 18,
        "cohort_ic_pearson": -0.0643, "cohort_ic_spearman": -0.1335,
    }
    base.update(kw)
    return base


TODAY = date(2026, 8, 17)


class TestTheIntervalReachesTheReader(unittest.TestCase):

    def test_the_confidence_interval_is_shown(self):
        out = format_evidence_banner(_ev(), today=TODAY)
        self.assertIn("-0.10", out)
        self.assertIn("0.24", out)

    def test_a_ci_containing_zero_says_so_in_words(self):
        out = format_evidence_banner(_ev(), today=TODAY).lower()
        self.assertIn("zero", out,
                      "the banner reports -0.12 without saying the interval "
                      "contains zero")

    def test_the_fold_count_is_shown(self):
        out = format_evidence_banner(_ev(), today=TODAY)
        self.assertIn("11", out)
        self.assertIn("18", out)

    def test_the_pooled_estimate_is_still_there(self):
        out = format_evidence_banner(_ev(), today=TODAY)
        self.assertIn("-0.12", out)
        self.assertIn("n=232", out)


class TestItDoesNotOverclaimInEitherDirection(unittest.TestCase):

    def test_a_wholly_negative_interval_is_not_called_indistinguishable(self):
        out = format_evidence_banner(
            _ev(fold_ic_mean=-0.20, fold_ic_ci_95=[-0.31, -0.09],
                folds_ic_positive=2), today=TODAY).lower()
        self.assertNotIn("not distinguishable", out)

    def test_a_wholly_positive_interval_is_not_called_indistinguishable(self):
        out = format_evidence_banner(
            _ev(fold_ic_mean=0.20, fold_ic_ci_95=[0.09, 0.31],
                folds_ic_positive=16), today=TODAY).lower()
        self.assertNotIn("not distinguishable", out)

    def test_an_interval_touching_zero_counts_as_containing_it(self):
        out = format_evidence_banner(
            _ev(fold_ic_ci_95=[0.0, 0.31]), today=TODAY).lower()
        self.assertIn("zero", out)


class TestOlderArtifactsStillRender(unittest.TestCase):
    """Reports written before the fold fields existed must not break the
    banner — `reports/` still holds artifacts from 2026-05."""

    def test_missing_fold_fields_degrade_to_the_old_line(self):
        out = format_evidence_banner(
            _ev(fold_ic_mean=None, fold_ic_ci_95=None,
                folds_ic_positive=None, n_folds=None), today=TODAY)
        self.assertIn("-0.12", out)
        self.assertIn("n=232", out)

    def test_a_malformed_interval_is_ignored_not_crashed_on(self):
        for bad in ([], [0.1], "nope", [None, None]):
            out = format_evidence_banner(_ev(fold_ic_ci_95=bad), today=TODAY)
            self.assertIn("-0.12", out)

    def test_no_walk_forward_at_all_still_renders(self):
        out = format_evidence_banner(
            _ev(pooled_ic=None, p_value=None, fold_ic_ci_95=None), today=TODAY)
        self.assertIn("n/a", out)


if __name__ == "__main__":
    unittest.main()
