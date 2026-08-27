"""The report refuses without a matching prereg, and states its own limits."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.catalyst.backtest import report
from src.catalyst.backtest.study import Result

fmt._COLOR_ENABLED = False


def a_result(verdict="NO EVIDENCE", key="H1"):
    return Result(key=key, label="funded through", n_true=40, n_false=22,
                  mean_true=0.031, mean_false=-0.012, diff=0.043,
                  ci_lo=-0.02, ci_hi=0.11, verdict=verdict)


class TestClusterCountsAreShown(unittest.TestCase):
    """A row count that hides the cluster count is how this went wrong.

    "n = 2137" reads as 2,137 observations. When those rows come from 659
    clusters the honest sample size is 659, and a reader who is shown only the
    row count cannot tell the difference.
    """

    def _clustered(self):
        return Result(key="H1", label="funded through", n_true=180,
                      n_false=96, mean_true=0.031, mean_false=-0.012,
                      diff=0.043, ci_lo=-0.09, ci_hi=0.18,
                      verdict="NO EVIDENCE", k_true=41, k_false=23)

    def test_the_cluster_count_appears_beside_the_row_count(self):
        out = report.render([self._clustered()], horizon_counts={6: 12},
                            dropped_delisted=12, prereg_ok=True)
        self.assertIn("41", out)
        self.assertIn("23", out)
        self.assertIn("ticker", out.lower())

    def test_a_result_with_no_cluster_count_does_not_invent_one(self):
        # k=0 means "not measured". Printing "0 tickers" would be a claim.
        out = report.render([a_result()], horizon_counts={6: 12},
                            dropped_delisted=12, prereg_ok=True)
        self.assertNotIn("0 tickers", out)

    def test_it_says_the_ci_is_cluster_robust(self):
        out = report.render([self._clustered()], horizon_counts={6: 12},
                            dropped_delisted=12, prereg_ok=True)
        self.assertIn("cluster", out.lower())


class TestRefusal(unittest.TestCase):
    def test_refuses_without_a_matching_prereg(self):
        out = report.render([a_result()], horizon_counts={6: 12},
                            dropped_delisted=12, prereg_ok=False)
        self.assertIn("REFUS", out.upper())

    def test_refusal_shows_no_numbers(self):
        out = report.render([a_result()], horizon_counts={6: 12},
                            dropped_delisted=12, prereg_ok=False)
        self.assertNotIn("0.043", out)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.out = report.render([a_result()], horizon_counts={6: 12},
                                 dropped_delisted=12, prereg_ok=True)

    def test_states_the_verdict(self):
        self.assertIn("NO EVIDENCE", self.out)

    def test_shows_both_arm_sizes(self):
        self.assertIn("40", self.out)
        self.assertIn("22", self.out)

    def test_prints_the_vintage_count_per_horizon(self):
        self.assertIn("12", self.out)

    def test_states_the_survivorship_drop(self):
        self.assertIn("delisted", self.out.lower())

    def test_states_the_market_cap_compromise(self):
        self.assertIn("market cap", self.out.lower())

    def test_states_that_vintages_are_not_independent(self):
        self.assertIn("independent", self.out.lower())

    def test_labels_exploratory_results(self):
        out = report.render([a_result(key="H2")], horizon_counts={6: 12},
                            dropped_delisted=0, prereg_ok=True)
        self.assertIn("EXPLORATORY", out.upper())

    def test_primary_is_not_labelled_exploratory(self):
        line = [l for l in self.out.splitlines() if "H1" in l][0]
        self.assertNotIn("EXPLORATORY", line.upper())

    def test_underpowered_says_so_instead_of_showing_a_ci(self):
        out = report.render([a_result(verdict="UNDERPOWERED")],
                            horizon_counts={6: 12}, dropped_delisted=0,
                            prereg_ok=True)
        self.assertIn("UNDERPOWERED", out)
        self.assertNotIn("95% CI", out)


class TestNotComputableRendering(unittest.TestCase):
    def test_states_not_computable_and_shows_no_ci(self):
        r = Result(key="H4", label="implied vs realised — no historical chains",
                   n_true=0, n_false=0, mean_true=0.0, mean_false=0.0,
                   diff=0.0, ci_lo=0.0, ci_hi=0.0, verdict="NOT COMPUTABLE")
        out = report.render([r], horizon_counts={6: 12}, dropped_delisted=0,
                            prereg_ok=True)
        self.assertIn("NOT COMPUTABLE", out)
        self.assertNotIn("95% CI", out)
        self.assertIn("no historical chains", out)


class TestMain(unittest.TestCase):
    def test_returns_two_when_prereg_is_missing(self):
        from src.catalyst.backtest import __main__ as cli
        with tempfile.TemporaryDirectory() as d:
            rc = cli.main(["--prereg", os.path.join(d, "nope.md"),
                           "--db", os.path.join(d, "pit.db")])
        self.assertEqual(rc, 2)

class TestTheHorizonCounterIsLabelledHonestly(unittest.TestCase):
    """It counts OBSERVATIONS, and printed them as "vintages".

    There are 12 vintages. The 2026-08-27 run printed "3mo: 2103 vintages",
    which is the observation count wearing the wrong noun — the same defect
    shape this repo keeps paying for.
    """

    def test_it_does_not_call_observation_counts_vintages(self):
        out = report.render([a_result()], horizon_counts={6: 2103},
                            dropped_delisted=0, prereg_ok=True)
        self.assertIn("2103", out)
        self.assertNotIn("2103 vintages", out)

    def test_it_names_them_observations(self):
        out = report.render([a_result()], horizon_counts={6: 2103},
                            dropped_delisted=0, prereg_ok=True)
        self.assertIn("observation", out.lower())

if __name__ == "__main__":
    unittest.main()
