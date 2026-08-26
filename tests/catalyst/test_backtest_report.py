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


class TestRefusal(unittest.TestCase):
    def test_refuses_without_a_matching_prereg(self):
        out = report.render([a_result()], vintage_counts={6: 12},
                            dropped_delisted=12, prereg_ok=False)
        self.assertIn("REFUS", out.upper())

    def test_refusal_shows_no_numbers(self):
        out = report.render([a_result()], vintage_counts={6: 12},
                            dropped_delisted=12, prereg_ok=False)
        self.assertNotIn("0.043", out)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.out = report.render([a_result()], vintage_counts={6: 12},
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
        out = report.render([a_result(key="H2")], vintage_counts={6: 12},
                            dropped_delisted=0, prereg_ok=True)
        self.assertIn("EXPLORATORY", out.upper())

    def test_primary_is_not_labelled_exploratory(self):
        line = [l for l in self.out.splitlines() if "H1" in l][0]
        self.assertNotIn("EXPLORATORY", line.upper())

    def test_underpowered_says_so_instead_of_showing_a_ci(self):
        out = report.render([a_result(verdict="UNDERPOWERED")],
                            vintage_counts={6: 12}, dropped_delisted=0,
                            prereg_ok=True)
        self.assertIn("UNDERPOWERED", out)
        self.assertNotIn("95% CI", out)


class TestMain(unittest.TestCase):
    def test_returns_two_when_prereg_is_missing(self):
        from src.catalyst.backtest import __main__ as cli
        with tempfile.TemporaryDirectory() as d:
            rc = cli.main(["--prereg", os.path.join(d, "nope.md"),
                           "--db", os.path.join(d, "pit.db")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
