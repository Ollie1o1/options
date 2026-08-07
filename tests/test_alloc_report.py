"""A result is only promotable if it survives deflation, clustering and BROAD.

Every threshold here exists because something got past a weaker version of it.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_report -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.alloc.engine import Trade
from src.alloc.fills import Leg
from src.alloc.report import (clustered_tstat, promotion_verdict, summarise)

LEGS = [Leg("2024-03-15", 100.0, "put", "sell"),
        Leg("2024-03-15", 95.0, "put", "buy")]


def _t(entry, pnl, car=400.0, stratum="broad", exit_="2024-03-15",
       reason="expiry"):
    return Trade(symbol="AAA", entry_date=entry, entry_price=1.0,
                 capital_at_risk=car, legs=LEGS, expiration="2024-03-15",
                 exit_date=exit_, exit_price=-0.5, pnl=pnl,
                 exit_reason=reason, stratum=stratum)


def _res(dsr=0.9, tc=3.5, broad_pnl=50.0, pbo=0.2, n=200):
    return {"n": n, "dsr": dsr, "pbo": pbo, "tstat_clustered": tc,
            "by_stratum": {"legacy": {"pnl": 100.0},
                           "liquid": {"pnl": 80.0},
                           "broad": {"pnl": broad_pnl}},
            "insufficient": False}


class SampleFloorTest(unittest.TestCase):
    """A handful of trades cannot earn a verdict, however good they look.

    Measured the hard way: a starved run of index_put_spread_w25 closed 3 trades
    at 100% wins, reported DSR 0.996 and t=12.56, and was graded `liquid_only`.
    Every number was arithmetically correct and the conclusion was worthless.
    """

    def test_a_tiny_sample_cannot_be_promoted(self):
        self.assertEqual(promotion_verdict(_res(n=3)), "insufficient")

    def test_a_tiny_sample_that_looks_brilliant_still_cannot(self):
        self.assertEqual(
            promotion_verdict(_res(n=3, dsr=0.996, tc=12.56)), "insufficient")

    def test_insufficient_is_not_reject(self):
        """'Not measured' and 'measured and failed' are different claims."""
        self.assertNotEqual(promotion_verdict(_res(n=3)), "reject")

    def test_the_floor_lets_a_real_sample_through(self):
        self.assertEqual(promotion_verdict(_res(n=200)), "promote")

    def test_a_result_without_an_n_is_judged_on_its_statistics(self):
        """Callers that pass no sample size keep the old behaviour."""
        r = _res()
        r.pop("n")
        self.assertEqual(promotion_verdict(r), "promote")


class VerdictTest(unittest.TestCase):
    def test_all_criteria_met_promotes(self):
        self.assertEqual(promotion_verdict(_res()), "promote")

    def test_negative_broad_is_liquid_only_not_promoted(self):
        self.assertEqual(promotion_verdict(_res(broad_pnl=-50.0)),
                         "liquid_only")

    def test_low_dsr_rejects(self):
        self.assertEqual(promotion_verdict(_res(dsr=0.1)), "reject")

    def test_high_pbo_rejects(self):
        self.assertEqual(promotion_verdict(_res(pbo=0.7)), "reject")

    def test_low_clustered_tstat_rejects(self):
        """The naive t may look fine; the clustered one is what counts."""
        self.assertEqual(promotion_verdict(_res(tc=1.9)), "reject")

    def test_a_significantly_NEGATIVE_result_still_rejects(self):
        """|t| >= 3 must not let a strongly losing strategy through."""
        self.assertEqual(promotion_verdict(_res(tc=-5.0)), "reject")

    def test_insufficient_data_rejects(self):
        self.assertEqual(promotion_verdict({"insufficient": True}), "reject")

    def test_the_25_wide_case_that_motivated_deflation(self):
        """DSR 0.921 undeflated, 0.432 deflated — must not promote."""
        self.assertEqual(promotion_verdict(_res(dsr=0.432, tc=0.9)), "reject")


class ClusteredTstatTest(unittest.TestCase):
    def test_same_day_trades_are_pooled(self):
        """Ten trades on one day are one observation, not ten."""
        same = [_t("2024-01-05", 50.0) for _ in range(10)]
        self.assertEqual(clustered_tstat(same), 0.0)

    def test_distinct_days_produce_a_statistic(self):
        spread = [_t(f"2024-01-{i+1:02d}", 40.0 + i) for i in range(10)]
        self.assertNotEqual(clustered_tstat(spread), 0.0)

    def test_clustering_reduces_significance(self):
        """The same P&L concentrated on fewer days is weaker evidence."""
        many_days = [_t(f"2024-01-{i+1:02d}", 50.0 + (i % 3)) for i in range(20)]
        few_days = [_t(f"2024-01-{(i % 3)+1:02d}", 50.0 + (i % 3))
                    for i in range(20)]
        self.assertGreater(abs(clustered_tstat(many_days)),
                           abs(clustered_tstat(few_days)))

    def test_too_few_days_is_zero_not_a_crash(self):
        self.assertEqual(clustered_tstat([_t("2024-01-05", 10.0)]), 0.0)


class SummariseTest(unittest.TestCase):
    def _trades(self, n=40):
        return [_t(f"2024-01-{(i % 28)+1:02d}", 40.0 if i % 4 else -100.0)
                for i in range(n)]

    def test_reports_the_core_fields(self):
        s = summarise(self._trades(), n_trials=10)
        for k in ("n", "win_rate", "mean_return_on_capital", "sharpe",
                  "tstat", "tstat_clustered", "skew", "dsr", "capacity"):
            self.assertIn(k, s)

    def test_trial_count_is_carried_with_the_result(self):
        """A deflated Sharpe is meaningless without the size of the search."""
        self.assertEqual(summarise(self._trades(), n_trials=17)["n_trials"], 17)

    def test_more_trials_lowers_the_deflated_sharpe(self):
        few = summarise(self._trades(), n_trials=1)["dsr"]
        many = summarise(self._trades(), n_trials=5000)["dsr"]
        self.assertGreaterEqual(few, many)

    def test_ticker_ended_trades_are_excluded(self):
        """Forced closes are an artifact of the data ending, not a result."""
        t = self._trades()
        t.append(_t("2024-02-01", -9999.0, reason="ticker_ended"))
        self.assertEqual(summarise(t, 10)["n"], len(self._trades()))

    def test_open_trades_are_excluded(self):
        t = self._trades()
        t.append(_t("2024-02-01", None, exit_=None, reason=None))
        self.assertEqual(summarise(t, 10)["n"], len(self._trades()))

    def test_stratum_split_is_reported(self):
        t = ([_t(f"2024-01-{i+1:02d}", 50.0, stratum="broad") for i in range(5)]
             + [_t(f"2024-02-{i+1:02d}", 50.0, stratum="legacy")
                for i in range(5)])
        s = summarise(t, 10)
        self.assertEqual(set(s["by_stratum"]), {"broad", "legacy"})

    def test_too_few_trades_is_flagged_not_computed(self):
        self.assertTrue(summarise([_t("2024-01-05", 10.0)], 10)["insufficient"])

    def test_default_max_capital_branch(self):
        self.assertIn("capacity", summarise(self._trades(), n_trials=10))


if __name__ == "__main__":
    unittest.main()
