"""Tests for src/dolt_research.py — entry-filter factories + harness wiring.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_research -v
"""
import unittest
from unittest import mock

from src import dolt_research as dr


class FilterTest(unittest.TestCase):
    def test_low_vix(self):
        with mock.patch("src.dolt_research.vix_on", return_value=15.0):
            self.assertTrue(dr.low_vix(20.0)({"date": "2024-03-01"}))
        with mock.patch("src.dolt_research.vix_on", return_value=25.0):
            self.assertFalse(dr.low_vix(20.0)({"date": "2024-03-01"}))
        with mock.patch("src.dolt_research.vix_on", return_value=None):
            self.assertFalse(dr.low_vix(20.0)({"date": "2024-03-01"}))

    def test_high_vix(self):
        with mock.patch("src.dolt_research.vix_on", return_value=25.0):
            self.assertTrue(dr.high_vix(22.0)({"date": "x"}))
        with mock.patch("src.dolt_research.vix_on", return_value=20.0):
            self.assertFalse(dr.high_vix(22.0)({"date": "x"}))

    def test_low_iv(self):
        self.assertTrue(dr.low_iv(0.30)({"entry_iv": 0.25}))
        self.assertFalse(dr.low_iv(0.30)({"entry_iv": 0.40}))
        self.assertFalse(dr.low_iv(0.30)({"entry_iv": None}))

    def test_trend_up(self):
        spots = {f"2024-03-{d:02d}": 100.0 + d for d in range(1, 11)}  # rising
        sdates = sorted(spots)
        ctx = {"spots": spots, "sdates": sdates, "date": "2024-03-10", "spot": spots["2024-03-10"]}
        self.assertTrue(dr.trend_up(5)(ctx))   # latest > MA of rising series
        # falling series → below MA
        spots2 = {f"2024-03-{d:02d}": 200.0 - d for d in range(1, 11)}
        ctx2 = {"spots": spots2, "sdates": sorted(spots2), "date": "2024-03-10", "spot": spots2["2024-03-10"]}
        self.assertFalse(dr.trend_up(5)(ctx2))

    def test_combine(self):
        always = lambda ctx: True
        never = lambda ctx: False
        self.assertTrue(dr.combine(always, always)({}))
        self.assertFalse(dr.combine(always, never)({}))

    def test_battery_has_baseline(self):
        b = dr.standard_battery()
        self.assertIn("baseline", b)
        self.assertIsNone(b["baseline"])


class HarnessWiringTest(unittest.TestCase):
    def test_compare_calls_backtest_per_filter(self):
        fake = {"n": 10, "win_rate": 0.4, "avg_return": 0.05,
                "median_return": -0.1, "profit_factor": 1.1}
        with mock.patch("src.dolt_research.run_cohort_backtest", return_value=fake) as m:
            out = dr.compare(["AAPL"], "2024-01-01", "2024-02-01",
                             {"baseline": None, "low_vix_20": dr.low_vix(20.0)})
        self.assertEqual(m.call_count, 2)
        self.assertEqual(out["baseline"]["n"], 10)
        self.assertEqual(out["low_vix_20"]["profit_factor"], 1.1)


class TrainTestStrategyTest(unittest.TestCase):
    def test_strategy_registry_has_three(self):
        self.assertEqual(set(dr.STRATEGIES), {"long_call", "short_put", "put_spread"})

    def test_train_test_default_uses_cohort(self):
        tr = {"n": 50, "win_rate": 0.6, "avg_return": 0.1,
              "median_return": 0.05, "profit_factor": 1.3}
        te = {"n": 25, "win_rate": 0.55, "avg_return": 0.08,
              "median_return": 0.02, "profit_factor": 1.2}
        with mock.patch("src.dolt_research.run_cohort_backtest", side_effect=[tr, te]) as m:
            out = dr.train_test(["SPY"], None)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(out["train"]["profit_factor"], 1.3)
        self.assertEqual(out["test"]["profit_factor"], 1.2)

    def test_train_test_put_spread_routes_to_spread_runner(self):
        tr = {"n": 40, "win_rate": 0.8, "avg_return": 0.2,
              "median_return": 0.3, "profit_factor": 4.0}
        te = {"n": 18, "win_rate": 0.75, "avg_return": 0.15,
              "median_return": 0.2, "profit_factor": 2.5}
        with mock.patch("src.dolt_research.run_spread_backtest", side_effect=[tr, te]) as ms, \
             mock.patch("src.dolt_research.run_cohort_backtest") as mc:
            out = dr.train_test(["SPY"], None, strategy="put_spread")
        self.assertEqual(ms.call_count, 2)
        mc.assert_not_called()
        self.assertEqual(out["train"]["profit_factor"], 4.0)
        self.assertEqual(out["test"]["profit_factor"], 2.5)

    def test_train_test_short_put_routes_to_short_runner(self):
        row = {"n": 30, "win_rate": 0.7, "avg_return": 0.1,
               "median_return": 0.1, "profit_factor": 1.26}
        with mock.patch("src.dolt_research.run_short_backtest", return_value=row) as ms:
            out = dr.train_test(["NVDA"], None, strategy="short_put")
        self.assertEqual(ms.call_count, 2)
        # short_put must be invoked as puts, not calls
        for c in ms.call_args_list:
            self.assertEqual(c.kwargs.get("opt_type"), "put")
        self.assertEqual(out["test"]["profit_factor"], 1.26)


if __name__ == "__main__":
    unittest.main()
