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


if __name__ == "__main__":
    unittest.main()
