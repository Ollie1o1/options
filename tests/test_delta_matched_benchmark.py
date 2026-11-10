"""PF and the cluster bootstrap behind docs/DECISION_RULE_20261015.md §2.

No ledger, no network: synthetic frames only.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_delta_matched_benchmark -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.delta_matched_benchmark import cluster_boot, pf, pf_verdict


class PFTest(unittest.TestCase):
    def test_pf_is_gains_over_losses(self):
        self.assertAlmostEqual(pf(np.array([0.3, -0.1, 0.1, -0.2])), 0.4 / 0.3)

    def test_pf_with_no_losses_is_infinite(self):
        self.assertEqual(pf(np.array([0.1, 0.2])), float("inf"))

    def test_pf_with_no_gains_is_zero(self):
        self.assertEqual(pf(np.array([-0.1, -0.2])), 0.0)

    def test_pf_ignores_scale(self):
        v = np.array([0.5, -0.25, 0.1])
        self.assertAlmostEqual(pf(v), pf(10 * v))

    def test_pf_verdict_needs_the_lower_bound_above_one(self):
        self.assertEqual(pf_verdict({"lo": 1.01, "hi": 2.0}), "profitable")
        self.assertEqual(pf_verdict({"lo": 0.99, "hi": 2.0}), "inconclusive")
        self.assertEqual(pf_verdict({"lo": 0.5, "hi": 0.9}), "unprofitable")


class ClusterBootTest(unittest.TestCase):
    def _frame(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame({"x": rng.normal(0.01, 0.1, 60),
                             "week": np.repeat([f"W{i}" for i in range(10)], 6)})

    def test_default_statistic_is_the_mean(self):
        df = self._frame()
        a = cluster_boot(df, "x", "week")
        b = cluster_boot(df, "x", "week", stat=np.mean)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["est"], df.x.mean())

    def test_a_custom_statistic_is_used_for_estimate_and_interval(self):
        df = self._frame()
        r = cluster_boot(df, "x", "week", stat=pf)
        self.assertAlmostEqual(r["est"], pf(df.x.values))
        self.assertLessEqual(r["lo"], r["est"])
        self.assertGreaterEqual(r["hi"], r["est"])
        self.assertEqual(r["k"], 10)


if __name__ == "__main__":
    unittest.main()


class SpotAtExitFreshnessTest(unittest.TestCase):
    """A price cache that ends before the exit must not price the exit.

    `fetch_prices` reuses `px_1d.pkl` / `px_1h.pkl` whenever they exist. Run in
    January against a cache directory left from October, `spot_at_exit` took
    the last bar on or before the exit — October's — and returned it as the
    exit spot with no warning.
    """

    def _px(self, days):
        idx = pd.DatetimeIndex(days, tz="America/New_York")
        return pd.DataFrame({"SPY": np.linspace(500.0, 510.0, len(idx))}, index=idx)

    def test_a_stale_cache_yields_no_spot(self):
        from scripts.delta_matched_benchmark import spot_at_exit
        px1d = self._px(["2026-10-12", "2026-10-13", "2026-10-14"])
        px1h = self._px(["2026-10-14 15:30"])
        self.assertIsNone(spot_at_exit(px1h, px1d, "SPY", "2027-01-08 10:20:00"))

    def test_the_last_session_before_a_holiday_exit_is_still_a_spot(self):
        from scripts.delta_matched_benchmark import spot_at_exit
        px1d = self._px(["2026-10-12", "2026-10-13", "2026-10-14"])
        px1h = self._px(["2026-10-14 15:30"])
        self.assertAlmostEqual(
            spot_at_exit(px1h, px1d, "SPY", "2026-10-16 10:20:00"), 510.0)

    def test_an_hourly_bar_on_the_exit_day_wins(self):
        from scripts.delta_matched_benchmark import spot_at_exit
        px1d = self._px(["2026-10-13", "2026-10-14"])
        px1h = self._px(["2026-10-14 09:30", "2026-10-14 10:30"])
        self.assertAlmostEqual(
            spot_at_exit(px1h, px1d, "SPY", "2026-10-14 10:20:00"), 500.0)
