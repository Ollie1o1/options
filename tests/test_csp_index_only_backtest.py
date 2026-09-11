"""Tests for scripts/csp_index_only_backtest.py's pure helpers.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_csp_index_only_backtest -v
"""
import importlib.util
import os
import sys
import unittest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "csp_index_only_backtest.py")
_SPEC = importlib.util.spec_from_file_location("csp_index_only_backtest", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["csp_index_only_backtest"] = mod
_SPEC.loader.exec_module(mod)


class IvRankSeriesTest(unittest.TestCase):
    def test_refuses_to_rank_without_enough_trailing_history(self):
        # Fewer than IV_RANK_MIN_HISTORY prior observations: None, not a
        # guess — the same NULL-is-not-zero convention as everywhere else.
        hist = {f"2024-01-{d:02d}": 0.20 for d in range(1, 10)}
        ranks = mod._iv_rank_series(hist)
        self.assertIsNone(ranks["2024-01-09"])

    def test_a_days_own_value_never_enters_its_own_window(self):
        # A flat 30-day trailing history has hi==lo regardless of what today
        # does — a spike to 0.50 must NOT widen the window to [0.20, 0.50]
        # and rank itself as a trivial 100 by having contaminated its own
        # reference range. It must fall through the same zero-variance guard
        # data_fetching.get_iv_rank_percentile_from_history uses (a flat
        # window carries no information to rank against, so the neutral 50
        # is returned) — a window that saw today's value would instead show
        # hi=0.50 != lo=0.20 and take the other branch entirely.
        hist = {f"day{i:03d}": 0.20 for i in range(30)}
        hist["day030"] = 0.50
        ranks = mod._iv_rank_series(hist)
        self.assertEqual(ranks["day030"], 50.0)

    def test_rank_is_zero_at_the_trailing_minimum(self):
        hist = {f"day{i:03d}": 0.01 * (i + 1) for i in range(30)}  # rising
        hist["day030"] = 0.0001   # below everything seen so far
        ranks = mod._iv_rank_series(hist)
        self.assertEqual(ranks["day030"], 0.0)

    def test_window_is_only_the_trailing_N_observations(self):
        # A value far outside a stale, long-ago range must not still anchor
        # today's rank once it has scrolled out of the trailing window.
        hist = {}
        for i in range(60):
            hist[f"2024-01-{i:03d}" if i >= 100 else f"day{i:03d}"] = 0.10
        # Keys must sort in date order for this synthetic test; use zero-padded
        # sortable keys directly instead.
        hist = {f"day{i:03d}": 0.10 for i in range(60)}
        hist["day000"] = 0.90   # an old spike, 59 observations back
        hist["day060"] = 0.10   # today, flat like the recent window
        ranks = mod._iv_rank_series(hist)
        # IV_RANK_WINDOW=52: the old spike at day000 is outside the trailing
        # 52-observation window for day060 (observations day008..day059), so
        # today's flat 0.10 against an all-flat recent window reads 50, not
        # pulled down by the stale outlier.
        self.assertEqual(ranks["day060"], 50.0)


class EntryFilterTest(unittest.TestCase):
    def test_selected_filter_refuses_outside_dte_window(self):
        f = mod._make_filter({"2024-01-01": 80.0}, require_iv_rank=True)
        self.assertFalse(f({"date": "2024-01-01", "dte": 24}))
        self.assertFalse(f({"date": "2024-01-01", "dte": 61}))
        self.assertTrue(f({"date": "2024-01-01", "dte": 25}))
        self.assertTrue(f({"date": "2024-01-01", "dte": 60}))

    def test_selected_filter_refuses_below_iv_rank_floor(self):
        f = mod._make_filter({"2024-01-01": 49.9}, require_iv_rank=True)
        self.assertFalse(f({"date": "2024-01-01", "dte": 30}))

    def test_selected_filter_refuses_unrankable_date_rather_than_admitting(self):
        f = mod._make_filter({"2024-01-01": None}, require_iv_rank=True)
        self.assertFalse(f({"date": "2024-01-01", "dte": 30}))
        f2 = mod._make_filter({}, require_iv_rank=True)
        self.assertFalse(f2({"date": "2024-01-01", "dte": 30}))

    def test_control_filter_ignores_iv_rank_entirely(self):
        f = mod._make_filter({}, require_iv_rank=False)
        self.assertTrue(f({"date": "2024-01-01", "dte": 30}))


class MeasureTest(unittest.TestCase):
    def test_empty_trades_is_n_zero(self):
        self.assertEqual(mod._measure([]), {"n": 0})

    def test_reports_the_shape_a_landed_result_needs(self):
        trades = [
            {"ret": 0.05, "entry_date": "2024-01-05", "exit_date": "2024-02-01"},
            {"ret": -0.03, "entry_date": "2024-02-02", "exit_date": "2024-03-01"},
            {"ret": 0.04, "entry_date": "2024-03-02", "exit_date": "2024-04-01"},
        ]
        out = mod._measure(trades)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_trials"], 1)
        self.assertAlmostEqual(out["win_rate"], 200 / 3, places=2)
        self.assertIn("dsr", out)
        self.assertIn("sharpe", out)
        self.assertIn("tstat_clustered", out)


class ClusteredTstatTest(unittest.TestCase):
    def test_same_day_entries_collapse_to_one_cluster(self):
        # Four trades on ONE entry day carry one day's worth of independent
        # information, not four — collapsing to day-means must produce a
        # single cluster, which (< 3 clusters) reads as 0.0, not a real t.
        trades = [{"ret": r, "entry_date": "2024-01-05"} for r in
                 (0.05, 0.04, 0.06, 0.05)]
        self.assertEqual(mod._clustered_tstat(trades), 0.0)

    def test_matches_report_pys_clustered_tstat_on_priced_trades(self):
        # Same day-mean t-statistic src.alloc.report.clustered_tstat computes,
        # just fed from this script's plain ret/entry_date dicts instead of
        # trade objects with pnl/capital_at_risk — same arithmetic, different
        # input shape, so the two numbers must agree.
        from src.alloc.report import clustered_tstat as _report_clustered_tstat

        class _T:
            def __init__(self, ret, entry_date):
                self.pnl = ret
                self.capital_at_risk = 1.0
                self.entry_date = entry_date

        rets_by_day = [("2024-01-05", 0.05), ("2024-01-05", 0.03),
                      ("2024-01-12", -0.02), ("2024-01-19", 0.04),
                      ("2024-01-19", 0.06), ("2024-01-26", -0.01)]
        trades = [{"ret": r, "entry_date": d} for d, r in rets_by_day]
        objs = [_T(r, d) for d, r in rets_by_day]
        self.assertAlmostEqual(mod._clustered_tstat(trades),
                               _report_clustered_tstat(objs), places=6)


class VerdictTest(unittest.TestCase):
    def test_below_min_n_is_insufficient_regardless_of_the_statistics(self):
        # The exact incident report.py's MIN_N guards against: a thin sample
        # with a flattering DSR must never read as promoted or even rejected.
        stats = {"n": 19, "dsr": 0.99, "tstat_clustered": 12.0}
        self.assertEqual(mod._verdict(stats), "insufficient")

    def test_weak_dsr_at_full_n_is_reject_not_insufficient(self):
        stats = {"n": 25, "dsr": 0.3, "tstat_clustered": 1.0}
        self.assertEqual(mod._verdict(stats), "reject")

    def test_negative_tstat_is_reject_even_with_high_dsr(self):
        stats = {"n": 25, "dsr": 0.8, "tstat_clustered": -4.0}
        self.assertEqual(mod._verdict(stats), "reject")

    def test_clears_every_bar_is_promote(self):
        stats = {"n": 25, "dsr": 0.8, "tstat_clustered": 4.0}
        self.assertEqual(mod._verdict(stats), "promote")


if __name__ == "__main__":
    unittest.main()
