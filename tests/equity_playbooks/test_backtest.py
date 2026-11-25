"""Unit tests for src/equity_playbooks (indicators + clustered backtest engine)."""
from __future__ import annotations

import unittest
import numpy as np

from src.equity_playbooks.indicators import (
    bad_jump_mask,
    mom_12_1_series,
    sma,
    trailing_median_dollar_vol,
    trailing_rv,
    wilder_rsi,
)
from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    cluster_bootstrap_stats,
    iso_week_key,
    moving_block_bootstrap_stats,
    simulate_capacity_book,
)
from src.alloc.report import summarise


class TestIndicators(unittest.TestCase):
    def test_sma_causal(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        res = sma(arr, 3)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        np.testing.assert_allclose(res[2:], [20.0, 30.0, 40.0])

    def test_wilder_rsi_extremes(self):
        up = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        dn = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
        self.assertAlmostEqual(wilder_rsi(up, 2)[-1], 100.0)
        self.assertAlmostEqual(wilder_rsi(dn, 2)[-1], 0.0)

    def test_wilder_rsi_does_not_look_ahead(self):
        base = np.array([100.0, 102.0, 101.0, 98.0, 95.0, 99.0, 105.0])
        r_full = wilder_rsi(base, 2)
        r_trunc = wilder_rsi(base[:5], 2)
        np.testing.assert_allclose(r_full[:5], r_trunc, equal_nan=True)

    def test_mom_12_1_exact(self):
        arr = np.linspace(100.0, 200.0, 260)
        m = mom_12_1_series(arr, lookback=252, skip=21)
        self.assertTrue(np.isnan(m[251]))
        expected = arr[252 - 21] / arr[0] - 1.0
        self.assertAlmostEqual(m[252], expected, places=10)

    def test_trailing_median_dollar_vol(self):
        c = np.array([10.0, 10.0, 10.0, 20.0, 20.0])
        v = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        dv = trailing_median_dollar_vol(c, v, window=3)
        self.assertTrue(np.isnan(dv[1]))
        self.assertAlmostEqual(dv[2], 2000.0)
        self.assertAlmostEqual(dv[3], 3000.0)

    def test_bad_jump_mask(self):
        c = np.array([100.0, 105.0, 40.0, 41.0])
        m = bad_jump_mask(c, max_abs_daily_ret=0.45)
        self.assertEqual(m.tolist(), [False, False, True, False])

    def test_trailing_rv_positive(self):
        rng = np.random.default_rng(0)
        c = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=50)))
        rv = trailing_rv(c, window=20)
        self.assertTrue(np.isnan(rv[19]))
        self.assertGreater(rv[20], 0.05)


class TestBacktestPrimitives(unittest.TestCase):
    def _make_trades(self, n_weeks: int = 45, mean_r: float = 0.02):
        trades = []
        import datetime as dt
        base = dt.date(2023, 1, 2)
        for i in range(n_weeks):
            ed = base + dt.timedelta(days=7 * i)
            xd = ed + dt.timedelta(days=5)
            r = (mean_r + 0.02) if (i % 3 != 0) else -0.01
            trades.append(EquityTrade(
                ticker=f"T{i % 5}",
                signal_date=(ed - dt.timedelta(days=1)).isoformat(),
                entry_date=ed.isoformat(),
                exit_date=xd.isoformat(),
                entry_price=100.0,
                exit_price=100.0 * (1.0 + r),
                net_return=r,
                bench_return=0.002,
                exit_reason="target",
            ))
        return trades

    def test_cluster_and_moving_block_bootstrap_positive(self):
        trades = self._make_trades(45, mean_r=0.03)
        c_res = cluster_bootstrap_stats(trades, lambda t: iso_week_key(t.entry_date), n_boot=1000, min_clusters=30)
        self.assertFalse(c_res["insufficient"])
        self.assertGreater(c_res["mean_ret_ci"][0], 0.0)
        self.assertGreater(c_res["pf_ci"][0], 1.0)
        self.assertGreater(c_res["mean_excess_ci"][0], 0.0)

        mb_res = moving_block_bootstrap_stats(trades, lambda t: iso_week_key(t.entry_date), block_len=3, n_boot=1000, min_clusters=30)
        self.assertFalse(mb_res["insufficient"])
        self.assertGreater(mb_res["mean_ret_ci"][0], 0.0)

    def test_alloc_report_compatibility(self):
        trades = self._make_trades(30, mean_r=0.02)
        summary = summarise(trades, n_trials=1)
        self.assertFalse(summary["insufficient"])
        self.assertEqual(summary["n"], 30)
        self.assertGreater(summary["tstat_clustered"], 0.0)

    def test_benchmark_asof_lookup(self):
        b = BenchmarkSeries(["2024-01-02", "2024-01-03", "2024-01-05"], [100.0, 102.0, 105.0])
        self.assertAlmostEqual(b.window_return("2024-01-02", "2024-01-04"), 0.02)
        self.assertAlmostEqual(b.window_return("2024-01-02", "2024-01-05"), 0.05)

    def test_capacity_book_respects_max_concurrent(self):
        trades = self._make_trades(20, mean_r=0.02)
        book = simulate_capacity_book(trades, max_concurrent=2, slot_weight=0.50)
        self.assertGreater(book["n_taken"], 0)
        self.assertGreater(book["total_ret"], 0.0)


if __name__ == "__main__":
    unittest.main()
