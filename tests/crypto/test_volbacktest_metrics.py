import unittest

from src.crypto.volbacktest.metrics import (
    summarize, block_bootstrap_ci, max_drawdown, cvar, newey_west_tstat,
)


class TestMetrics(unittest.TestCase):
    def test_max_drawdown_simple(self):
        eq = [0, 100, 50, 150, 50]
        self.assertAlmostEqual(max_drawdown(eq), 100.0)

    def test_cvar_is_mean_of_worst_tail(self):
        pnl = [-10, -8, -6, 1, 2, 3, 4, 5, 6, 7]
        self.assertAlmostEqual(cvar(pnl, q=0.2), -9.0)

    def test_summarize_basic_stats(self):
        pnl = [100.0, -50.0, 100.0, -50.0]
        s = summarize(pnl)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean"], 25.0)
        self.assertAlmostEqual(s["hit_rate"], 0.5)
        self.assertAlmostEqual(s["profit_factor"], 2.0)

    def test_bootstrap_ci_brackets_mean(self):
        pnl = [10.0] * 50 + [-1.0] * 50
        lo, hi = block_bootstrap_ci(pnl, block=5, iters=500, seed=1)
        self.assertLess(lo, 4.5)
        self.assertGreater(hi, 4.5)

    def test_newey_west_tstat_positive_for_clear_edge(self):
        pnl = [5.0, 4.0, 6.0, 5.0, 4.0, 6.0] * 10
        self.assertGreater(newey_west_tstat(pnl, lag=2), 3.0)

    def test_empty_summarize(self):
        self.assertEqual(summarize([]), {"n": 0})


if __name__ == "__main__":
    unittest.main()
