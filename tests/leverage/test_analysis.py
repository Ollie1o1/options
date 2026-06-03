import unittest
from src.leverage.backtest import BacktestResult
from src.leverage.analysis import analyze, render_analysis


def _result(trades, sides=None, reasons=None, max_dd=0.0):
    return BacktestResult(trades=list(trades), n=len(trades),
                          sides=list(sides or []),
                          exit_reasons=list(reasons or []), max_dd=max_dd)


class TestAnalysis(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(analyze(_result([]))["n"], 0)
        self.assertIn("no trades", render_analysis(_result([])))

    def test_profit_factor_and_winrate(self):
        a = analyze(_result([0.02, 0.01, -0.01]))
        self.assertAlmostEqual(a["win_rate"], 2 / 3)
        self.assertAlmostEqual(a["profit_factor"], 0.03 / 0.01)
        self.assertGreater(a["expectancy"], 0)

    def test_sharpe_sign_follows_mean(self):
        self.assertGreater(analyze(_result([0.01, 0.02, 0.015]))["sharpe_per_trade"], 0)
        self.assertLess(analyze(_result([-0.01, -0.02, -0.015]))["sharpe_per_trade"], 0)

    def test_per_side_breakdown(self):
        a = analyze(_result([0.02, -0.01, 0.03, -0.02],
                            sides=["long", "long", "short", "short"]))
        self.assertEqual(a["by_side"]["long"]["n"], 2)
        self.assertEqual(a["by_side"]["short"]["n"], 2)

    def test_exit_mix_sums_to_one(self):
        a = analyze(_result([0.01, -0.01, 0.0],
                            reasons=["target", "stop", "time"]))
        self.assertAlmostEqual(sum(a["exit_mix"].values()), 1.0)

    def test_render_contains_verdict(self):
        txt = render_analysis(_result([0.02, 0.01], reasons=["target", "target"]),
                              label="X")
        self.assertIn("EDGE", txt)
        self.assertIn("profit factor", txt)


if __name__ == "__main__":
    unittest.main()
