import unittest

from src.crypto.volbacktest.report import format_report, annualized_sharpe


class TestReport(unittest.TestCase):
    def test_annualized_sharpe(self):
        self.assertAlmostEqual(annualized_sharpe(1.0, 1.0, 52.0), 52 ** 0.5, places=6)
        self.assertEqual(annualized_sharpe(1.0, 0.0, 52.0), 0.0)

    def test_report_contains_key_lines(self):
        stats = {"n": 50, "mean": 12.3, "hit_rate": 0.68, "profit_factor": 1.4,
                 "total": 615.0, "max_drawdown": 200.0, "cvar5": -180.0,
                 "std": 90.0, "worst": -300.0, "best": 400.0}
        txt = format_report("BTC", stats, tstat=2.1, ci=(3.0, 21.0),
                            cost_breakeven_mult=2.8, mark_rmse=1.4,
                            regime={"bull": 5.0, "bear": -3.0, "chop": 8.0})
        for token in ("BTC", "Sharpe", "t-stat", "CVaR", "breakeven", "RMSE", "bull"):
            self.assertIn(token, txt)


if __name__ == "__main__":
    unittest.main()
