import unittest
import pandas as pd
from src.leverage.signals import Params
from src.leverage.backtest import walk_forward_windows, robustness_params


class TestWalkForward(unittest.TestCase):
    def test_windows_roll_train_test(self):
        # ~12 months of 5m bars
        idx = pd.date_range("2025-01-01", periods=12 * 30 * 24 * 12, freq="5min",
                            tz="UTC")
        wins = walk_forward_windows(idx, train_months=6, test_months=2)
        self.assertGreater(len(wins), 0)
        for tr_start, tr_end, te_start, te_end in wins:
            self.assertLess(tr_end, te_end)
            self.assertLessEqual(tr_end, te_start)

    def test_robustness_params_perturbs_all(self):
        variants = robustness_params(Params(), pct=0.20)
        self.assertGreater(len(variants), 6)   # baseline + 2 per numeric param
        self.assertIn(Params(), variants)


if __name__ == "__main__":
    unittest.main()
