import unittest
import math
from src.leverage.risk import DailyLossLimit, expected_worst_streak


class TestDailyLossLimit(unittest.TestCase):
    def test_allows_until_limit(self):
        lim = DailyLossLimit(equity=1500, max_daily_loss_frac=0.06)  # $90
        lim.record_pnl(-30)
        self.assertTrue(lim.can_trade())
        lim.record_pnl(-30)
        self.assertTrue(lim.can_trade())

    def test_blocks_after_limit_breached(self):
        lim = DailyLossLimit(equity=1500, max_daily_loss_frac=0.06)
        lim.record_pnl(-50)
        lim.record_pnl(-50)  # -100 < -90
        self.assertFalse(lim.can_trade())

    def test_reset_clears(self):
        lim = DailyLossLimit(equity=1500, max_daily_loss_frac=0.06)
        lim.record_pnl(-100)
        lim.reset()
        self.assertTrue(lim.can_trade())


class TestRuinMonitor(unittest.TestCase):
    def test_expected_worst_streak(self):
        s = expected_worst_streak(n_trades=200, p_win=0.42)
        self.assertAlmostEqual(s, math.log(200) / math.log(1 / 0.58), places=4)
        self.assertGreater(s, 9.0)
        self.assertLess(s, 11.0)


if __name__ == "__main__":
    unittest.main()
