import unittest
import pandas as pd
from src.leverage.signals import Signal
from src.leverage.backtest import simulate_trade


class TestSimulateTrade(unittest.TestCase):
    def _path(self, prices):
        idx = pd.date_range("2026-05-01 13:35", periods=len(prices), freq="5min",
                            tz="UTC")
        return pd.DataFrame({"open": prices, "high": [p + 1 for p in prices],
                             "low": [p - 1 for p in prices], "close": prices,
                             "volume": 100.0}, index=idx)

    def _sig(self):
        return Signal("BTCUSDT", "long", pd.Timestamp("2026-05-01 13:35", tz="UTC"),
                      entry=100, atr=10, stop=88, target=122, trail_trigger=115,
                      session="us-open", confidence=0.5)

    def test_long_hits_target(self):
        df = self._path([100, 105, 130])  # bar 3 high=131 >= target 122
        res = simulate_trade(self._sig(), df, funding=None, taker=0.0, slippage=0.0)
        self.assertGreater(res["pnl_pct"], 0)
        self.assertEqual(res["exit_reason"], "target")

    def test_long_stop_wins_when_bar_touches_both(self):
        idx = pd.date_range("2026-05-01 13:35", periods=2, freq="5min", tz="UTC")
        df = pd.DataFrame({"open": [100, 105], "high": [101, 125],
                           "low": [99, 80], "close": [100, 105], "volume": 100.0},
                          index=idx)
        res = simulate_trade(self._sig(), df, funding=None, taker=0.0, slippage=0.0)
        self.assertEqual(res["exit_reason"], "stop")  # worst-case ordering
        self.assertLess(res["pnl_pct"], 0)

    def test_costs_reduce_pnl(self):
        df = self._path([100, 105, 130])
        gross = simulate_trade(self._sig(), df, funding=None, taker=0.0,
                               slippage=0.0)["pnl_pct"]
        net = simulate_trade(self._sig(), df, funding=None, taker=0.0006,
                             slippage=0.0002)["pnl_pct"]
        self.assertLess(net, gross)


if __name__ == "__main__":
    unittest.main()
