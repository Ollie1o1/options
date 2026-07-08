import unittest
import numpy as np
import pandas as pd
from src.leverage.candidates import Trade, TrendCandidate


def _trending_df(n=400, seed=1):
    """A gently up-then-down series so both long and short breakouts occur."""
    rng = np.random.default_rng(seed)
    up = np.cumsum(rng.normal(0.4, 1.0, n // 2)) + 100
    down = up[-1] + np.cumsum(rng.normal(-0.4, 1.0, n - n // 2))
    close = np.concatenate([up, down])
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


class TestTrendCandidate(unittest.TestCase):
    def test_trade_dataclass_fields(self):
        t = Trade("BTC", "long", 0, 1, 100.0, 110.0, 0.09, 1.5, 3, "stop")
        self.assertEqual(t.symbol, "BTC")
        self.assertEqual(t.side, "long")

    def test_walk_forward_returns_two_trade_lists(self):
        frames = {"BTC": _trending_df()}
        funding = {"BTC": pd.Series(0.0, index=frames["BTC"].index)}
        costs = {"BTC": 0.0013}
        is_t, oos_t = TrendCandidate().walk_forward(frames, funding, costs)
        self.assertIsInstance(is_t, list)
        self.assertIsInstance(oos_t, list)
        self.assertTrue(all(isinstance(x, Trade) for x in is_t + oos_t))
        # some breakouts should have fired on a trending series
        self.assertGreater(len(is_t) + len(oos_t), 0)

    def test_higher_cost_lowers_net_return(self):
        frames = {"BTC": _trending_df()}
        funding = {"BTC": pd.Series(0.0, index=frames["BTC"].index)}
        _, oos_lo = TrendCandidate().walk_forward(frames, funding, {"BTC": 0.0})
        _, oos_hi = TrendCandidate().walk_forward(frames, funding, {"BTC": 0.02})
        if oos_lo and oos_hi:
            self.assertGreater(sum(t.ret_net for t in oos_lo),
                               sum(t.ret_net for t in oos_hi))


if __name__ == "__main__":
    unittest.main()
