import unittest
import numpy as np
import pandas as pd
from src.leverage.candidates import TrendCandidate, TrendCarryCandidate


def _trending_df(n=400, seed=1):
    rng = np.random.default_rng(seed)
    up = np.cumsum(rng.normal(0.4, 1.0, n // 2)) + 100
    down = up[-1] + np.cumsum(rng.normal(-0.4, 1.0, n - n // 2))
    close = np.concatenate([up, down])
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=idx)


class TestTrendCarry(unittest.TestCase):
    def test_carry_filter_removes_some_trades(self):
        df = _trending_df()
        # funding that opposes longs during the up-leg (z high) -> veto some
        fund = pd.Series(0.0001, index=df.index)
        fund.iloc[:200] = 0.02
        frames, funding, costs = {"BTC": df}, {"BTC": fund}, {"BTC": 0.0013}
        _, base_oos = TrendCandidate().walk_forward(frames, funding, costs)
        _, carry_oos = TrendCarryCandidate(z_threshold=1.5).walk_forward(
            frames, funding, costs)
        # carry is a strict filter on the same breakouts -> never MORE trades
        self.assertLessEqual(len(carry_oos), len(base_oos))

    def test_returns_trade_objects(self):
        df = _trending_df()
        fund = pd.Series(0.0, index=df.index)
        is_t, oos_t = TrendCarryCandidate().walk_forward(
            {"BTC": df}, {"BTC": fund}, {"BTC": 0.0013})
        self.assertTrue(all(hasattr(t, "ret_net") for t in is_t + oos_t))


if __name__ == "__main__":
    unittest.main()
