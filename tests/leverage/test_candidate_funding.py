import unittest
import numpy as np
import pandas as pd
from src.leverage.candidates import Trade, FundingContrarianCandidate


def _flat_df(n=200, price=100.0, seed=2):
    rng = np.random.default_rng(seed)
    close = price + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1.0}, index=idx)


class TestFundingContrarian(unittest.TestCase):
    def test_short_on_crowded_long_funding_spike(self):
        df = _flat_df()
        fund = pd.Series(0.0001, index=df.index)
        fund.iloc[40:60] = 0.02      # sustained crowded-long funding
        cand = FundingContrarianCandidate(z_threshold=1.5, horizon=5)
        is_t, oos_t = cand.walk_forward({"BTC": df}, {"BTC": fund}, {"BTC": 0.0})
        trades = is_t + oos_t
        self.assertTrue(any(t.side == "short" for t in trades))

    def test_no_trades_when_funding_flat(self):
        df = _flat_df()
        fund = pd.Series(0.0001, index=df.index)
        cand = FundingContrarianCandidate()
        is_t, oos_t = cand.walk_forward({"BTC": df}, {"BTC": fund}, {"BTC": 0.0})
        self.assertEqual(len(is_t) + len(oos_t), 0)

    def test_cost_reduces_net_return(self):
        df = _flat_df()
        fund = pd.Series(0.0001, index=df.index)
        fund.iloc[40:60] = 0.02
        cand = FundingContrarianCandidate()
        _, oos0 = cand.walk_forward({"BTC": df}, {"BTC": fund}, {"BTC": 0.0})
        _, oosc = cand.walk_forward({"BTC": df}, {"BTC": fund}, {"BTC": 0.01})
        if oos0 and oosc:
            self.assertGreater(sum(t.ret_net for t in oos0),
                               sum(t.ret_net for t in oosc))


if __name__ == "__main__":
    unittest.main()
