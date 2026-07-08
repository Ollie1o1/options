import unittest
import numpy as np
import pandas as pd
from src.leverage.xsect import CrossSectionalCandidate
from src.leverage.candidates import Trade


def _df_from_close(close, seed=0):
    n = len(close)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=idx)


class TestCrossSectional(unittest.TestCase):
    def test_longs_strong_shorts_weak(self):
        n = 200
        strong = _df_from_close(100 + np.arange(n) * 0.5)   # rising
        weak = _df_from_close(100 - np.arange(n) * 0.3)     # falling
        flat = _df_from_close(100 + np.zeros(n))
        frames = {"BTC": strong, "ETH": flat, "SOL": weak}
        funding = {k: pd.Series(0.0, index=v.index) for k, v in frames.items()}
        costs = {k: 0.0 for k in frames}
        is_t, oos_t = CrossSectionalCandidate(lookback=20, hold=10).walk_forward(
            frames, funding, costs)
        trades = is_t + oos_t
        self.assertTrue(trades)
        longs = {t.symbol for t in trades if t.side == "long"}
        shorts = {t.symbol for t in trades if t.side == "short"}
        self.assertIn("BTC", longs)   # strongest is longed
        self.assertIn("SOL", shorts)  # weakest is shorted

    def test_cost_reduces_net_return(self):
        n = 200
        strong = _df_from_close(100 + np.arange(n) * 0.5)
        weak = _df_from_close(100 - np.arange(n) * 0.3)
        frames = {"BTC": strong, "SOL": weak}
        funding = {k: pd.Series(0.0, index=v.index) for k, v in frames.items()}
        _, oos0 = CrossSectionalCandidate(20, 10).walk_forward(
            frames, funding, {k: 0.0 for k in frames})
        _, oosc = CrossSectionalCandidate(20, 10).walk_forward(
            frames, funding, {k: 0.01 for k in frames})
        if oos0 and oosc:
            self.assertGreater(sum(t.ret_net for t in oos0),
                               sum(t.ret_net for t in oosc))


if __name__ == "__main__":
    unittest.main()
