import unittest
import numpy as np
import pandas as pd
from src.leverage.__main__ import build_reports


def _df(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1.0}, index=idx)


class TestBuildReports(unittest.TestCase):
    def test_reports_cover_all_candidates(self):
        frames = {"BTC": _df(seed=1), "ETH": _df(seed=2), "SOL": _df(seed=3)}
        funding = {k: pd.Series(0.0, index=v.index) for k, v in frames.items()}
        costs = {k: 0.0013 for k in frames}
        reports = build_reports(frames, funding, costs)
        names = {r.name for r in reports}
        self.assertEqual(names, {"trend_breakout", "funding_contrarian",
                                 "trend_carry", "xsect_momentum"})
        for r in reports:
            self.assertIn(r.verdict,
                          {"PROMOTE", "DEAD", "UNDERPOWERED"})


if __name__ == "__main__":
    unittest.main()
