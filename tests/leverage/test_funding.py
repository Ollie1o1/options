import unittest
import numpy as np
import pandas as pd
from src.leverage import funding as F


def _daily_index(n):
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


class TestFunding(unittest.TestCase):
    def test_align_daily_sums_intraday_to_day(self):
        # three 8h funding prints on day 0, one on day 1
        ts = pd.to_datetime([
            "2024-01-01T00:00Z", "2024-01-01T08:00Z", "2024-01-01T16:00Z",
            "2024-01-02T00:00Z"])
        fund = pd.Series([0.0001, 0.0001, 0.0002, 0.0005], index=ts)
        idx = _daily_index(2)
        daily = F.align_daily(fund, idx)
        self.assertEqual(len(daily), 2)
        self.assertAlmostEqual(daily.iloc[0], 0.0004, places=8)  # summed
        self.assertAlmostEqual(daily.iloc[1], 0.0005, places=8)

    def test_align_daily_missing_days_are_zero(self):
        fund = pd.Series([0.0001], index=pd.to_datetime(["2024-01-01T00:00Z"]))
        idx = _daily_index(3)
        daily = F.align_daily(fund, idx)
        self.assertEqual(list(daily.values[1:]), [0.0, 0.0])

    def test_zscore_flags_extreme(self):
        base = pd.Series([0.0001] * 40, index=_daily_index(40))
        base.iloc[-1] = 0.01  # a big funding spike
        z = F.zscore(base, window=30)
        self.assertGreater(z.iloc[-1], 3.0)

    def test_zscore_is_zero_for_flat_series(self):
        flat = pd.Series([0.0002] * 40, index=_daily_index(40))
        z = F.zscore(flat, window=30)
        self.assertTrue(np.isnan(z.iloc[-1]) or abs(z.iloc[-1]) < 1e-9)


if __name__ == "__main__":
    unittest.main()
