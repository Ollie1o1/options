"""Sector leadership must keep its meaning in falling and flat markets."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sector_analyzer import SectorAnalyzer, _roc_20


def closes(spy_return, a_return, b_return):
    # The last 21 points have exact known 20-session returns; earlier rows
    # provide a 50-session moving-average window.
    def values(ret):
        return np.r_[np.full(39, 100.), np.linspace(100, 100 * (1 + ret), 21)]
    frame = pd.DataFrame({"SPY": values(spy_return), "XLK": values(a_return), "XLE": values(b_return)},
                         index=pd.bdate_range("2026-01-01", periods=60))
    frame.columns = pd.MultiIndex.from_product([["Close"], frame.columns])
    return frame


class RelativeStrengthTests(unittest.TestCase):
    def compute(self, raw):
        with patch("yfinance.download", return_value=raw):
            return SectorAnalyzer()._fetch_and_compute()

    def test_falling_market_does_not_reverse_the_leaders(self):
        result = self.compute(closes(-.05, -.01, -.1))
        self.assertEqual(result.top_sectors[0], "XLK")
        self.assertGreater(result.rs_matrix["XLK"], 1)
        self.assertLess(result.rs_matrix["XLE"], 1)

    def test_flat_market_is_a_valid_relative_comparison(self):
        result = self.compute(closes(0, .03, -.02))
        self.assertAlmostEqual(result.rs_matrix["XLK"], 1.03)
        self.assertEqual(result.top_sectors[0], "XLK")

    def test_rising_market_preserves_outperformance_meaning(self):
        result = self.compute(closes(.05, .1, .01))
        self.assertAlmostEqual(result.rs_matrix["XLK"], round(1.1 / 1.05, 4))
        self.assertGreater(result.rs_matrix["XLK"], result.rs_matrix["XLE"])

    def test_missing_sector_candle_is_not_a_different_lookback(self):
        raw = closes(.05, .1, .01)
        raw.loc[raw.index[-8], ("Close", "XLK")] = np.nan
        self.assertNotIn("XLK", self.compute(raw).rs_matrix)

    def test_short_or_invalid_history_is_unknown_not_zero_return(self):
        self.assertIsNone(_roc_20(pd.Series([100.] * 20)))
        self.assertIsNone(_roc_20(pd.Series([0.] + [100.] * 20)))
        self.assertIsNone(_roc_20(pd.Series([100.] * 20 + [float("inf")])))
