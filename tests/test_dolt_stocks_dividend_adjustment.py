"""`close_history` returned DIVIDEND-ADJUSTED prices against RAW strikes.

`_yf_adjusted` called `yf.Ticker(s).history(period=...)`, whose `Close` is
adjusted for splits **and dividends** — the function's own comment said so.
`raw_from_adjusted` then un-adjusted **splits only**. The dividend adjustment
was never removed, so every historical spot came back too low, by an amount
that grows the further back you look and decays to zero at the present day.

Measured 2026-08-14 against SPY's true closes:

    2020-06-19   used 284.96   true 308.64    understated 8.31%
    2021-06-18   used 388.80   true 414.92    understated 6.72%
    2022-06-17   used 347.70   true 365.86    understated 5.22%
    2023-06-16   used 424.47   true 439.46    understated 3.53%
    2024-06-21   used 533.29   true 544.51    understated 2.10%
    2026-06-11   used 737.76   true 737.76    understated 0.00%

Spot decides strike-breach stops and, more importantly, EXPIRY SETTLEMENT. An
understated spot makes calls finish OTM more often and puts finish ITM more
often, so it inflates call-selling and deflates put-selling — on every Dolt
backtest ever run here.

Two properties make it worse than a constant bias:

* it is TIME-VARYING (8.3% in a 2020-2023 train window against ~2% in a
  2024-2026 holdout), so train and holdout are distorted by different amounts
  and any out-of-sample comparison inherits the difference;
* it is proportional to DIVIDEND YIELD, so it differs by symbol and by sector —
  XLRE and XLU carry roughly triple SPY's distortion.

Caught while verifying a SPY bear-call cell whose 93.8% win rate looked wrong:
the selector asked for 25-delta and, measured against a correct spot, the
contracts it picked sat at ~4 delta.

The fix is `auto_adjust=False`, whose `Close` is split-adjusted but NOT
dividend-adjusted; the existing split un-adjustment in `raw_from_adjusted` is
still needed and still correct.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.dolt_stocks import raw_from_adjusted


class TestTheFetcherAsksForUnadjustedCloses(unittest.TestCase):

    def _capture_history_kwargs(self):
        import pandas as pd

        from src import dolt_stocks
        captured = {}

        class _T:
            def history(self, **kw):
                captured.update(kw)
                idx = pd.to_datetime(["2020-06-19"])
                return pd.DataFrame({"Close": [308.64]}, index=idx)

        fake_yf = MagicMock()
        fake_yf.Ticker.return_value = _T()
        with patch.dict("sys.modules", {"yfinance": fake_yf}):
            dolt_stocks._yf_adjusted("SPY")
        return captured

    def test_auto_adjust_is_switched_off(self):
        """The whole defect in one keyword."""
        kw = self._capture_history_kwargs()
        self.assertIn("auto_adjust", kw,
                      "history() called without auto_adjust — Close will be "
                      "dividend-adjusted and will not match raw strikes")
        self.assertFalse(kw["auto_adjust"])


class TestSplitUnadjustmentStillApplies(unittest.TestCase):
    """`auto_adjust=False` removes the DIVIDEND adjustment only; Close is still
    split-adjusted, so this step remains load-bearing for split names."""

    def test_nvda_style_double_split_is_undone(self):
        # NVDA: 4:1 on 2021-07-20, 10:1 on 2024-06-10. A 2020 close of 9.26
        # as reported today is 9.26 * 4 * 10 = 370.4 as it actually traded.
        raw = raw_from_adjusted(
            {"2020-06-19": 9.26},
            [("2021-07-20", 4.0), ("2024-06-10", 10.0)])
        self.assertAlmostEqual(raw["2020-06-19"], 370.4, places=4)

    def test_a_date_after_every_split_is_untouched(self):
        raw = raw_from_adjusted(
            {"2025-01-02": 100.0},
            [("2021-07-20", 4.0), ("2024-06-10", 10.0)])
        self.assertAlmostEqual(raw["2025-01-02"], 100.0, places=9)

    def test_no_splits_is_the_identity(self):
        self.assertEqual(raw_from_adjusted({"2020-06-19": 308.64}, []),
                         {"2020-06-19": 308.64})


if __name__ == "__main__":
    unittest.main()
