"""The vol the EV is priced on.

`calculate_metrics` valued every contract at Black-Scholes on `hv_252d` — a
252-day trailing realized vol — for options that are 14-45 DTE. A one-year
backward window forecasting a one-month forward outcome.

That was not an arbitrary choice: the long window replaced EWMA-20 on
2026-08-04 because a stale earnings gap read 51.8% and turned a $5 edge into a
reported +$4,664. Short estimators are horizon-matched but unstable, so the
trade-off was real and had to be measured rather than argued.

`scripts/vol_forecast_study.py` measured it — pre-registered, 20 symbols, six
years, 1,180 NON-OVERLAPPING 21-trading-day windows, every estimator computed
from the repo's own functions with no lookahead:

    estimator            RMSE      MAE  spearman     bias  P(fc>1.5x real)
    50/50 252d+30d     0.1012   0.0695    0.7318  -0.0059           0.108
    hv_30d blend       0.1072   0.0714    0.7265  -0.0188           0.071
    hv_252d (was)      0.1110   0.0808    0.6560  +0.0069           0.184

The 50/50 blend beats the old basis on ALL FIVE pre-registered metrics, and it
ranks first in BOTH halves of a period split it was not chosen on (early RMSE
0.1009, late 0.1015) while every other challenger moves four or more ranks.

The tail column refuted the hypothesis that motivated the long window. A
252-day window carries a stale high-vol regime for months after vol
normalises, so it OVERSTATES vol more often than any other candidate tested —
18.4% of windows above 1.5x realized. Overstating vol is precisely what
manufactures phantom edge, so the long window was more prone to the failure it
was introduced to prevent, not less.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trade_analysis import ev_vol_basis


class TestTheBlend(unittest.TestCase):

    def test_both_present_gives_the_even_blend(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.30], "hv_30d": [0.20]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.25)

    def test_it_is_not_just_the_long_window(self):
        """The regression this exists to prevent."""
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.40], "hv_30d": [0.20]}))
        self.assertNotAlmostEqual(float(out.iloc[0]), 0.40)

    def test_it_is_not_just_the_short_window(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.40], "hv_30d": [0.20]}))
        self.assertNotAlmostEqual(float(out.iloc[0]), 0.20)

    def test_row_by_row_not_frame_wide(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.30, 0.50],
                                         "hv_30d": [0.20, 0.10]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.25)
        self.assertAlmostEqual(float(out.iloc[1]), 0.30)


class TestFallbacks(unittest.TestCase):
    """A blend of two numbers needs both. One estimator is still better than
    no EV, so a missing side degrades rather than voids."""

    def test_missing_long_window_uses_the_short_one(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [np.nan], "hv_30d": [0.22]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.22)

    def test_missing_short_window_uses_the_long_one(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.31], "hv_30d": [np.nan]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.31)

    def test_ewma_is_used_when_the_short_blend_is_absent(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.30], "hv_30d": [np.nan],
                                         "hv_ewma": [0.20]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.25)

    def test_both_missing_stays_missing(self):
        """NaN here is load-bearing: `calculate_metrics` nulls the EV on it
        rather than substituting a number nobody measured."""
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [np.nan], "hv_30d": [np.nan]}))
        self.assertTrue(pd.isna(out.iloc[0]))

    def test_a_frame_with_no_vol_columns_at_all(self):
        out = ev_vol_basis(pd.DataFrame({"strike": [100.0]}))
        self.assertEqual(len(out), 1)
        self.assertTrue(pd.isna(out.iloc[0]))

    def test_mixed_rows_degrade_independently(self):
        df = pd.DataFrame({"hv_252d": [0.30, np.nan, 0.40],
                           "hv_30d": [0.20, 0.25, np.nan]})
        out = ev_vol_basis(df)
        self.assertAlmostEqual(float(out.iloc[0]), 0.25)
        self.assertAlmostEqual(float(out.iloc[1]), 0.25)
        self.assertAlmostEqual(float(out.iloc[2]), 0.40)

    def test_non_positive_vol_is_not_a_measurement(self):
        out = ev_vol_basis(pd.DataFrame({"hv_252d": [0.0], "hv_30d": [0.20]}))
        self.assertAlmostEqual(float(out.iloc[0]), 0.20)


class TestItIsWiredIn(unittest.TestCase):

    def test_calculate_metrics_uses_the_named_basis(self):
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertIn("ev_vol_basis(df)", src,
                      "the EV no longer prices on the measured vol basis")

    def test_the_old_long_window_first_chain_is_gone(self):
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertNotIn('_hv_cols = [c for c in ("hv_252d", "hv_ewma", "hv_30d")', src,
                         "the long-window-first fallback chain is back")


if __name__ == "__main__":
    unittest.main()
