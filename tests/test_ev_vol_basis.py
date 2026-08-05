"""The vol the EV values options against, and the guard on implausible gaps.

`ev_per_contract` valued every option at Black-Scholes on a SHORT-window
realized vol (hv_ewma, span 20). Checked against live quotes 2026-08-04: a
163-DTE MSFT 535 call quoted $28.80 mid; BS at 252-day realized (31.6%) gives
$28.85 — a $5/contract edge. The screener reported +4,664, because MSFT's
30-day realized was 51.8% after an earnings gap against 35.1% implied.

A long window is the right basis for a 163-day option, and a realized/implied
ratio far from 1 is a data artifact rather than an opportunity.
"""
import math
import unittest

import numpy as np
import pandas as pd

from src.data_fetching import long_window_volatility
from src.trade_analysis import implausible_vol_gap


def _hist(n, daily_vol=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily_vol, n)
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rets))})


class LongWindowVolTest(unittest.TestCase):
    def test_it_annualises_a_known_daily_vol(self):
        hv = long_window_volatility(_hist(260, daily_vol=0.02))
        self.assertAlmostEqual(hv, 0.02 * math.sqrt(252), delta=0.05)

    def test_it_uses_the_full_year_when_available(self):
        """A 163-day option should not be valued off 20 days of returns."""
        calm = _hist(300, daily_vol=0.01, seed=1)
        spiked = calm.copy()
        spiked.iloc[-15:, 0] *= 1.6                     # a recent gap
        long_hv = long_window_volatility(spiked)
        from src.data_fetching import calculate_ewma_volatility
        short_hv = calculate_ewma_volatility(spiked, span=20)
        self.assertLess(long_hv, short_hv)

    def test_it_falls_back_to_the_longest_window_actually_present(self):
        self.assertIsNotNone(long_window_volatility(_hist(150)))

    def test_too_little_history_returns_none_rather_than_a_noisy_number(self):
        self.assertIsNone(long_window_volatility(_hist(40)))

    def test_an_empty_frame_is_handled(self):
        self.assertIsNone(long_window_volatility(pd.DataFrame({"Close": []})))


class ImplausibleGapTest(unittest.TestCase):
    """Nulling the EV, the same way it is already nulled when HV is missing."""

    def test_a_realistic_gap_is_allowed(self):
        self.assertFalse(implausible_vol_gap(hv=0.32, iv=0.35))
        self.assertFalse(implausible_vol_gap(hv=0.45, iv=0.35))

    def test_realized_far_above_implied_is_refused(self):
        """The MSFT case: 51.8% realized against 35.1% implied is a stale
        earnings gap, and it manufactured a 900x overstated edge."""
        self.assertTrue(implausible_vol_gap(hv=0.90, iv=0.35))

    def test_realized_far_below_implied_is_refused(self):
        self.assertTrue(implausible_vol_gap(hv=0.10, iv=0.60))

    def test_missing_inputs_are_refused_rather_than_assumed_fine(self):
        for hv, iv in ((None, 0.3), (0.3, None), (0.0, 0.3), (0.3, 0.0)):
            self.assertTrue(implausible_vol_gap(hv=hv, iv=iv))

    def test_it_is_vectorised_for_the_scan_path(self):
        hv = np.array([0.32, 0.90, 0.10])
        iv = np.array([0.35, 0.35, 0.60])
        np.testing.assert_array_equal(implausible_vol_gap(hv=hv, iv=iv),
                                      np.array([False, True, True]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
