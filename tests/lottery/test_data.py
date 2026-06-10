"""Tests for the lottery option-data layer.

Provider-agnostic interface (free yfinance now, paid Polygon later) plus the
calibration math that measures REAL volatility-risk-premium and skew from live
chains so the backtest no longer relies on guessed constants. Pure pieces are
tested offline; network providers are not hit here.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.lottery.test_data -v
"""
from __future__ import annotations

import math
import unittest

from src.lottery.data import (
    atm_iv_from_chain,
    measure_vrp,
    measure_skew_per_sigma,
    calibrate_from_chains,
    get_provider,
    YFinanceProvider,
    PolygonProvider,
)


def _synthetic_call_wing(spot, atm_iv, T, skew, n_max=3):
    """Calls where IV = atm_iv*(1 + skew*sigma_otm); sigma unit = atm_iv*sqrt(T)."""
    unit = atm_iv * math.sqrt(T)
    pts = []
    for n in range(0, n_max + 1):
        strike = spot * math.exp(unit * n)
        iv = atm_iv * (1.0 + skew * n)
        pts.append({"strike": strike, "iv": iv, "open_interest": 100, "last": 1.0})
    return pts


class CalibrationTests(unittest.TestCase):

    def test_atm_iv_is_strike_nearest_spot(self):
        chain = [{"strike": 90, "iv": 0.55}, {"strike": 101, "iv": 0.40},
                 {"strike": 130, "iv": 0.70}]
        self.assertAlmostEqual(atm_iv_from_chain(chain, spot=100.0), 0.40)

    def test_vrp_is_implied_over_realized_minus_one(self):
        self.assertAlmostEqual(measure_vrp(atm_iv=0.40, realized_vol=0.32), 0.25, places=6)

    def test_skew_slope_recovered_from_wing(self):
        spot, atm_iv, T, skew = 100.0, 0.40, 0.25, 0.10
        wing = _synthetic_call_wing(spot, atm_iv, T, skew)
        est = measure_skew_per_sigma(wing, spot=spot, atm_iv=atm_iv, t_years=T)
        self.assertAlmostEqual(est, skew, places=2)

    def test_calibrate_aggregates_across_tickers(self):
        spot, atm_iv, T = 100.0, 0.40, 0.25
        samples = [
            {"atm_iv": 0.40, "realized_vol": 0.32, "spot": spot, "t_years": T,
             "calls": _synthetic_call_wing(spot, atm_iv, T, 0.10)},
            {"atm_iv": 0.50, "realized_vol": 0.40, "spot": spot, "t_years": T,
             "calls": _synthetic_call_wing(spot, 0.50, T, 0.12)},
        ]
        cal = calibrate_from_chains(samples)
        self.assertGreater(cal["vrp"], 0.0)
        self.assertGreater(cal["skew_per_sigma"], 0.0)
        self.assertIn("n_samples", cal)


class ProviderTests(unittest.TestCase):

    def test_default_provider_is_free_yfinance(self):
        p = get_provider()
        self.assertIsInstance(p, YFinanceProvider)
        self.assertEqual(p.name, "yfinance")
        self.assertTrue(p.is_free)

    def test_polygon_stub_is_not_free_and_errors_without_subscription(self):
        p = get_provider("polygon")
        self.assertIsInstance(p, PolygonProvider)
        self.assertFalse(p.is_free)
        with self.assertRaises(NotImplementedError):
            p.get_chain("AAPL")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            get_provider("bloomberg")


if __name__ == "__main__":
    unittest.main()
