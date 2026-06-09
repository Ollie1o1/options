"""Tests for the IV solver and cross-validation in src/data_quality.py."""

import unittest

import numpy as np
import pandas as pd

from src.utils import bs_price, bs_vega
from src.data_quality import implied_vol_from_price, cross_validate_iv


class TestImpliedVolRoundTrip(unittest.TestCase):
    def test_round_trip_grid(self):
        r = 0.04
        S = 100.0
        # moneyness incl. deep ITM/OTM, DTE incl. short and long, vol range
        asserted = 0
        for opt in ("call", "put"):
            for m in (0.80, 0.90, 1.00, 1.10, 1.25):
                K = S / m  # m = S/K ; deep ITM call when m>1, OTM when m<1
                for dte in (7, 30, 180, 365):
                    T = dte / 365.0
                    for sigma in (0.10, 0.25, 0.50, 1.00):
                        price = float(bs_price(opt, S, K, T, r, sigma))
                        solved = implied_vol_from_price(opt, S, K, T, r, price)
                        # The solver must never raise; it returns float or None.
                        self.assertTrue(solved is None or isinstance(solved, float))
                        # IV is only identifiable to high precision where the
                        # price is actually sensitive to vol. Deep ITM/OTM
                        # short-dated contracts have vega ~ 0 and are NOT
                        # invertible — assert precision only where vega is real.
                        # bs_vega is per 1% IV move; gate on the price being
                        # genuinely vol-sensitive (dPrice/dSigma = vega*100 > 5).
                        vega = float(bs_vega(S, K, T, r, sigma))
                        if price < 0.01 or vega < 0.05:
                            continue
                        self.assertIsNotNone(
                            solved, f"no solve for {opt} m={m} dte={dte} sig={sigma}"
                        )
                        self.assertAlmostEqual(
                            solved, sigma, delta=1e-4,
                            msg=f"{opt} m={m} dte={dte} sig={sigma} -> {solved}",
                        )
                        asserted += 1
        # sanity: the grid actually exercised the tight path many times,
        # including deep ITM/OTM at longer DTE.
        self.assertGreater(asserted, 40)

    def test_price_below_intrinsic_returns_none(self):
        # Call worth at least ~ S-K (discounted); price well below intrinsic.
        self.assertIsNone(implied_vol_from_price("call", 100, 50, 1.0, 0.04, 5.0))
        # Put intrinsic ~ K-S
        self.assertIsNone(implied_vol_from_price("put", 50, 100, 1.0, 0.04, 5.0))

    def test_price_above_model_max_returns_none(self):
        # No sigma in [0.005, 5] reaches a call price above the underlying.
        self.assertIsNone(implied_vol_from_price("call", 100, 100, 1.0, 0.04, 150.0))

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(implied_vol_from_price("call", 0, 100, 1.0, 0.04, 5.0))
        self.assertIsNone(implied_vol_from_price("call", 100, 100, 0.0, 0.04, 5.0))
        self.assertIsNone(implied_vol_from_price("call", 100, 100, 1.0, 0.04, 0.0))
        self.assertIsNone(implied_vol_from_price("call", 100, 100, 1.0, 0.04, float("nan")))


class TestCrossValidateIV(unittest.TestCase):
    def _row(self, opt, S, K, T, sigma, yahoo_iv):
        price = float(bs_price(opt, S, K, T, 0.04, sigma))
        return dict(type=opt, underlying=S, strike=K, T_years=T,
                    mid=price, impliedVolatility=yahoo_iv)

    def test_columns_and_flags(self):
        rows = [
            # yahoo IV matches the price -> verified True
            self._row("call", 100, 100, 0.5, 0.30, 0.30),
            # yahoo IV way off vs the price -> verified False (corrected candidate)
            self._row("call", 100, 105, 0.5, 0.30, 0.80),
            # no solvable mid (price 0) -> unsolvable, verified None
            dict(type="call", underlying=100, strike=100, T_years=0.5,
                 mid=np.nan, impliedVolatility=0.30),
        ]
        df = pd.DataFrame(rows)
        out = cross_validate_iv(df, r=0.04)
        for c in ("iv_solved", "iv_residual_pct", "iv_verified"):
            self.assertIn(c, out.columns)

        # row 0: solved ~ 0.30, residual ~ 0 -> verified
        self.assertAlmostEqual(out.loc[0, "iv_solved"], 0.30, delta=1e-3)
        self.assertTrue(bool(out.loc[0, "iv_verified"]))
        self.assertLess(abs(out.loc[0, "iv_residual_pct"]), 0.15)

        # row 1: solved ~ 0.30 but yahoo 0.80 -> not verified
        self.assertAlmostEqual(out.loc[1, "iv_solved"], 0.30, delta=1e-3)
        self.assertFalse(bool(out.loc[1, "iv_verified"]))
        self.assertGreater(abs(out.loc[1, "iv_residual_pct"]), 0.15)

        # row 2: unsolvable -> iv_solved NaN, iv_verified None
        self.assertTrue(pd.isna(out.loc[2, "iv_solved"]))
        self.assertIsNone(out.loc[2, "iv_verified"])


if __name__ == "__main__":
    unittest.main()
