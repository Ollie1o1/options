import unittest
import numpy as np
import pandas as pd
from src.iv_surface import fit_svi_slice, SVIParams, fit_svi_surface


class TestFitSviSlice(unittest.TestCase):
    def test_recovers_flat_smile(self):
        S, T = 100.0, 0.25
        strikes = np.linspace(80, 120, 21)
        ivs = np.full_like(strikes, 0.30)
        params = fit_svi_slice(strikes, ivs, T, S)
        self.assertIsInstance(params, SVIParams)
        k = np.log(strikes / S)
        fitted = params.iv(k)
        self.assertTrue(np.allclose(fitted, 0.30, atol=0.01))

    def test_thin_slice_returns_none(self):
        self.assertIsNone(fit_svi_slice(np.array([100.0, 101.0]),
                                        np.array([0.3, 0.3]), 0.25, 100.0))

    def test_surface_still_fits(self):
        # Regression: the public refactor must not change fit_svi_surface output.
        rows = []
        for K in np.linspace(80, 120, 21):
            rows.append({"strike": K, "underlying": 100.0,
                         "impliedVolatility": 0.30, "T_years": 0.25,
                         "expiration": "2026-09-18"})
        df = fit_svi_surface(pd.DataFrame(rows))
        self.assertIn("iv_surface_residual", df.columns)
        self.assertTrue(df["iv_surface_fitted"].any())


if __name__ == "__main__":
    unittest.main()
