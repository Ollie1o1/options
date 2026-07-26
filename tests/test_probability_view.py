import unittest
import numpy as np
from src.probability_lab.rnd import rnd_from_smile
from src.probability_lab.view import apply_view


class TestView(unittest.TestCase):
    def setUp(self):
        S, T, r, sig = 100.0, 0.25, 0.04, 0.30
        strikes = np.linspace(60, 160, 41)
        self.market = rnd_from_smile(strikes, np.full_like(strikes, sig), T, S, r)

    def test_identity(self):
        v = apply_view(self.market, 0.0, 1.0)
        self.assertAlmostEqual(v.mean(), self.market.mean(), delta=0.5)

    def test_drift_shifts_mean_up(self):
        v = apply_view(self.market, 0.03, 1.0)
        self.assertAlmostEqual(v.mean() / self.market.mean(), 1.03, delta=0.01)

    def test_vol_mult_widens(self):
        base = self.market.quantile(0.9) - self.market.quantile(0.1)
        wide = apply_view(self.market, 0.0, 1.5)
        span = wide.quantile(0.9) - wide.quantile(0.1)
        self.assertGreater(span, base * 1.3)

    def test_still_integrates_to_one(self):
        v = apply_view(self.market, 0.05, 0.8)
        self.assertAlmostEqual(np.trapezoid(v.pdf, v.K), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
