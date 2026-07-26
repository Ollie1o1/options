import unittest
import numpy as np
from scipy.stats import norm
from src.iv_surface import _svi_iv
from src.probability_lab.rnd import rnd_from_smile, Density


def _logret_skew(d, S):
    """Skewness of log-returns ln(S_T/S) under the density. Isolates the
    vol-skew effect from the lognormal baseline (flat smile -> ~0)."""
    K, pdf = d.K, d.pdf
    m = K > 1e-6
    K, pdf = K[m], pdf[m]
    pdf = pdf / np.trapezoid(pdf, K)
    x = np.log(K / S)
    mu = np.trapezoid(x * pdf, K)
    var = np.trapezoid((x - mu) ** 2 * pdf, K)
    m3 = np.trapezoid((x - mu) ** 3 * pdf, K)
    return m3 / var ** 1.5


def lognormal_prob_above(S, x, sigma, T, r):
    # Closed-form risk-neutral P(S_T > x) for GBM.
    d2 = (np.log(S / x) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d2))


class TestRND(unittest.TestCase):
    def setUp(self):
        self.S, self.T, self.r, self.sig = 100.0, 0.25, 0.04, 0.30
        strikes = np.linspace(60, 160, 41)
        ivs = np.full_like(strikes, self.sig)
        self.d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)

    def test_integrates_to_one(self):
        self.assertAlmostEqual(np.trapezoid(self.d.pdf, self.d.K), 1.0, places=2)

    def test_matches_lognormal_prob(self):
        for x in (90, 100, 110, 120):
            got = self.d.prob_above(x)
            exp = lognormal_prob_above(self.S, x, self.sig, self.T, self.r)
            self.assertAlmostEqual(got, exp, delta=0.02)

    def test_mean_is_forward(self):
        fwd = self.S * np.exp(self.r * self.T)
        self.assertAlmostEqual(self.d.mean(), fwd, delta=fwd * 0.02)

    def test_flat_smile_logret_skew_near_zero(self):
        # Flat vol => Gaussian log-returns => ~zero skew (baseline sanity).
        self.assertAlmostEqual(_logret_skew(self.d, self.S), 0.0, delta=0.05)

    def test_put_skew_gives_negative_logret_skew(self):
        # Negative-rho SVI (downward-sloping/put skew) => left-skewed log-returns.
        strikes = np.linspace(50, 180, 51)
        k = np.log(strikes / self.S)
        ivs = _svi_iv(k, self.T, 0.02, 0.10, -0.6, 0.0, 0.20)
        d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)
        self.assertLess(_logret_skew(d, self.S), -0.2)

    def test_call_skew_gives_positive_logret_skew(self):
        # Positive-rho SVI (upward-sloping/call skew) => right-skewed log-returns.
        strikes = np.linspace(50, 180, 51)
        k = np.log(strikes / self.S)
        ivs = _svi_iv(k, self.T, 0.02, 0.10, 0.6, 0.0, 0.20)
        d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)
        self.assertGreater(_logret_skew(d, self.S), 0.2)


if __name__ == "__main__":
    unittest.main()
