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


class RiskNeutralMomentsTest(unittest.TestCase):
    """Q-skew and Q-kurtosis, validated against a lognormal's closed form.

    The basis is a REQUIRED argument. Skewness of the price distribution and
    skewness of log-returns are different numbers from the same density — for
    a flat smile the first is +0.456 and the second is 0.0 — and a default
    would let one be quoted as the other.
    """

    def setUp(self):
        self.S, self.T, self.r, self.sig = 100.0, 0.25, 0.04, 0.30
        strikes = np.linspace(40, 260, 111)
        ivs = np.full_like(strikes, self.sig)
        self.d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)

    def _lognormal_price_moments(self):
        s2 = self.sig ** 2 * self.T
        e = np.exp(s2)
        skew = (e + 2.0) * np.sqrt(e - 1.0)
        kurt = np.exp(4 * s2) + 2 * np.exp(3 * s2) + 3 * np.exp(2 * s2) - 3.0
        return float(skew), float(kurt)

    def test_the_basis_must_be_named(self):
        with self.assertRaises(TypeError):
            self.d.moments()          # type: ignore[call-arg]

    def test_an_unknown_basis_is_refused(self):
        with self.assertRaises(ValueError):
            self.d.moments("returns")

    def test_price_skew_and_kurtosis_match_the_lognormal(self):
        exp_skew, exp_kurt = self._lognormal_price_moments()
        got = self.d.moments("price")
        self.assertAlmostEqual(got["skewness"], exp_skew, delta=0.05)
        self.assertAlmostEqual(got["kurtosis"], exp_kurt, delta=0.20)

    def test_log_return_moments_are_gaussian_for_a_flat_smile(self):
        got = self.d.moments("logret")
        self.assertAlmostEqual(got["skewness"], 0.0, delta=0.05)
        self.assertAlmostEqual(got["kurtosis"], 3.0, delta=0.20)

    def test_the_two_bases_genuinely_differ(self):
        """If these ever coincide the basis argument is pointless."""
        self.assertGreater(
            abs(self.d.moments("price")["skewness"]
                - self.d.moments("logret")["skewness"]), 0.2)

    def test_variance_is_consistent_with_the_mean(self):
        got = self.d.moments("price")
        self.assertGreater(got["variance"], 0.0)
        self.assertAlmostEqual(got["mean"], self.d.mean(), delta=1e-6)
        fwd = self.S * np.exp(self.r * self.T)
        self.assertAlmostEqual(got["mean"], fwd, delta=fwd * 0.02)

    def test_put_skew_makes_log_return_skew_negative(self):
        """The sign must track the smile, not just the lognormal baseline."""
        strikes = np.linspace(50, 180, 51)
        k = np.log(strikes / self.S)
        ivs = _svi_iv(k, self.T, 0.02, 0.10, -0.6, 0.0, 0.20)
        d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)
        self.assertLess(d.moments("logret")["skewness"], -0.2)

    def test_a_fat_tailed_smile_lifts_kurtosis_above_the_flat_case(self):
        strikes = np.linspace(50, 180, 51)
        k = np.log(strikes / self.S)
        ivs = _svi_iv(k, self.T, 0.02, 0.30, 0.0, 0.0, 0.10)   # strong curvature
        d = rnd_from_smile(strikes, ivs, self.T, self.S, self.r)
        self.assertGreater(d.moments("logret")["kurtosis"],
                           self.d.moments("logret")["kurtosis"])


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
