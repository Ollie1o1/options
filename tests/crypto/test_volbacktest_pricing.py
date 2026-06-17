import unittest

from src.crypto.volbacktest import pricing as P


class TestPricing(unittest.TestCase):
    def test_straddle_price_is_call_plus_put(self):
        v = P.straddle(S=100, K=100, T=0.25, r=0.0, sigma=0.5)
        from src.utils import bs_call, bs_put
        self.assertAlmostEqual(
            v,
            bs_call(100, 100, 0.25, 0.0, 0.5) + bs_put(100, 100, 0.25, 0.0, 0.5),
            places=8,
        )

    def test_atm_straddle_delta_near_zero(self):
        # ATM (S=K) straddle delta = 2*N(d1)-1, nonzero due to the sigma^2/2 drift
        # in d1 (delta-neutral point sits just above spot). Small vs a single
        # call's ~0.55, which is what matters: it starts ~delta-neutral.
        d = P.straddle_delta(S=100, K=100, T=0.25, r=0.0, sigma=0.5)
        self.assertLess(abs(d), 0.12)

    def test_straddle_gamma_vega_positive(self):
        self.assertGreater(P.straddle_gamma(100, 100, 0.25, 0.0, 0.5), 0)
        self.assertGreater(P.straddle_vega(100, 100, 0.25, 0.0, 0.5), 0)

    def test_zero_T_straddle_is_intrinsic(self):
        self.assertAlmostEqual(P.straddle(110, 100, 0.0, 0.0, 0.5), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
