import unittest
import numpy as np
from src.utils import bs_call, bs_put, bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        # Gold standard parameters
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_accuracy(self):
        """Test accuracy against standard reference values."""
        call_price = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        put_price = bs_put(self.S, self.K, self.T, self.r, self.sigma)
        
        # Expected values from standard BS calculator
        # Call: 10.4506, Put: 5.5735
        self.assertAlmostEqual(call_price, 10.45058, places=4)
        self.assertAlmostEqual(put_price, 5.57352, places=4)

    def test_vectorization(self):
        """Test that the functions correctly handle NumPy arrays."""
        S_arr = np.array([100.0, 110.0, 120.0])
        K_arr = np.array([100.0, 100.0, 100.0])
        
        call_prices = bs_call(S_arr, K_arr, self.T, self.r, self.sigma)
        
        self.assertIsInstance(call_prices, np.ndarray)
        self.assertEqual(call_prices.shape, (3,))
        self.assertAlmostEqual(call_prices[0], 10.45058, places=4)
        
        # Test delta vectorization with mixed types
        types = np.array(["call", "put", "call"])
        deltas = bs_delta(types, S_arr, K_arr, self.T, self.r, self.sigma)
        self.assertEqual(deltas.shape, (3,))
        self.assertGreater(deltas[0], 0) # Call delta > 0
        self.assertLess(deltas[1], 0)    # Put delta < 0

    def test_edge_cases(self):
        """Test edge cases like very small T and deep ITM/OTM."""
        # Very small T (approaching expiration)
        small_T = 0.00001
        # ATM at expiration should be roughly 0.5 delta for call
        d_atm = bs_delta("call", 100, 100, small_T, self.r, self.sigma)
        self.assertAlmostEqual(d_atm, 0.5, places=2)
        
        # Deep ITM Call
        d_itm = bs_delta("call", 500, 100, self.T, self.r, self.sigma)
        self.assertAlmostEqual(d_itm, 1.0, places=4)
        
        # Deep OTM Call
        d_otm = bs_delta("call", 10, 100, self.T, self.r, self.sigma)
        self.assertAlmostEqual(d_otm, 0.0, places=4)
        
        # Deep ITM Put
        dp_itm = bs_delta("put", 10, 100, self.T, self.r, self.sigma)
        self.assertAlmostEqual(dp_itm, -1.0, places=4)

    def test_greeks_accuracy(self):
        """Verify other greeks for standard scenario."""
        # Gamma: 0.0187
        gamma = bs_gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(gamma, 0.01876, places=4)
        
        # Vega: 0.3752
        vega = bs_vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(vega, 0.37524, places=4)
        
        # Theta Call (Annual/365): -0.0175
        theta = bs_theta("call", self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(theta, -0.01756, places=4)

if __name__ == "__main__":
    unittest.main()
