import unittest
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from src.utils import bs_call, bs_put, bs_delta, bs_gamma, bs_theta, bs_vega

# Hypothesis strategies for option parameters
# Prices should be positive, reasonable magnitudes
S_strategy = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)
K_strategy = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)
# Time to expiration: avoid absolute 0 to prevent div-by-zero if not handled
T_strategy = st.floats(min_value=1e-5, max_value=10.0, allow_nan=False, allow_infinity=False)
# Risk-free rate: reasonable bounds
r_strategy = st.floats(min_value=-0.05, max_value=0.2, allow_nan=False, allow_infinity=False)
# Volatility: positive, avoiding 0 exactly
sigma_strategy = st.floats(min_value=1e-4, max_value=5.0, allow_nan=False, allow_infinity=False)

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
        
        types = np.array(["call", "put", "call"])
        deltas = bs_delta(types, S_arr, K_arr, self.T, self.r, self.sigma)
        self.assertEqual(deltas.shape, (3,))
        self.assertGreater(deltas[0], 0) # Call delta > 0
        self.assertLess(deltas[1], 0)    # Put delta < 0

    def test_greeks_accuracy(self):
        """Verify other greeks for standard scenario."""
        gamma = bs_gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(gamma, 0.01876, places=4)
        vega = bs_vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(vega, 0.37524, places=4)
        theta = bs_theta("call", self.S, self.K, self.T, self.r, self.sigma)
        self.assertAlmostEqual(theta, -0.01756, places=4)

@given(S=S_strategy, K=K_strategy, T=T_strategy, r=r_strategy, sigma=sigma_strategy)
@settings(max_examples=200, deadline=None)
def test_hypothesis_prices_are_valid(S, K, T, r, sigma):
    """Property: Prices should never be negative, NaN, or Inf."""
    call = bs_call(S, K, T, r, sigma)
    put = bs_put(S, K, T, r, sigma)
    
    assert not np.isnan(call) and not np.isnan(put), "Price is NaN"
    assert not np.isinf(call) and not np.isinf(put), "Price is Inf"
    assert call >= -1e-10, f"Call price negative: {call}"
    assert put >= -1e-10, f"Put price negative: {put}"

@given(S=S_strategy, K=K_strategy, T=T_strategy, r=r_strategy, sigma=sigma_strategy)
@settings(max_examples=200, deadline=None)
def test_hypothesis_put_call_parity(S, K, T, r, sigma):
    """Property: C - P = S - K * e^(-rT)"""
    call = bs_call(S, K, T, r, sigma)
    put = bs_put(S, K, T, r, sigma)
    
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    
    # Parity might have float precision issues on extreme bounds
    np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-3)

@given(S=S_strategy, K=K_strategy, T=T_strategy, r=r_strategy, sigma=sigma_strategy)
@settings(max_examples=200, deadline=None)
def test_hypothesis_delta_bounds(S, K, T, r, sigma):
    """Property: Call delta in [0, 1], Put delta in [-1, 0]."""
    c_delta = bs_delta("call", S, K, T, r, sigma)
    p_delta = bs_delta("put", S, K, T, r, sigma)
    
    assert not np.isnan(c_delta) and not np.isnan(p_delta), "Delta is NaN"
    assert not np.isinf(c_delta) and not np.isinf(p_delta), "Delta is Inf"
    
    assert -1e-5 <= c_delta <= 1.0 + 1e-5, f"Call delta out of bounds: {c_delta}"
    assert -1.0 - 1e-5 <= p_delta <= 1e-5, f"Put delta out of bounds: {p_delta}"

if __name__ == "__main__":
    unittest.main()
