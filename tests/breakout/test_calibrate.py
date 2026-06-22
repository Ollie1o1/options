"""Tests for in-house isotonic (PAV) calibration."""
from __future__ import annotations
import unittest
import numpy as np
from src.breakout.calibrate import fit_isotonic, apply_isotonic


class IsotonicTests(unittest.TestCase):
    def test_output_is_monotone(self):
        rng = np.random.default_rng(0)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(float)
        cal = fit_isotonic(p, y)
        xs = np.linspace(0, 1, 50)
        out = [apply_isotonic(cal, x) for x in xs]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(out, out[1:])))

    def test_corrects_overconfident_probs(self):
        # raw probs are 2x too confident: true rate = p/2
        rng = np.random.default_rng(1)
        p = rng.random(4000)
        y = (rng.random(4000) < p / 2).astype(float)
        cal = fit_isotonic(p, y)
        self.assertAlmostEqual(apply_isotonic(cal, 0.8), 0.4, delta=0.08)

    def test_bounds_clamped(self):
        cal = fit_isotonic(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
        self.assertGreaterEqual(apply_isotonic(cal, -5.0), 0.0)
        self.assertLessEqual(apply_isotonic(cal, 5.0), 1.0)


if __name__ == "__main__":
    unittest.main()
