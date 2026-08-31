"""A fit's reported quality must describe the parameters it actually returns.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_iv_surface -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.iv_surface import (SVIParams, _fit_single_expiry, _svi_objective,
                            calendar_arbitrage)

# A realistic skew (IV 0.15 -> 0.49 across k in [-0.78, 0.52]) on which the
# Nelder-Mead search escapes to a degenerate corner: b -> 0, rho -> -1,
# sigma -> 5.4e6, m -> 9.4e4. Those flatten w(k) to a constant, so the fitted
# smile carries no information — yet the reported fit_quality was 0.9905,
# because it was computed from the optimiser's objective at its own iterate
# rather than from the parameters handed back after projection.
_DEGENERATE_T = 0.16883
_DEGENERATE_K = [-0.78248, -0.689032, -0.57741, -0.576886, -0.457813,
                 -0.41652, -0.410586, -0.155672, -0.109184, 0.024298,
                 0.042237, 0.068103, 0.182999, 0.272295, 0.426333,
                 0.496005, 0.515791]
_DEGENERATE_IV = [0.283268, 0.270753, 0.243208, 0.243064, 0.20042, 0.147585,
                  0.196269, 0.146788, 0.167162, 0.239323, 0.187105, 0.234874,
                  0.275126, 0.316855, 0.410707, 0.494572, 0.494298]


def _quality_of(params, k, market_var) -> float:
    """fit_quality recomputed from the parameters actually returned."""
    budget = float(np.mean(market_var)) * len(k)
    sse = float(_svi_objective(np.asarray(params, dtype=float), k, market_var))
    return max(0.0, 1.0 - sse / max(budget, 1e-10))


class FitQualityDescribesTheReturnedParamsTest(unittest.TestCase):
    """The defect: quality measured at one point, parameters returned from another.

    `_fit_single_expiry` reported `1 - res.fun / budget`, where `res.fun` is the
    PENALISED objective at the optimiser's own iterate `res.x`. It then returned
    `_enforce_constraints(res.x)` — a different point. Nothing checked that the
    number and the parameters referred to the same thing.
    """

    def test_a_degenerate_fit_is_not_reported_as_a_good_one(self):
        k = np.array(_DEGENERATE_K)
        iv = np.array(_DEGENERATE_IV)
        params, quality = _fit_single_expiry(k, iv, _DEGENERATE_T)
        if params is None:
            self.assertEqual(quality, 0.0)
            return
        market_var = iv ** 2 * _DEGENERATE_T
        self.assertAlmostEqual(
            quality, _quality_of(params, k, market_var), places=6,
            msg=("reported fit_quality does not describe the returned "
                 "parameters — the number and the model disagree"))

    def test_quality_matches_the_returned_params_across_many_slices(self):
        """Property, not anecdote: it must hold for every slice that fits."""
        rng = np.random.default_rng(7)
        checked = 0
        for _ in range(300):
            n = int(rng.integers(6, 30))
            k = np.sort(rng.uniform(-0.8, 0.8, n))
            T = float(rng.uniform(0.02, 1.0))
            base = rng.uniform(0.10, 0.80)
            iv = np.clip(
                np.abs(base + rng.uniform(0, 0.6) * k ** 2
                       + rng.uniform(-0.4, 0.4) * k)
                + rng.normal(0, 0.02, n), 1e-3, None)
            params, quality = _fit_single_expiry(k, iv, T)
            if params is None:
                self.assertEqual(quality, 0.0)
                continue
            checked += 1
            self.assertAlmostEqual(
                quality, _quality_of(params, k, iv ** 2 * T), places=6,
                msg="quality and returned params disagree on this slice")
        self.assertGreater(checked, 20, "too few slices fitted to be a test")

    def test_an_accepted_fit_actually_tracks_its_data(self):
        """Acceptance must be judged on what is returned, not on an iterate.

        A fit whose returned parameters miss the data by more than the whole
        variance budget is not a fit, whatever the optimiser reported about a
        point it visited on the way.
        """
        rng = np.random.default_rng(7)
        for _ in range(300):
            n = int(rng.integers(6, 30))
            k = np.sort(rng.uniform(-0.8, 0.8, n))
            T = float(rng.uniform(0.02, 1.0))
            base = rng.uniform(0.10, 0.80)
            iv = np.clip(
                np.abs(base + rng.uniform(0, 0.6) * k ** 2
                       + rng.uniform(-0.4, 0.4) * k)
                + rng.normal(0, 0.02, n), 1e-3, None)
            params, quality = _fit_single_expiry(k, iv, T)
            if params is None:
                continue
            self.assertGreater(
                _quality_of(params, k, iv ** 2 * T), 0.0,
                "an accepted fit scores zero against its own data")


class ConvergenceFlagIsNotFitQualityTest(unittest.TestCase):
    """`res.success` was a hard gate. It measures the wrong thing.

    Nelder-Mead reports success only if it meets xatol=1e-8 AND fatol=1e-10 on
    a 5-parameter, badly-scaled problem. It usually cannot, so it exhausts
    maxiter and reports failure — while sitting on an excellent fit. Measured
    over 120 realistic slices: 62% reported failure, and 100% of those scored
    above 0.95 against their own data (median 0.9999).

    The flag is uninformative in BOTH directions. The degenerate corner in
    `FitQualityDescribesTheReturnedParamsTest` converged with success=True.
    Fit adequacy is what the SSE budget check measures; the flag adds only
    false refusals.
    """

    def _realistic_slices(self, n_slices=120):
        rng = np.random.default_rng(11)
        for _ in range(n_slices):
            n = int(rng.integers(8, 25))
            k = np.sort(rng.uniform(-0.5, 0.5, n))
            T = float(rng.choice([0.02, 0.05, 0.1, 0.25, 0.5, 1.0]))
            iv = np.clip(0.25 - 0.25 * k + 0.45 * k ** 2
                         + rng.normal(0, 0.015, n), 1e-3, None)
            yield k, iv, T

    def test_most_realistic_slices_now_produce_a_fit(self):
        """Before: 38% fitted. A signal NaN on most expiries is not a signal."""
        fitted = sum(1 for k, iv, T in self._realistic_slices()
                     if _fit_single_expiry(k, iv, T)[0] is not None)
        self.assertGreater(
            fitted, 100,
            f"only {fitted}/120 realistic slices fitted — the convergence flag "
            f"is discarding good fits again")

    def test_every_accepted_fit_still_tracks_its_data(self):
        """Recovering fits must not mean accepting bad ones."""
        for k, iv, T in self._realistic_slices():
            params, quality = _fit_single_expiry(k, iv, T)
            if params is None:
                continue
            self.assertGreater(
                quality, 0.90,
                "a recovered fit does not actually track its data")
            self.assertAlmostEqual(
                quality, _quality_of(params, k, iv ** 2 * T), places=6)

    def test_the_degenerate_corner_is_still_refused(self):
        """Dropping the flag must not readmit the fit that started all this."""
        k = np.array(_DEGENERATE_K)
        iv = np.array(_DEGENERATE_IV)
        params, quality = _fit_single_expiry(k, iv, _DEGENERATE_T)
        if params is not None:
            self.assertAlmostEqual(
                quality, _quality_of(params, k, iv ** 2 * _DEGENERATE_T),
                places=6)
            self.assertGreater(quality, 0.0)


class ButterflyWingBoundTest(unittest.TestCase):
    """`b(1+|rho|) < 4` bounds the asymptotic slope of total variance.

    w(k)/|k| -> b(1 +/- rho) as k -> +/-inf, so an unbounded b drives the wings
    steeper than any arbitrage-free surface allows. Measured before enforcing:
    6% of realistic equity smiles violated it, 42% across a wider sweep, with a
    worst case of 419 — so this is a live constraint, not a decorative one.
    """

    def _fits(self, slope_r, curv_r, base_r, n_trials=200, seed=11):
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            m = int(rng.integers(8, 25))
            k = np.sort(rng.uniform(-0.5, 0.5, m))
            T = float(rng.choice([0.02, 0.05, 0.1, 0.25, 0.5, 1.0]))
            iv = np.clip(np.abs(rng.uniform(*base_r)
                                + rng.uniform(*slope_r) * k
                                + rng.uniform(*curv_r) * k ** 2)
                         + rng.normal(0, 0.015, m), 1e-3, None)
            params, quality = _fit_single_expiry(k, iv, T)
            if params is not None:
                yield params, quality, k, iv, T

    def test_no_accepted_fit_violates_the_wing_bound(self):
        checked = 0
        for params, _q, _k, _iv, _T in self._fits((-0.6, -0.1), (0.1, 1.0),
                                                  (0.12, 0.60)):
            checked += 1
            _a, b, rho, _sigma, _m = params
            self.assertLess(
                b * (1 + abs(rho)), 4.0,
                "an accepted fit has arbitrageable wings")
        self.assertGreater(checked, 50, "too few fits to be a test")

    def test_enforcing_the_bound_does_not_gut_realistic_fits(self):
        """The constraint must bind on pathology, not on ordinary smiles."""
        fits = list(self._fits((-0.30, -0.15), (0.3, 0.6), (0.18, 0.35)))
        self.assertGreater(len(fits), 150,
                           f"only {len(fits)}/200 realistic smiles survived the "
                           f"wing bound — it is rejecting ordinary data")

    def test_quality_still_describes_the_returned_params(self):
        """The new projection must not reintroduce the #83 defect."""
        for params, quality, k, iv, T in self._fits((-0.6, -0.1), (0.1, 1.0),
                                                    (0.12, 0.60), n_trials=80):
            self.assertAlmostEqual(quality, _quality_of(params, k, iv ** 2 * T),
                                   places=6)


class CalendarArbitrageTest(unittest.TestCase):
    """Total variance must not decrease with maturity at any log-moneyness.

    w(k, T2) >= w(k, T1) for T2 > T1. A dip means a calendar spread prices
    below zero.
    """

    def _slice(self, T, a, b=0.1, rho=-0.3, sigma=0.3, m=0.0):
        return SVIParams(a=a, b=b, rho=rho, sigma=sigma, m=m, T=T,
                         fit_quality=1.0)

    def test_a_rising_surface_is_arbitrage_free(self):
        rep = calendar_arbitrage([self._slice(0.25, 0.02),
                                  self._slice(0.50, 0.04),
                                  self._slice(1.00, 0.08)])
        self.assertTrue(rep["arbitrage_free"])
        self.assertEqual(rep["n_violations"], 0)

    def test_a_dipping_surface_is_caught(self):
        rep = calendar_arbitrage([self._slice(0.25, 0.08),
                                  self._slice(0.50, 0.02)])
        self.assertFalse(rep["arbitrage_free"])
        self.assertGreater(rep["n_violations"], 0)
        self.assertGreater(rep["worst_drop"], 0.0)

    def test_slices_are_ordered_by_maturity_not_by_input_order(self):
        """An unsorted input must not read as a violation."""
        rep = calendar_arbitrage([self._slice(1.00, 0.08),
                                  self._slice(0.25, 0.02),
                                  self._slice(0.50, 0.04)])
        self.assertTrue(rep["arbitrage_free"])

    def test_fewer_than_two_slices_cannot_violate(self):
        self.assertTrue(calendar_arbitrage([])["arbitrage_free"])
        self.assertTrue(
            calendar_arbitrage([self._slice(0.25, 0.02)])["arbitrage_free"])
        self.assertEqual(calendar_arbitrage([])["n_violations"], 0)

    def test_the_report_names_which_pair_dipped(self):
        rep = calendar_arbitrage([self._slice(0.25, 0.08),
                                  self._slice(0.50, 0.02)])
        self.assertEqual(rep["violations"][0]["T_lo"], 0.25)
        self.assertEqual(rep["violations"][0]["T_hi"], 0.50)


class GoodFitsStillFitTest(unittest.TestCase):
    """The correction must not start refusing fits that were always fine."""

    def test_a_clean_synthetic_smile_still_fits_well(self):
        k = np.linspace(-0.4, 0.4, 21)
        T = 0.25
        w = 0.04 + 0.1 * (-0.3 * k + np.sqrt(k ** 2 + 0.04))
        iv = np.sqrt(w / T)
        params, quality = _fit_single_expiry(k, iv, T)
        self.assertIsNotNone(params, "a clean SVI smile must still fit")
        self.assertGreater(quality, 0.90)


if __name__ == "__main__":
    unittest.main()
