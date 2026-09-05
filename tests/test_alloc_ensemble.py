"""The ridge ensemble: combining several credit-richness-controlled features.

Every feature tested alone in this repo has died the same way once
`residual_ic` controlled for credit richness. These tests check the two
things that would make a combined score dishonest in the same way: that the
control happens INSIDE the fit (a renamed control must not dominate the
combination), and that cross-validation for the regularization strength
never crosses a fold boundary via a shuffle (the same leak `split_by_time`
guards against, at CV-fold scale).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_ensemble -v
"""
from __future__ import annotations

import random
import unittest

import numpy as np

from src.alloc.ensemble import (DEFAULT_ALPHAS, EnsembleModel, _blocked_folds,
                                _design_matrix, _ridge_fit, ensemble_ic,
                                fit_ensemble, score_ensemble)


class _T:
    """Minimal stand-in for engine.Trade, matching test_alloc_attribution's."""

    def __init__(self, entry_date, roc, capital_at_risk=100.0, **feats):
        self.entry_date = entry_date
        self.capital_at_risk = capital_at_risk
        self.pnl = roc * capital_at_risk
        self.features = feats
        self.exit_date = "2024-12-31"


def _dates(n):
    return [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)]


def _combo_trades(n=240, seed=11, k_own=3, own_weight=0.25, noise=0.35):
    """Outcome driven by credit richness (should be controlled away) plus
    several independently-weak `own_i` features (should combine)."""
    rng = random.Random(seed)
    out = []
    for i, d in enumerate(_dates(n)):
        rich = rng.random()
        owns = [rng.random() for _ in range(k_own)]
        roc = 0.4 * (rich - 0.5) + sum(own_weight * (o - 0.5) for o in owns) \
            + noise * (rng.random() - 0.5)
        feats = {"credit_pct_width": rich, "atm_iv": rich}
        feats.update({f"own{j}": o for j, o in enumerate(owns)})
        out.append(_T(d, roc=roc, **feats))
    return out


class RidgeFitTest(unittest.TestCase):

    def test_alpha_zero_recovers_ols_on_a_simple_fit(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 2))
        true_coef = np.array([1.5, -0.7])
        y = X @ true_coef + rng.normal(scale=0.01, size=200)
        coef, intercept = _ridge_fit(X, y, alpha=0.0)
        np.testing.assert_allclose(coef, true_coef, atol=0.05)
        self.assertAlmostEqual(intercept, float(y.mean()), places=6)

    def test_larger_alpha_shrinks_coefficients_toward_zero(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 3))
        y = X @ np.array([1.0, 1.0, 1.0]) + rng.normal(scale=0.5, size=200)
        small, _ = _ridge_fit(X, y, alpha=0.1)
        large, _ = _ridge_fit(X, y, alpha=1000.0)
        self.assertLess(np.sum(large ** 2), np.sum(small ** 2))


class BlockedFoldsTest(unittest.TestCase):

    def test_folds_are_contiguous_and_cover_every_row_once(self):
        folds = _blocked_folds(23, 5)
        seen = np.concatenate(folds)
        self.assertEqual(sorted(seen.tolist()), list(range(23)))
        for f in folds:
            self.assertTrue(np.all(np.diff(f) == 1) or f.size <= 1)

    def test_asks_for_more_blocks_than_rows_still_returns_at_most_n(self):
        folds = _blocked_folds(3, 10)
        self.assertLessEqual(len([f for f in folds if f.size > 0]), 3)


class DesignMatrixTest(unittest.TestCase):

    def test_drops_a_feature_that_is_a_renamed_control(self):
        trades = _combo_trades()
        for t in trades:
            t.features["renamed"] = t.features["credit_pct_width"]
        dm = _design_matrix(trades, ["own0", "renamed"],
                            controls=("credit_pct_width", "atm_iv"))
        self.assertIn("own0", dm["features"])
        self.assertNotIn("renamed", dm["features"])

    def test_none_when_no_feature_is_measurable(self):
        trades = _combo_trades(n=4)  # below MIN_TRADES
        dm = _design_matrix(trades, ["own0"], controls=("credit_pct_width",))
        self.assertIsNone(dm)

    def test_intersects_trades_across_features_with_different_gaps(self):
        trades = _combo_trades(n=100)
        for t in trades[:20]:
            t.features.pop("own0")
        for t in trades[50:70]:
            t.features.pop("own1")
        dm = _design_matrix(trades, ["own0", "own1"],
                            controls=("credit_pct_width", "atm_iv"))
        self.assertEqual(dm["X"].shape, (len(dm["keep"]), 2))
        self.assertLessEqual(len(dm["keep"]), 60)


class FitEnsembleTest(unittest.TestCase):

    def test_too_few_trades_returns_none(self):
        self.assertIsNone(fit_ensemble(_combo_trades(n=5), ["own0", "own1"]))

    def test_a_feature_that_is_pure_credit_richness_does_not_dominate(self):
        trades = _combo_trades()
        for t in trades:
            t.features["echo"] = t.features["credit_pct_width"]
        model = fit_ensemble(trades, ["own0", "own1", "own2", "echo"])
        self.assertIsNotNone(model)
        self.assertNotIn("echo", model.features)

    def test_combining_several_weak_features_beats_any_one_alone(self):
        trades = _combo_trades(n=400, k_own=4, own_weight=0.22, noise=0.3)
        model = fit_ensemble(trades, [f"own{i}" for i in range(4)])
        self.assertIsNotNone(model)
        ens = ensemble_ic(trades, model)
        from src.alloc.attribution import feature_ic
        singles = [abs(feature_ic(trades, f"own{i}")["ic"] or 0.0)
                  for i in range(4)]
        self.assertGreater(abs(ens["ic"]), max(singles))

    def test_the_chosen_alpha_is_one_of_the_offered_ones(self):
        model = fit_ensemble(_combo_trades(n=300), ["own0", "own1", "own2"])
        self.assertIn(model.alpha, DEFAULT_ALPHAS)

    def test_cv_scores_are_reported_for_every_alpha_tried(self):
        model = fit_ensemble(_combo_trades(n=300), ["own0", "own1"])
        self.assertEqual(len(model.cv_ic_by_alpha), len(DEFAULT_ALPHAS))


class ScoreEnsembleTest(unittest.TestCase):

    def test_scoring_a_model_with_none_of_its_features_present_is_empty(self):
        train = _combo_trades(n=200, seed=1)
        model = fit_ensemble(train, ["own0", "own1"])
        other = _combo_trades(n=50, seed=2)
        for t in other:
            t.features.pop("own0")
            t.features.pop("own1")
        keep, scores = score_ensemble(model, other)
        self.assertEqual(keep, [])
        self.assertEqual(scores, [])

    def test_scoring_a_fresh_sample_from_the_same_process_correlates_with_return(self):
        train = _combo_trades(n=400, seed=5, k_own=3, own_weight=0.3, noise=0.25)
        test = _combo_trades(n=200, seed=6, k_own=3, own_weight=0.3, noise=0.25)
        model = fit_ensemble(train, ["own0", "own1", "own2"])
        self.assertIsNotNone(model)
        result = ensemble_ic(test, model)
        self.assertIsNotNone(result["ic"])
        self.assertGreater(result["ic"], 0.1)


class EnsembleIcTest(unittest.TestCase):

    def test_shape_matches_feature_ic_for_direct_comparison(self):
        trades = _combo_trades(n=300)
        model = fit_ensemble(trades, ["own0", "own1", "own2"])
        row = ensemble_ic(trades, model)
        for key in ("ic", "p", "t", "t_clustered", "n"):
            self.assertIn(key, row)

    def test_too_few_scored_trades_is_none_not_zero(self):
        trades = _combo_trades(n=300)
        model = fit_ensemble(trades, ["own0", "own1"])
        row = ensemble_ic(trades[:3], model)
        self.assertIsNone(row["ic"])


if __name__ == "__main__":
    unittest.main()
