"""Combine many individually-weak entry features into one ridge score.

Every feature this repo has tested alone has died the same way: it looked
predictive until `residual_ic` regressed `credit_pct_width`/`atm_iv` out of
it, then it collapsed or flipped (`docs/HOLDOUT_20260809.md`). That is the
credit-richness identity — for a held-to-expiry credit spread, return on
capital is close to a function of the credit received, and the credit is
close to a function of implied vol, so anything correlated with vol inherits
an IC that carries no information of its own.

A naive multi-feature model built on the same 18 features would just
rediscover that identity in a different shape. So the control is INSIDE this
module, not applied after the fact: every candidate feature is residualised
against the same two controls (`_residual_values`, the exact function
`residual_ic` uses) BEFORE it is allowed to enter the combined score. What
gets combined is "the part of each feature that isn't credit richness", not
the raw feature.

Ridge over lasso: several candidates here are known-collinear proxies for
"how volatile is this name" (`atm_iv`, `rv`, `vol_of_vol`, `iv_rank`). Ridge
shrinks a correlated group together; lasso would pick one of them almost
arbitrarily and call the others zero, which is a worse basis for combining
weak, correlated signals.

THE LIMITATION, stated rather than hidden: `_residual_values` computes ranks
and residuals WITHIN whatever trade set it is given — that is how
`residual_ic` already behaves, and this module keeps that convention rather
than inventing a different one. So scoring a held-out set does not replay a
frozen rank-transform from the fit sample; only the ridge WEIGHTS on the
standardised residual columns are frozen and carried over. Read a holdout
result as "do these combination weights, chosen on one sample, still line up
with return on a different sample" — not as a fully leak-free pipeline in
every respect.

See `docs/PREREG_ENSEMBLE_20260905.md` for the frozen design this
implements. This module fits and scores; it is never called against holdout
data by anything in this PR — that is deliberately a separate, later step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.alloc.attribution import _sps  # guarded import, matches attribution.py
from src.alloc.attribution import (MIN_TRADES, RESIDUAL_CONTROLS,
                                   _clustered_t, _residual_values)

# Log-spaced: ridge's effective degrees of freedom drops fast near 0 and
# levels off past ~100 for standardised columns at this sample size, so a
# log grid probes the useful range without wasting trials on either tail.
DEFAULT_ALPHAS: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_N_BLOCKS = 6


@dataclass(frozen=True)
class EnsembleModel:
    """A frozen ridge combination. `predict` is `score_ensemble`, not a
    method, so scoring stays a pure function of (model, trades) — the model
    itself carries no reference back to the data it was fit on."""
    features: Tuple[str, ...]       # order matches coef/feature_mean/feature_std
    controls: Tuple[str, ...]
    alpha: float
    coef: Tuple[float, ...]
    intercept: float
    feature_mean: Tuple[float, ...]  # standardisation stats, frozen at fit time
    feature_std: Tuple[float, ...]
    n_fit: int
    cv_ic_by_alpha: Tuple[Tuple[float, Optional[float]], ...]


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float
              ) -> Tuple[np.ndarray, float]:
    """Closed-form ridge. `X` columns must already be standardised (mean 0,
    std 1) by the caller — an unstandardised column would be penalised in
    proportion to its native scale, not its actual redundancy."""
    yc = y - y.mean()
    k = X.shape[1]
    coef = np.linalg.solve(X.T @ X + alpha * np.eye(k), X.T @ yc)
    return coef, float(y.mean())


def _blocked_folds(n: int, n_blocks: int) -> List[np.ndarray]:
    """Row indices for each of `n_blocks` contiguous, non-overlapping folds.

    Contiguous, not shuffled: rows here are pre-sorted by entry date by the
    caller, and a shuffled fold would put trades from the same week on both
    sides of a split — the exact leak `split_by_time`'s docstring warns
    about, at the scale of a CV fold instead of the whole holdout.
    """
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]


def _design_matrix(trades: Sequence[Any], features: Sequence[str],
                   controls: Sequence[str]
                   ) -> Optional[Dict[str, Any]]:
    """Residualise every feature against `controls`, then intersect down to
    the trades every included feature could be measured on.

    A feature `_residual_values` cannot measure at all (too few trades,
    collinear with the controls, absent from the data) is DROPPED from the
    ensemble rather than failing the whole fit — the same "report what could
    not be measured" convention as the rest of this module, applied to
    dropping a column instead of returning None for one.
    """
    per_feature: Dict[str, Dict[str, Any]] = {}
    for f in features:
        r = _residual_values(trades, f, controls)
        if r is not None:
            per_feature[f] = r
    if not per_feature:
        return None

    common = None
    for r in per_feature.values():
        ids = {id(t) for t in r["keep"]}
        common = ids if common is None else (common & ids)
    if not common or len(common) < MIN_TRADES:
        return None

    used = sorted(per_feature)          # deterministic column order
    first = per_feature[used[0]]
    order = [i for i, t in enumerate(first["keep"]) if id(t) in common]
    keep = [first["keep"][i] for i in order]
    ys = np.asarray([first["ys"][i] for i in order], dtype=float)

    cols = []
    for f in used:
        r = per_feature[f]
        pos = {id(t): j for j, t in enumerate(r["keep"])}
        cols.append([float(r["resid"][pos[id(t)]]) for t in keep])
    X = np.column_stack(cols) if cols else np.empty((len(keep), 0))
    return {"features": used, "keep": keep, "X": X, "ys": ys}


def fit_ensemble(trades: Sequence[Any], features: Sequence[str],
                 controls: Sequence[str] = RESIDUAL_CONTROLS,
                 alphas: Sequence[float] = DEFAULT_ALPHAS,
                 n_blocks: int = DEFAULT_N_BLOCKS
                 ) -> Optional[EnsembleModel]:
    """Fit a ridge combination of `features` (residualised against
    `controls`) on `trades`, choosing alpha by blocked time-series CV
    WITHIN this sample only. Callers pass the in-sample window here and
    nothing else — see the module docstring and the frozen prereg doc.

    Returns None if fewer than `MIN_TRADES` trades survive residualisation,
    or if too few blocks would result (need at least 2 non-empty folds to
    cross-validate at all).
    """
    dm = _design_matrix(trades, features, controls)
    if dm is None or dm["X"].shape[1] == 0:
        return None
    keep, X, ys = dm["keep"], dm["X"], dm["ys"]

    order = np.argsort([str(t.entry_date) for t in keep], kind="stable")
    X, ys = X[order], ys[order]
    keep = [keep[i] for i in order]
    n = len(keep)
    blocks = [b for b in _blocked_folds(n, n_blocks) if b.size > 0]
    if len(blocks) < 2:
        return None

    mean, std = X.mean(axis=0), X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    Xs = (X - mean) / std_safe

    def _cv_ic(alpha: float) -> Optional[float]:
        ics = []
        for i, test_idx in enumerate(blocks):
            train_idx = np.concatenate([b for j, b in enumerate(blocks) if j != i])
            if train_idx.size < MIN_TRADES or test_idx.size < 2:
                continue
            coef, intercept = _ridge_fit(Xs[train_idx], ys[train_idx], alpha)
            pred = Xs[test_idx] @ coef + intercept
            if np.std(pred) == 0 or np.std(ys[test_idx]) == 0:
                continue
            ic = float(np.corrcoef(_rank(pred), _rank(ys[test_idx]))[0, 1])
            if ic == ic:                # not NaN
                ics.append(ic)
        return float(np.mean(ics)) if ics else None

    cv_scores = {float(a): _cv_ic(a) for a in alphas}
    scored = [(a, s) for a, s in cv_scores.items() if s is not None]
    if not scored:
        return None
    # Ties favour the LARGER alpha — more shrinkage, the more conservative
    # choice when the data cannot tell two regularisation strengths apart.
    best_ic = max(s for _, s in scored)
    alpha = max(a for a, s in scored if s == best_ic)

    coef, intercept = _ridge_fit(Xs, ys, alpha)
    return EnsembleModel(
        features=tuple(dm["features"]), controls=tuple(controls),
        alpha=alpha, coef=tuple(float(c) for c in coef),
        intercept=intercept,
        feature_mean=tuple(float(m) for m in mean),
        feature_std=tuple(float(s) for s in std_safe),
        n_fit=n, cv_ic_by_alpha=tuple(sorted(cv_scores.items())))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks


def score_ensemble(model: EnsembleModel, trades: Sequence[Any]
                   ) -> Tuple[List[Any], List[float]]:
    """Apply a frozen model to a (possibly different) trade set.

    Residualisation is recomputed WITHIN `trades` — see the module
    docstring's limitation note. Only `model.coef`/`intercept` and the
    standardisation stats are carried over unmodified from the fit sample.
    """
    dm = _design_matrix(trades, model.features, model.controls)
    if dm is None:
        return [], []
    # A holdout run may not measure every fit-time feature (e.g. a feature
    # with no variance in that window) — score only on what both windows
    # share, never by inventing a value for a dropped column.
    common = [f for f in model.features if f in dm["features"]]
    if not common:
        return [], []
    idx = [dm["features"].index(f) for f in common]
    X = dm["X"][:, idx]
    mean = np.array([model.feature_mean[model.features.index(f)] for f in common])
    std = np.array([model.feature_std[model.features.index(f)] for f in common])
    coef = np.array([model.coef[model.features.index(f)] for f in common])
    Xs = (X - mean) / std
    scores = Xs @ coef + model.intercept
    return dm["keep"], [float(s) for s in scores]


def ensemble_ic(trades: Sequence[Any], model: EnsembleModel,
                n_trials: int = 1) -> Dict[str, Any]:
    """Spearman IC (naive and day-clustered) of the ensemble score against
    realized return, mirroring `feature_ic`'s reporting shape so the two
    are directly comparable in the same table."""
    keep, scores = score_ensemble(model, trades)
    out: Dict[str, Any] = {"feature": "ensemble", "n": len(keep), "ic": None,
                           "p": None, "t": 0.0, "t_clustered": 0.0,
                           "n_trials": n_trials,
                           "n_features": len(model.features)}
    if len(keep) < MIN_TRADES or _sps is None:
        return out
    ys = [float(t.pnl or 0.0) / float(t.capital_at_risk) for t in keep]
    if len(set(scores)) < 2 or len(set(ys)) < 2:
        return out
    ic, p = _sps.spearmanr(scores, ys)
    if ic != ic:
        return out
    n = len(scores)
    denom = max(1.0 - ic * ic, 1e-12)
    out.update({
        "ic": round(float(ic), 4), "p": round(float(p), 4),
        "t": round(float(ic * np.sqrt((n - 2) / denom)), 3),
        "t_clustered": round(_clustered_t(keep, scores, ys), 3),
    })
    return out
