"""Leakage-controlled cross-validation and search-aware statistics.

Two jobs, both aimed at the same failure: believing a number that a large enough
search would have produced from noise.

CPCV builds many purged, embargoed train/test paths instead of one walk-forward
split, so the out-of-sample estimate has a distribution rather than a point.

Deflated Sharpe then discounts the result by how many configurations were tried,
and by skew and kurtosis — short premium's many-small-wins/rare-large-loss shape
inflates a naive Sharpe badly, which is exactly the shape that looks best right
before it fails.

PBO estimates how often the in-sample winner underperforms out of sample. A high
PBO on a good-looking strategy is the clearest available warning that the search
found the noise rather than the signal.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

DEFAULT_BLOCKS = 8
DEFAULT_K = 2
DEFAULT_EMBARGO = 5
_EULER = 0.5772156649015329


def cpcv_splits(n_samples: int, n_blocks: int = DEFAULT_BLOCKS,
                k: int = DEFAULT_K) -> List[Tuple[List[int], List[int]]]:
    """All C(n_blocks, k) train/test partitions over contiguous time blocks."""
    if n_samples <= 0 or n_blocks <= 0 or k <= 0 or k >= n_blocks:
        return []
    edges = np.linspace(0, n_samples, n_blocks + 1).astype(int)
    blocks = [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
    out = []
    for combo in itertools.combinations(range(n_blocks), k):
        test = [i for b in combo for i in blocks[b]]
        train = [i for b in range(n_blocks) if b not in combo for i in blocks[b]]
        out.append((train, test))
    return out


def purge_embargo(train_idx: Sequence[int], test_idx: Sequence[int],
                  holding_days: int,
                  embargo_days: int = DEFAULT_EMBARGO) -> List[int]:
    """Drop training samples that leak into the test block.

    Purge: a sample entered `holding_days` before the test block still has its
    outcome determined inside it.
    Embargo: samples just after the test block are correlated with its tail.
    """
    if not test_idx:
        return list(train_idx)
    lo, hi = min(test_idx), max(test_idx)
    purge_from = lo - max(0, holding_days)
    embargo_to = hi + max(0, embargo_days)
    return [i for i in train_idx if not (purge_from <= i <= embargo_to)]


def sharpe(returns: Union[Sequence[float], Any]) -> float:
    """Per-observation Sharpe.

    The zero-variance guard is relative, not `sd > 0`: a constant series has a
    floating-point standard deviation around 1e-18 rather than exactly zero, so
    an absolute test lets mean/sd explode to ~1e15 and report a flat series as
    the best strategy ever measured.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    scale = max(abs(float(r.mean())), 1.0)
    if sd <= 1e-12 * scale:
        return 0.0
    return float(r.mean() / sd)


def expected_max_sharpe(n_trials: int, trial_variance: float = 1.0) -> float:
    """Highest Sharpe a search of `n_trials` zero-skill strategies would produce.

    This is the bar a real result has to clear. Bailey & Lopez de Prado's
    expression for the expected maximum of N draws from a normal.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    v = math.sqrt(max(trial_variance, 1e-12))
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(v * ((1.0 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe(returns: Union[Sequence[float], Any], n_trials: int,
                    trial_variance: Optional[float] = None) -> float:
    """P(true Sharpe > 0) given the size of the search, plus skew and kurtosis.

    Returns a probability. Above 0.95 is a strong result; below 0.5 says the
    search alone could plausibly have produced this.

    `trial_variance` is the variance of the SHARPE ESTIMATES across trials, not
    of the returns. Defaulting it to 1.0 is a units error that makes the bar
    roughly sqrt(n) times too high and rejects everything: for a
    per-observation Sharpe over n observations the sampling variance is ~1/n.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    sr = sharpe(r)
    if sr == 0.0:
        return 0.0
    if trial_variance is None:
        trial_variance = 1.0 / n
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, trial_variance)

    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(
        is_matrix: Sequence[Sequence[float]]) -> float:
    """Fraction of CPCV paths where the in-sample best is below median OOS.

    `is_matrix` is [path][strategy] of in-sample scores; the same ordering is
    assumed for the out-of-sample half via `pbo_from_pairs` below.
    """
    m = np.asarray(is_matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return 0.0
    return float((m > 0).mean())


def pbo_from_pairs(pairs: Sequence[Tuple[Sequence[float], Sequence[float]]]
                   ) -> float:
    """PBO from (in_sample_scores, out_of_sample_scores) per path.

    For each path: pick the strategy that won in sample, find its OOS rank, and
    count it as an overfit event when it lands in the bottom half.
    """
    events = []
    for is_scores, oos_scores in pairs:
        a = np.asarray(is_scores, dtype=float)
        b = np.asarray(oos_scores, dtype=float)
        if a.size == 0 or a.size != b.size:
            continue
        best = int(np.argmax(a))
        rank = float((b < b[best]).sum()) / max(1, b.size - 1)
        events.append(1.0 if rank < 0.5 else 0.0)
    return float(np.mean(events)) if events else 0.0
