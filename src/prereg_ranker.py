"""Statistics for the pre-registered ranker test.

The question: among gate survivors, does `ev_net` order `pnl_pct`? It is
answerable only because PR #50 made entry selection random — a cohort selected
by rule X cannot test rule X, which is why the 1,005-row ledger never could.

Every threshold lives in `docs/PREREG_RANKER_TEST.md` and is read from there.
This module computes; it does not decide.

See docs/PREREG_RANKER_TEST_SPEC.md.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A cell of 2 contributes demeaned ranks of exactly +/-0.5 on both variables,
# so it adds a +/-1 correlation pair carrying almost no information while
# counting as two observations. Pre-registered analysis parameter.
MIN_CELL_ROWS = 3


def demeaned_ranks(df: pd.DataFrame, cols: Sequence[str],
                   cell_cols: Sequence[str]) -> pd.DataFrame:
    """Rank each column within its cell, then subtract the cell mean rank.

    The demeaning IS the pairing: it removes every between-cell difference, so
    only ordering *within* a (day, strategy) cell can contribute. Without it a
    feature that merely separates strategies looks predictive — the artifact
    that made carry read +0.104 whole-book while running -0.282 inside Iron
    Condor.

    Cells below `MIN_CELL_ROWS` are dropped.
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    work = df.dropna(subset=list(cols) + list(cell_cols)).copy()
    if len(work) == 0:
        return work

    sizes = work.groupby(list(cell_cols))[cols[0]].transform("size")
    work = work[sizes >= MIN_CELL_ROWS].copy()
    if len(work) == 0:
        return work

    for col in cols:
        ranked = work.groupby(list(cell_cols))[col].rank(method="average")
        cell_mean = ranked.groupby(
            [work[c] for c in cell_cols]).transform("mean")
        work[f"_dm_{col}"] = ranked - cell_mean
    return work


def rank_ic(df: pd.DataFrame, feature: str, outcome: str,
            cell_cols: Sequence[str]) -> Optional[float]:
    """Pooled correlation of within-cell demeaned ranks, or None if undefined."""
    work = demeaned_ranks(df, [feature, outcome], cell_cols)
    if work is None or len(work) < 2:
        return None
    x = work[f"_dm_{feature}"].to_numpy(dtype="float64")
    y = work[f"_dm_{outcome}"].to_numpy(dtype="float64")
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def cluster_bootstrap_ci(df: pd.DataFrame, feature: str, outcome: str,
                         cell_cols: Sequence[str], cluster_col: str,
                         n_boot: int = 10000, alpha: float = 0.05,
                         seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for the rank IC, resampling whole clusters.

    The same contract recorded on five scans is one piece of information, not
    five. Resampling rows would treat it as five and report an interval far too
    narrow — the same overcounting that made the `n >= 50` gate trigger
    systematically early (ICC 0.08-0.11, design effect 1.23-1.27).

    Cells are re-derived inside each resample, because a resampled frame has
    different cell composition.
    """
    if df is None or len(df) == 0 or cluster_col not in df.columns:
        return (None, None)

    groups = {k: g for k, g in df.groupby(cluster_col)}
    keys = list(groups)
    if not keys:
        return (None, None)

    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(int(n_boot)):
        drawn = rng.integers(0, len(keys), size=len(keys))
        sample = pd.concat([groups[keys[i]] for i in drawn], ignore_index=True)
        ic = rank_ic(sample, feature, outcome, cell_cols)
        if ic is not None and ic == ic:
            stats.append(ic)

    if len(stats) < 2:
        return (None, None)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ── Power arithmetic ─────────────────────────────────────────────────────────

def required_effective_n(target_ic: float, power: float = 0.80,
                         alpha: float = 0.05) -> float:
    """Effective observations needed to detect `target_ic`, by Fisher-z.

        n = ((z_{alpha/2} + z_{power}) / atanh(rho))^2 + 3

    This is the arithmetic the LC gate got wrong. It demanded `IC >= 0.08 AND
    p < 0.05` at a trigger of n >= 50, when detecting 0.08 needs n ~ 1224 — so
    the 0.08 was decorative and the p-clause silently bound. Here n* is powered
    FOR the threshold, which makes the two conditions unable to disagree.
    """
    from scipy.stats import norm
    rho = abs(float(target_ic))
    if not 0 < rho < 1:
        raise ValueError("target_ic must be strictly between 0 and 1")
    z_a = float(norm.ppf(1 - alpha / 2))
    z_b = float(norm.ppf(power))
    return ((z_a + z_b) / math.atanh(rho)) ** 2 + 3


def icc_oneway(df: pd.DataFrame, value_col: str,
               cluster_col: str) -> Optional[float]:
    """One-way random-effects ICC by ANOVA, clamped to [0, 1].

    A negative ANOVA estimate means "no more agreement within clusters than
    between", which is an ICC of 0 rather than a negative correlation.
    """
    if df is None or len(df) == 0:
        return None
    work = df.dropna(subset=[value_col, cluster_col])
    groups = [g[value_col].to_numpy(dtype="float64")
              for _, g in work.groupby(cluster_col)]
    groups = [g for g in groups if len(g)]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return None

    grand = float(np.mean(np.concatenate(groups)))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups)
    ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    # Mean cluster size for the unbalanced case.
    sizes = np.array([len(g) for g in groups], dtype="float64")
    m0 = (n - (float(np.sum(sizes ** 2)) / n)) / (k - 1)
    denom = ms_between + (m0 - 1) * ms_within
    if m0 <= 0 or denom == 0:
        return None
    icc = (ms_between - ms_within) / denom
    return float(min(1.0, max(0.0, icc)))


def design_effect(df: pd.DataFrame, value_col: str,
                  cluster_col: str) -> Optional[float]:
    """1 + (mean cluster size - 1) * ICC.

    When every cluster is a singleton the design effect is exactly 1 whatever
    the ICC, and `icc_oneway` correctly declines to estimate one — there is no
    within-cluster variance to estimate it from. Returning None there would
    propagate a None into the power arithmetic on exactly the early data this
    is first run against, so the degenerate case is answered directly.
    """
    if df is None or len(df) == 0:
        return None
    work = df.dropna(subset=[value_col, cluster_col])
    k = int(work[cluster_col].nunique())
    if not k:
        return None
    mean_size = len(work) / k
    if mean_size <= 1.0:
        return 1.0

    icc = icc_oneway(df, value_col, cluster_col)
    if icc is None:
        return None
    return float(1.0 + (mean_size - 1.0) * icc)


def effective_n(nominal: float, design_effect_value: float) -> float:
    """Nominal observations divided by the design effect."""
    de = float(design_effect_value)
    return float(nominal) / de if de > 0 else float(nominal)
