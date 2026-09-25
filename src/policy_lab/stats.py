"""Paired statistics for policy comparisons.

Both arms replay the same entry on the same path, so the underlying's move is
common to both and cancels in `d = r_alt - r_base`. Testing `d` rather than `r`
is what turns an unanswerable question (n ~ 3,500 at sd 43.65%) into a tractable
one — but only if the pairing actually removes variance, which
`variance_reduction` measures instead of assuming.

`src/prereg_ranker.py::cluster_bootstrap_ci` is NOT reusable here: it computes a
rank IC, and this needs a bootstrap on a paired mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.policy_lab.types import Outcome, PricePath
from src.walk_forward import Trade, purge_overlapping

# Below this ratio of sd(d) to sd(r_base) the pairing is doing its job. At or
# above it the variance reduction is too small to rescue the power problem and
# every result in the run is labelled UNDERPOWERED.
PAIRING_INEFFECTIVE_RATIO = 0.5

CLUSTER_COL = "cluster"

# Matches MIN_N in src/alloc/report.py.
MIN_CLUSTERS = 20

# Family-wise error rate bar for "the best policy in this grid is real".
MAX_FWER_P = 0.05


def paired_frame(paths: Sequence[PricePath],
                 base_outcomes: Dict[str, Outcome],
                 alt_outcomes: Dict[str, Outcome]) -> pd.DataFrame:
    """One row per position present in BOTH arms.

    The clustering unit is `(symbol, entry_date)`. Rows are never the unit:
    the same underlying on the same day is one piece of information however
    many contracts it produced.
    """
    rows = []
    for path in paths:
        base = base_outcomes.get(path.position_id)
        alt = alt_outcomes.get(path.position_id)
        if base is None or alt is None:
            continue
        rows.append({
            "position_id": path.position_id,
            "symbol": path.symbol,
            "entry_date": path.entry_date,
            CLUSTER_COL: f"{path.symbol}|{path.entry_date}",
            "r_base": base.pnl_frac_car,
            "r_alt": alt.pnl_frac_car,
            "d": alt.pnl_frac_car - base.pnl_frac_car,
        })
    return pd.DataFrame(rows, columns=[
        "position_id", "symbol", "entry_date", CLUSTER_COL,
        "r_base", "r_alt", "d"])


def cluster_bootstrap_mean_ci(df: pd.DataFrame, value_col: str,
                              cluster_col: str = CLUSTER_COL,
                              n_boot: int = 10000, alpha: float = 0.05,
                              seed: int = 0
                              ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for the mean of `value_col`, resampling whole clusters.

    Resampling rows would treat one ticker-day's many contracts as many
    independent observations and report an interval far too narrow. That
    overcounting is the mistake this repo has made three times.
    """
    if df is None or len(df) == 0 or cluster_col not in df.columns:
        return (None, None)
    groups = {k: g[value_col].to_numpy(dtype="float64")
              for k, g in df.groupby(cluster_col)}
    keys = list(groups)
    if not keys:
        return (None, None)

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        drawn = rng.integers(0, len(keys), size=len(keys))
        vals = np.concatenate([groups[keys[i]] for i in drawn])
        if vals.size:
            stats.append(float(vals.mean()))
    if len(stats) < 2:
        return (None, None)
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def cluster_bootstrap_mean_ci_many(
        per_policy_d: Dict[str, Sequence[float]],
        cluster_ids: Sequence[str],
        n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
        ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """`cluster_bootstrap_mean_ci`, vectorized across many policies at once.

    PERFORMANCE (2026-09-25): the real sweep scores a ~750-policy grid, and
    the family-wise screen at `report.py:440` lets nearly all of them through
    to bootstrapping (the grid's cells are heavily correlated, so once the
    best clears, most others do). At `n_boot=10000` the single-policy
    `cluster_bootstrap_mean_ci` costs ~4s each in a per-boot Python loop —
    750 * 4s = 48 minutes, measured as the whole cost of a 105,392-path,
    1,870-cluster run. This function computes every policy's CI from ONE set
    of resampled cluster weights instead of resampling once per policy.

    Every `d` vector in `per_policy_d` must be row-aligned to the single
    shared `cluster_ids` array (same length, same order) — this is a
    stricter contract than the single-policy function, which groups each
    policy's own frame independently, and callers with ragged per-policy row
    sets must batch policies that share an alignment before calling this
    (see `report.py::sweep`, which batches by row signature for exactly this
    reason: two grid policies can drop different positions when a path
    opened at `entry_price == 0.0` and only one of the two arms a
    take-profit/stop — see `replay.py`).

    Size-weighted resampling (`cluster_bootstrap_mean_ci`'s docstring): a
    resampled mean is `sum(selected clusters' sums) / sum(selected
    clusters' counts)`. Precompute, once, a sums matrix `S` (n_policies x
    n_clusters) and a shared counts vector `c` (n_clusters); for `n_boot`
    draws, a multinomial weight matrix `W` (n_clusters x n_boot) counts how
    many times each cluster was drawn per replicate (matching the
    single-policy function's `rng.integers(0, n_clusters, size=n_clusters)`
    draw of n_clusters cluster-picks per boot in distribution, though NOT in
    RNG call sequence — the two do not produce bit-identical resamples for
    the same seed; see `test_cluster_bootstrap_many_matches_single_
    statistically` in `tests/policy_lab/test_stats_paired.py`, which is why
    this repo asserts statistical rather than exact numerical equivalence).
    `M = (S @ W) / (c @ W)` gives every policy's resampled means at once.

    `W` at realistic scale (1,870 clusters x 10,000 boots, float64) is
    ~150MB, so the `n_boot` dimension is chunked rather than materialized
    whole.
    """
    if not per_policy_d:
        return {}
    ids = np.asarray(cluster_ids)
    n_rows = ids.shape[0]
    names = list(per_policy_d)
    if n_rows == 0:
        return {name: (None, None) for name in names}

    d_matrix = np.empty((len(names), n_rows), dtype="float64")
    for i, name in enumerate(names):
        d = np.asarray(per_policy_d[name], dtype="float64")
        if d.shape[0] != n_rows:
            raise ValueError(
                f"policy {name!r} has {d.shape[0]} d-values but cluster_ids "
                f"has {n_rows}; every policy must be aligned to the same "
                "cluster_ids row order")
        d_matrix[i] = d

    uniq_clusters, inverse = np.unique(ids, return_inverse=True)
    n_clusters = int(uniq_clusters.shape[0])
    if n_clusters == 0:
        return {name: (None, None) for name in names}

    # S[i, k]: policy i's sum of d over rows in cluster k.
    inverse = inverse.astype(np.intp, copy=False)
    S = np.zeros((len(names), n_clusters), dtype="float64")
    for i in range(len(names)):
        np.add.at(S[i], inverse, d_matrix[i])
    # c[k]: row count of cluster k — shared, because every policy's d vector
    # is aligned to the same `cluster_ids`.
    c = np.bincount(inverse, minlength=n_clusters).astype("float64")

    n_boot = int(n_boot)
    if n_boot < 1:
        return {name: (None, None) for name in names}
    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / n_clusters)
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)

    chunk_size = 1000
    replicate_chunks: List[np.ndarray] = []
    remaining = n_boot
    while remaining > 0:
        take = min(chunk_size, remaining)
        # (n_clusters, take): how many times each cluster was drawn, per
        # bootstrap replicate in this chunk.
        W = rng.multinomial(n_clusters, probs, size=take).T
        num = S @ W                 # (n_policies, take)
        den = c @ W                 # (take,) — never zero: every column of
                                     # W sums to n_clusters (>=1) over
                                     # clusters that each have count >= 1.
        replicate_chunks.append(num / den)
        remaining -= take
    replicates = np.concatenate(replicate_chunks, axis=1)  # (n_policies, n_boot)

    result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for i, name in enumerate(names):
        row = replicates[i]
        row = row[np.isfinite(row)]
        if row.size < 2:
            result[name] = (None, None)
            continue
        result[name] = (float(np.percentile(row, lo_q)),
                        float(np.percentile(row, hi_q)))
    return result


def variance_reduction(df: pd.DataFrame) -> float:
    """sd(d) / sd(r_base) — how much of the common factor the pairing removed.

    Returns 1.0 when the baseline has no dispersion to remove, which reports as
    "pairing did not help" rather than as a division by zero. An optimistic
    default (e.g. 0.0, "pairing worked perfectly") would hide a broken pairing
    behind the one input configuration that can't demonstrate it — 1.0 is the
    safe direction because it always reads as "did not help" and never as a
    false pass.
    """
    if df is None or len(df) < 2:
        return 1.0
    sd_base = float(np.std(df["r_base"].to_numpy(dtype="float64"), ddof=1))
    sd_d = float(np.std(df["d"].to_numpy(dtype="float64"), ddof=1))
    if sd_base == 0.0:
        return 1.0
    return sd_d / sd_base


def family_wise_p(cluster_means: np.ndarray, n_perm: int = 2000,
                  seed: int = 0) -> float:
    """Max-statistic sign-flip permutation p-value across a policy grid.

    `cluster_means` is a (n_policies, n_clusters) array: row g, column c is
    policy g's mean paired difference within cluster c. Under the null a
    policy's differences are symmetric about zero per cluster, so flipping a
    cluster's sign is an exchange the null permits. Flipping whole clusters
    (a single sign per COLUMN, applied to every policy) preserves the
    correlation between policies, which is what makes the max statistic the
    correct family-wise correction rather than a per-policy one.

    Returns the fraction of permutations whose max |mean| across policies is
    at least the observed max — the family-wise error rate for the claim
    "the best policy in this grid is real".

    This replaces the deflated Sharpe as the promotion gate. A deflated Sharpe
    treats each row as an independent time period; this design's paired
    difference is cross-sectional (the same path scored two ways, with the
    market factor already removed), and the ticker-day cluster — not the row,
    and not a calendar period — is the unit of information.
    """
    arr = np.asarray(cluster_means, dtype="float64")
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return 1.0
    n_clusters = arr.shape[1]
    obs = float(np.abs(arr.mean(axis=1)).max())
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype="float64")
    for i in range(int(n_perm)):
        s = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        null[i] = np.abs((arr * s).mean(axis=1)).max()
    return float((null >= obs).mean())


def embargo_purge(train: List[Trade], test: List[Trade],
                  embargo_days: int = 5) -> List[Trade]:
    """Purge overlapping trades, then drop a gap after the test window.

    `src/walk_forward.py::purge_overlapping` removes trades whose holding
    period overlaps the test block, but leaves a trade entered the day after it
    closes — which is still priced off the same autocorrelated tape. The
    embargo default of 5 days is at or above the 95th percentile holding period
    on this book (median hold is 1.5-3.5 days for every strategy except Iron
    Condor).
    """
    kept = purge_overlapping(train, test)
    if not test or embargo_days <= 0:
        return kept
    hi = max(date.fromisoformat(str(t.exit_date)[:10]) for t in test)
    barrier = hi + timedelta(days=int(embargo_days))
    min_test_entry = min(date.fromisoformat(str(x.entry_date)[:10])
                         for x in test)
    return [t for t in kept
            if date.fromisoformat(str(t.entry_date)[:10]) > barrier
            or date.fromisoformat(str(t.exit_date)[:10]) < min_test_entry]


def leave_one_symbol_out_stable(df: pd.DataFrame) -> bool:
    """Does the sign of the mean difference survive dropping any one symbol?

    Bull Put looked significant at t=3.48 until MU and AMD — 68% of its dollars
    during a semiconductor melt-up — were removed, at which point it fell to
    t=1.61. A result one ticker can overturn is a result about that ticker.

    PERFORMANCE (2026-09-25): rewritten to compute every symbol's leave-one-out
    mean from one groupby, instead of re-filtering the whole `df` per symbol.
    The original `for sym in symbols: df[df["symbol"] != sym]` is
    O(n_symbols * n_rows); on the real Corpus A Bull Put frame (105,392 rows,
    hundreds of distinct symbols) that took 70.9s of a 118s profiled 60-policy
    sweep (cProfile-measured) — extrapolated across the full 750-policy grid
    this alone made the CLI's smoke test fail to finish inside an hour. This
    version is mathematically identical (leave-one-out mean recovered as
    `(total_sum - symbol_sum) / (total_n - symbol_n)`, verified against the
    row-filtering version — see `tests/policy_lab/test_stats_verdict.py`'s
    `TestLeaveOneSymbolOut`, unchanged and still passing) but is
    O(n_rows + n_symbols).
    """
    if df is None or len(df) == 0 or "symbol" not in df.columns:
        return False
    grp = df.groupby("symbol")["d"].agg(["sum", "count"])
    if len(grp) < 2:
        return False
    total_sum = float(df["d"].sum())
    total_n = len(df)
    full = total_sum / total_n
    if full == 0.0:
        return False
    rest_n = total_n - grp["count"]
    if (rest_n <= 0).any():
        return False
    rest_mean = (total_sum - grp["sum"]) / rest_n
    return bool((np.sign(rest_mean.to_numpy()) == np.sign(full)).all())


@dataclass(frozen=True)
class PolicyResult:
    """Everything the promotion bar needs to judge one knob setting.

    `dsr` and `pbo` are still computed and carried as diagnostics — n_eff for
    this corpus is 9 regardless of row count (see `family_wise_p`), so `dsr`
    cannot clear 0.95 by construction and is NOT a gate. `fwer_p` is the gate.
    `skew_d` is carried for the report, not gated on: the sign-flip null in
    `family_wise_p` assumes symmetry per cluster, and a strongly skewed `d`
    would make that test anti-conservative, which the report needs to show
    even though this function does not act on it.
    """
    policy_name: str
    n_clusters: int
    mean_d_cross: float
    mean_d_surface: float
    ci_lo: float
    ci_hi: float
    dsr: float
    pbo: float
    fwer_p: float
    skew_d: float
    beats_all_nulls: bool
    corpus_b_sign_agrees: bool
    loso_stable: bool
    variance_ratio: float


def policy_verdict(result: PolicyResult) -> str:
    """`promote`, `reject`, `insufficient` or `underpowered`.

    `underpowered` is checked FIRST: if the pairing did not remove variance,
    no other condition here is trustworthy, and reporting `reject` would claim
    a measurement that was never made.

    `insufficient` is deliberately not `reject` — "we could not measure this"
    and "we measured it and it failed" are different claims.

    The gate is `fwer_p < MAX_FWER_P`, a max-statistic sign-flip permutation
    test across the policy grid — NOT `dsr`/`pbo`. `dsr` requires an
    independent-observation count (`effective_n`) that is 9 for every
    strategy on this corpus regardless of row count, because it is a function
    of the 26-day window and the 3-11 day holding period; at n_eff=9 a single
    hypothesis with zero multiple-testing penalty already fails 0.95. `pbo` is
    still carried and printed (unlike `src/alloc/report.py::promotion_verdict`,
    whose docstring records that its PBO key was never set and so never
    fired), but neither `dsr` nor `pbo` gates here.
    """
    if result.variance_ratio >= PAIRING_INEFFECTIVE_RATIO:
        return "underpowered"
    if result.n_clusters < MIN_CLUSTERS:
        return "insufficient"
    if result.mean_d_cross <= 0.0 or result.mean_d_surface <= 0.0:
        return "reject"
    if result.ci_lo <= 0.0:
        return "reject"
    if result.fwer_p >= MAX_FWER_P:
        return "reject"
    if not result.beats_all_nulls:
        return "reject"
    if not result.corpus_b_sign_agrees:
        return "reject"
    if not result.loso_stable:
        return "reject"
    return "promote"
