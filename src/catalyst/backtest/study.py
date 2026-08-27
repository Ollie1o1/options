"""Effect sizes with bootstrap confidence intervals.

A CI CONTAINING ZERO IS "NO EVIDENCE". It is not a weaker version of a
finding, it is not re-sliced by horizon until it clears, and it does not become
a result because the point estimate has the hoped-for sign. This repo's
pre-registered rank race returned exactly this outcome across every candidate
key, and the correct response was to stop ranking.

Percentile bootstrap on the difference of means, seeded so a rerun reproduces.
No scipy — the venv is deliberately thin and `random` is sufficient here.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

MIN_ARM = 8

#: Minimum CLUSTERS per arm. The row count is not the sample size when rows
#: repeat within a cluster: 60 rows from 3 tickers is 3 observations. Counting
#: rows is the error that let the ranker test's 2,137 rows read as 150% of a
#: target its 659 clusters had not reached.
MIN_CLUSTERS = 8

#: One observation: (value, cluster label, which arm it belongs to).
Observation = Tuple[float, str, bool]


@dataclass(frozen=True)
class Result:
    key: str
    label: str
    n_true: int
    n_false: int
    mean_true: float
    mean_false: float
    diff: float
    ci_lo: float
    ci_hi: float
    verdict: str
    # Clusters behind each arm. Default 0 so the unclustered constructors are
    # unchanged; a 0 here means "not measured", never "no clusters".
    k_true: int = 0
    k_false: int = 0


def bootstrap_ci(a: Sequence[float], b: Sequence[float], seed: int = 0,
                 iters: int = 2000) -> Tuple[float, float]:
    """Percentile CI for mean(a) - mean(b). Deterministic for a given seed."""
    rng = random.Random(seed)
    la, lb = list(a), list(b)
    diffs: List[float] = []
    for _ in range(iters):
        ra = [rng.choice(la) for _ in la]
        rb = [rng.choice(lb) for _ in lb]
        diffs.append(statistics.fmean(ra) - statistics.fmean(rb))
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return lo, hi


def _by_cluster(obs: Sequence[Observation]
                ) -> Dict[str, Tuple[List[float], List[float]]]:
    """Cluster label -> (values in the True arm, values in the False arm).

    A cluster that appears in BOTH arms stays one entry, because it is one
    unit. Splitting it per arm would let the bootstrap draw the same ticker
    independently into each side and destroy exactly the correlation this
    estimator exists to preserve.
    """
    out: Dict[str, Tuple[List[float], List[float]]] = {}
    for value, cluster, arm in obs:
        a, b = out.setdefault(str(cluster), ([], []))
        (a if arm else b).append(float(value))
    return out


def cluster_bootstrap_ci(obs: Sequence[Observation], seed: int = 0,
                         iters: int = 2000) -> Tuple[float, float]:
    """Percentile CI for mean(True arm) - mean(False arm), resampling CLUSTERS.

    The unit of resampling is the cluster, not the row. In this study the
    outcome is a forward return on the TICKER — `outcomes_for` never sees an
    nct_id — so every trial on one ticker at one vintage contributes a
    byte-identical value. Resampling rows counts those copies as independent
    evidence and returns an interval too narrow by roughly the square root of
    the design effect, which on the 2026-08-26 panel was 1.8x to 2.9x.

    Duplicating a cluster's rows therefore cannot narrow this interval, which
    is the property the tests pin.
    """
    grouped = _by_cluster(obs)
    clusters = sorted(grouped)
    if not clusters:
        return 0.0, 0.0
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(iters):
        drawn = [grouped[rng.choice(clusters)] for _ in clusters]
        a = [v for arm_true, _ in drawn for v in arm_true]
        b = [v for _, arm_false in drawn for v in arm_false]
        if not a or not b:
            # A draw that empties an arm carries no difference to report.
            # Skipped rather than scored as zero, which would drag the
            # interval toward a difference the data never showed.
            continue
        diffs.append(statistics.fmean(a) - statistics.fmean(b))
    if not diffs:
        return 0.0, 0.0
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return lo, hi


def compare_clustered(obs: Sequence[Observation], key: str, label: str,
                      seed: int = 0) -> Result:
    """Difference of means with a CLUSTER-robust CI and an explicit verdict.

    UNDERPOWERED is decided on the cluster count, not the row count.
    """
    grouped = _by_cluster(obs)
    a = [v for arm_true, _ in grouped.values() for v in arm_true]
    b = [v for _, arm_false in grouped.values() for v in arm_false]
    k_true = sum(1 for arm_true, _ in grouped.values() if arm_true)
    k_false = sum(1 for _, arm_false in grouped.values() if arm_false)
    mean_a = statistics.fmean(a) if a else 0.0
    mean_b = statistics.fmean(b) if b else 0.0

    if k_true < MIN_CLUSTERS or k_false < MIN_CLUSTERS:
        return Result(key, label, len(a), len(b), mean_a, mean_b,
                      mean_a - mean_b, 0.0, 0.0, "UNDERPOWERED",
                      k_true, k_false)
    lo, hi = cluster_bootstrap_ci(obs, seed=seed)
    verdict = "NO EVIDENCE" if lo <= 0.0 <= hi else "SEPARATES"
    return Result(key, label, len(a), len(b), mean_a, mean_b,
                  mean_a - mean_b, lo, hi, verdict, k_true, k_false)


def not_computable(key: str, label: str, reason: str) -> Result:
    """A declared hypothesis the available data cannot test.

    Reported explicitly rather than omitted: "we could not run this" and "this
    came back empty" are different claims, and dropping a pre-registered
    hypothesis from the output silently narrows the study after the fact.
    """
    return Result(key, f"{label} — {reason}", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  "NOT COMPUTABLE")


def compare(values_true: Sequence[float], values_false: Sequence[float],
            key: str, label: str, seed: int = 0) -> Result:
    """Difference of means with a CI and an explicit verdict."""
    a, b = list(values_true), list(values_false)
    if len(a) < MIN_ARM or len(b) < MIN_ARM:
        mean_a = statistics.fmean(a) if a else 0.0
        mean_b = statistics.fmean(b) if b else 0.0
        return Result(key, label, len(a), len(b), mean_a, mean_b,
                      mean_a - mean_b, 0.0, 0.0, "UNDERPOWERED")
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    lo, hi = bootstrap_ci(a, b, seed=seed)
    verdict = "NO EVIDENCE" if lo <= 0.0 <= hi else "SEPARATES"
    return Result(key, label, len(a), len(b), mean_a, mean_b,
                  mean_a - mean_b, lo, hi, verdict)
