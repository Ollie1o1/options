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
from typing import List, Sequence, Tuple

MIN_ARM = 8


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
