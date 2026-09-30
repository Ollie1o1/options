"""Run manifest, sweep runner and markdown emission.

A run without a manifest is not a result: without the git SHA, the grid size
and the clustering unit, a number on a page cannot be checked by anyone,
including its author a month later.

This module also carries the MANDATORY CORRECTIONS to the original Task 11
brief, established by measurement against the real databases (2026-09-22):

1. Corpus A's calibration threshold is 0.30, not the 0.10 default in
   `calibrate.py` — see `calibration_max_fail_rate`.
2. Corpus A's baseline is the RECORDER's policy, not the live book's — see
   `CORPUS_A_RECORDER_SHORT` / `corpus_a_baseline`.
3. The promotion gate is `fwer_p`, computed ONCE for the whole grid via a
   max-statistic sign-flip permutation test — not `dsr`/`pbo`, which are
   still computed and reported as diagnostics.
4. Only survivors (`fwer_p < MAX_FWER_P`), the null policies and the
   baseline get a bootstrap CI; every other policy's `ci_lo`/`ci_hi` is
   `float("nan")`, rendered as the literal string "not computed" (chosen
   over widening `PolicyResult` to `Optional[float]` so `stats.py`, which
   belongs to an earlier task, does not need to change).
5. `bounded_walk_benchmark` measures how much of a promoted policy's edge is
   just the zero floor on option prices, not skill.
6. The report leads with calibration reproduction (+ by-reason breakdown),
   the fraction of imputed-spread path points, and the measured detection
   floor.
7. Grid cells whose `time_exit_dte` is 21 or 28 are labelled as "exits at
   the first available mark" on Corpus A (median entry DTE 17).
8. `_LIMITS` carries the single-regime, non-ground-truth, path-depth,
   Short-Put-coverage and null-result caveats.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from datetime import date, timedelta, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew as _scipy_skew

from src.alloc.validate import deflated_sharpe, effective_n, pbo_from_pairs
from src.policy_lab.costs import CostModel
from src.policy_lab.calibrate import CalibrationReport
from src.policy_lab.calibrate import MAX_FAIL_RATE as CORPUS_B_MAX_FAIL_RATE
from src.policy_lab.calibrate import calibration_report
from src.policy_lab.policies import (
    ExitPolicy, LIVE_BASELINE_LONG, NULL_POLICIES, grid_cardinality,
)
from src.policy_lab.replay import replay
from src.policy_lab.stats import (
    CLUSTER_COL, MAX_FWER_P, PolicyResult, cluster_bootstrap_mean_ci_many,
    family_wise_p, leave_one_symbol_out_stable, paired_frame, policy_verdict,
    variance_reduction,
)
from src.policy_lab.types import PathPoint, PricePath

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def run_manifest(corpus: str, grid: Sequence[ExitPolicy], costs: str,
                 seed: int, row_counts: Dict[str, Any],
                 corpus_fingerprint: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Everything needed to reproduce or challenge this run.

    `corpus_fingerprint` records what the corpus looked like AT READ TIME —
    `max_ts` and `terminal_count` from `paths.corpus_a_fingerprint`/
    `corpus_b_fingerprint` — alongside `loaded`, the count that actually
    survived the loader's invariants (see `row_counts['loaded']`), so both
    the offered and the used counts are visible on the same manifest. The
    live scheduler writes `data/candidates.db` continuously while the lab
    reads it, so two runs' `rows: {'loaded': N}` alone cannot say whether
    they read the same corpus — this can. When the caller has not computed
    one (every pre-existing call site before this fingerprint existed),
    `max_ts`/`terminal_count` are `None` rather than absent, so the key is
    always present and always has all three sub-keys.
    """
    fp: Dict[str, Any] = dict(corpus_fingerprint) if corpus_fingerprint else {}
    fp.setdefault("max_ts", None)
    fp.setdefault("terminal_count", None)
    fp.setdefault("loaded", row_counts.get("loaded"))
    return {
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "n_trials": grid_cardinality(grid),
        "costs": costs,
        "seed": seed,
        "cluster_unit": "(symbol, entry_date)",
        "row_counts": dict(row_counts),
        "corpus_fingerprint": fp,
    }


def manifests_comparable(a: Dict[str, Any], b: Dict[str, Any]
                         ) -> Tuple[bool, str]:
    """Whether two manifests' numbers may be compared, and why not if not.

    A differing git SHA disqualifies on its own, even when the corpus lines
    up exactly, because the code that produced the two sets of numbers is
    different — see the module docstring's account of a real error this
    caused (a corpus-drift difference attributed to a policy knob). Beyond
    that, two runs are comparable only when their `corpus_fingerprint`s
    agree on `strategy`, `max_ts` and `terminal_count` for the same
    `corpus` (`"A"`/`"B"`) — anything else means the corpus moved between,
    or during, the two runs. `terminal_count` is compared with `!=`, never
    bare truthiness, so a corpus that legitimately offers zero terminal
    rows for a strategy is never confused with a missing count.
    """
    sha_a, sha_b = a.get("git_sha"), b.get("git_sha")
    if sha_a != sha_b:
        return False, f"different git sha: {sha_a} vs {sha_b}"

    corpus_a, corpus_b = a.get("corpus"), b.get("corpus")
    if corpus_a != corpus_b:
        return False, f"different corpus: {corpus_a} vs {corpus_b}"

    fp_a = a.get("corpus_fingerprint") or {}
    fp_b = b.get("corpus_fingerprint") or {}

    strat_a, strat_b = fp_a.get("strategy"), fp_b.get("strategy")
    if strat_a != strat_b:
        return False, f"different strategy: {strat_a} vs {strat_b}"

    ts_a, ts_b = fp_a.get("max_ts"), fp_b.get("max_ts")
    if ts_a != ts_b:
        return False, f"corpus drifted: max_ts {ts_a} -> {ts_b}"

    tc_a, tc_b = fp_a.get("terminal_count"), fp_b.get("terminal_count")
    if tc_a != tc_b:
        grew = tc_a is not None and tc_b is not None and tc_b > tc_a
        verb = "grew" if grew else "changed"
        return False, f"corpus {verb}: terminal_count {tc_a} -> {tc_b}"

    return True, "same git sha, same corpus fingerprint"


# ---------------------------------------------------------------------------
# Correction (1) + (2): Corpus A's calibration bar and baseline
# ---------------------------------------------------------------------------

# Measured 2026-09-22 reproduction on 6,000 real Corpus A Bull Put positions:
# 77.9% under mid costs, 41.6% under the recorder's own policy with imputed
# costs. 0.30 is comfortably below the mid-cost ceiling (still catches a
# harness regression) without pretending 90% was ever reachable here.
CORPUS_A_MAX_FAIL_RATE = 0.30

# Recovered from 3,817 real take-profits (exit/entry p95 = 0.495) and 1,831
# stops (min = 2.010) on Corpus A. NOT `LIVE_BASELINE_SHORT`: that policy
# arms `time_exit_dte=21` against a corpus whose median entry DTE is 17, so
# it would fire on the first interior point for most positions and make
# every alternative policy beat a degenerate baseline.
CORPUS_A_RECORDER_SHORT = ExitPolicy("corpusA_recorder", 0.50, 2.0, None, None)


def corpus_a_baseline(is_long: bool) -> ExitPolicy:
    """The baseline to replay Corpus A against.

    Only the short-premium recorder policy was recovered from the real
    corpus (see `CORPUS_A_RECORDER_SHORT`'s docstring above) — no equivalent
    measurement exists for Long Call/Long Put. For long premium this falls
    back to `LIVE_BASELINE_LONG`, which carries the same `time_exit_dte=21`
    risk the correction above exists to avoid; that gap is unresolved and is
    called out again in `_LIMITS`.
    """
    return LIVE_BASELINE_LONG if is_long else CORPUS_A_RECORDER_SHORT


# Recovered from the ledger's own recorded exits on the 45 real Corpus B
# Bull Put positions (2026-09-25, after the loader's short-leg/spread fix):
# 30 Take Profit exits at exit/credit median 0.346, max 0.498 -> threshold
# 0.50; 9 Stop Loss exits at exit/credit min 2.021, median 2.274 -> 2.0x.
# The 26 `Time Exit (Nd to expiry)` labels record the DTE the exit
# HAPPENED at (4, 7, 8, 11, 14, 15, 18, 21 -- heterogeneous), not a fixed
# rule, so they cannot be expressed as one `time_exit_dte` threshold;
# `hold_to_end` already covers them. NOT `LIVE_BASELINE_SHORT`: Corpus B's
# entry DTE is p5=8, median=16, p95=30, with 71% entering at DTE <= 21, so
# a `dte=21` knob exits on the first interior point for most of this
# corpus too -- the same degenerate-baseline defect `CORPUS_A_RECORDER_SHORT`
# exists to avoid for Corpus A, just recovered here for Corpus B from the
# ledger instead of assumed.
CORPUS_B_LEDGER_SHORT = ExitPolicy("corpusB_ledger", 0.50, 2.0, None, None)


def corpus_b_baseline(is_long: bool) -> ExitPolicy:
    """The baseline to replay Corpus B against.

    Mirrors `corpus_a_baseline`: only the short-premium policy was
    recovered from the real ledger (see `CORPUS_B_LEDGER_SHORT`'s docstring
    above) — no equivalent measurement exists for Long Call/Long Put. For
    long premium this falls back to `LIVE_BASELINE_LONG`, which carries the
    same `time_exit_dte=21` risk the correction above exists to avoid; that
    gap is unresolved, mirroring `corpus_a_baseline`'s.
    """
    return LIVE_BASELINE_LONG if is_long else CORPUS_B_LEDGER_SHORT


def calibration_max_fail_rate(corpus: str) -> float:
    """0.30 for Corpus A, the strict 0.10 default for Corpus B."""
    return CORPUS_A_MAX_FAIL_RATE if corpus.strip().upper() == "A" \
        else CORPUS_B_MAX_FAIL_RATE


def calibrate_for_run(paths: Sequence[PricePath], baseline: ExitPolicy,
                      sweep_costs: CostModel,
                      max_fail_rate: float) -> CalibrationReport:
    """Calibration checks the harness against the RECORDED policy, always at
    MID — never at `sweep_costs` (the sweep's own cost setting, `--costs` on
    the CLI, default `cross`).

    The recorder priced its exits at MID (`candidate_positions.exit_price`
    equals the exit-day mark mid in 89.6% of cases), so a mid replay is the
    only like-for-like comparison to what was actually recorded. Replaying
    at `cross`/`surface` instead measures the crossing cost, not whether the
    harness reproduces the record — measured on the real Corpus A: 79.7%
    reproduction at mid vs. 48.8% at cross, using the identical baseline and
    positions. `sweep_costs` is accepted (and intentionally ignored) so the
    call site is explicit about the two settings' independence rather than
    silently dropping the argument — do not "simplify" this by passing
    `sweep_costs` through; that is the exact regression this function exists
    to prevent (see `tests/policy_lab/test_report.py::
    TestCalibrationCostIndependence`).

    This is a different concern from `render_markdown`'s refusal to render a
    report built from `mid` costs: that guards the RESEARCH answer (must be
    `cross`); this guards the MACHINERY check against a mid-priced record
    (must be `mid`). Neither should be made to match the other.
    """
    del sweep_costs  # deliberately unused — see docstring
    return calibration_report(paths, baseline, CostModel.mid(),
                              max_fail_rate=max_fail_rate)


# ---------------------------------------------------------------------------
# Correction (6): spread-imputed fraction
# ---------------------------------------------------------------------------


def spread_imputed_fraction(paths: Sequence[PricePath]) -> float:
    """Fraction of all path points whose bid/ask were synthesized from mid.

    100% for Corpus A Bull Put: every cost figure for that cohort rests on
    the modelled `imputed_half_spread`, not an observed quote.
    """
    total = 0
    imputed = 0
    for p in paths:
        for pt in p.points:
            total += 1
            if pt.spread_imputed:
                imputed += 1
    return (imputed / total) if total else 0.0


# ---------------------------------------------------------------------------
# Correction (5): bounded-walk benchmark
# ---------------------------------------------------------------------------


def _calibrate_step_sd(paths: Sequence[PricePath]) -> Optional[float]:
    """sd of (q2.mid - q1.mid) / entry_price across consecutive real points."""
    deltas: List[float] = []
    for p in paths:
        if p.entry_price == 0.0:
            continue
        for a, b in zip(p.points, p.points[1:]):
            deltas.append((b.mid - a.mid) / p.entry_price)
    if len(deltas) < 2:
        return None
    return float(np.std(np.asarray(deltas, dtype="float64"), ddof=1))


def bounded_walk_benchmark(paths: Sequence[PricePath], base_policy: ExitPolicy,
                           alt_policy: ExitPolicy, costs: CostModel,
                           seed: int) -> float:
    """Mean paired difference `alt - base` on a SYNTHETIC bounded walk.

    Real option prices are bounded below by zero, so near that floor any
    early-exit policy gains a real advantage that has nothing to do with
    edge (see `tests/policy_lab/test_planted_effect.py::
    test_a_lower_barrier_gives_early_exit_a_real_advantage`, 11/20 false
    positives at sd=0.45 with a floor vs. 0/20 without one).

    Calibrates a per-step sd from the real paths, generates the same number
    of synthetic paths at the corpus's median point count with
    `max(0.01, prev + normal(0, sd))`, and replays both policies over them.
    Returns `float("nan")` when there is nothing to calibrate against.
    """
    sd = _calibrate_step_sd(paths)
    usable = [p for p in paths if p.entry_price != 0.0 and p.capital_at_risk > 0.0]
    if sd is None or not usable:
        return float("nan")

    median_len = int(round(float(np.median([len(p.points) for p in usable]))))
    median_len = max(median_len, 2)

    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for p in usable:
        entry_d = date.fromisoformat(p.entry_date[:10])
        start_dte = p.points[0].dte
        pts: List[PathPoint] = []
        frac = 1.0
        for i in range(median_len):
            if i == 0:
                mid = p.entry_price
            else:
                frac = max(0.01, frac + float(rng.normal(0.0, sd)))
                mid = frac * p.entry_price
            d = (entry_d + timedelta(days=i)).isoformat()
            pts.append(PathPoint(
                date=d, bid=mid, ask=mid, mid=mid, spot=None,
                dte=max(0, start_dte - i), spread_imputed=True))
        synth = PricePath(
            position_id=f"synthetic:{p.position_id}", symbol=p.symbol,
            strategy=p.strategy, entry_date=pts[0].date,
            entry_price=p.entry_price, capital_at_risk=p.capital_at_risk,
            is_credit=p.is_credit, points=tuple(pts),
            actual_exit_date=pts[-1].date, actual_pnl_frac=0.0,
            corpus=p.corpus)
        try:
            base_o = replay(synth, base_policy, costs)
            alt_o = replay(synth, alt_policy, costs)
        except ValueError:
            continue
        diffs.append(alt_o.pnl_frac_car - base_o.pnl_frac_car)

    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Sweep: fwer gate once for the grid, bootstrap only survivors, dsr/pbo as
# diagnostics, corpus-B sign check.
# ---------------------------------------------------------------------------


def _infer_corpus_b_baseline(baseline: ExitPolicy) -> ExitPolicy:
    """Which live baseline Corpus B should be replayed against.

    Corpus B always keeps its OWN recovered baseline
    (`CORPUS_B_LEDGER_SHORT`/`LIVE_BASELINE_LONG`), mirroring
    `corpus_b_baseline` — never `LIVE_BASELINE_SHORT`, which is a
    degenerate `time_exit_dte=21` baseline on this corpus (see
    `CORPUS_B_LEDGER_SHORT`'s docstring) and is never selected as a replay
    baseline for any corpus. `sweep` is not told the strategy family
    directly, so this infers it from the sweep's own `baseline` argument:
    every baseline this lab uses (`CORPUS_A_RECORDER_SHORT`,
    `CORPUS_B_LEDGER_SHORT`, `LIVE_BASELINE_LONG`) sets `stop_mult`, and
    only the long-premium convention expresses it as a fraction below 1.0
    (a multiple-of-credit stop is always >= 1.0).
    """
    if baseline.stop_mult is not None and baseline.stop_mult < 1.0:
        return LIVE_BASELINE_LONG
    return CORPUS_B_LEDGER_SHORT


def _maxT_pvalues(matrix: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Per-policy family-wise-corrected p-values, one per grid permutation run.

    Mirrors `stats.family_wise_p`'s own null-generation exactly (whole-cluster
    sign flips of the full `(n_policies, n_clusters)` matrix, applied to every
    policy at once, so the correlation between policies is preserved) so that
    the smallest value this returns always equals
    `family_wise_p(matrix, n_perm, seed)` — both read off the same null
    distribution, from the same seed. Correction (3) requires the permutation
    to run ONCE for the whole grid; this is that one run, generalized to give
    every policy its own family-wise-corrected p instead of only the winner's
    (`family_wise_p` itself only ever answers "is the single best policy
    real").
    """
    if matrix.size == 0 or matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.ones(matrix.shape[0] if matrix.ndim == 2 else 0)
    n_clusters = matrix.shape[1]
    obs = np.abs(matrix.mean(axis=1))
    rng = np.random.default_rng(seed)
    null_max: np.ndarray = np.empty(int(n_perm), dtype="float64")
    for i in range(int(n_perm)):
        s = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        null_max[i] = np.abs((matrix * s).mean(axis=1)).max()
    return np.array([(null_max >= o).mean() for o in obs])


def _paired(paths: Sequence[PricePath], base_out: Dict[str, Any],
           policy: ExitPolicy, costs: CostModel) -> pd.DataFrame:
    alt_out = {}
    for p in paths:
        if p.position_id not in base_out:
            continue
        try:
            alt_out[p.position_id] = replay(p, policy, costs)
        except ValueError:
            continue
    return paired_frame(paths, base_out, alt_out)


def _grid_pbo(cluster_means: Dict[str, "pd.Series"]) -> float:
    """A single, whole-grid PBO diagnostic from one chronological split.

    Real combinatorial PBO needs many in-sample/out-of-sample folds; at
    `effective_n=9` (see `deflated_sharpe`'s docstring) this corpus cannot
    support more than one meaningful split, so this uses exactly one: the
    common clusters, sorted by the entry_date embedded in the cluster key
    (`"symbol|entry_date"`), split in half. Diagnostic only — `policy_verdict`
    never reads `PolicyResult.pbo`.
    """
    if len(cluster_means) < 2:
        return 0.0
    common = sorted(
        set.intersection(*(set(s.index) for s in cluster_means.values())),
        key=lambda k: k.split("|", 1)[-1])
    if len(common) < 4:
        return 0.0
    mid = len(common) // 2
    is_c, oos_c = common[:mid], common[mid:]
    names = list(cluster_means)
    is_scores = [float(cluster_means[n].loc[is_c].mean()) for n in names]
    oos_scores = [float(cluster_means[n].loc[oos_c].mean()) for n in names]
    return pbo_from_pairs([(is_scores, oos_scores)])


def sweep(paths: Sequence[PricePath], grid: Sequence[ExitPolicy],
         baseline: ExitPolicy, costs: CostModel,
         corpus_b_paths: Optional[Sequence[PricePath]] = None,
         seed: int = 0, n_boot: int = 10000,
         n_perm: int = 2000) -> List[PolicyResult]:
    """Score every policy in `grid` (except `baseline`) against `baseline`.

    Bootstraps a CI only for survivors of the whole-grid family-wise
    permutation test (`fwer_p < MAX_FWER_P`), plus the null policies and the
    baseline's own comparison set — see correction (4). Every other policy's
    `ci_lo`/`ci_hi` is `float("nan")` ("not computed", never a literal 0.0).
    """
    if not paths:
        return []

    base_out = {}
    for p in paths:
        try:
            base_out[p.position_id] = replay(p, baseline, costs)
        except ValueError:
            continue
    if not base_out:
        return []

    policies = [pol for pol in grid if pol != baseline]
    per_policy_df: Dict[str, pd.DataFrame] = {}
    for i, policy in enumerate(policies):
        df = _paired(paths, base_out, policy, costs)
        if len(df) == 0:
            continue
        per_policy_df[policy.name] = df
        if (i + 1) % 100 == 0:
            print(f"[policy_lab] replay sweep: {i + 1}/{len(policies)} "
                  "policies", file=sys.stderr)

    if not per_policy_df:
        return []

    cluster_means = {name: df.groupby(CLUSTER_COL)["d"].mean()
                     for name, df in per_policy_df.items()}
    policy_names = list(cluster_means)
    common = sorted(set.intersection(
        *(set(s.index) for s in cluster_means.values())))
    if len(common) >= 2:
        matrix = np.array(
            [[cluster_means[name].loc[c] for c in common]
             for name in policy_names])
        # Correction (3): the permutation runs ONCE for the whole grid.
        # `family_wise_p` itself answers only "is the single best policy in
        # this grid real"; `_maxT_pvalues` reuses the identical null
        # distribution to give every policy its own family-wise-corrected p
        # (its minimum always equals `family_wise_p`'s scalar — see that
        # helper's docstring).
        grid_best_fwer_p = family_wise_p(matrix, n_perm=n_perm, seed=seed)
        p_by_policy = dict(zip(
            policy_names, _maxT_pvalues(matrix, n_perm=n_perm, seed=seed)))
    else:
        grid_best_fwer_p = 1.0
        p_by_policy = {name: 1.0 for name in policy_names}

    # Null policies, paired against the SAME baseline over the SAME corpus.
    null_means: Dict[str, float] = {}
    for np_pol in NULL_POLICIES:
        if np_pol == baseline:
            continue
        ndf = _paired(paths, base_out, np_pol, costs)
        if len(ndf):
            null_means[np_pol.name] = float(ndf["d"].mean())

    # Corpus B sign check (independent corpus; always uses the LIVE baseline,
    # never the Corpus-A recorder — see `_infer_corpus_b_baseline`).
    b_means: Dict[str, float] = {}
    if corpus_b_paths:
        b_baseline = _infer_corpus_b_baseline(baseline)
        b_base_out = {}
        for p in corpus_b_paths:
            try:
                b_base_out[p.position_id] = replay(p, b_baseline, costs)
            except ValueError:
                continue
        if b_base_out:
            for policy in policies:
                bdf = _paired(corpus_b_paths, b_base_out, policy, costs)
                if len(bdf):
                    b_means[policy.name] = float(bdf["d"].mean())

    # Diagnostics: dsr (per policy, cluster-level Sharpe deflated by the
    # grid size and this corpus's effective_n) and skew_d. NEITHER gates.
    starts = [p.entry_date for p in paths]
    ends = [p.actual_exit_date for p in paths]
    n_eff = effective_n(starts, ends)
    n_trials = grid_cardinality(grid)
    dsr_by_policy: Dict[str, float] = {}
    skew_by_policy: Dict[str, float] = {}
    for name, s in cluster_means.items():
        arr = s.to_numpy(dtype="float64")
        dsr_by_policy[name] = (
            deflated_sharpe(arr, n_trials, n_eff) if arr.size >= 3 else 0.0)
        skew_by_policy[name] = float(_scipy_skew(arr)) if arr.size >= 3 else 0.0

    pbo_grid = _grid_pbo(cluster_means)

    survivors = {name for name, p in p_by_policy.items() if p < MAX_FWER_P}
    ci_needed = survivors | set(null_means)
    print(f"[policy_lab] permutation test done: {len(survivors)}/"
          f"{len(policy_names)} policies survive fwer_p<{MAX_FWER_P}",
          file=sys.stderr)

    # Vectorized bootstrap (2026-09-25): the single-policy
    # `cluster_bootstrap_mean_ci` costs ~4s each at n_boot=10000, and nearly
    # every policy in `ci_needed` survives the family-wise screen (the grid's
    # cells are heavily correlated), so a per-policy loop here is the 48
    # minutes measured on the real 750-policy grid. `cluster_bootstrap_mean_
    # ci_many` resamples cluster weights ONCE and scores every policy against
    # them, but requires every policy's `d` vector to share one row-aligned
    # `cluster_ids` array — so policies are batched by their exact
    # position_id sequence first. In practice every `ci_needed` policy shares
    # one alignment (all are paired against the same `base_out`), so this
    # is a single batched call; the only way two policies diverge is a path
    # opened at `entry_price == 0.0` combined with only one policy arming a
    # take-profit/stop (see `replay.py`), which is handled correctly but
    # would cost a second, smaller batch rather than the full vectorization.
    ci_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    batches: Dict[Tuple[str, ...], List[str]] = {}
    for name in ci_needed:
        df = per_policy_df.get(name)
        if df is None:
            continue
        sig = tuple(df["position_id"])
        batches.setdefault(sig, []).append(name)
    for names in batches.values():
        rep_df = per_policy_df[names[0]]
        batch_cluster_ids = rep_df[CLUSTER_COL].to_numpy()
        batch_d = {name: per_policy_df[name]["d"].to_numpy(dtype="float64")
                  for name in names}
        ci_cache.update(cluster_bootstrap_mean_ci_many(
            batch_d, batch_cluster_ids, n_boot=n_boot, seed=seed))
    print(f"[policy_lab] bootstrap done: {len(ci_cache)} policies",
          file=sys.stderr)

    results: List[PolicyResult] = []
    for policy in policies:
        df = per_policy_df.get(policy.name)
        if df is None:
            continue
        mean_d = float(df["d"].mean())
        n_clusters = int(df[CLUSTER_COL].nunique())
        lo, hi = ci_cache.get(policy.name, (None, None))
        ci_lo = lo if lo is not None else float("nan")
        ci_hi = hi if hi is not None else float("nan")
        beats_nulls = (bool(null_means) and
                       all(mean_d > nv for nv in null_means.values()))
        b_mean = b_means.get(policy.name)
        sign_agrees = (b_mean is not None and mean_d != 0.0 and b_mean != 0.0
                       and (b_mean > 0) == (mean_d > 0))
        # `mean_d_surface` == `mean_d_cross`: costs.py's `close_price` treats
        # "cross" and "surface" identically until the Corpus B fit lands.
        results.append(PolicyResult(
            policy_name=policy.name, n_clusters=n_clusters,
            mean_d_cross=mean_d, mean_d_surface=mean_d,
            ci_lo=ci_lo, ci_hi=ci_hi,
            dsr=dsr_by_policy.get(policy.name, 0.0), pbo=pbo_grid,
            fwer_p=p_by_policy.get(policy.name, 1.0),
            skew_d=skew_by_policy.get(policy.name, 0.0),
            beats_all_nulls=beats_nulls, corpus_b_sign_agrees=sign_agrees,
            loso_stable=leave_one_symbol_out_stable(df),
            variance_ratio=variance_reduction(df)))
    return results


# ---------------------------------------------------------------------------
# Correction (7): degenerate DTE cells
# ---------------------------------------------------------------------------

_DEGENERATE_DTE_RE = re.compile(r"_dte(21|28)_")


def grid_wide_fwer_p(results: Sequence[PolicyResult]) -> float:
    """The p-value for "the single best policy in this grid is real".

    Always the minimum of every `PolicyResult.fwer_p` in `results`, because
    both `family_wise_p` and the per-policy `fwer_p` values `sweep` assigns
    come from the identical max-statistic null distribution (see
    `_maxT_pvalues`'s docstring) — this is the same number `sweep` gets from
    calling `family_wise_p(matrix, ...)` directly, recovered here without
    needing the raw cluster matrix.
    """
    if not results:
        return 1.0
    return min(r.fwer_p for r in results)


def is_degenerate_dte_cell(policy_name: str, corpus: str) -> bool:
    """True when this cell's `time_exit_dte` (21 or 28) fires on the first
    interior mark for most Corpus A positions (median entry DTE 17; 94.1%
    reach DTE<=21 by exit). Not meaningful — and not flagged — for Corpus B.
    """
    return corpus.strip().upper() == "A" and bool(
        _DEGENERATE_DTE_RE.search(policy_name))


# ---------------------------------------------------------------------------
# Correction (6) + (8): the report body
# ---------------------------------------------------------------------------

_DETECTION_FLOOR = (
    "- **Detection floor: ~+0.012 of capital at risk per trade.** Validated "
    "by planting a synthetic effect: +0.008 CAR/trade gives family-wise "
    "p=0.474 (invisible), +0.012 gives p=0.000 (detected). A null result "
    "below this line is not \"no effect\" — see the limitations below."
)

_LIMITS = """## What this run cannot tell you

- It is **not** a selection result. Nothing here ranks candidates, and a
  positive verdict is evidence that a policy knob paid — **not evidence that
  the system has an edge**.
- Corpus A spans 26 trading days of a single low-vol regime, and Corpus B
  covers the same period. This is **one regime**: SPY's worst drawdown
  across the whole life of the book was -4.49%, against a long-run drawdown
  of -55.2%, so no result here says anything about a volatility event.
- Corpus A's own recorded P&L reproduces at only ~78% under mid costs
  (calibrate.py), with stop-loss exits failing at roughly double the rate of
  other exit reasons. Corpus A's recorded P&L must **never** be quoted as
  ground truth.
- Path depth averages 5.75 marks per contract, so coarse threshold questions
  are answerable and intraday timing questions are not.
- Short Put has zero terminal priced rows in Corpus A and n=20 in Corpus B.
- A null result here means "no knob worth more than ~1.2% of capital at
  risk per trade was found at this grid size" — it does **not** mean "no
  effect". See the detection floor above.
- Grid cells with `time_exit_dte` of 21 or 28 on Corpus A are labelled
  `[first-mark exit]` below: at a median entry DTE of 17, those cells exit
  at the first available mark for most positions rather than measuring
  time decay.
"""


def _fmt_ci(lo: float, hi: float) -> str:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "not computed"
    return f"[{lo:+.4f}, {hi:+.4f}]"


def render_markdown(results: Sequence[PolicyResult], manifest: Dict[str, Any],
                    calibration: Optional[Any],
                    benchmark: Optional[Dict[str, Tuple[float, float, float]]] = None
                    ) -> str:
    """The POLICY_LAB_RESULT_*.md body.

    Refuses to render when the run used `mid` costs: replaying exits at mid
    manufactures edge, so a promotion decision from it would be a fiction.
    """
    if manifest.get("costs") == "mid":
        raise ValueError(
            "refusing to render a report from `mid` costs: mid replay invents "
            "edge that no one could have traded. Re-run with --costs cross.")

    corpus = str(manifest.get("corpus", ""))
    lines = [
        "# Policy Lab Result",
        "",
        f"- git: `{manifest['git_sha']}`",
        f"- generated: {manifest['generated_at']}",
        f"- corpus: {manifest['corpus']}  |  costs: {manifest['costs']}  "
        f"|  seed: {manifest['seed']}",
        f"- trials in grid (n_trials for DSR): **{manifest['n_trials']}**",
        f"- clustering unit: {manifest['cluster_unit']}",
        f"- rows: {manifest['row_counts']}",
        f"- corpus fingerprint: {manifest.get('corpus_fingerprint', {})}",
        "",
    ]

    # Correction (6): calibration + by_reason, spread-imputed fraction, and
    # the detection floor, all near the top.
    if calibration is not None:
        lines.append(
            f"- **calibration: {calibration.passed}/{calibration.checked} "
            f"positions reproduce their recorded P&L "
            f"({100 * calibration.fail_rate:.1f}% fail)**")
        if getattr(calibration, "by_reason", None):
            lines.append("  - by reason/strategy:")
            for label, (checked, passed) in sorted(calibration.by_reason.items()):
                failed = checked - passed
                rate = (100.0 * failed / checked) if checked else 0.0
                lines.append(
                    f"    - `{label}`: {passed}/{checked} reproduce "
                    f"({rate:.1f}% fail)")
        lines.append("")

    row_counts = manifest.get("row_counts") or {}
    spread_frac = row_counts.get("spread_imputed_frac")
    if spread_frac is not None:
        lines.append(
            f"- **{100 * spread_frac:.1f}% of path points had no observed "
            f"two-sided quote** and were widened from mid by the modelled "
            f"half-spread instead of a real quote (see `CostModel."
            f"imputed_half_spread`).")
        lines.append("")

    lines.append(_DETECTION_FLOOR)
    lines.append("")

    if results:
        lines.append(
            f"- whole-grid max-stat p (\"is the single best policy in this "
            f"grid real\"): **{grid_wide_fwer_p(results):.4f}**")
        lines.append("")

    lines += [
        "## Results",
        "",
        "| policy | verdict | clusters | mean d (cross) | mean d (surface) "
        "| 95% CI | fwer_p | dsr | pbo | skew_d | var ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tag = " `[first-mark exit]`" if is_degenerate_dte_cell(
            r.policy_name, corpus) else ""
        lines.append(
            f"| `{r.policy_name}`{tag} | **{policy_verdict(r)}** | "
            f"{r.n_clusters} | {r.mean_d_cross:+.4f} | {r.mean_d_surface:+.4f} "
            f"| {_fmt_ci(r.ci_lo, r.ci_hi)} | {r.fwer_p:.4f} | {r.dsr:.3f} "
            f"| {r.pbo:.3f} | {r.skew_d:+.2f} | {r.variance_ratio:.2f} |")

    # Correction (5): bounded-walk benchmark, per promoted policy.
    promoted = [r for r in results if policy_verdict(r) == "promote"]
    if promoted:
        lines += ["", "## Bounded-walk benchmark (promoted policies only)", ""]
        lines.append(
            "Real option prices are bounded below by zero; near that floor "
            "ANY early exit gains a real advantage unrelated to edge. Each "
            "row replays the SAME two policies on a synthetic bounded walk "
            "calibrated to this corpus's own step volatility.")
        lines.append("")
        lines.append(
            "| policy | real mean d | synthetic mean d | "
            "difference-of-differences |")
        lines.append("|---|---|---|---|")
        for r in promoted:
            b = (benchmark or {}).get(r.policy_name)
            if b is None:
                lines.append(f"| `{r.policy_name}` | {r.mean_d_cross:+.4f} "
                             f"| not computed | not computed |")
                continue
            real_d, synth_d, dod = b
            synth_str = "n/a" if math.isnan(synth_d) else f"{synth_d:+.4f}"
            dod_str = "n/a" if math.isnan(dod) else f"{dod:+.4f}"
            lines.append(f"| `{r.policy_name}` | {real_d:+.4f} | {synth_str} "
                         f"| {dod_str} |")

    lines += ["", _LIMITS]
    return "\n".join(lines)
