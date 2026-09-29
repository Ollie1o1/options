"""Does the live gate separate forward outcomes? — the pre-registered test.

`prereg_ranker.load_cohort` restricts itself to gate survivors and records that
"the refused population belongs to the separate removal question, which needs
its own pre-registration." This module answers that question.

Three design facts, each forced by something this corpus does:

  * **The cell is the pairing.** Within one `(symbol, expiration, scan_date)`
    the underlying's forward move is common to the passed and refused arms and
    cancels. A cell holding only one arm carries no comparison.
  * **The symbol is the cluster.** The corpus spans 5.4 weeks against 3-11 day
    holds, so the same ticker on adjacent days is the same bet. Rows are never
    the unit — this repo has overcounted correlated rows three times.
  * **The primary statistic is rank-based.** `candidate_positions.pnl_pct` is
    return on PREMIUM, which is unbounded below for credit structures: Bull Put
    reaches -168.231, Bear Call +659.0. A mean of that column is a statement
    about a handful of near-zero-credit rows. Probability-of-superiority cannot
    be moved by them.

Every threshold lives in `docs/PREREG_GATE_SEPARATION_20260929.md` and is
mirrored here. This module computes; it does not decide.

The registration's provenance section is not optional reading: the Long Put
inversion this was built to examine was seen in this same corpus first.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Pre-registered parameters (docs/PREREG_GATE_SEPARATION_20260929.md) ──────

CELL_COLS: Sequence[str] = ("symbol", "expiration", "scan_date")
CLUSTER_COL = "symbol"
OUTCOME_COL = "pnl_pct"
ARM_COL = "gate_passed"

MIN_CLUSTERS = 20          # matches policy_lab.stats.MIN_CLUSTERS, alloc MIN_N
MAX_FWER_P = 0.05
ALPHA = 0.05
N_BOOT = 10_000
N_PERM = 10_000
SEED = 20260929
WINSOR: Tuple[float, float] = (0.01, 0.99)
NEGATIVE_CONTROL_SHUFFLES = 200
# The negative control's null must be centred here or the estimator is broken
# and the whole run is void.
MAX_NULL_MEAN = 0.01


# ── Primary statistic ────────────────────────────────────────────────────────

def _auc(passed: np.ndarray, refused: np.ndarray) -> float:
    """P(passed > refused) + 0.5 * P(tie), via the rank identity.

    Computed from ranks rather than from the |P| x |R| pair matrix: a cell can
    hold hundreds of refused contracts against a handful of passed ones, and
    the pairwise form is O(|P| * |R|) where this is O(n log n). Average ranks
    give the 0.5-per-tie convention exactly, which is why `rankdata`'s default
    is the right one here and not merely convenient.
    """
    n_p, n_r = passed.size, refused.size
    if n_p == 0 or n_r == 0:
        return float("nan")
    from scipy.stats import rankdata
    combined = np.concatenate([passed, refused])
    ranks = rankdata(combined)
    u = float(ranks[:n_p].sum()) - n_p * (n_p + 1) / 2.0
    return u / (n_p * n_r)


def cell_superiority(df: pd.DataFrame, outcome: str = OUTCOME_COL,
                     arm: str = ARM_COL,
                     cell_cols: Sequence[str] = CELL_COLS) -> pd.DataFrame:
    """One row per cell holding BOTH arms, carrying `d_auc` in [-0.5, +0.5].

    Cells with a single arm are dropped rather than imputed: there is nothing
    to compare them against, and giving them a zero would quietly shrink every
    strategy's statistic toward the null in proportion to how decisively its
    gate acts.
    """
    cols = list(cell_cols)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols + ["d_auc", "n_pass", "n_ref"])
    work = df.dropna(subset=[outcome, arm] + cols)
    if len(work) == 0:
        return pd.DataFrame(columns=cols + ["d_auc", "n_pass", "n_ref"])

    out: List[Dict[str, object]] = []
    for key, g in work.groupby(cols, sort=False):
        is_pass = g[arm].to_numpy() == 1
        passed = g.loc[is_pass, outcome].to_numpy(dtype="float64")
        refused = g.loc[~is_pass, outcome].to_numpy(dtype="float64")
        if passed.size == 0 or refused.size == 0:
            continue
        row = dict(zip(cols, key if isinstance(key, tuple) else (key,)))
        row["d_auc"] = _auc(passed, refused) - 0.5
        row["n_pass"] = int(passed.size)
        row["n_ref"] = int(refused.size)
        out.append(row)
    return pd.DataFrame(out, columns=cols + ["d_auc", "n_pass", "n_ref"])


# ── Secondary statistic ──────────────────────────────────────────────────────

def winsorize(s: pd.Series, lo: float = WINSOR[0],
              hi: float = WINSOR[1]) -> pd.Series:
    """Clip to the [lo, hi] quantiles of `s` itself."""
    vals = pd.to_numeric(s, errors="coerce")
    if vals.dropna().empty:
        return vals
    return vals.clip(lower=vals.quantile(lo), upper=vals.quantile(hi))


def cell_mean_difference(df: pd.DataFrame, outcome: str = OUTCOME_COL,
                         arm: str = ARM_COL,
                         cell_cols: Sequence[str] = CELL_COLS,
                         winsor: Optional[Tuple[float, float]] = WINSOR
                         ) -> pd.DataFrame:
    """`mean(passed) - mean(refused)` per cell, in fractions of premium.

    Winsorized across the whole frame BEFORE differencing, so the clip points
    are a property of the strategy's distribution rather than of whichever cell
    happened to contain the outlier.

    Carried for economic magnitude only. It has no decision authority, because
    the units are return-on-premium and therefore not comparable between a
    credit spread and a long option.
    """
    cols = list(cell_cols)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols + ["d_mean"])
    work = df.dropna(subset=[outcome, arm] + cols).copy()
    if len(work) == 0:
        return pd.DataFrame(columns=cols + ["d_mean"])
    if winsor is not None:
        work[outcome] = winsorize(work[outcome], winsor[0], winsor[1])

    out: List[Dict[str, object]] = []
    for key, g in work.groupby(cols, sort=False):
        is_pass = g[arm].to_numpy() == 1
        passed = g.loc[is_pass, outcome].to_numpy(dtype="float64")
        refused = g.loc[~is_pass, outcome].to_numpy(dtype="float64")
        if passed.size == 0 or refused.size == 0:
            continue
        row = dict(zip(cols, key if isinstance(key, tuple) else (key,)))
        row["d_mean"] = float(passed.mean() - refused.mean())
        out.append(row)
    return pd.DataFrame(out, columns=cols + ["d_mean"])


# ── Aggregation ──────────────────────────────────────────────────────────────

def cluster_means(cells: pd.DataFrame, value_col: str,
                  cluster_col: str = CLUSTER_COL) -> pd.Series:
    """Mean of `value_col` within each cluster, indexed by cluster."""
    if cells is None or len(cells) == 0 or cluster_col not in cells.columns:
        return pd.Series(dtype="float64")
    return cells.groupby(cluster_col)[value_col].mean().dropna()


def strategy_statistic(means: pd.Series) -> float:
    """Mean over clusters — NOT over rows.

    Cluster weighting is what stops one heavily-scanned ticker from carrying a
    strategy. SPY alone can appear on every scan of every day.
    """
    if means is None or len(means) == 0:
        return float("nan")
    return float(np.mean(means.to_numpy(dtype="float64")))


def cluster_bootstrap_ci(cells: pd.DataFrame, value_col: str,
                         cluster_col: str = CLUSTER_COL,
                         n_boot: int = N_BOOT, alpha: float = ALPHA,
                         seed: int = SEED
                         ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for `strategy_statistic`, resampling whole symbols.

    Resampling cells would treat one ticker's many scan-days as independent and
    report an interval far too narrow.
    """
    means = cluster_means(cells, value_col, cluster_col)
    if len(means) < 2:
        return (None, None)
    arr = means.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    stats = arr[draws].mean(axis=1)
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


# ── Family-wise correction across the six strategies ─────────────────────────

def masked_row_means(arr: np.ndarray) -> np.ndarray:
    """Row means ignoring NaN, so each strategy keeps its own denominator.

    A strategy absent from a symbol must contribute nothing rather than a zero:
    a zero would shrink that strategy's statistic in proportion to how many
    symbols it never traded, which is a property of the universe and not of the
    gate.
    """
    arr = np.asarray(arr, dtype="float64")
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(arr), axis=1)
        sums = np.nansum(arr, axis=1)
        out = np.divide(sums, counts, out=np.full(arr.shape[0], np.nan),
                        where=counts > 0)
    return out


def family_wise_p_masked(cluster_means_matrix: np.ndarray,
                         n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Max-statistic sign-flip permutation across the strategy family.

    `cluster_means_matrix` is (n_strategies, n_clusters): row g, column c is
    strategy g's mean statistic within cluster c, NaN where that strategy never
    appeared on that symbol.

    Under the null a cluster's statistic is symmetric about zero, so flipping a
    cluster's sign is an exchange the null permits. One sign per COLUMN applied
    to every strategy at once preserves the correlation between strategies,
    which is what makes the max statistic the correct family-wise correction
    rather than six separate ones. Long Put was chosen as the largest of six
    looks; this is the price of that choice.

    Reduces EXACTLY to `policy_lab.stats.family_wise_p` when nothing is masked
    — same RNG call sequence, and `nanmean == mean` with no NaN present. That
    equivalence is asserted in `tests/test_gate_separation.py`.
    """
    arr = np.asarray(cluster_means_matrix, dtype="float64")
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return 1.0
    arr = arr[~np.all(np.isnan(arr), axis=1)]
    if arr.shape[0] == 0:
        return 1.0

    n_clusters = arr.shape[1]
    obs = float(np.nanmax(np.abs(masked_row_means(arr))))
    if not np.isfinite(obs):
        return 1.0

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype="float64")
    for i in range(int(n_perm)):
        s = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        null[i] = float(np.nanmax(np.abs(masked_row_means(arr * s))))
    return float((null >= obs).mean())


def family_matrix(per_strategy_means: Dict[str, pd.Series]
                  ) -> Tuple[List[str], np.ndarray]:
    """Align every strategy's cluster means onto one shared symbol axis."""
    names = sorted(per_strategy_means)
    symbols = sorted({s for m in per_strategy_means.values() for s in m.index})
    arr = np.full((len(names), len(symbols)), np.nan)
    index = {s: i for i, s in enumerate(symbols)}
    for g, name in enumerate(names):
        for sym, val in per_strategy_means[name].items():
            arr[g, index[sym]] = val
    return names, arr


# ── Guards ───────────────────────────────────────────────────────────────────

def both_arm_rows(df: pd.DataFrame, arm: str = ARM_COL,
                  cell_cols: Sequence[str] = CELL_COLS) -> pd.DataFrame:
    """Rows in cells holding both arms — the only rows any statistic here uses.

    Filtering is EXACTLY equivalent for the permutation null, not merely close:
    permuting the arm label within a cell preserves that cell's label multiset,
    so a both-arm cell stays a both-arm cell and a single-arm cell stays
    single-arm under every permutation. Nothing can cross the boundary.

    It matters because it is the difference between a negative control that
    runs in seconds and one that runs for ten minutes: Bull Put has 111,583
    closed rows but only 287 cells that carry a comparison at all.
    """
    cols = list(cell_cols)
    if df is None or len(df) == 0:
        return df
    work = df.dropna(subset=[arm] + cols)
    if len(work) == 0:
        return work
    flags = work.groupby(cols)[arm].transform(
        lambda s: (s == 1).any() and (s == 0).any())
    return work[flags.fillna(False).astype(bool)]


def negative_control(df: pd.DataFrame, outcome: str = OUTCOME_COL,
                     arm: str = ARM_COL,
                     cell_cols: Sequence[str] = CELL_COLS,
                     cluster_col: str = CLUSTER_COL,
                     n_shuffles: int = NEGATIVE_CONTROL_SHUFFLES,
                     seed: int = SEED) -> Dict[str, float]:
    """Permute the ARM LABEL within cell and re-measure. Must return null.

    This tests the test. A bug in the cell construction or the cluster
    weighting that manufactures separation is otherwise indistinguishable from
    a finding.

    The arm label is permuted rather than the outcome, because the arm is what
    the hypothesis is about: shuffling it preserves every cell's outcome
    distribution and its arm sizes, and destroys only the gate's assignment.
    """
    observed = strategy_statistic(
        cluster_means(cell_superiority(df, outcome, arm, cell_cols),
                      "d_auc", cluster_col))
    rng = np.random.default_rng(seed)
    cols = list(cell_cols)
    # Equivalent to shuffling the whole frame — see `both_arm_rows`.
    work = both_arm_rows(df.dropna(subset=[outcome, arm] + cols),
                         arm, cell_cols).copy()

    stats: List[float] = []
    for _ in range(int(n_shuffles)):
        shuffled = work.copy()
        shuffled[arm] = shuffled.groupby(cols)[arm].transform(
            lambda s: s.to_numpy()[rng.permutation(len(s))])
        stat = strategy_statistic(
            cluster_means(cell_superiority(shuffled, outcome, arm, cell_cols),
                          "d_auc", cluster_col))
        if np.isfinite(stat):
            stats.append(stat)

    arr = np.array(stats, dtype="float64") if stats else np.array([0.0])
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(arr)),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
        "n_shuffles": float(len(stats)),
    }


def loso_sign_stable(means: pd.Series) -> bool:
    """Does the sign of the statistic survive dropping any one symbol?

    Bull Put looked significant at t=3.48 in the policy lab until MU and AMD —
    68% of its dollars during a semiconductor melt-up — were removed. A result
    one ticker can overturn is a result about that ticker.
    """
    if means is None or len(means) < 2:
        return False
    arr = means.to_numpy(dtype="float64")
    full = float(arr.mean())
    if full == 0.0:
        return False
    total, n = arr.sum(), arr.size
    rest = (total - arr) / (n - 1)
    return bool(np.all(np.sign(rest) == np.sign(full)))


def half_split(df: pd.DataFrame, date_col: str = "scan_date"
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the median scan date. Replication, not a gate on its own."""
    if df is None or len(df) == 0 or date_col not in df.columns:
        empty = df.iloc[0:0] if df is not None else pd.DataFrame()
        return (empty, empty)
    dates = sorted(pd.Series(df[date_col]).dropna().unique())
    if len(dates) < 2:
        return (df.iloc[0:0], df)
    cut = dates[len(dates) // 2]
    return (df[df[date_col] < cut], df[df[date_col] >= cut])


# ── Result and decision ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyResult:
    """Everything the decision rule needs for one strategy."""
    strategy: str
    n_rows: int
    n_cells: int
    n_clusters: int
    stat: float
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    secondary: float
    fwer_p: float
    loso_stable: bool
    half_a: Optional[float]
    half_b: Optional[float]
    skew: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def verdict(r: StrategyResult) -> str:
    """`separates`, `inverted`, `null` or `insufficient`.

    `insufficient` is checked first and is deliberately not `null`: "we could
    not measure this" and "we measured it and found nothing" are different
    claims, and collapsing them is how a scarcity problem gets reported as a
    negative result.

    No verdict here authorises a configuration change. The registration's
    provenance section explains why: the inversion that motivated this test was
    seen in this same corpus.
    """
    if r.n_clusters < MIN_CLUSTERS:
        return "insufficient"
    if r.ci_lo is None or r.ci_hi is None:
        return "null"
    if r.half_a is None or r.half_b is None:
        return "null"
    if not np.isfinite(r.half_a) or not np.isfinite(r.half_b):
        return "null"
    if not r.loso_stable or r.fwer_p >= MAX_FWER_P:
        return "null"
    halves_agree = np.sign(r.half_a) == np.sign(r.half_b)
    if not halves_agree:
        return "null"
    if r.ci_lo > 0 and r.secondary > 0 and np.sign(r.half_a) > 0:
        return "separates"
    if r.ci_hi < 0 and r.secondary < 0 and np.sign(r.half_a) < 0:
        return "inverted"
    return "null"


# ── The one I/O boundary ─────────────────────────────────────────────────────

COHORT_COLUMNS = ["strategy_name", "symbol", "expiration", "scan_date",
                  "contract_key", "gate_passed", "pnl_pct"]

UNLABELLED = "(unlabelled)"


def _derive_strategy(name: object, mode: object, opt_type: object) -> str:
    """Recorded name, else derived from `(mode, opt_type)`, else unlabelled.

    `candidates.strategy_name` is never written for the single-leg Premium
    Selling and Discovery boards — about 100,000 closed rows spanning the whole
    corpus, not a legacy prefix. Left alone they form a single nameless bucket
    that is really Short Put, Long Put and Long Call mixed together, and mixing
    them is not cosmetic: a put sold and a call bought on the same day are not
    exchangeable, which is the same reason `prereg_ranker.load_cohort` derives
    its label rather than using `candidate_positions.family`.

    Rows carrying neither a name nor a mode (2,354, all on 2026-08-19) stay
    unlabelled — deriving a label for them would be inventing one.
    """
    from .trade_analysis import strategy_label_for_mode
    if isinstance(name, str) and name.strip():
        return name.strip()
    if not mode or not opt_type:
        return UNLABELLED
    try:
        return strategy_label_for_mode(str(mode), opt_type)
    except Exception:
        return UNLABELLED


def load_cohort(db_path: str) -> pd.DataFrame:
    """Closed, marked candidate positions joined to their recorded decision.

    THE ONLY I/O IN THIS MODULE.

    `gate_passed IS NULL` is excluded: 8,454 such rows span the whole corpus
    rather than a legacy prefix, and a row whose arm is unknown cannot be
    assigned to one.
    """
    import sqlite3
    sql = (
        "SELECT c.strategy_name, c.mode, c.opt_type, c.symbol, c.expiration,"
        "       date(c.ts) AS scan_date, c.contract_key,"
        "       c.gate_passed, p.pnl_pct "
        "FROM candidate_positions p JOIN candidates c "
        "  ON c.scan_id = p.scan_id AND c.board = p.board "
        " AND c.contract_key = p.contract_key "
        "WHERE p.status = 'CLOSED' AND p.pnl_pct IS NOT NULL "
        "  AND c.gate_passed IS NOT NULL"
    )
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            df = pd.read_sql(sql, conn)
    except Exception:
        log.warning("cohort unreadable at %s", db_path, exc_info=True)
        return pd.DataFrame(columns=COHORT_COLUMNS)
    if len(df) == 0:
        return pd.DataFrame(columns=COHORT_COLUMNS)
    df["strategy_name"] = [
        _derive_strategy(r.strategy_name, r.mode, r.opt_type)
        for r in df.itertuples(index=False)]
    return df[COHORT_COLUMNS]


def corpus_fingerprint(db_path: str) -> Dict[str, object]:
    """Identify the corpus a run was computed against.

    Both databases in this project are live; one grew 17,378 closed rows in a
    single session. Two runs whose fingerprints differ are not comparable, and
    without this recorded the difference looks like a finding.
    """
    import sqlite3
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows, = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
            lo, hi = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM candidates").fetchone()
            closed, = conn.execute(
                "SELECT COUNT(*) FROM candidate_positions "
                "WHERE status = 'CLOSED'").fetchone()
    except Exception:
        log.warning("fingerprint unreadable at %s", db_path, exc_info=True)
        return {"error": "unreadable"}
    digest = hashlib.sha256(
        f"{rows}|{lo}|{hi}|{closed}".encode()).hexdigest()[:16]
    return {"candidate_rows": int(rows), "closed_positions": int(closed),
            "ts_min": lo, "ts_max": hi, "digest": digest}
