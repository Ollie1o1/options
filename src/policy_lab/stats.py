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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.policy_lab.types import Outcome, PricePath

# Below this ratio of sd(d) to sd(r_base) the pairing is doing its job. At or
# above it the variance reduction is too small to rescue the power problem and
# every result in the run is labelled UNDERPOWERED.
PAIRING_INEFFECTIVE_RATIO = 0.5

CLUSTER_COL = "cluster"


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
