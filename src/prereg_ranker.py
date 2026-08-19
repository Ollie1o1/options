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
