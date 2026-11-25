"""The live VALIDATED sleeve, computed exactly as it was backtested.

`VM_OUTLOOK_TOP3` (configs/playbooks.json → `vol_managed_outlook`) was
validated by `scripts/backtest_vol_managed_outlook.py` under
`docs/PREREG_VOL_MANAGED_OUTLOOK_20261124.md`. At each monthly rebalance date
t that estimator:

  1. builds each ETF's factor row with `src.outlook.backtest._features`
     (`mom_12_1`, `trend_score` = close / SMA200 − 1, `reversal_1m`,
     63-day `relative_strength` vs SPY) on a SPY-date-aligned close matrix;
  2. ranks the 16-ETF universe with `rank_universe(.., DEFAULT_OUTLOOK_CONFIG)`
     — the frozen default weights, never a config.json override;
  3. holds the top 3 **with a positive composite**, 1/3 each (a non-positive
     name's third stays in cash — it is not redistributed);
  4. scales that basket by clip(0.14 / RV20(basket), 0.25, 1.00), where
     RV20 is the annualised 20-day realised vol of the *selected basket*
     (not SPY's — scaling by SPY's vol is the `VM_SPY` control's rule).

Until 2026-11-25 the Radar re-implemented this inline and differed on all of
(3), (4) and the trend formula, so the allocation shown as VALIDATED was not
the rule that had been validated. This module is now the one implementation
the live sleeve uses, and `tests/outlook/test_vm_rotation_parity.py`
replays the backtest's own `_simulate_arms` against it period by period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE, _features
from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG, rank_universe

TARGET_VOL = 0.14
MIN_SCALE = 0.25
MAX_SCALE = 1.00
TOP_K = 3
RV_WINDOW = 20
HISTORY = 252  # finite closes required over [t-252, t]

UNIVERSE: Tuple[str, ...] = tuple(DEFAULT_UNIVERSE)

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]


@dataclass
class RotationTarget:
    """Target weights for one rebalance date, with everything needed to audit them."""

    as_of: str
    ranked: List[Dict[str, Any]]
    weights: Dict[str, float]          # final (vol-scaled) weight per ETF
    unscaled: Dict[str, float]         # Arm A weights (1/top_k per positive name)
    scale: float
    basket_rv20: Optional[float]
    eligible: List[str]
    excluded: Dict[str, str] = field(default_factory=dict)

    @property
    def cash(self) -> float:
        return max(0.0, 1.0 - sum(self.weights.values()))


def align_closes(
    ohlcv: OhlcvMap,
    tickers: Sequence[str],
    *,
    min_date: str = "0000-00-00",
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """SPY-date-aligned close columns, NaN where a ticker has no bar.

    Identical to the backtest's `_align_matrix(.., strict_common=False)`.
    """
    spy_dates, spy_closes, _ = ohlcv[BENCH]
    base_idx = [i for i, d in enumerate(spy_dates) if d >= min_date]
    dates = [spy_dates[i] for i in base_idx]
    pos = {d: i for i, d in enumerate(dates)}
    cols: Dict[str, np.ndarray] = {BENCH: np.asarray(spy_closes, dtype=float)[base_idx]}
    for tk in tickers:
        if tk not in ohlcv:
            continue
        ds, cs, _ = ohlcv[tk]
        arr: np.ndarray = np.full(len(dates), np.nan, dtype=float)
        for d, c in zip(ds, cs):
            p = pos.get(d)
            if p is not None:
                arr[p] = float(c)
        cols[tk] = arr
    return dates, cols


def basket_rv(
    cols: Dict[str, np.ndarray], weights: Dict[str, float], t: int, window: int = RV_WINDOW,
) -> Optional[float]:
    """Annualised trailing realised vol of the weight-normalised basket.

    Same arithmetic as the backtest's `_trailing_basket_rv`.
    """
    if t < window or not weights:
        return None
    w_sum = sum(weights.values())
    if w_sum <= 0:
        return None
    port: np.ndarray = np.zeros(window, dtype=float)
    for tk, w in weights.items():
        seg = cols[tk][t - window: t + 1]
        if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
            return None
        port += (w / w_sum) * (np.diff(seg) / seg[:-1])
    sd = float(np.std(port, ddof=1))
    return sd * math.sqrt(252.0) if sd > 0 else None


def eligible_at(
    cols: Dict[str, np.ndarray], universe: Sequence[str], t: int,
) -> Tuple[List[str], Dict[str, str]]:
    """Names with finite closes over [t-252, t]; the rest with the reason."""
    ok: List[str] = []
    why: Dict[str, str] = {}
    for tk in universe:
        if tk not in cols:
            why[tk] = "not in price archive"
            continue
        if t < HISTORY:
            why[tk] = f"needs {HISTORY + 1} sessions of history"
            continue
        if not np.all(np.isfinite(cols[tk][t - HISTORY: t + 1])):
            why[tk] = "gap in last 253 sessions"
            continue
        ok.append(tk)
    return ok, why


def target_weights(
    dates: Sequence[str],
    cols: Dict[str, np.ndarray],
    t: int,
    *,
    universe: Sequence[str] = UNIVERSE,
    top_k: int = TOP_K,
    target_vol: float = TARGET_VOL,
) -> Optional[RotationTarget]:
    """The backtested Arm B (`VM_OUTLOOK_TOP3`) target at index t, or None if it can't be formed."""
    avail, excluded = eligible_at(cols, universe, t)
    if len(avail) < top_k:
        return None
    bench_list = cols[BENCH][: t + 1].tolist()
    feats = {tk: _features(cols[tk][: t + 1].tolist(), bench_list, t) for tk in avail}
    ranked = rank_universe(feats, DEFAULT_OUTLOOK_CONFIG)
    pos_top = [str(r["ticker"]) for r in ranked[:top_k] if float(r["score"]) > 0.0]
    w_a = {tk: 1.0 / float(top_k) for tk in pos_top}
    rv = basket_rv(cols, w_a, t)
    scale = float(np.clip(target_vol / rv, MIN_SCALE, MAX_SCALE)) if (rv and rv > 0 and w_a) else 1.0
    return RotationTarget(
        as_of=str(dates[t]),
        ranked=[dict(r) for r in ranked],
        weights={tk: v * scale for tk, v in w_a.items()},
        unscaled=w_a,
        scale=scale,
        basket_rv20=rv,
        eligible=avail,
        excluded=excluded,
    )


def latest_complete_index(cols: Dict[str, np.ndarray], universe: Sequence[str], lookback: int = 5) -> Optional[int]:
    """Latest SPY-aligned index at which every universe name that traded recently has a bar.

    The archive is refreshed per symbol, so SPY can be a session ahead of some
    ETFs. Ranking on SPY's last date would silently drop the stragglers (their
    last bar is NaN there); instead rank on the newest date where coverage is as
    complete as it has been over the last `lookback` sessions.
    """
    if BENCH not in cols or len(cols[BENCH]) == 0:
        return None
    n = len(cols[BENCH])
    lo = max(0, n - lookback)
    names = [tk for tk in universe if tk in cols]
    counts = [sum(1 for tk in names if np.isfinite(cols[tk][i])) for i in range(lo, n)]
    if not counts:
        return None
    best = max(counts)
    for off in range(len(counts) - 1, -1, -1):
        if counts[off] == best:
            return lo + off
    return None


def live_target(ohlcv: OhlcvMap, *, universe: Sequence[str] = UNIVERSE) -> Optional[RotationTarget]:
    """Today's sleeve from the local archive, on the latest fully-covered session."""
    if BENCH not in ohlcv:
        return None
    dates, cols = align_closes(ohlcv, universe)
    t = latest_complete_index(cols, universe)
    if t is None:
        return None
    return target_weights(dates, cols, t, universe=universe)


__all__ = [
    "TARGET_VOL", "MIN_SCALE", "MAX_SCALE", "TOP_K", "UNIVERSE",
    "RotationTarget", "align_closes", "basket_rv", "eligible_at",
    "target_weights", "latest_complete_index", "live_target",
]
