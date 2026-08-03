"""Matched low-SI controls for the IV-premium measurement.

Without a control this test rediscovers that shorted stocks are volatile.
Trailing realised vol is the covariate doing the real work — it is the dominant
driver of implied vol — with size and price alongside it. Deciles 6-9 are
excluded from the control pool by the caller: partial treatment would shrink
the measured gap toward zero.

Matching is nearest-neighbour on standardised covariates under hard calipers,
k=3 with replacement. With replacement because the pool of 90-vol, sub-$1B,
low-SI optionable names is thin; the reuse is legitimate and its dependence is
absorbed by the date-level clustering in the bootstrap, but it is logged so it
can be inspected. A fitted propensity model was rejected: four covariates and a
transparent decision beat a model fitted on a small forward sample.

Five-day return is matched too, and that is not cosmetic. The treated arm is
defined partly BY momentum — top short interest that has already run — so a
control pool without a momentum condition would make the contrast "short
interest plus whatever momentum contributes" rather than short interest. The
live pricing arm matches without any momentum term, and two halves measuring
different populations cannot be subtracted from one another.

Treated units with no in-caliper control are dropped AND counted. Silent
dropping would select the cohort toward whatever happens to be matchable.
"""
from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

K_CONTROLS = 3
CALIPER_RV_REL = 0.20      # trailing realised vol, relative
CALIPER_LOG_MCAP = 1.0
CALIPER_LOG_PRICE = 0.7
CALIPER_RET5D = 0.05       # five-day return, absolute (fractions, not percent)
MAX_DROP_RATE = 0.30       # above this a cycle is flagged
MAX_SMD = 0.25             # standardised mean difference, post-match
_SD_FLOOR_REL = 1e-12      # below this, the covariate is constant to float precision

_COVARIATES = ("rv", "log_mcap", "log_price", "ret_5d")


class Unit(NamedTuple):
    key: str
    rv: float
    log_mcap: float
    log_price: float
    ret_5d: float


class MatchResult(NamedTuple):
    pairs: Dict[str, List[str]]
    dropped: List[str]
    drop_rate: float
    smd: Dict[str, float]
    reuse: Dict[str, int]


def _matrix(units: Sequence[Unit]) -> np.ndarray:
    return np.array([[u.rv, u.log_mcap, u.log_price, u.ret_5d] for u in units],
                    dtype=float)


def _within_calipers(t: Unit, c: Unit) -> bool:
    if t.rv <= 0 or c.rv <= 0:
        return False
    if abs(c.rv - t.rv) / t.rv > CALIPER_RV_REL:
        return False
    if abs(c.log_mcap - t.log_mcap) > CALIPER_LOG_MCAP:
        return False
    if abs(c.log_price - t.log_price) > CALIPER_LOG_PRICE:
        return False
    if abs(c.ret_5d - t.ret_5d) > CALIPER_RET5D:
        return False
    return True


def match(treated: Sequence[Unit], controls: Sequence[Unit],
          k: int = K_CONTROLS) -> MatchResult:
    """Nearest-neighbour matched controls, with replacement, under calipers."""
    pairs: Dict[str, List[str]] = {}
    dropped: List[str] = []
    reuse: Dict[str, int] = {}

    if not treated:
        return MatchResult({}, [], 0.0, {}, {})
    if not controls:
        return MatchResult({}, [u.key for u in treated], 1.0, {}, {})

    pooled = np.vstack([_matrix(treated), _matrix(controls)])
    scale = pooled.std(axis=0, ddof=0)
    scale[scale <= 0] = 1.0
    c_scaled = _matrix(controls) / scale

    for t in treated:
        eligible = [i for i, c in enumerate(controls) if _within_calipers(t, c)]
        if not eligible:
            dropped.append(t.key)
            continue
        t_scaled = np.array([t.rv, t.log_mcap, t.log_price, t.ret_5d],
                            dtype=float) / scale
        dist = np.linalg.norm(c_scaled[eligible] - t_scaled, axis=1)
        order = np.argsort(dist, kind="stable")[:k]
        chosen = [controls[eligible[i]].key for i in order]
        pairs[t.key] = chosen
        for key in chosen:
            reuse[key] = reuse.get(key, 0) + 1

    drop_rate = len(dropped) / len(treated)
    return MatchResult(pairs=pairs, dropped=dropped, drop_rate=drop_rate,
                       smd=_smd(treated, controls, pairs), reuse=reuse)


def _smd(treated: Sequence[Unit], controls: Sequence[Unit],
         pairs: Dict[str, List[str]]) -> Dict[str, float]:
    """Standardised mean difference per covariate, post-match."""
    if not pairs:
        return {name: float("inf") for name in _COVARIATES}
    by_key = {c.key: c for c in controls}
    t_units = [t for t in treated if t.key in pairs]
    c_units = [by_key[key] for keys in pairs.values() for key in keys]

    out: Dict[str, float] = {}
    for idx, name in enumerate(_COVARIATES):
        t_vals = np.array([[u.rv, u.log_mcap, u.log_price, u.ret_5d][idx]
                           for u in t_units])
        c_vals = np.array([[u.rv, u.log_mcap, u.log_price, u.ret_5d][idx]
                           for u in c_units])
        pooled_sd = math.sqrt((t_vals.var(ddof=0) + c_vals.var(ddof=0)) / 2.0)
        # The floor is relative, not `<= 0`: the variance of a repeated
        # constant carries float noise (~1e-32), so a covariate identical
        # across both arms once yielded pooled_sd ~1e-16, cleared the zero
        # guard, and turned noise-over-noise into a sqrt(2) "imbalance" that
        # rejected every observation date. A pooled SD at float-noise scale
        # means the covariate is constant and imbalance is undefined; the
        # honest value is 0. Relative because rv (~0.9) and log_mcap (~20)
        # live at different magnitudes.
        scale = max(1.0, abs(float(t_vals.mean())), abs(float(c_vals.mean())))
        if pooled_sd <= _SD_FLOOR_REL * scale:
            out[name] = 0.0
        else:
            out[name] = float(abs(t_vals.mean() - c_vals.mean()) / pooled_sd)
    return out


def is_valid(result: MatchResult) -> bool:
    """Both committed match-quality tripwires, per the spec's section 4.5."""
    if result.drop_rate > MAX_DROP_RATE:
        return False
    return is_balanced(result)


def is_balanced(result: MatchResult) -> bool:
    """Covariate balance alone — `is_valid` without the drop-rate arm.

    D_hist's estimand was changed on 2026-08-03 to the matchable subsample
    (see status/DECISIONS.md), which makes an unmatchable treated unit a
    selection to be documented rather than a defect in the comparison. Balance
    between the units that DID match is the part that still has to hold: it is
    the whole reason treated-minus-control is a fair difference. The drop rate
    is still computed and still reported — it is the size of the selection —
    it just no longer invalidates.

    An empty `smd` means nothing matched, so there is no comparison to be
    balanced. `all()` over an empty dict is True, which would read as "no
    imbalance detected"; the honest answer is False.
    """
    if not result.smd:
        return False
    return all(v <= MAX_SMD for v in result.smd.values())
