"""D_hist — the payoff half of the expected-value decomposition.

Expected net return per premium dollar splits as
``E = D_hist - P_live - F_live``. This module computes the first term from
underlying prices alone, on the 205 settlement dates the asymmetry study
already validated, which is what makes the forward test short: only the
pricing terms have to be gathered live, and an IV gap converges in cycles
rather than years.

The synthetic call is priced under the sleeve's OWN exit ladder rather than
some cleaner standardised exit. A test run under a different exit measures a
different strategy than the one it would authorise.

Resampling follows the precedent set by the asymmetry study: never resample
rows, resample DATES, in contiguous blocks. Every name on a settlement date
shares that day's market move, and settlement dates ~11 trading days apart
observe overlapping futures at horizons out to 42.

The synthetic call is priced at a FAIR vol — the row's own trailing daily
realised vol, annualised as ``sigma_d * sqrt(252)`` — never at market IV, and
the vol is derived here rather than accepted per row so the convention cannot
drift. The decomposition only avoids double-counting because the market's IV
markup lives exclusively in P_live: feed market IV into D_hist and the premium
is subtracted twice, biasing the gate toward STOP; let the two arms price on
different conventions and the bias direction is unknown.

Rows arrive already labelled into arms. Cohort policy — who counts as treated,
who is an eligible control, and who is excluded as partially treated — lives in
``cohort.py`` and deliberately not here: this module does the statistics and
should not also hold an opinion about short-interest percentiles.

THE ESTIMAND IS THE MATCHABLE SUBSAMPLE (operator decision, 2026-08-03). The
original estimand was the effect on a uniformly-drawn treated name, and it is
not measurable on this panel: a top-5%-SI name that has just run +10% sits at
the top of the realised-vol distribution, and the low-SI control pool populates
that band only at its lower edge, so matched controls come out systematically
less volatile. Across 44 tested configurations the post-match SMD on realised
vol never fell below its 0.25 bar. That is missing common support between two
populations, not a threshold that needs tuning — see status/DECISIONS.md.

So a treated unit with no in-caliper control is now a SELECTION rather than a
failure: it is dropped, counted, and characterised against the units that did
match. Only covariate BALANCE still invalidates a cycle. This buys a measurable
quantity at the price of a narrower one, and the price is real — the sample is
selected on matchability, which is exactly the bias the matched design was
built to avoid. Every consumer therefore gets ``selection`` alongside the
estimate, and the report prints it before the number.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from src.squeeze.sleeve import matching, payoff

SEED = 12345
N_BOOT = 4000
BLOCK_BY_HORIZON = {21: 2, 42: 4}

TREATED_ARM = "treated"
CONTROL_ARM = "control"

_COVARIATES = ("rv", "log_mcap", "log_price", "ret_5d")


def _empty_selection() -> Dict[str, object]:
    return {"treated_eligible": 0, "treated_matched": 0,
            "dates_over_drop_bar": 0, "drop_rates": [],
            "matched": {c: [] for c in _COVARIATES},
            "dropped": {c: [] for c in _COVARIATES}}


def _record_selection(sel: Dict[str, Any], units: Sequence[matching.Unit],
                      result: matching.MatchResult) -> None:
    """Accumulate who was matched, who was not, and how they differed.

    This is the documentation half of the matchable-subsample estimand. A
    selected sample whose selection is not characterised is just a biased
    sample, so the covariates of the DROPPED treated units are kept alongside
    the matched ones and reported side by side.
    """
    dropped = set(result.dropped)
    sel["treated_eligible"] += len(units)
    sel["treated_matched"] += len(result.pairs)
    sel["drop_rates"].append(result.drop_rate)
    if result.drop_rate > matching.MAX_DROP_RATE:
        sel["dates_over_drop_bar"] += 1
    for unit in units:
        bucket = sel["dropped"] if unit.key in dropped else sel["matched"]
        for name in _COVARIATES:
            bucket[name].append(float(getattr(unit, name)))


def _finalise_selection(sel: Dict[str, object]) -> Dict[str, object]:
    """Collapse the accumulated lists into the numbers the report prints."""
    rates = sel.pop("drop_rates")
    eligible = sel["treated_eligible"]
    out: Dict[str, object] = {
        "treated_eligible": eligible,
        "treated_matched": sel["treated_matched"],
        "coverage": (sel["treated_matched"] / eligible) if eligible else float("nan"),
        "dates_over_drop_bar": sel["dates_over_drop_bar"],
        "median_drop_rate": float(np.median(rates)) if rates else float("nan"),
    }
    for label in ("matched", "dropped"):
        vals = sel[label]
        out[label + "_mean"] = {
            name: (float(np.mean(vals[name])) if vals[name] else float("nan"))
            for name in _COVARIATES}
    out["n_dropped"] = len(sel["dropped"][_COVARIATES[0]])
    return out


def _unit(row: dict) -> matching.Unit:
    return matching.Unit(key=row["symbol"], rv=float(row["rv"]),
                         log_mcap=float(row["log_mcap"]),
                         log_price=float(row["log_price"]),
                         ret_5d=float(row["ret_5d"]))


def _mean_return(rows: Sequence[dict], keys: Sequence[str], horizon: int,
                 variant: str) -> Sequence[float]:
    by_symbol = {r["symbol"]: r for r in rows}
    out: List[float] = []
    for key in keys:
        row = by_symbol.get(key)
        if row is None:
            continue
        sigma_d = float(row["sigma_d"])
        # Fair-vol pricing by construction: both arms use the row's own
        # annualised realised vol, so no per-row field can smuggle market IV
        # (and its premium markup, which belongs to P_live) into D_hist.
        got = payoff.synthetic_call_return(
            row["path"], _entry_spot(row),
            sigma_d, sigma_d * math.sqrt(252.0),
            horizon_bars=horizon, variant=variant)
        if got is not None:
            out.append(got)
    return out


def _entry_spot(row: dict) -> float:
    """Entry spot, always supplied explicitly.

    There is no fallback to ``path[0]``. The path begins one bar AFTER entry, so
    inferring the entry level from it would shift every trade forward a day and
    misprice every strike and every ladder threshold — with no test failing and
    no number looking wrong. A missing ``spot0`` is a caller bug and says so.
    """
    return float(row["spot0"])


def compute(rows: Sequence[dict], horizon: int, variant: str = "central",
            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Matched treated-minus-control mean call return, with a block-bootstrap CI."""
    by_date: Dict[str, List[dict]] = {}
    for row in rows:
        by_date.setdefault(row["date"], []).append(row)

    dates: List[str] = []
    per_date: List[tuple] = []
    flagged: List[str] = []
    used: set = set()
    sel = _empty_selection()

    for date in sorted(by_date):
        day = by_date[date]
        treated = [r for r in day if r.get("arm") == TREATED_ARM]
        controls = [r for r in day if r.get("arm") == CONTROL_ARM]
        if not treated:
            # No treated units: genuinely not an observation of the effect.
            continue
        if not controls:
            # Treated units with an EMPTY eligible-control pool is a matching
            # failure, same as a caliper miss — skipping it silently would
            # under-count the flags that feed the majority-flagged tripwire,
            # weakening a validity check in the direction that flatters the
            # strategy.
            flagged.append(date)
            continue
        t_units = [_unit(r) for r in treated]
        result = matching.match(t_units, [_unit(r) for r in controls])
        # Balance, not full validity: under the matchable-subsample estimand a
        # high drop rate is the size of the selection, not a defect in the
        # comparison. It is recorded below and reported, never silently.
        if not matching.is_balanced(result):
            flagged.append(date)
            continue
        _record_selection(sel, t_units, result)

        t_keys = list(result.pairs)
        c_keys = [k for keys in result.pairs.values() for k in keys]
        t_rets = _mean_return(treated, t_keys, horizon, variant)
        c_rets = _mean_return(controls, c_keys, horizon, variant)
        if not t_rets or not c_rets:
            flagged.append(date)
            continue

        used.update(t_keys)
        used.update(c_keys)
        dates.append(date)
        per_date.append((float(np.mean(t_rets)), float(np.mean(c_rets)),
                         len(t_rets), len(c_rets)))

    if not dates:
        # Same nine keys as the success path: a fully-flagged panel is a real
        # state the gate has to describe (the INVALID verdict), and a dict
        # that changes shape underneath its consumer is the wrong way to say
        # "no data". NaN, not None — the success path already says NaN for an
        # unobtainable interval, and one sentinel convention is enough.
        return {"n_dates": 0, "treat_n": 0, "control_n": 0,
                "observed": float("nan"), "draws": np.array([]),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "flagged_dates": flagged, "used_symbols": sorted(used),
                "selection": _finalise_selection(sel)}

    arr = np.array([(t, c) for t, c, _, _ in per_date], dtype=float)
    observed = float(arr[:, 0].mean() - arr[:, 1].mean())

    block = BLOCK_BY_HORIZON.get(horizon, 4)
    rng = np.random.default_rng(seed)
    n_dates = len(dates)
    block = max(1, min(block, n_dates))
    n_blocks = max(1, n_dates // block)
    offsets = np.arange(block)

    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n_dates - block + 1), size=n_blocks)
        idx = np.clip((starts[:, None] + offsets).ravel(), 0, n_dates - 1)
        sample = arr[idx]
        draws[b] = sample[:, 0].mean() - sample[:, 1].mean()
    draws = draws[np.isfinite(draws)]

    # A degenerate resampling space cannot support an interval. With
    # ``n_dates <= block`` every start index is zero, so all n_boot draws are
    # the identical full sample and the percentile band collapses onto the
    # observed value — a zero-width "95% CI" that is fabricated precision, not
    # a confidence interval. Zero variance across the draws is the same
    # degeneracy detected after the fact. Refuse rather than fabricate: NaN is
    # already this module's convention for an unobtainable value. The point
    # estimate stays — it is real — only the interval is unsupportable.
    degenerate = (n_dates <= block or draws.size == 0
                  or bool(np.all(draws == draws[0])))

    return {
        "n_dates": n_dates,
        "treat_n": int(sum(t for _, _, t, _ in per_date)),
        "control_n": int(sum(c for _, _, _, c in per_date)),
        "observed": observed,
        "draws": draws,
        "ci_lo": float("nan") if degenerate else float(np.percentile(draws, 2.5)),
        "ci_hi": float("nan") if degenerate else float(np.percentile(draws, 97.5)),
        "flagged_dates": flagged,
        "used_symbols": sorted(used),
        "selection": _finalise_selection(sel),
    }
