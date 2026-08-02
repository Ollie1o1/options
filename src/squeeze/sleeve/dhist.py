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
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.squeeze.sleeve import matching, payoff

SEED = 12345
N_BOOT = 4000
BLOCK_BY_HORIZON = {21: 2, 42: 4}

TREATED_DECILE = 10
CONTROL_DECILES = (1, 2, 3, 4, 5)   # 6-9 excluded: partially treated


def _unit(row: dict) -> matching.Unit:
    return matching.Unit(key=row["symbol"], rv=float(row["rv"]),
                         log_mcap=float(row["log_mcap"]),
                         log_price=float(row["log_price"]))


def _mean_return(rows: Sequence[dict], keys: Sequence[str], horizon: int,
                 variant: str) -> Sequence[float]:
    by_symbol = {r["symbol"]: r for r in rows}
    out: List[float] = []
    for key in keys:
        row = by_symbol.get(key)
        if row is None:
            continue
        got = payoff.synthetic_call_return(
            row["path"], _entry_spot(row),
            float(row["sigma_d"]), float(row["iv"]),
            horizon_bars=horizon, variant=variant)
        if got is not None:
            out.append(got)
    return out


def _entry_spot(row: dict) -> float:
    """Entry spot. Rows carry the post-entry path, so the entry level is
    supplied explicitly when present and otherwise inferred as the level the
    path starts from."""
    if row.get("spot0") is not None:
        return float(row["spot0"])
    return float(row["path"][0])


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

    for date in sorted(by_date):
        day = by_date[date]
        treated = [r for r in day if r["si_decile"] == TREATED_DECILE]
        controls = [r for r in day if r["si_decile"] in CONTROL_DECILES]
        if not treated or not controls:
            continue
        result = matching.match([_unit(r) for r in treated],
                                [_unit(r) for r in controls])
        if not matching.is_valid(result):
            flagged.append(date)
            continue

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
                "flagged_dates": flagged, "used_symbols": sorted(used)}

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

    return {
        "n_dates": n_dates,
        "treat_n": int(sum(t for _, _, t, _ in per_date)),
        "control_n": int(sum(c for _, _, _, c in per_date)),
        "observed": observed,
        "draws": draws,
        "ci_lo": float(np.percentile(draws, 2.5)) if draws.size else float("nan"),
        "ci_hi": float(np.percentile(draws, 97.5)) if draws.size else float("nan"),
        "flagged_dates": flagged,
        "used_symbols": sorted(used),
    }
