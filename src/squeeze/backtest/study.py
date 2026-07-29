"""Statistics for the squeeze study.

The central quantity is the **σ-normalised right-tail rate**: the share of
observations whose forward path maximum reaches k standard deviations above the
entry price, where σ is that name's own trailing realised volatility scaled to
the horizon. Comparing that rate across grades, inside one high-short-interest
universe, isolates the grade's contribution from the fact that shorted names are
volatile.

Inference is clustered by settlement date, and that is not a formality. Every
name on a given date shares market beta, and squeeze episodes arrive in bursts —
treating ~10^5 overlapping observations as independent would shrink the standard
error by more than an order of magnitude and manufacture significance out of a
handful of distinct events. So the bootstrap resamples **dates**, never rows.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GRADES = ("SETUP", "WATCH", "NONE")
MEME_WINDOW = ("2021-01-01", "2021-03-31")


def select_top_by_si(panel: Sequence[dict], n_by_date: Dict[str, int]) -> List[dict]:
    """The n highest short-interest names on each date, n taken per date.

    This is the null the grader has to beat: no scoring, no factors, just rank
    by short interest and take the same number of names the grader took. Sizing
    per date rather than globally matters because inference bootstraps over
    dates — a baseline drawing its names from a different set of dates would not
    be comparable however well the totals matched.

    Rows without a short-interest ratio are ungradeable and therefore absent
    from the grader's cohort, so they are not selectable here either. Ties break
    on symbol so the cohort does not depend on panel ordering.
    """
    by_date: Dict[str, List[dict]] = {}
    for row in panel:
        date = row.get("date")
        if date is None or date not in n_by_date or row.get("si_ratio") is None:
            continue
        by_date.setdefault(date, []).append(row)

    picked: List[dict] = []
    for date, rows in by_date.items():
        rows.sort(key=lambda r: (-float(r["si_ratio"]), str(r.get("symbol") or "")))
        picked.extend(rows[: n_by_date[date]])
    return picked


def cohort_overlap(a: Sequence[dict], b: Sequence[dict]) -> float:
    """Share of cohort ``a`` also present in ``b``, keyed on (date, symbol).

    A high overlap means the comparison is close to a paired one: the two
    selections are largely the same names, and any difference in outcome comes
    from the minority they disagree about.
    """
    def keys(rows):
        return {(r.get("date"), r.get("symbol")) for r in rows}

    ka = keys(a)
    if not ka:
        return 0.0
    return len(ka & keys(b)) / len(ka)


def _cohort_date_counts(rows: Sequence[dict], horizon: int, k: float
                        ) -> Dict[str, Tuple[int, int, int]]:
    """Per-date (n, up-crossings, down-crossings) for one cohort."""
    zk, zdk = f"z_{horizon}", f"zdn_{horizon}"
    acc: Dict[str, List[int]] = {}
    for r in rows:
        z, zd = r.get(zk), r.get(zdk)
        if z is None or zd is None:
            continue
        cell = acc.setdefault(r["date"], [0, 0, 0])
        cell[0] += 1
        cell[1] += 1 if z >= k else 0
        cell[2] += 1 if zd <= -k else 0
    return {d: (c[0], c[1], c[2]) for d, c in acc.items()}


def si_only_comparison(panel: Sequence[dict], horizon: int, k: float = 2.0,
                       n_boot: int = 2000, seed: int = 7, block: int = 4) -> dict:
    """Does ranking on short interest alone beat the multi-factor grader?

    Takes the grader's SETUP count on each date, selects that many names by
    short interest alone, and compares the two cohorts' asymmetry
    (P(+kσ) − P(−kσ)) with a moving-block bootstrap over settlement dates.

    The bootstrap is **paired**: each draw takes a block of dates and computes
    both cohorts on those same dates. Resampling them independently would carry
    the between-date variance twice and could hide a real difference behind an
    interval built mostly out of market conditions common to both.

    A baseline that wins here is not a tuning opportunity. It means the scoring
    is decoration over its own gate and should be deleted rather than adjusted.
    """
    setup_by_date: Dict[str, int] = {}
    for r in panel:
        if r.get("grade") == "SETUP":
            setup_by_date[r["date"]] = setup_by_date.get(r["date"], 0) + 1

    grader_rows = [r for r in panel if r.get("grade") == "SETUP"]
    si_rows = select_top_by_si(panel, setup_by_date)

    g_counts = _cohort_date_counts(grader_rows, horizon, k)
    s_counts = _cohort_date_counts(si_rows, horizon, k)
    dates = sorted(set(g_counts) & set(s_counts))
    if not dates:
        return {"n_dates": 0, "grader_n": 0, "si_only_n": 0}

    arr = np.array([[*g_counts[d], *s_counts[d]] for d in dates], dtype=float)

    def _asym(block_sums: np.ndarray) -> Tuple[float, float]:
        gn, gu, gd, sn, su, sd = block_sums
        g = (gu / gn - gd / gn) if gn else float("nan")
        s = (su / sn - sd / sn) if sn else float("nan")
        return g, s

    grader_asym, si_asym = _asym(arr.sum(axis=0))

    rng = np.random.default_rng(seed)
    n_dates = len(dates)
    n_blocks = max(1, math.ceil(n_dates / block))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n_dates - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n_dates)) for s in starts])
        g, s = _asym(arr[idx].sum(axis=0))
        if not (math.isnan(g) or math.isnan(s)):
            diffs.append(s - g)

    diffs_arr = np.array(diffs) if diffs else np.array([0.0])
    return {
        "n_dates": n_dates,
        "grader_n": int(arr[:, 0].sum()),
        "si_only_n": int(arr[:, 3].sum()),
        "grader_asymmetry": float(grader_asym),
        "si_only_asymmetry": float(si_asym),
        "difference": float(si_asym - grader_asym),
        "ci_lo": float(np.percentile(diffs_arr, 2.5)),
        "ci_hi": float(np.percentile(diffs_arr, 97.5)),
        "overlap": cohort_overlap(si_rows, grader_rows),
    }


def _mask(panel: List[dict], grade: Optional[str] = None,
          date_lo: Optional[str] = None, date_hi: Optional[str] = None,
          exclude_lo: Optional[str] = None, exclude_hi: Optional[str] = None) -> List[dict]:
    out = panel
    if grade is not None:
        out = [r for r in out if r.get("grade") == grade]
    if date_lo:
        out = [r for r in out if r["date"] >= date_lo]
    if date_hi:
        out = [r for r in out if r["date"] <= date_hi]
    if exclude_lo and exclude_hi:
        out = [r for r in out if not (exclude_lo <= r["date"] <= exclude_hi)]
    return out


def tail_rate(rows: Sequence[dict], horizon: int, k: float = 2.0) -> Tuple[float, int]:
    """Share of rows whose path maximum reached k·σ, and the count used."""
    key = f"z_{horizon}"
    vals = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    if not vals:
        return float("nan"), 0
    arr = np.asarray(vals)
    return float((arr >= k).mean()), arr.size


def down_rate(rows: Sequence[dict], horizon: int, k: float = 2.0) -> Tuple[float, int]:
    """Share of rows whose path *minimum* fell k·σ. The kurtosis control."""
    key = f"zdn_{horizon}"
    vals = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    if not vals:
        return float("nan"), 0
    arr = np.asarray(vals)
    return float((arr <= -k).mean()), arr.size


def asymmetry(rows: Sequence[dict], horizon: int, k: float = 2.0) -> Tuple[float, int]:
    """Up-tail rate minus down-tail rate.

    This is the measure that actually isolates the squeeze claim. A junkier,
    more kurtotic name reaches k·σ more often in *both* directions, so a raw
    up-tail lift can be pure fat-tailedness. Only an excess of upside over
    downside is evidence that trapped shorts push the distribution's right side
    out further than its left.
    """
    up, n_up = tail_rate(rows, horizon, k)
    dn, n_dn = down_rate(rows, horizon, k)
    if not n_up or not n_dn:
        return float("nan"), 0
    return up - dn, min(n_up, n_dn)


def describe(rows: Sequence[dict], horizon: int) -> dict:
    """Full outcome profile for one group at one horizon."""
    zk, zdk = f"z_{horizon}", f"zdn_{horizon}"
    mx, end = f"max_{horizon}", f"end_{horizon}"
    z = np.asarray([r[zk] for r in rows if r.get(zk) is not None and math.isfinite(r[zk])])
    zd = np.asarray([r[zdk] for r in rows if r.get(zdk) is not None and math.isfinite(r[zdk])])
    m = np.asarray([r[mx] for r in rows if r.get(mx) is not None and math.isfinite(r[mx])])
    e = np.asarray([r[end] for r in rows if r.get(end) is not None and math.isfinite(r[end])])
    if z.size == 0:
        return {"n": 0}
    p_up = float((z >= 2.0).mean())
    p_dn = float((zd <= -2.0).mean()) if zd.size else float("nan")
    return {
        "n": int(z.size),
        "p_2sig": p_up,
        "p_3sig": float((z >= 3.0).mean()),
        "p_2sig_down": p_dn,
        "asymmetry": p_up - p_dn if p_dn == p_dn else float("nan"),
        "median_max": float(np.median(m)) if m.size else float("nan"),
        "median_end": float(np.median(e)) if e.size else float("nan"),
        "mean_end": float(e.mean()) if e.size else float("nan"),
        "p_up20": float((m >= 0.20).mean()) if m.size else float("nan"),
    }


def _by_date(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["date"], []).append(r)
    return out


METRICS = {"up": tail_rate, "down": down_rate, "asym": asymmetry}


def _date_aggregates(panel: Sequence[dict], horizon: int, k: float,
                     treat: str, control: str):
    """Per-settlement-date counts and threshold crossings for both grades.

    The bootstrap only ever needs sums, so collapsing each date to six integers
    up front turns a draw from O(rows) of Python dict work into O(dates) of
    numpy addition. On a 700k-row panel that is the difference between the
    report finishing in seconds and not finishing at all.
    """
    zk, zdk = f"z_{horizon}", f"zdn_{horizon}"
    acc: Dict[str, List[float]] = {}
    for r in panel:
        g = r.get("grade")
        if g != treat and g != control:
            continue
        z = r.get(zk)
        if z is None or not math.isfinite(z):
            continue
        zd = r.get(zdk)
        slot = acc.get(r["date"])
        if slot is None:
            slot = acc[r["date"]] = [0, 0, 0, 0, 0, 0]
        off = 0 if g == treat else 3
        slot[off] += 1
        if z >= k:
            slot[off + 1] += 1
        if zd is not None and math.isfinite(zd) and zd <= -k:
            slot[off + 2] += 1
    dates = sorted(acc)
    arr = np.array([acc[d] for d in dates], dtype=float) if dates else np.zeros((0, 6))
    return dates, arr


def _rates_from_counts(sums: np.ndarray, metric: str):
    """(treat_rate, control_rate) for a metric, from summed per-date counts."""
    t_n, t_up, t_dn, c_n, c_up, c_dn = sums
    if t_n <= 0 or c_n <= 0:
        return None, None
    if metric == "up":
        return t_up / t_n, c_up / c_n
    if metric == "down":
        return t_dn / t_n, c_dn / c_n
    return (t_up / t_n - t_dn / t_n), (c_up / c_n - c_dn / c_n)


def lift_bootstrap(panel: List[dict], horizon: int, k: float = 2.0,
                   treat: str = "SETUP", control: str = "NONE",
                   n_boot: int = 2000, seed: int = 7,
                   metric: str = "up", block: int = 4) -> dict:
    """Difference in *metric* between two grades, with a block-bootstrap CI.

    Two dependencies have to be respected or the interval is fiction:

    * **Cross-sectional** — every name on a settlement date shares market beta,
      so a draw takes a date and *all* of its rows, never individual rows.
    * **Serial** — settlement dates are ~11 trading days apart while horizons run
      to 42, so neighbouring dates observe overlapping futures. Draws therefore
      take contiguous *blocks* of ``block`` dates (moving-block bootstrap);
      ``block=1`` degenerates to the plain date-clustered version.
    """
    dates, counts = _date_aggregates(panel, horizon, k, treat, control)
    if not dates:
        return {"n_dates": 0, "treat_n": 0, "control_n": 0}

    totals = counts.sum(axis=0)
    t_rate, c_rate = _rates_from_counts(totals, metric)
    t_n, c_n = int(totals[0]), int(totals[3])
    if t_rate is None:
        return {"n_dates": len(dates), "treat_n": t_n, "control_n": c_n}
    observed = t_rate - c_rate

    rng = np.random.default_rng(seed)
    n_dates = len(dates)
    block = max(1, min(block, n_dates))
    n_blocks = max(1, n_dates // block)
    # Index matrix of contiguous blocks: row b holds the dates one draw uses.
    offsets = np.arange(block)
    lifts = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n_dates - block + 1), size=n_blocks)
        idx = np.clip((starts[:, None] + offsets).ravel(), 0, n_dates - 1)
        sums = counts[idx].sum(axis=0)
        tr, cr = _rates_from_counts(sums, metric)
        if tr is not None:
            lifts[b] = tr - cr
    lifts = lifts[np.isfinite(lifts)]
    if lifts.size == 0:
        return {"n_dates": len(dates), "observed": observed,
                "treat_rate": t_rate, "control_rate": c_rate,
                "treat_n": t_n, "control_n": c_n}

    return {
        "n_dates": len(dates),
        "treat_rate": t_rate, "control_rate": c_rate,
        "treat_n": t_n, "control_n": c_n,
        "observed": observed,
        "ci_lo": float(np.percentile(lifts, 2.5)),
        "ci_hi": float(np.percentile(lifts, 97.5)),
        # Share of resamples on the wrong side of zero — a bootstrap p-value for
        # "the lift is not positive".
        "p_le_zero": float((lifts <= 0).mean()),
    }


def monotonicity(panel: List[dict], horizon: int, k: float = 2.0) -> List[dict]:
    """Tail rate by raw evidence points — the shape test the grade alone hides."""
    out = []
    graded = [r for r in panel if r.get("points") is not None]
    by_points: Dict[int, List[dict]] = {}
    for r in graded:
        by_points.setdefault(int(r["points"]), []).append(r)
    for p in sorted(by_points):
        rate, n = tail_rate(by_points[p], horizon, k)
        dn, _ = down_rate(by_points[p], horizon, k)
        if n >= 30:
            out.append({"points": p, "rate": rate, "down": dn, "n": n})
    return out


def spearman_clustered(panel: List[dict], horizon: int,
                       n_boot: int = 300, seed: int = 11, block: int = 4) -> dict:
    """Rank correlation between evidence points and the σ-normalised outcome.

    The continuous analogue of the grade comparison: does *more* evidence track a
    bigger normalised move? Block-bootstrapped over dates for the same reason.
    Rows are held in flat numpy arrays with per-date slices so a draw is index
    concatenation rather than Python list building.
    """
    key = f"z_{horizon}"
    rows = [r for r in panel
            if r.get(key) is not None and math.isfinite(r[key])
            and r.get("points") is not None]
    if len(rows) < 100:
        return {"n": len(rows)}

    rows.sort(key=lambda r: r["date"])
    x_all = np.array([r["points"] for r in rows], dtype=float)
    y_all = np.array([r[key] for r in rows], dtype=float)
    dates, starts = [], []
    for i, r in enumerate(rows):
        if not dates or r["date"] != dates[-1]:
            dates.append(r["date"])
            starts.append(i)
    starts.append(len(rows))
    slices = [np.arange(starts[i], starts[i + 1]) for i in range(len(dates))]

    def rho(xs, ys):
        if xs.size < 10 or xs.std() == 0 or ys.std() == 0:
            return np.nan
        xr = np.argsort(np.argsort(xs)).astype(float)
        yr = np.argsort(np.argsort(ys)).astype(float)
        return float(np.corrcoef(xr, yr)[0, 1])

    observed = rho(x_all, y_all)
    rng = np.random.default_rng(seed)
    n_dates = len(dates)
    block = max(1, min(block, n_dates))
    n_blocks = max(1, n_dates // block)
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts_b = rng.integers(0, max(1, n_dates - block + 1), size=n_blocks)
        picks = np.clip((starts_b[:, None] + np.arange(block)).ravel(), 0, n_dates - 1)
        idx = np.concatenate([slices[j] for j in picks])
        vals[b] = rho(x_all[idx], y_all[idx])
    vals = vals[np.isfinite(vals)]
    return {"n": len(rows), "rho": observed,
            "ci_lo": float(np.percentile(vals, 2.5)) if vals.size else float("nan"),
            "ci_hi": float(np.percentile(vals, 97.5)) if vals.size else float("nan")}


def coverage_bias(panel: List[dict], horizon: int, k: float = 2.0) -> dict:
    """Do rows lost to missing shares-outstanding differ in outcome?

    The gate needs EDGAR shares outstanding, which is absent for ETFs and funds
    (correctly — they are not squeeze candidates) but also for some companies
    that delisted, which is the dangerous kind of missing. Because prices exist
    for dropped rows even when shares data does not, their forward outcomes are
    observable, so the exclusion can be tested rather than assumed harmless.

    Similar tail rates on both sides mean the join is roughly random with
    respect to outcome. A large gap means the headline result is conditioned on
    a survivor-flavoured subsample and should be read with that in mind.
    """
    kept = [r for r in panel if r.get("si_ratio") is not None]
    lost = [r for r in panel if r.get("si_ratio") is None]
    k_rate, k_n = tail_rate(kept, horizon, k)
    l_rate, l_n = tail_rate(lost, horizon, k)
    kd, _ = down_rate(kept, horizon, k)
    ld, _ = down_rate(lost, horizon, k)
    return {
        "kept_n": k_n, "lost_n": l_n,
        "kept_pct": 100.0 * k_n / (k_n + l_n) if (k_n + l_n) else float("nan"),
        "kept_up": k_rate, "lost_up": l_rate,
        "kept_down": kd, "lost_down": ld,
        "up_gap": k_rate - l_rate if l_n else float("nan"),
    }


def robustness(panel: List[dict], horizon: int, k: float = 2.0,
               n_boot: int = 1000, metric: str = "up") -> Dict[str, dict]:
    """The same lift under cuts that would expose a one-episode result."""
    cuts = {
        "full sample": panel,
        "ex 2021 meme window": _mask(panel, exclude_lo=MEME_WINDOW[0], exclude_hi=MEME_WINDOW[1]),
        "2018-2022 (train)": _mask(panel, date_hi="2022-12-31"),
        "2023-2026 (holdout)": _mask(panel, date_lo="2023-01-01"),
    }
    return {name: lift_bootstrap(rows, horizon, k, n_boot=n_boot, metric=metric)
            for name, rows in cuts.items() if rows}
