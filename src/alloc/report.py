"""Evaluate a strategy end to end, and decide whether it may be promoted.

The promotion bar is deliberately hard to clear, because the failure mode this
system keeps hitting is a plausible-looking result produced by search rather
than by edge. Every threshold here exists because something got past a weaker
version of it:

  DSR > 0.5   a $25-wide spread read DSR 0.921 alone and 0.432 deflated
  t >= 3.0    Harvey's hurdle, not the conventional 2.0, given how many factors
              have already been tested against this data
  clustered t positions sharing an entry day share that day's move, and
              correcting for it roughly halves every t-statistic
  BROAD > 0   an edge that lives only in the famous names is an attention
              artifact, not a premium

  (PBO is not measured — see `promotion_verdict`.)
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as _sps

from src.alloc.portfolio import apply_capacity, capacity_stats
from src.alloc.validate import deflated_sharpe, effective_n, sharpe

MIN_DSR = 0.5
MIN_TSTAT = 3.0

# Below this many closed trades a verdict is refused outright, whatever the
# statistics say. Learned from a starved run that closed 3 trades, reported
# DSR 0.996 and t=12.56, and was graded `liquid_only` — every number correct,
# the conclusion worthless. A deflated Sharpe is a distributional claim, and 3
# observations do not carry one.
MIN_N = 20


def _returns(trades: Sequence[Any]) -> np.ndarray:
    return np.array([float(t.pnl or 0.0) / float(t.capital_at_risk)
                     for t in trades if t.capital_at_risk], dtype=float)


def clustered_tstat(trades: Sequence[Any]) -> float:
    """t-statistic over entry-day means rather than individual trades.

    Trades opened on the same day share that day's market move, so treating
    them as independent overstates significance — by roughly a factor of two on
    this data.
    """
    by_day: Dict[str, List[float]] = collections.defaultdict(list)
    for t in trades:
        if t.capital_at_risk:
            by_day[str(t.entry_date)].append(
                float(t.pnl or 0.0) / float(t.capital_at_risk))
    days = np.array([np.mean(v) for v in by_day.values()], dtype=float)
    if days.size < 3 or days.std(ddof=1) == 0:
        return 0.0
    return float(days.mean() / (days.std(ddof=1) / np.sqrt(days.size)))


def summarise(trades: Sequence[Any], n_trials: int,
              max_capital: float = 4000.0,
              max_concurrent: int = 3) -> Dict[str, Any]:
    """Everything needed to judge a strategy, including what it cannot do."""
    closed = [t for t in trades
              if t.exit_date and t.exit_reason != "ticker_ended"]
    if len(closed) < 3:
        return {"n": len(closed), "insufficient": True}

    r = _returns(closed)
    # `_returns` drops trades with no capital_at_risk, so the cluster count
    # must be taken over the SAME subset that formed `r` — counting clusters
    # of rows that are not in the return series is the identical defect one
    # level down.
    priced = [t for t in closed if t.capital_at_risk]
    n_eff = effective_n([t.entry_date for t in priced],
                        [t.exit_date for t in priced])
    wins = [t for t in closed if (t.pnl or 0) > 0]
    naive_t = (float(r.mean() / (r.std(ddof=1) / np.sqrt(r.size)))
               if r.std(ddof=1) > 0 else 0.0)

    by_stratum: Dict[str, Dict[str, float]] = {}
    for t in closed:
        s = t.stratum or "unknown"
        d = by_stratum.setdefault(s, {"n": 0, "pnl": 0.0})
        d["n"] += 1
        d["pnl"] += float(t.pnl or 0.0)

    return {
        "n": len(closed),
        "win_rate": round(100.0 * len(wins) / len(closed), 2),
        "mean_return_on_capital": round(float(r.mean()), 6),
        "sharpe": round(sharpe(r), 4),
        "tstat": round(naive_t, 3),
        "tstat_clustered": round(clustered_tstat(closed), 3),
        "skew": round(float(_sps.skew(r)), 3),
        "n_trials": n_trials,
        "n_eff": n_eff,
        "dsr": round(deflated_sharpe(r, n_trials, n_eff), 4),
        "dsr_undeflated": round(deflated_sharpe(r, 1, n_eff), 4),
        "by_stratum": by_stratum,
        # Capacity is measured on the trades an account of this size could
        # ACTUALLY have held, not on every signal the engine generated.
        "capacity": capacity_stats(
            apply_capacity(closed, max_concurrent, max_capital)[0],
            max_capital),
        "insufficient": False,
    }


def promotion_verdict(result: Dict[str, Any]) -> str:
    """`promote`, `liquid_only`, `reject`, or `insufficient`.

    A strategy clearing everything except BROAD is recorded as `liquid_only`
    and never silently promoted: "works on the names we already knew about" is
    a materially weaker claim than "works".

    `insufficient` is deliberately NOT `reject`: "we could not measure this" and
    "we measured this and it failed" are different claims, and collapsing them
    loses the one that tells you to go and get more data.

    PBO is deliberately NOT among these conditions. It gated here for a long
    time via `result.get("pbo", 0.0)`, but `summarise` never set that key, so
    the check always read 0.0 and never once fired. Measuring it for real needs
    in-sample/out-of-sample pairs across CPCV paths, which this system does not
    build. Three conditions that run beat four where one is decorative.
    """
    if result.get("insufficient"):
        return "reject"
    n = result.get("n")
    if n is not None and int(n) < MIN_N:
        return "insufficient"
    if (result.get("dsr", 0.0) < MIN_DSR
            or abs(result.get("tstat_clustered", 0.0)) < MIN_TSTAT
            or result.get("tstat_clustered", 0.0) < 0):
        return "reject"
    broad = result.get("by_stratum", {}).get("broad", {})
    if broad.get("pnl", 0.0) <= 0:
        return "liquid_only"
    return "promote"


def format_summary(label: str, result: Dict[str, Any]) -> str:
    if result.get("insufficient"):
        return f"{label:<28} n={result['n']} — too few trades to judge"
    cap = result["capacity"]
    return (
        f"{label:<28} n={result['n']:>5} win={result['win_rate']:>5.1f}% "
        f"RoC={100*result['mean_return_on_capital']:>6.2f}% "
        f"t={result['tstat']:>5.2f} tc={result['tstat_clustered']:>5.2f} "
        f"skew={result['skew']:>6.2f} DSR={result['dsr']:.3f} "
        f"[{promotion_verdict(result)}]  "
        f"{cap['trades_per_year']:.0f} trades/yr, "
        f"{100*cap['return_on_cap']:.1f}%/yr on account"
    )
