"""Shared purged backtest primitives, clustered bootstrap, and capacity simulation.

Designed to integrate directly with `src/alloc/report.py::summarise` and
`promotion_verdict` (via matching `EquityTrade` fields) while supporting
equity-specific cluster units (ISO weeks, calendar months, settlement dates,
and distinct tickers).
"""
from __future__ import annotations

import bisect
import collections
import datetime as dt
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EquityTrade:
    """Single simulated equity/ETF trade compatible with `src.alloc.report`."""
    ticker: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return: float                 # fractional net return after friction
    bench_return: Optional[float]     # matched-window benchmark return
    exit_reason: str
    stratum: str = "broad"
    capital_at_risk: float = 1000.0   # notional basis for equal-weighted alloc stats
    score_tiebreak: float = 0.0       # pre-declared capacity tie-breaker (lower wins)
    cluster_override: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        return self.net_return * self.capital_at_risk

    @property
    def excess_return(self) -> Optional[float]:
        if self.bench_return is None:
            return None
        return self.net_return - self.bench_return


def iso_week_key(date_str: str) -> str:
    d = dt.date.fromisoformat(date_str[:10])
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def month_key(date_str: str) -> str:
    return date_str[:7]


class BenchmarkSeries:
    """Fast exact-date / as-of benchmark close lookup (SPY, XBI, IWM, etc.)."""

    def __init__(self, dates: Sequence[str], closes: Sequence[float]):
        self.dates = list(dates)
        self.closes = np.asarray(closes, dtype=float)
        self._index = {d: i for i, d in enumerate(self.dates)}

    def close_asof(self, date_str: str) -> Optional[float]:
        i = self._index.get(date_str)
        if i is None:
            pos = bisect.bisect_right(self.dates, date_str) - 1
            if pos < 0:
                return None
            i = pos
        val = float(self.closes[i])
        return val if math.isfinite(val) and val > 0.0 else None

    def window_return(self, entry_date: str, exit_date: str) -> Optional[float]:
        c0 = self.close_asof(entry_date)
        c1 = self.close_asof(exit_date)
        if c0 is None or c1 is None or c0 <= 0.0:
            return None
        return c1 / c0 - 1.0


def load_equity_ohlcv(
    db_path: str = "data/equity_ohlcv.db",
    symbols: Optional[Iterable[str]] = None,
) -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    """Load `{ticker: (dates, closes, volumes)}` from `data/equity_ohlcv.db`."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        keep = {s.upper() for s in symbols} if symbols is not None else None
        cur = con.execute(
            "SELECT ticker, date, close, volume FROM ohlcv ORDER BY ticker, date"
        )
        dates: Dict[str, List[str]] = {}
        closes: Dict[str, List[float]] = {}
        vols: Dict[str, List[float]] = {}
        for sym, d, c, v in cur:
            s = str(sym).upper()
            if keep is not None and s not in keep:
                continue
            dates.setdefault(s, []).append(str(d))
            closes.setdefault(s, []).append(float(c))
            vols.setdefault(s, []).append(float(v) if v is not None else 0.0)
    finally:
        con.close()
    return {
        s: (dates[s], np.asarray(closes[s], dtype=float), np.asarray(vols[s], dtype=float))
        for s in dates
    }


def _profit_factor(returns: np.ndarray) -> float:
    gains = float(np.sum(returns[returns > 0.0]))
    losses = float(np.abs(np.sum(returns[returns < 0.0])))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def cluster_bootstrap_stats(
    trades: Sequence[EquityTrade],
    cluster_fn: Callable[[EquityTrade], str],
    *,
    n_boot: int = 10_000,
    seed: int = 20261124,
    alpha: float = 0.05,
    min_clusters: int = 20,
) -> Dict[str, Any]:
    """Cluster-resampled bootstrap for mean net return, PF, and benchmark excess.

    Each draw resamples entire clusters with replacement (preserving within-cluster
    correlation across simultaneous market moves) and recomputes trade-weighted
    mean, PF, and excess over the drawn clusters.
    """
    by_cluster: Dict[str, List[EquityTrade]] = collections.defaultdict(list)
    for t in trades:
        by_cluster[cluster_fn(t)].append(t)

    keys = sorted(by_cluster.keys())
    k = len(keys)
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "n_trades": 0, "n_clusters": 0, "insufficient": True,
            "mean_ret": None, "mean_ret_ci": None,
            "pf": None, "pf_ci": None,
            "mean_excess": None, "mean_excess_ci": None,
            "win_rate": None, "median_ret": None,
        }

    all_r = np.array([t.net_return for t in trades], dtype=float)
    all_x = np.array([t.excess_return for t in trades if t.excess_return is not None], dtype=float)
    point_mean = float(np.mean(all_r))
    point_med = float(np.median(all_r))
    point_pf = _profit_factor(all_r)
    point_wr = float(np.mean(all_r > 0.0))
    point_excess = float(np.mean(all_x)) if all_x.size else None

    out: Dict[str, Any] = {
        "n_trades": n_trades,
        "n_clusters": k,
        "insufficient": k < min_clusters,
        "mean_ret": point_mean,
        "median_ret": point_med,
        "win_rate": point_wr,
        "pf": point_pf,
        "mean_excess": point_excess,
        "mean_ret_ci": None,
        "pf_ci": None,
        "mean_excess_ci": None,
        "alpha": alpha,
    }
    if k < min_clusters:
        return out

    # Pre-aggregate per cluster for O(n_boot * k) speed even with 100k+ trades!
    c_n = np.empty(k, dtype=float)
    c_sum_r = np.empty(k, dtype=float)
    c_pos_r = np.empty(k, dtype=float)
    c_neg_r = np.empty(k, dtype=float)
    c_nx = np.empty(k, dtype=float)
    c_sum_x = np.empty(k, dtype=float)

    for idx, key in enumerate(keys):
        ct = by_cluster[key]
        rs = np.array([t.net_return for t in ct], dtype=float)
        xs = np.array([t.excess_return for t in ct if t.excess_return is not None], dtype=float)
        c_n[idx] = rs.size
        c_sum_r[idx] = float(np.sum(rs))
        c_pos_r[idx] = float(np.sum(rs[rs > 0.0]))
        c_neg_r[idx] = float(np.abs(np.sum(rs[rs < 0.0])))
        c_nx[idx] = xs.size
        c_sum_x[idx] = float(np.sum(xs)) if xs.size else 0.0

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, k, size=(n_boot, k))

    tot_n = np.sum(c_n[idx_mat], axis=1)
    boot_mean = np.sum(c_sum_r[idx_mat], axis=1) / tot_n
    tot_pos = np.sum(c_pos_r[idx_mat], axis=1)
    tot_neg = np.sum(c_neg_r[idx_mat], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_pf = np.where(tot_neg > 0.0, tot_pos / tot_neg, np.inf)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    pf_method = "nearest" if np.any(np.isinf(boot_pf)) else "linear"
    out["mean_ret_ci"] = (float(np.percentile(boot_mean, lo_q)), float(np.percentile(boot_mean, hi_q)))
    out["pf_ci"] = (
        float(np.percentile(boot_pf, lo_q, method=pf_method)),
        float(np.percentile(boot_pf, hi_q, method=pf_method)),
    )

    tot_nx = np.sum(c_nx[idx_mat], axis=1)
    if np.all(tot_nx > 0):
        boot_ex = np.sum(c_sum_x[idx_mat], axis=1) / tot_nx
        out["mean_excess_ci"] = (float(np.percentile(boot_ex, lo_q)), float(np.percentile(boot_ex, hi_q)))
    return out


def moving_block_bootstrap_stats(
    trades: Sequence[EquityTrade],
    cluster_fn: Callable[[EquityTrade], str],
    *,
    block_len: int = 4,
    n_boot: int = 10_000,
    seed: int = 20261124,
    alpha: float = 0.05,
    min_clusters: int = 20,
) -> Dict[str, Any]:
    """Moving-block bootstrap over chronologically ordered clusters.

    Preserves multi-period overlap dependence (e.g., 42-day holds across
    bi-monthly FINRA settlement dates or 90-day holds across monthly entries).
    """
    by_cluster: Dict[str, List[EquityTrade]] = collections.defaultdict(list)
    for t in trades:
        by_cluster[cluster_fn(t)].append(t)

    keys = sorted(by_cluster.keys())
    k = len(keys)
    n_trades = len(trades)
    if n_trades == 0:
        return cluster_bootstrap_stats(trades, cluster_fn, n_boot=n_boot, seed=seed, alpha=alpha, min_clusters=min_clusters)

    base = cluster_bootstrap_stats(
        trades, cluster_fn, n_boot=1, seed=seed, alpha=alpha, min_clusters=k + 1
    )
    base["insufficient"] = k < min_clusters
    base["block_len"] = block_len
    if k < min_clusters:
        return base

    c_n = np.empty(k, dtype=float)
    c_sum_r = np.empty(k, dtype=float)
    c_pos_r = np.empty(k, dtype=float)
    c_neg_r = np.empty(k, dtype=float)
    c_nx = np.empty(k, dtype=float)
    c_sum_x = np.empty(k, dtype=float)
    for idx, key in enumerate(keys):
        ct = by_cluster[key]
        rs = np.array([t.net_return for t in ct], dtype=float)
        xs = np.array([t.excess_return for t in ct if t.excess_return is not None], dtype=float)
        c_n[idx] = rs.size
        c_sum_r[idx] = float(np.sum(rs))
        c_pos_r[idx] = float(np.sum(rs[rs > 0.0]))
        c_neg_r[idx] = float(np.abs(np.sum(rs[rs < 0.0])))
        c_nx[idx] = xs.size
        c_sum_x[idx] = float(np.sum(xs)) if xs.size else 0.0

    b = max(1, min(block_len, k))
    n_blocks = int(math.ceil(k / b))
    rng = np.random.default_rng(seed)
    # Circular moving blocks over ordered clusters 0..k-1
    starts = rng.integers(0, k, size=(n_boot, n_blocks))
    offsets = np.arange(b)[None, None, :]
    idx_mat = ((starts[:, :, None] + offsets) % k).reshape(n_boot, n_blocks * b)[:, :k]

    tot_n = np.sum(c_n[idx_mat], axis=1)
    boot_mean = np.sum(c_sum_r[idx_mat], axis=1) / tot_n
    tot_pos = np.sum(c_pos_r[idx_mat], axis=1)
    tot_neg = np.sum(c_neg_r[idx_mat], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_pf = np.where(tot_neg > 0.0, tot_pos / tot_neg, np.inf)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    pf_method = "nearest" if np.any(np.isinf(boot_pf)) else "linear"
    base["mean_ret_ci"] = (float(np.percentile(boot_mean, lo_q)), float(np.percentile(boot_mean, hi_q)))
    base["pf_ci"] = (
        float(np.percentile(boot_pf, lo_q, method=pf_method)),
        float(np.percentile(boot_pf, hi_q, method=pf_method)),
    )
    tot_nx = np.sum(c_nx[idx_mat], axis=1)
    if np.all(tot_nx > 0):
        boot_ex = np.sum(c_sum_x[idx_mat], axis=1) / tot_nx
        base["mean_excess_ci"] = (float(np.percentile(boot_ex, lo_q)), float(np.percentile(boot_ex, hi_q)))
    return base


def simulate_capacity_book(
    trades: Sequence[EquityTrade],
    *,
    max_concurrent: int = 5,
    slot_weight: float = 0.20,
    spy_series: Optional[BenchmarkSeries] = None,
) -> Dict[str, Any]:
    """Simulate a non-leveraged, capacity-capped portfolio of equity trades.

    Deterministic selection when >available slots fire on the same `entry_date`:
    sorts ascending by `(score_tiebreak, ticker)`. Books realized return on exit
    date onto compounded equity (conservative: no intraday compounding).
    """
    if not trades:
        return {"n_taken": 0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total_ret": 0.0}

    by_entry: Dict[str, List[EquityTrade]] = collections.defaultdict(list)
    for t in trades:
        by_entry[t.entry_date].append(t)

    active: List[EquityTrade] = []
    taken: List[EquityTrade] = []
    for day in sorted(by_entry.keys()):
        active = [t for t in active if t.exit_date > day]
        open_tickers = {t.ticker for t in active}
        free = max_concurrent - len(active)
        if free <= 0:
            continue
        cands = sorted(by_entry[day], key=lambda x: (x.score_tiebreak, x.ticker))
        for c in cands:
            if free <= 0:
                break
            if c.ticker in open_tickers:
                continue
            active.append(c)
            taken.append(c)
            open_tickers.add(c.ticker)
            free -= 1

    if not taken:
        return {"n_taken": 0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total_ret": 0.0}

    # Build daily realized-on-exit equity curve across trading days.
    by_exit: Dict[str, List[float]] = collections.defaultdict(list)
    for t in taken:
        by_exit[t.exit_date].append(slot_weight * t.net_return)

    first_day = min(t.entry_date for t in taken)
    last_day = max(t.exit_date for t in taken)
    if spy_series is not None:
        cal_days = [d for d in spy_series.dates if first_day <= d <= last_day]
    else:
        cal_days = sorted(by_exit.keys())

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    daily_rets: List[float] = []
    for d in cal_days:
        day_r = sum(by_exit.get(d, ()))
        daily_rets.append(day_r)
        equity *= (1.0 + day_r)
        if equity > peak:
            peak = equity
        dd = equity / peak - 1.0
        if dd < max_dd:
            max_dd = dd

    years = max((dt.date.fromisoformat(last_day) - dt.date.fromisoformat(first_day)).days / 365.25, 1e-3)
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    dr = np.asarray(daily_rets, dtype=float)
    sh = float((dr.mean() / dr.std(ddof=1)) * np.sqrt(252.0)) if dr.size > 5 and dr.std(ddof=1) > 0 else 0.0

    spy_total = spy_series.window_return(first_day, last_day) if spy_series is not None else None
    spy_cagr = ((1.0 + spy_total) ** (1.0 / years) - 1.0) if (spy_total is not None and spy_total > -1.0) else None
    return {
        "n_taken": len(taken),
        "first_day": first_day,
        "last_day": last_day,
        "years": round(years, 2),
        "total_ret": equity - 1.0,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sh,
        "spy_cagr": spy_cagr,
    }
