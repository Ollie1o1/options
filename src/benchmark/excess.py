"""Per-trade and daily excess of the book over each null, plus the one
dollar figure that carries no sampling noise: cumulative realised P&L.

**This module does not compute a verdict.** A paired comparison between the
book's trades and SPY has no shared path to cancel (unlike
`policy_lab`, where both arms replay the SAME position on the SAME path) —
the book's per-trade noise dwarfs SPY's window-to-window move, so the
per-trade and daily excess estimates below carry confidence intervals wide
enough to contain zero on this corpus, and `n_needed_for_power` says how
much more data would be needed to resolve them. None of that makes the
comparison worthless: it is an honest, reproducible point estimate and a
permanent fixture for accumulating forward observations, not a hypothesis
test with a pass/fail bar. Reporting it as a significance test would be
exactly the fabricated-precision failure this whole engine exists to avoid.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.benchmark.prices import DEFAULT_TOLERANCE_DAYS, nearest_price
from src.policy_lab.stats import (
    CLUSTER_COL, cluster_bootstrap_mean_ci, paired_frame,
)
from src.policy_lab.types import Outcome, PricePath

# ---------------------------------------------------------------------------
# Loading closed trades
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    """What the loader kept and what it threw away, by reason."""
    loaded: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


@dataclass(frozen=True)
class Trade:
    """One closed position from `paper_trades.db::trades`.

    `entry_date`/`exit_date` are ISO `YYYY-MM-DD` (the ledger's `exit_date`
    carries a timestamp; only the calendar date is used for SPY matching).
    """
    position_id: str
    symbol: str
    entry_date: str
    exit_date: str
    pnl_usd: float
    capital_at_risk: float


def _ro(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_closed_trades(ledger_db_path: str) -> Tuple[List[Trade], LoadReport]:
    """Every `status='CLOSED'` row, kept only when fully usable.

    Drop reasons (each counted, never silently skipped):
      - `exit_date_missing`: `exit_date IS NULL`.
      - `capital_at_risk_missing_or_zero`: `capital_at_risk IS NULL` or
        `<= 0` — an unbounded or absent risk figure cannot form a return.
        Zero is a drop reason here, never confused with "no data".
      - `pnl_usd_missing`: `pnl_usd IS NULL` — `0.0` is a real, kept result
        (a scratch trade), so this checks `IS NULL`, never bare truthiness.
      - `entry_date_missing`: `date IS NULL`.
    """
    con = _ro(ledger_db_path)
    try:
        rows = con.execute(
            """SELECT entry_id, ticker, date, exit_date, pnl_usd, capital_at_risk
                 FROM trades
                WHERE status = 'CLOSED'"""
        ).fetchall()
    finally:
        con.close()

    report = LoadReport()
    trades: List[Trade] = []
    for entry_id, ticker, entry_date, exit_date, pnl_usd, car in rows:
        if entry_date is None:
            report.drop("entry_date_missing")
            continue
        if exit_date is None:
            report.drop("exit_date_missing")
            continue
        if car is None or car <= 0:
            report.drop("capital_at_risk_missing_or_zero")
            continue
        if pnl_usd is None:
            report.drop("pnl_usd_missing")
            continue
        trades.append(Trade(
            position_id=f"{ticker}|{entry_date[:10]}|{entry_id}",
            symbol=ticker,
            entry_date=entry_date[:10],
            exit_date=exit_date[:10],
            pnl_usd=float(pnl_usd),
            capital_at_risk=float(car),
        ))
        report.loaded += 1
    return trades, report


# ---------------------------------------------------------------------------
# Per-trade paired frame (reuses policy_lab.stats machinery)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchStats:
    """How many trades matched the SPY series exactly vs. within tolerance.

    A trade needs BOTH its entry and exit date matched; `exact` requires
    both to be exact-date matches, `tolerant` means at least one used the
    date-tolerance fallback, and `missing` means at least one could not be
    matched at all within `tolerance_days` (dropped from every downstream
    calculation, and counted here rather than silently omitted).
    """
    exact: int = 0
    tolerant: int = 0
    missing: int = 0
    tolerance_days: int = DEFAULT_TOLERANCE_DAYS


NULL_CASH = "cash"
NULL_SPY = "spy_buy_hold"


def build_paired_frame(trades: Sequence[Trade], spy_series: Dict[str, float],
                        tolerance_days: int = DEFAULT_TOLERANCE_DAYS
                        ) -> Tuple[Dict[str, pd.DataFrame], MatchStats]:
    """One `PricePath`/`Outcome` triple per trade, reused through
    `policy_lab.stats.paired_frame` to get the `(symbol, entry_date)`
    clustering for free — this package does not reimplement that grouping.

    Returns `{null_name: paired_frame}`, where each frame's `d` column is
    `book_ret - null_ret` (achieved by putting the null in `base_outcomes`
    and the book in `alt_outcomes`: `paired_frame`'s `d = alt - base`).
    `book_ret = pnl_usd / capital_at_risk`; `spy_ret = spy[exit]/spy[entry]
    - 1` (a SAME-DAY round trip returning `0.0` is a real measurement, not a
    missing value); `cash_ret = 0.0` always.
    """
    paths: List[PricePath] = []
    book_out: Dict[str, Outcome] = {}
    cash_out: Dict[str, Outcome] = {}
    spy_out: Dict[str, Outcome] = {}

    exact = tolerant = missing = 0
    for t in trades:
        entry_match = nearest_price(spy_series, t.entry_date, tolerance_days)
        exit_match = nearest_price(spy_series, t.exit_date, tolerance_days)
        if entry_match is None or exit_match is None:
            missing += 1
            continue
        if entry_match.offset_days == 0 and exit_match.offset_days == 0:
            exact += 1
        else:
            tolerant += 1

        book_ret = t.pnl_usd / t.capital_at_risk
        spy_ret = exit_match.price / entry_match.price - 1.0

        paths.append(PricePath(
            position_id=t.position_id, symbol=t.symbol, strategy="benchmark",
            entry_date=t.entry_date, entry_price=0.0,
            capital_at_risk=t.capital_at_risk, is_credit=False, points=(),
            actual_exit_date=t.exit_date, actual_pnl_frac=book_ret,
            corpus="benchmark"))
        book_out[t.position_id] = Outcome(
            t.position_id, t.exit_date, "actual", book_ret, 0)
        cash_out[t.position_id] = Outcome(
            t.position_id, t.exit_date, "cash", 0.0, 0)
        spy_out[t.position_id] = Outcome(
            t.position_id, t.exit_date, "spy_buy_hold", spy_ret, 0)

    match_stats = MatchStats(exact=exact, tolerant=tolerant, missing=missing,
                              tolerance_days=tolerance_days)
    frames = {
        NULL_CASH: paired_frame(paths, cash_out, book_out),
        NULL_SPY: paired_frame(paths, spy_out, book_out),
    }
    return frames, match_stats


# ---------------------------------------------------------------------------
# Aggregation: descriptive, with CI shown, never a verdict
# ---------------------------------------------------------------------------

# Standard two-sided-alpha / power z-values for the sample-size estimate
# below (alpha=0.05 -> z=1.959964; power=0.80 -> z=0.841621).
_Z_ALPHA_2 = 1.959963984540054
_Z_POWER_80 = 0.8416212335729143


def n_needed_for_power(effect: float, sd: float, alpha: float = 0.05,
                        power: float = 0.80) -> float:
    """Sample size to resolve a mean `effect` at std-dev `sd`, two-sided.

    Standard one-sample-mean power formula:
    `n = ((z_(1-alpha/2) + z_power)^2 * sd^2) / effect^2`. Returns `inf`
    when `effect` is `0.0` (nothing to resolve) rather than dividing by
    zero.
    """
    if effect == 0.0:
        return float("inf")
    if alpha == 0.05 and power == 0.80:
        z_sum = _Z_ALPHA_2 + _Z_POWER_80
    else:
        z_sum = (scipy_stats.norm.ppf(1 - alpha / 2)
                 + scipy_stats.norm.ppf(power))
    return (z_sum ** 2) * (sd ** 2) / (effect ** 2)


@dataclass(frozen=True)
class AggregateResult:
    """A descriptive summary of one excess series — never a pass/fail verdict.

    `ci_lo`/`ci_hi` are `None` when they could not be computed (e.g. fewer
    than 2 clusters/observations); render that as "not computed", never as
    zero.
    """
    n_rows: int
    n_clusters: int
    mean: float
    sd: float
    mean_book: Optional[float]     # mean(r_alt) — the book's own mean return
    mean_null: Optional[float]     # mean(r_base) — the null's mean return
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    ci_contains_zero: Optional[bool]
    variance_ratio: float          # sd(d) / sd(book_ret); ~1.0 = pairing removed nothing
    corr_book_null: Optional[float]
    n_needed_80pct_power: float


def paired_aggregate(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0
                      ) -> AggregateResult:
    """Cluster-bootstrapped mean and 95% CI of `df["d"]` (the excess),
    clustered on `(symbol, entry_date)` via `policy_lab.stats`.
    """
    n_rows = len(df)
    if n_rows == 0:
        return AggregateResult(0, 0, 0.0, 0.0, None, None, None, None, None,
                                1.0, None, float("inf"))
    n_clusters = int(df[CLUSTER_COL].nunique())
    d = df["d"].to_numpy(dtype="float64")
    mean = float(d.mean())
    sd = float(np.std(d, ddof=1)) if n_rows >= 2 else 0.0
    ci_lo, ci_hi = cluster_bootstrap_mean_ci(df, "d", n_boot=n_boot, seed=seed)
    ci_contains_zero = (ci_lo <= 0.0 <= ci_hi) if (
        ci_lo is not None and ci_hi is not None) else None
    r_base = df["r_base"].to_numpy(dtype="float64")
    r_alt = df["r_alt"].to_numpy(dtype="float64")
    mean_null = float(r_base.mean())
    mean_book = float(r_alt.mean())
    # sd(d) / sd(book_ret): NOT policy_lab.stats.variance_reduction, which
    # divides by sd(r_base) -- here `r_base` is the null (Cash/SPY), whose
    # variance is tiny (SPY) or exactly zero (Cash) and so is the wrong
    # denominator: it would report a ratio near 35 for SPY (nonsense) and a
    # vacuous 1.0 for Cash (its "no dispersion to remove" default). The
    # quantity worth reporting is how much of the BOOK's own variance the
    # pairing removed, i.e. sd(d) against sd(book_ret) = sd(r_alt).
    sd_book = float(np.std(r_alt, ddof=1)) if n_rows >= 2 else 0.0
    var_ratio = (sd / sd_book) if sd_book > 0 else 1.0
    corr = None
    if n_rows >= 2 and np.std(r_base) > 0 and np.std(r_alt) > 0:
        corr = float(np.corrcoef(r_base, r_alt)[0, 1])
    n_needed = n_needed_for_power(mean, sd)
    return AggregateResult(n_rows, n_clusters, mean, sd, mean_book, mean_null,
                            ci_lo, ci_hi, ci_contains_zero, var_ratio, corr,
                            n_needed)


# ---------------------------------------------------------------------------
# Concurrent exposure (secondary capital convention)
# ---------------------------------------------------------------------------


def concurrent_exposure_daily(trades: Sequence[Trade]) -> "pd.Series":
    """Capital-at-risk deployed on each calendar day, as a step function.

    A position is counted as deployed on every day in `[entry_date,
    exit_date)` — entry day included, exit day excluded (the position is
    closed as of its exit, so does not tie up capital that day). Indexed by
    ISO date string, `min(entry_date)` to `max(exit_date)` inclusive.
    Empty trades -> empty series.
    """
    if not trades:
        return pd.Series(dtype="float64")
    min_d = min(date.fromisoformat(t.entry_date) for t in trades)
    max_d = max(date.fromisoformat(t.exit_date) for t in trades)
    n_days = (max_d - min_d).days + 1
    arr: np.ndarray = np.zeros(n_days, dtype="float64")
    for t in trades:
        start = (date.fromisoformat(t.entry_date) - min_d).days
        end = (date.fromisoformat(t.exit_date) - min_d).days  # exclusive
        if end > start:
            arr[start:end] += t.capital_at_risk
    idx = [(min_d + timedelta(days=i)).isoformat() for i in range(n_days)]
    return pd.Series(arr, index=idx)


@dataclass(frozen=True)
class ConcurrentStats:
    mean_deployed: float
    peak_deployed: float
    n_days: int


def concurrent_stats(trades: Sequence[Trade]) -> ConcurrentStats:
    series = concurrent_exposure_daily(trades)
    if len(series) == 0:
        return ConcurrentStats(0.0, 0.0, 0)
    return ConcurrentStats(float(series.mean()), float(series.max()), len(series))


# ---------------------------------------------------------------------------
# Daily portfolio time series
# ---------------------------------------------------------------------------


def daily_frame(trades: Sequence[Trade], spy_series: Dict[str, float]
                 ) -> pd.DataFrame:
    """One row per SPY trading day the book had capital deployed on.

    `book_ret` for a trading day `d` is the P&L realised since the previous
    trading day `prev_d` (summed over every trade whose `exit_date` falls in
    `(prev_d, d]` — a realised-on-exit approximation; the ledger records no
    intraday marks), divided by the capital deployed AT THE START of that
    interval (`concurrent_exposure_daily` on `prev_d`) — the balance that
    interval's return is actually earned on, not the balance left over
    afterward (which is why the divisor is `prev_d`'s exposure, not `d`'s: a
    position that fully exits inside the interval correctly still has its
    return measured against the capital it tied up, rather than against the
    zero left behind once it closed). `spy_ret` is SPY's return over the
    same interval. Intervals with zero deployed capital at the start are
    excluded (nothing to divide by); intervals with deployed capital but no
    exit still get `book_ret = 0.0` (a real "nothing realised this
    interval" result, not a missing value).
    """
    if not trades:
        return pd.DataFrame(columns=["trade_date", "book_ret", "spy_ret"])
    exposure = concurrent_exposure_daily(trades)
    pnl_by_day: Dict[str, float] = {}
    for t in trades:
        pnl_by_day[t.exit_date] = pnl_by_day.get(t.exit_date, 0.0) + t.pnl_usd

    # The upper bound is extended a little past the last exposure day: the
    # final position can exit on (or just before) a non-trading day, and its
    # P&L needs a NEXT trading day to attribute to that is still inside the
    # search window, not just the day the exposure series happens to end on.
    upper_bound = (date.fromisoformat(exposure.index[-1])
                   + timedelta(days=DEFAULT_TOLERANCE_DAYS * 2)).isoformat()
    trading_days = sorted(d for d in spy_series
                           if exposure.index[0] <= d <= upper_bound)
    rows = []
    for prev_d, d in zip(trading_days, trading_days[1:]):
        exp = float(exposure.get(prev_d, 0.0))
        if exp <= 0.0:
            continue
        # Attribute any P&L realised on a non-trading day (e.g. a weekend
        # mark) to the next trading day, matching how a daily return series
        # is normally constructed.
        realised = 0.0
        day_cursor = date.fromisoformat(prev_d) + timedelta(days=1)
        end_cursor = date.fromisoformat(d)
        while day_cursor <= end_cursor:
            realised += pnl_by_day.get(day_cursor.isoformat(), 0.0)
            day_cursor += timedelta(days=1)
        spy_ret = spy_series[d] / spy_series[prev_d] - 1.0
        rows.append({
            "trade_date": d,
            "book_ret": realised / exp,
            "spy_ret": spy_ret,
            "deployed": exp,
        })
    return pd.DataFrame(rows, columns=["trade_date", "book_ret", "spy_ret",
                                        "deployed"])


@dataclass(frozen=True)
class DailyAggregateResult:
    """A descriptive t-test summary of the daily excess series — not a gate."""
    n_days: int
    mean_book: float
    sd_book: float
    mean_spy: float
    sd_spy: float
    mean_excess: float
    sd_excess: float
    t_stat: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    ci_contains_zero: Optional[bool]
    variance_ratio: float
    n_needed_80pct_power: float


def daily_aggregate(df: pd.DataFrame) -> DailyAggregateResult:
    """Ordinary (non-clustered) paired t-test on the daily excess series.

    Days are not naturally clustered by symbol the way per-trade rows are
    (a day's book return already aggregates every position open that day),
    so this uses a plain one-sample t-test rather than the cluster
    bootstrap used for the per-trade series.
    """
    n = len(df)
    if n == 0:
        return DailyAggregateResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None,
                                     None, None, None, 1.0, float("inf"))
    book = df["book_ret"].to_numpy(dtype="float64")
    spy = df["spy_ret"].to_numpy(dtype="float64")
    excess = book - spy
    mean_book, mean_spy = float(book.mean()), float(spy.mean())
    sd_book = float(np.std(book, ddof=1)) if n >= 2 else 0.0
    sd_spy = float(np.std(spy, ddof=1)) if n >= 2 else 0.0
    mean_excess = float(excess.mean())
    sd_excess = float(np.std(excess, ddof=1)) if n >= 2 else 0.0

    t_stat = ci_lo = ci_hi = None
    if n >= 2 and sd_excess > 0.0:
        se = sd_excess / math.sqrt(n)
        t_stat = mean_excess / se
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_lo = mean_excess - t_crit * se
        ci_hi = mean_excess + t_crit * se
    ci_contains_zero = (ci_lo <= 0.0 <= ci_hi) if (
        ci_lo is not None and ci_hi is not None) else None
    var_ratio = (sd_excess / sd_book) if sd_book > 0 else 1.0
    n_needed = n_needed_for_power(mean_excess, sd_excess)
    return DailyAggregateResult(
        n, mean_book, sd_book, mean_spy, sd_spy, mean_excess, sd_excess,
        t_stat, ci_lo, ci_hi, ci_contains_zero, var_ratio, n_needed)


# ---------------------------------------------------------------------------
# Cumulative dollar comparison — the one figure with no per-observation noise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CumulativeComparison:
    """One realisation of one path: no confidence interval attaches to this.

    `book_pnl_usd` is the book's actual total realised P&L across every
    matched closed trade. The two SPY-equivalent figures answer "what would
    the same dollars have returned in SPY instead", under each capital
    convention (spec section "Capital convention").
    """
    n_trades: int
    book_pnl_usd: float
    total_capital_at_risk_usd: float
    spy_equiv_primary_usd: float       # sum_i car_i * spy_ret_i (per-trade window)
    mean_concurrent_usd: float
    peak_concurrent_usd: float
    spy_span_return: Optional[float]   # SPY return over the whole book span
    spy_equiv_secondary_usd: Optional[float]  # mean_concurrent * spy_span_return


def cumulative_dollar_comparison(
        trades: Sequence[Trade], spy_series: Dict[str, float],
        tolerance_days: int = DEFAULT_TOLERANCE_DAYS
        ) -> CumulativeComparison:
    """The headline dollar figure: total book P&L vs. total SPY-equivalent
    P&L on the same capital, under both capital conventions. No CI — this is
    a single historical realisation, not a sampled estimate.
    """
    if not trades:
        return CumulativeComparison(0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)

    book_pnl = sum(t.pnl_usd for t in trades)
    total_car = sum(t.capital_at_risk for t in trades)

    spy_equiv_primary = 0.0
    n_matched = 0
    for t in trades:
        entry_match = nearest_price(spy_series, t.entry_date, tolerance_days)
        exit_match = nearest_price(spy_series, t.exit_date, tolerance_days)
        if entry_match is None or exit_match is None:
            continue
        spy_ret = exit_match.price / entry_match.price - 1.0
        spy_equiv_primary += t.capital_at_risk * spy_ret
        n_matched += 1

    conc = concurrent_stats(trades)
    min_entry = min(t.entry_date for t in trades)
    max_exit = max(t.exit_date for t in trades)
    span_start = nearest_price(spy_series, min_entry, tolerance_days)
    span_end = nearest_price(spy_series, max_exit, tolerance_days)
    spy_span_return = None
    spy_equiv_secondary = None
    if span_start is not None and span_end is not None:
        spy_span_return = span_end.price / span_start.price - 1.0
        spy_equiv_secondary = conc.mean_deployed * spy_span_return

    return CumulativeComparison(
        n_trades=n_matched, book_pnl_usd=book_pnl,
        total_capital_at_risk_usd=total_car,
        spy_equiv_primary_usd=spy_equiv_primary,
        mean_concurrent_usd=conc.mean_deployed,
        peak_concurrent_usd=conc.peak_deployed,
        spy_span_return=spy_span_return,
        spy_equiv_secondary_usd=spy_equiv_secondary)
