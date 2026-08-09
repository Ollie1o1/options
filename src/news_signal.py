"""Does archived news sentiment predict forward returns?

The scorer has carried a `sentiment` feature at weight **0.0** since it was
written, because nothing here had ever been validated. `docs/INTEL_BACKTEST_
FINDINGS.md` lists news as "not backtestable from price", which was true when
the only news available was today's. It is no longer true: `news_archive.py`
has been accumulating point-in-time headlines with sentiment since 2026-06-20.

This module is the test that has to pass before any nonzero weight is
justified. It deliberately makes the pessimistic choice at every fork.

**The day is the unit of independence.** The archive holds thousands of
symbol-days, but they sit on a few dozen distinct dates, and every symbol on
one date shares that date's market move. So sentiment is measured the way a
factor is measured: rank the cross-section within each day, correlate with
forward return, and t-test the resulting series of daily ICs. Pooling instead
would inflate the sample by ~100x and manufacture significance.

**`archived_at`, never `published`.** Feeds revise and backfill publication
timestamps. The only stamp we control is when WE first saw the story, so that
is the only one that cannot leak the future.

**Power is reported alongside the result.** With a few dozen days, "p > 0.05"
means "could not measure", not "no effect" — a distinction this repo has been
bitten by before. `days_for_power` says how long the archive must accrue.
"""
from __future__ import annotations

import collections
import math
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy import stats as _sps
except ImportError:                     # pragma: no cover
    _sps = None                         # type: ignore[assignment]

DEFAULT_NEWS_DB = "data/news_archive.db"
DEFAULT_PRICE_DB = "data/squeeze_prices.db"
MIN_NAMES_PER_DAY = 10
READ_TIMEOUT_S = 120.0

Key = Tuple[str, str]                   # (date, symbol)


def daily_sentiment(news_db: str = DEFAULT_NEWS_DB,
                    ) -> Dict[Key, Dict[str, Any]]:
    """(archive_date, symbol) -> {score, n, relevance}.

    Keyed on the ARCHIVE date. A story published on the 1st that we did not
    see until the 5th is tradeable on the 5th and not before.

    The score is the relevance-weighted mean, matching what
    `news_fetcher._compute_aggregate_sentiment` shows the operator, so the
    thing measured here is the thing displayed.
    """
    conn = sqlite3.connect(news_db, timeout=READ_TIMEOUT_S)
    try:
        rows = conn.execute(
            "SELECT substr(archived_at,1,10), symbol, sentiment, relevance "
            "FROM news_archive WHERE symbol IS NOT NULL "
            "AND archived_at IS NOT NULL AND sentiment IS NOT NULL").fetchall()
    finally:
        conn.close()

    acc: Dict[Key, List[Tuple[float, float]]] = collections.defaultdict(list)
    for date, sym, sent, rel in rows:
        try:
            s = float(sent)
        except (TypeError, ValueError):
            continue
        try:
            w = float(rel) if rel is not None else 1.0
        except (TypeError, ValueError):
            w = 1.0
        acc[(date, str(sym).upper())].append((s, max(w, 0.0)))

    out: Dict[Key, Dict[str, Any]] = {}
    for key, pairs in acc.items():
        wsum = sum(w for _s, w in pairs)
        if wsum > 0:
            score = sum(s * w for s, w in pairs) / wsum
        else:                            # all-zero relevance: plain mean
            score = sum(s for s, _w in pairs) / len(pairs)
        vals = [s for s, _w in pairs]
        mean = sum(vals) / len(vals)
        # Dispersion, not tone: disagreement among headlines is a different
        # hypothesis from their average, and a better fit for a vol target.
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out[key] = {"score": score, "n": len(pairs),
                    "flow": float(len(pairs)),
                    "dispersion": math.sqrt(var),
                    "abs_score": abs(score),
                    "relevance": wsum / len(pairs)}
    return out


def forward_absolute_move(price_db: str, symbols: Sequence[str],
                          horizon: int = 5, start: Optional[str] = None
                          ) -> Dict[Key, float]:
    """(date, symbol) -> |forward return| over the horizon.

    The options-relevant target. This book does not need to know WHICH WAY a
    name moves to make money — a straddle, a condor and every
    buy-vs-sell-premium decision turn on HOW FAR it moves. Direction is also
    the thing the scorer has repeatedly failed to predict (IC ~0.03), whereas
    news arrival driving realized volatility is a far more plausible mechanism
    than news tone driving sign.
    """
    return {k: abs(v) for k, v in
            forward_returns(price_db, symbols, horizon, start).items()}


def forward_returns(price_db: str, symbols: Sequence[str], horizon: int = 5,
                    start: Optional[str] = None) -> Dict[Key, float]:
    """(date, symbol) -> close-to-close return over the next `horizon` rows.

    The horizon counts TRADING rows, not calendar days, so a Friday entry
    measures to the following week rather than into the weekend. The return
    starts at the decision date's own close: acting on day D's news means
    paying day D's close, never day D-1's.
    """
    conn = sqlite3.connect(price_db, timeout=READ_TIMEOUT_S)
    try:
        q = ("SELECT symbol, date, close FROM px WHERE close IS NOT NULL "
             "AND symbol IN (%s)" % ",".join("?" * len(symbols)))
        args = list(symbols)
        if start:
            q += " AND date >= ?"
            args.append(start)
        rows = conn.execute(q + " ORDER BY symbol, date", args).fetchall()
    finally:
        conn.close()

    series: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for sym, date, close in rows:
        series[str(sym).upper()].append((str(date)[:10], float(close)))

    out: Dict[Key, float] = {}
    for sym, pts in series.items():
        for i in range(len(pts) - horizon):
            d0, p0 = pts[i]
            _d1, p1 = pts[i + horizon]
            if p0 > 0:
                out[(d0, sym)] = (p1 / p0) - 1.0
    return out


def cross_sectional_ic(sentiment: Dict[Key, Dict[str, Any]],
                       returns: Dict[Key, float],
                       min_names: int = MIN_NAMES_PER_DAY,
                       field: str = "score",
                       ) -> List[Dict[str, Any]]:
    """One Spearman IC per day, over the names that had news that day.

    A day with fewer than `min_names` is dropped: a rank correlation over four
    names is noise with a decimal point. A day whose scores are all identical
    is dropped too — there is no cross-section to rank.
    """
    if _sps is None:
        return []
    by_day: Dict[str, List[Tuple[float, float]]] = collections.defaultdict(list)
    for (date, sym), rec in sentiment.items():
        r = returns.get((date, sym))
        if r is None:
            continue
        if rec.get(field) is None:
            continue
        by_day[date].append((float(rec[field]), float(r)))

    out: List[Dict[str, Any]] = []
    for date in sorted(by_day):
        pairs = by_day[date]
        if len(pairs) < min_names:
            continue
        xs = [a for a, _b in pairs]
        ys = [b for _a, b in pairs]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        ic, _p = _sps.spearmanr(xs, ys)
        if ic != ic:
            continue
        out.append({"date": date, "n": len(pairs), "ic": float(ic)})
    return out


def days_for_power(effect_ic: float, daily_sd: float = 0.15,
                   alpha: float = 0.05, power: float = 0.80) -> int:
    """Trading days of archive needed to detect a mean daily IC of `effect_ic`.

    A one-sample t-test on the series of daily ICs. `daily_sd` is the spread of
    the per-day IC, which on cross-sections of this width runs 0.10-0.20.

    This is the number that decides whether "not significant" means "no
    effect" or "not enough archive yet".
    """
    if effect_ic <= 0 or daily_sd <= 0:
        return 0
    z_a = 1.959963985 if alpha <= 0.05 else 1.644853627
    z_b = 0.841621234 if power >= 0.80 else 0.524400513
    return int(math.ceil(((z_a + z_b) * daily_sd / effect_ic) ** 2))


def validate(news_db: str = DEFAULT_NEWS_DB,
             price_db: str = DEFAULT_PRICE_DB,
             horizon: int = 5,
             min_names: int = MIN_NAMES_PER_DAY,
             field: str = "score",
             target: str = "return") -> Dict[str, Any]:
    """The whole test. Returns a verdict dict; never raises on thin data.

    `target` selects what is being predicted: "return" (direction) or
    "abs_return" (how far, the options-relevant one).
    """
    sent = daily_sentiment(news_db)
    symbols = sorted({sym for _d, sym in sent})
    dates = sorted({d for d, _s in sent})
    fn = forward_absolute_move if target == "abs_return" else forward_returns
    rets = fn(price_db, symbols, horizon=horizon,
              start=dates[0] if dates else None)
    series = cross_sectional_ic(sent, rets, min_names=min_names, field=field)

    out: Dict[str, Any] = {
        "field": field, "target": target,
        "horizon": horizon,
        "n_days": len(series),
        "n_observations": sum(s["n"] for s in series),
        "mean_ic": None, "t": None, "p": None,
        "days_needed_for_ic_0.03": days_for_power(0.03),
        "days_needed_for_ic_0.05": days_for_power(0.05),
        "series": series,
    }
    if not series or _sps is None:
        return out
    ics = [s["ic"] for s in series]
    out["mean_ic"] = round(sum(ics) / len(ics), 4)
    if len(ics) >= 3:
        t, p = _sps.ttest_1samp(ics, 0.0)
        if t == t:
            out["t"] = round(float(t), 3)
            out["p"] = round(float(p), 4)
        sd = float(_sps.tstd(ics)) if len(ics) > 1 else 0.0
        out["daily_ic_sd"] = round(sd, 4)
        if sd > 0:
            out["days_needed_at_observed_effect"] = days_for_power(
                abs(out["mean_ic"]) or 1e-9, daily_sd=sd)
    return out


def format_report(results: Sequence[Dict[str, Any]]) -> str:
    lines = ["NEWS SENTIMENT AS A CROSS-SECTIONAL FACTOR",
             "  (one Spearman IC per day; the DAY is the unit of independence)",
             "",
             f"  {'horizon':>8} {'days':>6} {'obs':>7} {'mean IC':>9} "
             f"{'t':>7} {'p':>8} {'daily sd':>9}"]
    for r in results:
        if r["mean_ic"] is None:
            lines.append(f"  {r['horizon']:>8}d {r['n_days']:>6} "
                         f"{r['n_observations']:>7}   (no overlapping days)")
            continue
        lines.append(
            f"  {r['horizon']:>8}d {r['n_days']:>6} {r['n_observations']:>7} "
            f"{r['mean_ic']:>9.4f} {r['t'] if r['t'] is not None else 0:>7.2f} "
            f"{r['p'] if r['p'] is not None else 1:>8.4f} "
            f"{r.get('daily_ic_sd', 0):>9.4f}")
    return "\n".join(lines)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--news-db", default=DEFAULT_NEWS_DB)
    ap.add_argument("--price-db", default=DEFAULT_PRICE_DB)
    ap.add_argument("--horizons", default="1,3,5,10")
    ap.add_argument("--min-names", type=int, default=MIN_NAMES_PER_DAY)
    args = ap.parse_args()

    hs = [int(h) for h in args.horizons.split(",") if h.strip()]
    results = [validate(args.news_db, args.price_db, horizon=h,
                        min_names=args.min_names) for h in hs]
    print(format_report(results))
    print()
    print(f"  Testing {len(hs)} horizons is a {len(hs)}-way search; the "
          f"smallest p must be read against that.")
    have = max((r["n_days"] for r in results), default=0)
    # Power at the sd actually observed, not an assumed one — the default 0.15
    # is a guess and this data has measured its own.
    sds = [r["daily_ic_sd"] for r in results if r.get("daily_ic_sd")]
    sd = sum(sds) / len(sds) if sds else 0.15
    print(f"  POWER at the OBSERVED daily IC sd of {sd:.3f}:")
    for eff in (0.10, 0.05, 0.03):
        need = days_for_power(eff, daily_sd=sd)
        verdict = "ALREADY POWERED" if have >= need else (
            f"needs {need - have} more trading days "
            f"(~{(need - have) / 21:.1f} months)")
        print(f"    mean daily IC {eff:.2f} -> {need:>4} days   {verdict}")
    print(f"  Archive supports {have} usable days today.")
    print(f"  A null at {have} days is NOT evidence of no effect for any "
          f"effect size needing more.")


if __name__ == "__main__":
    _cli()
