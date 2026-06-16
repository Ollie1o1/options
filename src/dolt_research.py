"""Quant research harness over the real-marks cohort backtest.

Tests economically-motivated ENTRY FILTERS one at a time, basket-wide, all-in,
with an optional train/test split — the disciplined way to look for an edge
without overfitting. Every filter has a reason to work; we keep the ones that
show a large, consistent effect that survives out-of-sample.

CLI:
    python -m src.dolt_research --sweep --symbols AAPL,SPY,QQQ,MSFT
    python -m src.dolt_research --train-test low_vix_20 --symbols AAPL,SPY,QQQ,MSFT
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from src.dolt_cohort import run_cohort_backtest
from src.dolt_short import run_short_backtest
from src.dolt_spread import run_spread_backtest


# ── External signals (cached) ───────────────────────────────────────────────
@lru_cache(maxsize=1)
def _vix_series() -> Dict[str, float]:
    """^VIX close keyed by ISO date (yfinance, free, no key)."""
    import warnings
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = yf.Ticker("^VIX").history(period="6y")["Close"]
    return {d.date().isoformat(): float(v) for d, v in h.items()}


def vix_on(date: str) -> Optional[float]:
    """VIX on a date, forward-filled up to 5 prior days for holidays."""
    s = _vix_series()
    d = _dt.date.fromisoformat(date)
    for back in range(0, 6):
        key = (d - _dt.timedelta(days=back)).isoformat()
        if key in s:
            return s[key]
    return None


def _ma(ctx, ma_days: int) -> Optional[float]:
    """Simple moving average of spot over the last ma_days trading rows up to date."""
    spots, sdates, date = ctx.get("spots"), ctx.get("sdates"), ctx.get("date")
    if not spots or not sdates or date not in spots:
        return None
    try:
        i = sdates.index(date)
    except ValueError:
        return None
    window = [spots[d] for d in sdates[max(0, i - ma_days + 1): i + 1] if d in spots]
    return sum(window) / len(window) if window else None


# ── Entry-filter factories (each returns ctx->bool) ─────────────────────────
def low_vix(thresh: float) -> Callable:
    def f(ctx):
        v = vix_on(ctx["date"])
        return v is not None and v < thresh
    return f


def high_vix(thresh: float) -> Callable:
    def f(ctx):
        v = vix_on(ctx["date"])
        return v is not None and v >= thresh
    return f


def trend_up(ma_days: int = 50) -> Callable:
    def f(ctx):
        m = _ma(ctx, ma_days)
        return m is not None and ctx["spot"] > m
    return f


def low_iv(thresh: float) -> Callable:        # absolute IV ceiling (cheap vol)
    def f(ctx):
        iv = ctx.get("entry_iv")
        return iv is not None and iv < thresh
    return f


def drawdown_on(ctx) -> Optional[float]:
    """Current drawdown of spot from its trailing peak (<= 0), using the spot
    history available in ctx. None if it can't be computed."""
    spots, sdates, date, spot = (ctx.get("spots"), ctx.get("sdates"),
                                 ctx.get("date"), ctx.get("spot"))
    if not spots or not sdates or date not in spots or spot is None:
        return None
    try:
        i = sdates.index(date)
    except ValueError:
        return None
    peak = max(spots[d] for d in sdates[: i + 1] if d in spots)
    if not peak:
        return None
    return spot / peak - 1.0


def realized_vol(ctx, lookback: int = 20) -> Optional[float]:
    """Annualized realized vol of spot over the trailing ``lookback`` trading
    rows up to ctx['date'] (close-to-close, 252d). None if too little history."""
    import math
    spots, sdates, date = ctx.get("spots"), ctx.get("sdates"), ctx.get("date")
    if not spots or not sdates or date not in spots:
        return None
    try:
        i = sdates.index(date)
    except ValueError:
        return None
    win = [spots[d] for d in sdates[max(0, i - lookback): i + 1] if d in spots]
    if len(win) < 5:
        return None
    rets = [math.log(win[k] / win[k - 1]) for k in range(1, len(win)) if win[k - 1] > 0]
    if len(rets) < 4:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def vrp_rich(min_ratio: float = 1.2, lookback: int = 20) -> Callable:
    """Entry only when implied vol is RICH vs realized — entry_iv / realized_vol
    >= min_ratio (variance-risk-premium harvesting; P1.4). Sells expensive vol,
    skips selling cheap vol (incl. tops just before a fall, where IV hasn't
    risen yet)."""
    def f(ctx):
        iv = ctx.get("entry_iv")
        rv = realized_vol(ctx, lookback)
        return iv is not None and rv is not None and rv > 0 and (iv / rv) >= min_ratio
    return f


def in_drawdown(min_dd: float) -> Callable:
    """Entry only when spot is at least ``min_dd`` (e.g. 0.10 = 10%) below its
    trailing peak — i.e. SELLING premium INTO market weakness, the worst-case
    stress for short premium (P0.2)."""
    def f(ctx):
        dd = drawdown_on(ctx)
        return dd is not None and dd <= -abs(min_dd)
    return f


def combine(*filters) -> Callable:
    def f(ctx):
        return all(flt(ctx) for flt in filters)
    return f


# Standard research battery (name -> filter). None = baseline (no filter).
def standard_battery() -> Dict[str, Optional[Callable]]:
    return {
        "baseline":            None,
        "low_vix_20":          low_vix(20.0),
        "low_vix_18":          low_vix(18.0),
        "high_vix_22":         high_vix(22.0),
        "trend_up_50":         trend_up(50),
        "low_iv_0.30":         low_iv(0.30),
        "lowvix20_trend50":    combine(low_vix(20.0), trend_up(50)),
    }


# ── Runners ─────────────────────────────────────────────────────────────────
def _row(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {k: stats.get(k) for k in ("n", "win_rate", "avg_return", "median_return", "profit_factor")}


def compare(symbols, start, end, filters: Dict[str, Optional[Callable]],
            db_path=None) -> Dict[str, Dict[str, Any]]:
    """Run each named filter on the same basket/window; return per-filter stats."""
    from src import dolt_options as _do
    dates = _do._date_range(start, end, weekly=True)
    out = {}
    for name, flt in filters.items():
        res = run_cohort_backtest(symbols, dates, db_path=db_path, entry_filter=flt)
        out[name] = _row(res)
    return out


# Universe segments — different vol character may favor different rules.
# NOTE: QQQ and IWM are NOT in the DoltHub options dataset (verified empty), so
# the "index" segment is SPY only. Only symbols with real chain data here.
# META, AMZN added to tech 2026-06-15 (probed: ~150 chain rows each) to thicken
# the per-segment sample — the binding constraint everywhere in this research.
# TSLA also has data (~138 rows) but is its own high-beta animal; left ungrouped
# until it earns a bucket. Broadened-basket verdicts are PROVISIONAL until a
# backtest with fetched histories confirms (needs a fetch pass; see P3.10).
SEGMENTS = {
    "index": ["SPY"],
    "tech":  ["AAPL", "MSFT", "GOOG", "META", "AMZN"],
    "semi":  ["NVDA", "AMD"],
}


def segment_battery() -> Dict[str, Optional[Callable]]:
    """Focused battery (regime hypothesis only) to limit multiple comparisons."""
    return {"baseline": None, "low_vix_20": low_vix(20.0),
            "low_vix_18": low_vix(18.0), "high_vix_22": high_vix(22.0)}


def segment_sweep(start, end, db_path=None, segments=None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the focused battery within each segment. Returns {segment: {filter: stats}}."""
    segments = segments or SEGMENTS
    return {seg: compare(syms, start, end, segment_battery(), db_path=db_path)
            for seg, syms in segments.items()}


# Strategy registry: name -> runner with the shared
# (symbols, dates, db_path=, entry_filter=) contract. short_put is the put
# variant of the short runner. Lets train_test holdout-validate ANY strategy,
# not just long calls (P0.1: the winning index put spread must survive a holdout).
# Late-bound (look up globals at call time) so tests can monkeypatch the runners.
STRATEGIES: Dict[str, Callable] = {
    "long_call":  lambda *a, **k: run_cohort_backtest(*a, **k),
    "short_put":  lambda *a, **k: run_short_backtest(*a, opt_type="put", **k),
    "put_spread": lambda *a, **k: run_spread_backtest(*a, **k),
}


def train_test(symbols, filter_fn, db_path=None,
               train=("2022-01-01", "2023-12-31"),
               test=("2024-01-01", "2024-12-31"),
               strategy="long_call") -> Dict[str, Any]:
    """Validate a filter/strategy: fit intuition on train, confirm on held-out
    test. ``strategy`` selects the backtest runner (see STRATEGIES)."""
    from src import dolt_options as _do
    runner = STRATEGIES[strategy]
    tr = runner(symbols, _do._date_range(*train, weekly=True),
                db_path=db_path, entry_filter=filter_fn)
    te = runner(symbols, _do._date_range(*test, weekly=True),
                db_path=db_path, entry_filter=filter_fn)
    return {"train": _row(tr), "test": _row(te)}


_ACTION = {
    "long_call":  "LONG  — buy calls",
    "short_put":  "SHORT — sell puts",
    "put_spread": "SHORT — sell put credit spreads (defined risk)",
    None:         "STAND DOWN — no edge here",
}


def recommend(symbols, start, end, db_path=None, min_pf=1.05, min_n=20) -> Dict[str, Any]:
    """The system's data-driven verdict for a name/segment: run long calls, short
    puts, and put spreads on real marks; recommend the highest-PF candidate that
    clears PF>=min_pf with positive expectancy, else STAND DOWN. The recommendation
    IS the backtest — nothing hardcoded."""
    from src import dolt_options as _do
    from src.dolt_cohort import run_cohort_backtest
    from src.dolt_short import run_short_backtest
    from src.dolt_spread import run_spread_backtest
    dates = _do._date_range(start, end, weekly=True)
    cands = {
        "long_call":  run_cohort_backtest(symbols, dates, db_path=db_path),
        "short_put":  run_short_backtest(symbols, dates, db_path=db_path, opt_type="put"),
        "put_spread": run_spread_backtest(symbols, dates, db_path=db_path),
    }
    eligible = [(k, v) for k, v in cands.items()
                if v.get("profit_factor") and v.get("n", 0) >= min_n]
    eligible.sort(key=lambda kv: kv[1]["profit_factor"], reverse=True)
    best = None
    for k, v in eligible:
        if v["profit_factor"] >= min_pf and (v.get("avg_return") or -1) > 0:
            best = k
            break
    return {"best": best, "action": _ACTION[best],
            "candidates": {k: _row(v) for k, v in cands.items()}}


def _fmt(row) -> str:
    def g(k, p=False):
        v = row.get(k)
        if v is None:
            return "  -  "
        return f"{v:+.1%}" if p else f"{v}"
    return (f"n={row.get('n'):>4}  win={g('win_rate')}  avg={g('avg_return', True):>7}  "
            f"med={g('median_return', True):>7}  PF={row.get('profit_factor')}")


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Quant research over the real-marks cohort")
    ap.add_argument("--symbols", default="AAPL,SPY,QQQ,MSFT")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--db", default=None)
    ap.add_argument("--sweep", action="store_true", help="Run the standard filter battery")
    ap.add_argument("--segments", action="store_true", help="Per-segment (etf/tech/semi) battery")
    ap.add_argument("--recommend", action="store_true", help="Long/short verdict for --symbols")
    ap.add_argument("--train-test", metavar="FILTER", help="Train/test a single battery filter")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="long_call",
                    help="Strategy to holdout-validate with --train-test")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = {}
    try:
        cfg = json.load(open("config.json")).get("dolt_options", {})
    except Exception:
        pass
    db = args.db or cfg.get("cache_path")

    if args.train_test:
        battery = standard_battery()
        flt = battery.get(args.train_test)
        out = train_test(syms, flt, db_path=db, strategy=args.strategy)
        print(f"Train/test [{args.strategy}]: {args.train_test}  ({', '.join(syms)})")
        print(f"  TRAIN 22-23: {_fmt(out['train'])}")
        print(f"  TEST  24   : {_fmt(out['test'])}")
        return

    if args.recommend:
        rec = recommend(syms, args.start, args.end, db_path=db)
        print(f"VERDICT for {syms}, {args.start}..{args.end}:  >>> {rec['action']} <<<")
        for name, row in rec["candidates"].items():
            mark = "  <-- chosen" if name == rec["best"] else ""
            print(f"  {name:12} {_fmt(row)}{mark}")
        return

    if args.segments:
        print(f"Per-segment battery, {args.start}..{args.end} (all-in real marks):")
        res = segment_sweep(args.start, args.end, db_path=db)
        for seg, battery in res.items():
            print(f"\n[{seg.upper()}]  {SEGMENTS[seg]}")
            for name, row in battery.items():
                print(f"  {name:14} {_fmt(row)}")
        return

    if args.sweep:
        print(f"Sweep on {syms}, {args.start}..{args.end} (all-in real marks):")
        res = compare(syms, args.start, args.end, standard_battery(), db_path=db)
        for name, row in res.items():
            print(f"  {name:20} {_fmt(row)}")


if __name__ == "__main__":
    _cli()
