#!/usr/bin/env python3
"""Preregistered estimator for Equity Playbook 1: Uptrend Pullback Mean Reversion.

Frozen registration: docs/PREREG_EQUITY_PULLBACKS_20261124.md
Zero parameter tuning. Runs on both:
  1. Broad survivorship-free universe (data/squeeze_prices.db)
  2. Auditable adjusted universe (data/equity_ohlcv.db)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.alloc.report import promotion_verdict, summarise
from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    cluster_bootstrap_stats,
    iso_week_key,
    load_equity_ohlcv,
    simulate_capacity_book,
)
from src.equity_playbooks.indicators import (
    bad_jump_mask,
    mom_12_1_series,
    sma,
    trailing_median_dollar_vol,
    wilder_rsi,
)

TRAIN_START = "2018-10-01"
TRAIN_END = "2022-12-31"
HOLDOUT_START = "2023-01-24"  # 15 trading-day purge embargo after 2022-12-31
MIN_CLUSTERS = 40
SEED = 20261124


def load_earnings_calendar(db_path: str = "data/dolt_options.db") -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = collections.defaultdict(list)
    if not Path(db_path).exists():
        return out
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for sym, d in con.execute("SELECT symbol, date FROM earnings_cal ORDER BY symbol, date"):
            out[str(sym).upper()].append(str(d)[:10])
    finally:
        con.close()
    return out


def _has_earnings_within(dates_sorted: List[str], sig_date: str, days: int = 10) -> bool:
    if not dates_sorted:
        return False
    end_d = (dt.date.fromisoformat(sig_date) + dt.timedelta(days=days)).isoformat()
    idx = bisect.bisect_left(dates_sorted, sig_date)
    return idx < len(dates_sorted) and dates_sorted[idx] <= end_d


def simulate_symbol_pullbacks(
    sym: str,
    dates: List[str],
    closes: np.ndarray,
    volumes: np.ndarray,
    spy: BenchmarkSeries,
    earnings_map: Dict[str, List[str]],
    *,
    stratum: str = "broad",
    rt_cost: float = 0.0010,
    next_day_fill: bool = True,
    min_price: float = 10.0,
    min_dollar_vol: float = 20_000_000.0,
) -> Tuple[List[EquityTrade], int]:
    """Run exact preregistered Playbook 1 rule on one ticker."""
    n = closes.size
    if n < 260:
        return [], 0

    jumps = bad_jump_mask(closes, max_abs_daily_ret=0.45)
    sma200 = sma(closes, 200)
    sma5 = sma(closes, 5)
    mom = mom_12_1_series(closes, 252, 21)
    rsi2 = wilder_rsi(closes, 2)
    med_dv = trailing_median_dollar_vol(closes, volumes, 20)

    sym_earns = earnings_map.get(sym, [])
    earn_covered = bool(sym_earns)

    trades: List[EquityTrade] = []
    skipped_jumps = 0
    in_trade_until = -1

    last_sig_idx = n - 3 if next_day_fill else n - 2
    for t in range(252, last_sig_idx):
        if t <= in_trade_until:
            continue
        sig_d = dates[t]
        if sig_d < TRAIN_START:
            continue
        c_t = closes[t]
        if c_t < min_price or not np.isfinite(c_t):
            continue
        if not np.isfinite(med_dv[t]) or med_dv[t] < min_dollar_vol:
            continue
        if not np.isfinite(sma200[t]) or c_t <= sma200[t]:
            continue
        if not np.isfinite(mom[t]) or mom[t] <= 0.0:
            continue
        if not np.isfinite(rsi2[t]) or rsi2[t] >= 10.0:
            continue
        if closes[t - 3] <= 0.0 or (c_t / closes[t - 3] - 1.0) > -0.03:
            continue
        # Guard trailing window against unadjusted split/bad ticks.
        if np.any(jumps[t - 20 : t + 1]):
            skipped_jumps += 1
            continue
        if earn_covered and _has_earnings_within(sym_earns, sig_d, days=10):
            continue

        entry_idx = t + 1 if next_day_fill else t
        if entry_idx >= n - 1 or jumps[entry_idx]:
            skipped_jumps += 1
            continue
        entry_px = float(closes[entry_idx])
        if entry_px <= 0.0 or not np.isfinite(entry_px):
            continue
        entry_d = dates[entry_idx]

        # Check exit daily starting at entry_idx + 1 (t+2 for next_day_fill, t+1 for MOC)
        exit_sig_idx = None
        exit_reason = "time_10d"
        max_k = min(entry_idx + 10, n - 2 if next_day_fill else n - 1)
        bad_forward = False
        for k in range(entry_idx + 1, max_k + 1):
            if jumps[k]:
                bad_forward = True
                break
            ck = float(closes[k])
            if np.isfinite(sma5[k]) and ck > sma5[k]:
                exit_sig_idx = k
                exit_reason = "target_sma5"
                break
            if np.isfinite(rsi2[k]) and rsi2[k] > 70.0:
                exit_sig_idx = k
                exit_reason = "target_rsi2"
                break
            if ck / entry_px - 1.0 <= -0.07:
                exit_sig_idx = k
                exit_reason = "stop_7pct"
                break
        if bad_forward:
            skipped_jumps += 1
            in_trade_until = max_k
            continue
        if exit_sig_idx is None:
            exit_sig_idx = max_k
            exit_reason = "time_10d"

        exit_idx = exit_sig_idx + 1 if next_day_fill else exit_sig_idx
        if exit_idx >= n or jumps[exit_idx]:
            skipped_jumps += 1
            in_trade_until = min(exit_idx, n - 1)
            continue
        exit_px = float(closes[exit_idx])
        if exit_px <= 0.0 or not np.isfinite(exit_px):
            continue
        exit_d = dates[exit_idx]

        net_r = (exit_px / entry_px - 1.0) - rt_cost
        bench_r = spy.window_return(entry_d, exit_d)
        trades.append(
            EquityTrade(
                ticker=sym,
                signal_date=sig_d,
                entry_date=entry_d,
                exit_date=exit_d,
                entry_price=entry_px,
                exit_price=exit_px,
                net_return=net_r,
                bench_return=bench_r,
                exit_reason=exit_reason,
                stratum=stratum,
                score_tiebreak=float(rsi2[t]),
                extra={"earn_covered": earn_covered, "rsi2": float(rsi2[t])},
            )
        )
        in_trade_until = exit_idx

    return trades, skipped_jumps


def run_on_squeeze_db(
    spy: BenchmarkSeries,
    earnings_map: Dict[str, List[str]],
    *,
    db_path: str = "data/squeeze_prices.db",
    rt_cost: float = 0.0010,
    next_day_fill: bool = True,
) -> Tuple[List[EquityTrade], int, int]:
    """Stream `data/squeeze_prices.db` one symbol at a time in O(1) memory."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    trades: List[EquityTrade] = []
    skipped_jumps = 0
    n_syms_scanned = 0
    try:
        cur = con.execute("SELECT symbol, date, close, volume FROM px ORDER BY symbol, date")
        cur_sym: Optional[str] = None
        dates: List[str] = []
        closes: List[float] = []
        vols: List[float] = []

        def _flush(sym_name: str, ds: List[str], cs: List[float], vs: List[float]):
            nonlocal skipped_jumps, n_syms_scanned
            if len(cs) < 260:
                return
            c_arr = np.asarray(cs, dtype=float)
            if float(np.nanmax(c_arr)) < 10.0:
                return
            n_syms_scanned += 1
            ts, sj = simulate_symbol_pullbacks(
                sym_name,
                ds,
                c_arr,
                np.asarray(vs, dtype=float),
                spy,
                earnings_map,
                stratum="broad",
                rt_cost=rt_cost,
                next_day_fill=next_day_fill,
            )
            trades.extend(ts)
            skipped_jumps += sj

        for sym, d, c, v in cur:
            s = str(sym).upper()
            if cur_sym is not None and s != cur_sym:
                _flush(cur_sym, dates, closes, vols)
                dates, closes, vols = [], [], []
            cur_sym = s
            if c is not None:
                dates.append(str(d))
                closes.append(float(c))
                vols.append(float(v) if v is not None else 0.0)
        if cur_sym is not None:
            _flush(cur_sym, dates, closes, vols)
    finally:
        con.close()
    return trades, skipped_jumps, n_syms_scanned


def run_on_ohlcv_db(
    spy: BenchmarkSeries,
    earnings_map: Dict[str, List[str]],
    *,
    db_path: str = "data/equity_ohlcv.db",
    rt_cost: float = 0.0010,
    next_day_fill: bool = True,
) -> Tuple[List[EquityTrade], int]:
    book = load_equity_ohlcv(db_path)
    trades: List[EquityTrade] = []
    skipped = 0
    for sym, (ds, cs, vs) in sorted(book.items()):
        ts, sj = simulate_symbol_pullbacks(
            sym, ds, cs, vs, spy, earnings_map, stratum="liquid",
            rt_cost=rt_cost, next_day_fill=next_day_fill,
        )
        trades.extend(ts)
        skipped += sj
    return trades, skipped


def evaluate_split(trades: Sequence[EquityTrade], spy: BenchmarkSeries, label: str) -> dict:
    train = [t for t in trades if TRAIN_START <= t.entry_date <= TRAIN_END]
    holdout = [t for t in trades if t.entry_date >= HOLDOUT_START]
    all_t = [t for t in trades if t.entry_date >= TRAIN_START]

    s_train = cluster_bootstrap_stats(train, lambda t: iso_week_key(t.entry_date), n_boot=10_000, seed=SEED, min_clusters=MIN_CLUSTERS)
    s_hold = cluster_bootstrap_stats(holdout, lambda t: iso_week_key(t.entry_date), n_boot=10_000, seed=SEED, min_clusters=MIN_CLUSTERS)
    s_all = cluster_bootstrap_stats(all_t, lambda t: iso_week_key(t.entry_date), n_boot=10_000, seed=SEED, min_clusters=MIN_CLUSTERS)

    alloc_hold = summarise(holdout, n_trials=1) if len(holdout) >= 3 else {"n": len(holdout), "insufficient": True}
    alloc_verdict = promotion_verdict(alloc_hold)

    # Binding H-A & H-B verdicts on holdout
    if s_hold["insufficient"]:
        ha = "INSUFFICIENT"
        hb = "INSUFFICIENT"
    else:
        ha_pass = (
            s_hold["mean_ret_ci"][0] > 0.0
            and s_hold["pf_ci"][0] > 1.0
            and (s_train["mean_ret"] is not None and s_train["mean_ret"] > 0.0)
        )
        ha = "PASS" if ha_pass else "FAIL"
        hb = "PASS" if (s_hold["mean_excess_ci"] is not None and s_hold["mean_excess_ci"][0] > 0.0) else "FAIL"

    # Per-year breakdown
    by_year: Dict[str, List[EquityTrade]] = collections.defaultdict(list)
    for t in all_t:
        by_year[t.entry_date[:4]].append(t)
    years_tbl = {}
    for y, yt in sorted(by_year.items()):
        rs = np.array([t.net_return for t in yt])
        xs = np.array([t.excess_return for t in yt if t.excess_return is not None])
        gains = float(np.sum(rs[rs > 0]))
        losses = float(np.abs(np.sum(rs[rs < 0])))
        years_tbl[y] = {
            "n": len(yt),
            "win_rate": round(float(np.mean(rs > 0)), 4),
            "mean_ret": round(float(np.mean(rs)), 5),
            "mean_excess": round(float(np.mean(xs)), 5) if xs.size else None,
            "pf": round(gains / losses, 3) if losses > 0 else None,
        }

    # Earnings coverage split on holdout
    known_clear = [t for t in holdout if t.extra.get("earn_covered")]
    unknown_earn = [t for t in holdout if not t.extra.get("earn_covered")]

    cap_hold = simulate_capacity_book(holdout, max_concurrent=5, slot_weight=0.20, spy_series=spy)
    cap_all = simulate_capacity_book(all_t, max_concurrent=5, slot_weight=0.20, spy_series=spy)

    return {
        "label": label,
        "verdict_HA": ha,
        "verdict_HB": hb,
        "alloc_promotion_verdict_holdout": alloc_verdict,
        "alloc_holdout": {k: v for k, v in alloc_hold.items() if k != "capacity"},
        "train": s_train,
        "holdout": s_hold,
        "full": s_all,
        "by_year": years_tbl,
        "holdout_earnings_split": {
            "known_clear_n": len(known_clear),
            "known_clear_mean": round(float(np.mean([t.net_return for t in known_clear])), 5) if known_clear else None,
            "unknown_n": len(unknown_earn),
            "unknown_mean": round(float(np.mean([t.net_return for t in unknown_earn])), 5) if unknown_earn else None,
        },
        "capacity_5slot_holdout": cap_hold,
        "capacity_5slot_full": cap_all,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest Equity Playbook 1 (Uptrend Pullbacks)")
    p.add_argument("--out", default="reports/backtest_equity_pullbacks_20261124.json")
    args = p.parse_args()

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db", ["SPY"])
    spy = BenchmarkSeries(ohlcv["SPY"][0], ohlcv["SPY"][1])
    earnings_map = load_earnings_calendar("data/dolt_options.db")

    print("Running Primary: Broad survivorship-free universe (next-day close fills, 10 bps RT)...")
    broad_t, broad_jumps, broad_syms = run_on_squeeze_db(spy, earnings_map, rt_cost=0.0010, next_day_fill=True)
    res_broad = evaluate_split(broad_t, spy, "broad_nextday_10bps")
    res_broad["symbols_scanned"] = broad_syms
    res_broad["skipped_jumps"] = broad_jumps

    print("Running Secondary: Auditable adjusted universe (equity_ohlcv.db, next-day close, 10 bps RT)...")
    liq_t, liq_jumps = run_on_ohlcv_db(spy, earnings_map, rt_cost=0.0010, next_day_fill=True)
    res_liq = evaluate_split(liq_t, spy, "liquid_ohlcv_nextday_10bps")
    res_liq["skipped_jumps"] = liq_jumps

    print("Running Sensitivities: MOC same-day close (10 bps) & 20 bps friction stress...")
    moc_t, _, _ = run_on_squeeze_db(spy, earnings_map, rt_cost=0.0010, next_day_fill=False)
    res_moc = evaluate_split(moc_t, spy, "sensitivity_broad_moc_10bps")

    stress_t = [
        EquityTrade(**{**t.__dict__, "net_return": t.net_return - 0.0010})
        for t in broad_t
    ]
    res_20bps = evaluate_split(stress_t, spy, "sensitivity_broad_nextday_20bps")

    payload = {
        "prereg": "docs/PREREG_EQUITY_PULLBACKS_20261124.md",
        "primary_broad": res_broad,
        "secondary_liquid_ohlcv": res_liq,
        "sensitivity_moc": res_moc,
        "sensitivity_20bps": res_20bps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for key in ("primary_broad", "secondary_liquid_ohlcv", "sensitivity_moc", "sensitivity_20bps"):
        r = payload[key]
        h = r["holdout"]
        tr = r["train"]
        print(
            f"\n[{r['label']}] H-A={r['verdict_HA']} H-B={r['verdict_HB']} alloc={r['alloc_promotion_verdict_holdout']}"
            f"\n  Train   : n={tr['n_trades']} wk={tr['n_clusters']} mean={tr['mean_ret']:+.4%} PF={tr['pf']:.3f} ex={tr['mean_excess']:+.4%}"
            f"\n  Holdout : n={h['n_trades']} wk={h['n_clusters']} mean={h['mean_ret']:+.4%} CI={h['mean_ret_ci']} PF={h['pf']:.3f} CI={h['pf_ci']} ex={h['mean_excess']:+.4%} CI={h['mean_excess_ci']}"
            f"\n  5-slot  : full CAGR={r['capacity_5slot_full']['cagr']:+.2%} (SPY {r['capacity_5slot_full']['spy_cagr']:+.2%}) Sh={r['capacity_5slot_full']['sharpe']:.2f} MaxDD={r['capacity_5slot_full']['max_dd']:+.2%}"
        )


if __name__ == "__main__":
    main()
