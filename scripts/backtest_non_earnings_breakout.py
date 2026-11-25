#!/usr/bin/env python3
"""Preregistered estimator for Non-Earnings Volume Breakout (Shares).

Frozen registration: docs/PREREG_NON_EARNINGS_BREAKOUT_20261124.md
Zero parameter tuning. Runs on both:
  1. Primary broad survivorship-free universe (`data/squeeze_prices.db`)
  2. Secondary auditable adjusted liquid universe (`data/equity_ohlcv.db`, 371 tickers)
"""
from __future__ import annotations

import bisect
import collections
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    iso_week_key,
    load_equity_ohlcv,
    moving_block_bootstrap_stats,
    simulate_capacity_book,
)
from src.equity_playbooks.indicators import bad_jump_mask, sma, trailing_median_dollar_vol

TRAIN_START = "2018-10-01"
TRAIN_END = "2022-12-31"
HOLDOUT_START = "2023-02-07"  # 25 trading-day purge embargo (> max_hold = 20d)
HORIZONS = (5, 10, 20)
RT_COST = 0.0010              # 10 bps round-trip shares
ALPHA_BONF = 0.05 / len(HORIZONS)
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


def _has_earnings_nearby(dates_sorted: List[str], sig_date: str, pad_days: int = 2) -> bool:
    if not dates_sorted:
        return False
    d0 = dt.date.fromisoformat(sig_date)
    lo = (d0 - dt.timedelta(days=pad_days)).isoformat()
    hi = (d0 + dt.timedelta(days=pad_days)).isoformat()
    idx = bisect.bisect_left(dates_sorted, lo)
    return idx < len(dates_sorted) and dates_sorted[idx] <= hi


def simulate_symbol_breakouts(
    sym: str,
    dates: List[str],
    closes: np.ndarray,
    volumes: np.ndarray,
    spy: BenchmarkSeries,
    earnings_map: Dict[str, List[str]],
    *,
    stratum: str = "broad",
    horizons: Sequence[int] = HORIZONS,
    rt_cost: float = RT_COST,
    min_price: float = 10.0,
    min_dollar_vol: float = 20_000_000.0,
) -> Tuple[Dict[int, List[EquityTrade]], int]:
    """Run exact preregistered Non-Earnings Volume Breakout rule on one ticker."""
    out: Dict[int, List[EquityTrade]] = {h: [] for h in horizons}
    n = closes.size
    max_h = max(horizons)
    if n < 260 + max_h:
        return out, 0

    jumps = bad_jump_mask(closes, max_abs_daily_ret=0.45)
    sma200 = sma(closes, 200)
    med_dv = trailing_median_dollar_vol(closes, volumes, 20)

    sym_earns = earnings_map.get(sym, [])
    earn_covered = bool(sym_earns)
    in_trade_until = {h: -1 for h in horizons}
    skipped_jumps = 0

    for t in range(252, n - 2):
        sig_d = dates[t]
        if sig_d < TRAIN_START:
            continue
        c_t = closes[t]
        c_prev = closes[t - 1]
        if not np.isfinite(c_t) or not np.isfinite(c_prev) or c_t < min_price or c_prev <= 0.0:
            continue
        if jumps[t]:
            continue
        # 1. Liquidity & uptrend
        if not np.isfinite(med_dv[t]) or med_dv[t] < min_dollar_vol:
            continue
        if not np.isfinite(sma200[t]) or c_t <= sma200[t]:
            continue
        # 2. +5.0% daily surge
        day_ret = c_t / c_prev - 1.0
        if day_ret < 0.05:
            continue
        # 3. >= 2.0x trailing 20d avg volume
        v_win = volumes[t - 20 : t]
        v_avg = float(np.mean(v_win))
        v_t = float(volumes[t])
        if not np.isfinite(v_avg) or v_avg <= 0.0 or v_t < 2.0 * v_avg:
            continue
        # 4. New 20d closing high
        prev_hi = float(np.max(closes[t - 20 : t]))
        if not np.isfinite(prev_hi) or c_t < prev_hi:
            continue
        # Guard trailing 20d against bad ticks
        if bool(np.any(jumps[t - 20 : t + 1])):
            skipped_jumps += 1
            continue
        # 5. Non-earnings filter (+/- 2 calendar days)
        if _has_earnings_nearby(sym_earns, sig_d, pad_days=2):
            continue

        entry_idx = t + 1
        if entry_idx >= n or jumps[entry_idx]:
            skipped_jumps += 1
            continue
        entry_px = float(closes[entry_idx])
        if not np.isfinite(entry_px) or entry_px <= 0.0:
            continue
        entry_d = dates[entry_idx]
        vol_ratio = v_t / v_avg

        for h in horizons:
            if t <= in_trade_until[h]:
                continue
            exit_idx = entry_idx + h
            if exit_idx >= n:
                continue
            if bool(np.any(jumps[entry_idx : exit_idx + 1])):
                skipped_jumps += 1
                in_trade_until[h] = exit_idx
                continue
            exit_px = float(closes[exit_idx])
            if not np.isfinite(exit_px) or exit_px <= 0.0:
                continue
            exit_d = dates[exit_idx]
            net_r = (exit_px / entry_px - 1.0) - rt_cost
            bench_r = spy.window_return(entry_d, exit_d)
            out[h].append(
                EquityTrade(
                    ticker=sym,
                    signal_date=sig_d,
                    entry_date=entry_d,
                    exit_date=exit_d,
                    entry_price=entry_px,
                    exit_price=exit_px,
                    net_return=net_r,
                    bench_return=bench_r,
                    exit_reason=f"hold_{h}d",
                    stratum=stratum,
                    score_tiebreak=-vol_ratio,  # highest relative volume first
                    extra={
                        "earn_covered": earn_covered,
                        "day_ret": round(day_ret, 4),
                        "vol_ratio": round(vol_ratio, 2),
                    },
                )
            )
            in_trade_until[h] = exit_idx

    return out, skipped_jumps


def run_on_squeeze_db(
    spy: BenchmarkSeries,
    earnings_map: Dict[str, List[str]],
    db_path: str = "data/squeeze_prices.db",
) -> Tuple[Dict[int, List[EquityTrade]], int, int]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    trades: Dict[int, List[EquityTrade]] = {h: [] for h in HORIZONS}
    skipped = 0
    n_syms = 0
    try:
        cur = con.execute("SELECT symbol, date, close, volume FROM px ORDER BY symbol, date")
        cur_sym: Optional[str] = None
        ds: List[str] = []
        cs: List[float] = []
        vs: List[float] = []

        def _flush(s: str, d_list: List[str], c_list: List[float], v_list: List[float]) -> None:
            nonlocal skipped, n_syms
            if len(c_list) < 280:
                return
            c_arr = np.asarray(c_list, dtype=float)
            if float(np.nanmax(c_arr)) < 10.0:
                return
            n_syms += 1
            res, sj = simulate_symbol_breakouts(
                s, d_list, c_arr, np.asarray(v_list, dtype=float), spy, earnings_map, stratum="broad"
            )
            for h in HORIZONS:
                trades[h].extend(res[h])
            skipped += sj

        for sym, d, c, v in cur:
            s = str(sym).upper()
            if s != cur_sym:
                if cur_sym is not None:
                    _flush(cur_sym, ds, cs, vs)
                cur_sym = s
                ds, cs, vs = [], [], []
            ds.append(str(d)[:10])
            cs.append(float(c) if c is not None else np.nan)
            vs.append(float(v) if v is not None else 0.0)
        if cur_sym is not None:
            _flush(cur_sym, ds, cs, vs)
    finally:
        con.close()
    return trades, skipped, n_syms


def evaluate_slice(trades: Sequence[EquityTrade], spy: BenchmarkSeries) -> Dict[str, object]:
    stats = moving_block_bootstrap_stats(
        trades,
        lambda t: iso_week_key(t.entry_date),
        block_len=4,
        n_boot=10_000,
        seed=SEED,
        alpha=ALPHA_BONF,
        min_clusters=MIN_CLUSTERS,
    )
    cap = simulate_capacity_book(trades, max_concurrent=5, slot_weight=0.20, spy_series=spy)
    pf_ci = stats.get("pf_ci")
    mr_ci = stats.get("mean_ret_ci")
    ex_ci = stats.get("mean_excess_ci")
    pass_ha = bool(
        not stats["insufficient"]
        and pf_ci is not None
        and mr_ci is not None
        and pf_ci[0] > 1.0
        and mr_ci[0] > 0.0
    )
    pass_hb = bool(
        not stats["insufficient"]
        and ex_ci is not None
        and ex_ci[0] > 0.0
    )
    stats["pass_ha"] = pass_ha
    stats["pass_hb"] = pass_hb
    stats["capacity_book"] = cap
    return stats


def main() -> None:
    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db")
    if "SPY" not in ohlcv:
        raise RuntimeError("SPY missing from data/equity_ohlcv.db")
    spy_dates, spy_closes, _ = ohlcv["SPY"]
    spy = BenchmarkSeries(spy_dates, spy_closes)
    earnings_map = load_earnings_calendar("data/dolt_options.db")

    print("Running Primary Broad Universe (data/squeeze_prices.db)...")
    broad_all, broad_skip, broad_syms = run_on_squeeze_db(spy, earnings_map)

    print(f"Running Secondary Liquid Universe (data/equity_ohlcv.db, {len(ohlcv)} symbols)...")
    liq_all: Dict[int, List[EquityTrade]] = {h: [] for h in HORIZONS}
    liq_skip = 0
    for sym, (ds, cs, vs) in sorted(ohlcv.items()):
        res, sj = simulate_symbol_breakouts(sym, ds, cs, vs, spy, earnings_map, stratum="liquid")
        for h in HORIZONS:
            liq_all[h].extend(res[h])
        liq_skip += sj

    summary = {
        "prereg": "docs/PREREG_NON_EARNINGS_BREAKOUT_20261124.md",
        "alpha_bonf": round(ALPHA_BONF, 5),
        "broad_syms_scanned": broad_syms,
        "broad_skipped_jumps": broad_skip,
        "liquid_syms_scanned": len(ohlcv),
        "liquid_skipped_jumps": liq_skip,
        "horizons": {},
    }

    for h in HORIZONS:
        b_train = [t for t in broad_all[h] if TRAIN_START <= t.signal_date <= TRAIN_END]
        b_hold = [t for t in broad_all[h] if t.signal_date >= HOLDOUT_START]
        b_hold_cov = [t for t in b_hold if t.extra.get("earn_covered")]
        l_train = [t for t in liq_all[h] if TRAIN_START <= t.signal_date <= TRAIN_END]
        l_hold = [t for t in liq_all[h] if t.signal_date >= HOLDOUT_START]
        l_hold_cov = [t for t in l_hold if t.extra.get("earn_covered")]

        summary["horizons"][str(h)] = {
            "broad_train": evaluate_slice(b_train, spy),
            "broad_holdout": evaluate_slice(b_hold, spy),
            "broad_holdout_earn_covered": evaluate_slice(b_hold_cov, spy),
            "liquid_train": evaluate_slice(l_train, spy),
            "liquid_holdout": evaluate_slice(l_hold, spy),
            "liquid_holdout_earn_covered": evaluate_slice(l_hold_cov, spy),
        }

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
