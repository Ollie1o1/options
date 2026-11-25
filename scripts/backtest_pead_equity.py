#!/usr/bin/env python3
"""Preregistered estimator for Equity Playbook 4 Arm A: Multi-Week PEAD Reaction Drift (Shares).

Frozen registration: docs/PREREG_PEAD_AND_REVISIONS_20261124.md
Uses data/dolt_options.db::earnings_cal + survivorship-free data/squeeze_prices.db
with strict D+1 close entry and non-earnings +5%/2x-vol negative control.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from src.alloc.report import promotion_verdict, summarise
from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    cluster_bootstrap_stats,
    load_equity_ohlcv,
    month_key,
)
from src.squeeze.backtest import PRICES_DB
from src.squeeze.backtest.panel import PriceBook

SEED = 20261124
RT_COST = 0.0010     # 10 bps round-trip
SPLIT_DATE = "2023-07-01"
MIN_EVENTS = 100
MIN_MONTHS = 24


def _eval_cohort(trades: List[EquityTrade], label: str) -> dict:
    h1 = [t for t in trades if t.entry_date < SPLIT_DATE]
    h2 = [t for t in trades if t.entry_date >= SPLIT_DATE]

    s_all = cluster_bootstrap_stats(trades, lambda t: month_key(t.signal_date), n_boot=10_000, seed=SEED, min_clusters=MIN_MONTHS)
    s_h1 = cluster_bootstrap_stats(h1, lambda t: month_key(t.signal_date), n_boot=10_000, seed=SEED, min_clusters=12)
    s_h2 = cluster_bootstrap_stats(h2, lambda t: month_key(t.signal_date), n_boot=10_000, seed=SEED, min_clusters=12)

    if s_all["insufficient"] or len(trades) < MIN_EVENTS:
        verdict = "INSUFFICIENT"
    else:
        same_sign = (
            s_h1["mean_excess"] is not None and s_h1["mean_excess"] > 0.0
            and s_h2["mean_excess"] is not None and s_h2["mean_excess"] > 0.0
        )
        ok = (
            s_all["mean_excess_ci"] is not None and s_all["mean_excess_ci"][0] > 0.0
            and s_all["pf_ci"] is not None and s_all["pf_ci"][0] > 1.0
            and same_sign
        )
        verdict = "PASS" if ok else "FAIL"

    alloc = summarise(trades, n_trials=1) if len(trades) >= 3 else {"n": len(trades), "insufficient": True}
    return {
        "label": label,
        "verdict": verdict,
        "alloc_promotion": promotion_verdict(alloc),
        "full": s_all,
        "half1_2020_2023H1": s_h1,
        "half2_2023H2_2026": s_h2,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest Equity Playbook 4 Arm A (Multi-Week PEAD)")
    p.add_argument("--out", default="reports/backtest_pead_equity_20261124.json")
    args = p.parse_args()

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db", ["SPY"])
    spy = BenchmarkSeries(ohlcv["SPY"][0], ohlcv["SPY"][1])

    con = sqlite3.connect("file:data/dolt_options.db?mode=ro", uri=True)
    ev_rows = con.execute(
        "SELECT symbol, date, whn FROM earnings_cal "
        "WHERE date >= '2020-01-22' AND date <= '2026-06-30' "
        "AND LOWER(whn) IN ('before market open', 'after market close', 'before-open', 'after-close') "
        "ORDER BY date, symbol"
    ).fetchall()
    con.close()

    syms = sorted({str(s).upper() for s, _, _ in ev_rows})
    print(f"Loaded {len(ev_rows)} timed earnings events across {len(syms)} symbols. Loading PriceBook...")
    book = PriceBook(PRICES_DB, syms)

    earn_reaction_keys: Set[Tuple[str, str]] = set()
    trades_20d: List[EquityTrade] = []
    trades_40d: List[EquityTrade] = []
    neg_react_20d: List[EquityTrade] = []

    for sym_raw, ev_date, whn in ev_rows:
        sym = str(sym_raw).upper()
        if sym not in book:
            continue
        dates = book._dates[sym]
        closes = book._close[sym]
        vols = book._vol[sym]

        # Reaction session D
        if "before" in str(whn).lower():
            d_idx = bisect.bisect_left(dates, ev_date)
        else:  # after-close
            d_idx = bisect.bisect_right(dates, ev_date)

        if d_idx < 25 or d_idx + 22 >= closes.size:
            continue
        d_date = dates[d_idx]
        earn_reaction_keys.add((sym, d_date))

        c_prev = float(closes[d_idx - 1])
        c_d = float(closes[d_idx])
        c_in = float(closes[d_idx + 1])
        if c_prev < 10.0 or c_d <= 0.0 or c_in <= 0.0:
            continue
        med_vol = float(np.median(vols[d_idx - 21 : d_idx]))
        med_dv = float(np.median(closes[d_idx - 21 : d_idx] * vols[d_idx - 21 : d_idx]))
        if med_vol <= 0.0 or med_dv < 20_000_000.0:
            continue
        react_ret = c_d / c_prev - 1.0
        vol_mult = float(vols[d_idx]) / med_vol

        # Positive PEAD breakout (primary)
        if react_ret >= 0.05 and vol_mult >= 2.0 and c_in >= c_d * 0.98:
            entry_d = dates[d_idx + 1]
            # 20d horizon (D+1 -> D+21)
            ex20_d = dates[d_idx + 21]
            ex20_px = float(closes[d_idx + 21])
            trades_20d.append(EquityTrade(
                ticker=sym, signal_date=d_date, entry_date=entry_d, exit_date=ex20_d,
                entry_price=c_in, exit_price=ex20_px,
                net_return=(ex20_px / c_in - 1.0) - RT_COST,
                bench_return=spy.window_return(entry_d, ex20_d),
                exit_reason="horizon_20d", stratum="broad",
                extra={"react_ret": react_ret, "vol_mult": vol_mult},
            ))
            # 40d horizon (D+1 -> D+41)
            if d_idx + 41 < closes.size:
                ex40_d = dates[d_idx + 41]
                ex40_px = float(closes[d_idx + 41])
                trades_40d.append(EquityTrade(
                    ticker=sym, signal_date=d_date, entry_date=entry_d, exit_date=ex40_d,
                    entry_price=c_in, exit_price=ex40_px,
                    net_return=(ex40_px / c_in - 1.0) - RT_COST,
                    bench_return=spy.window_return(entry_d, ex40_d),
                    exit_reason="horizon_40d", stratum="broad",
                ))

        # Negative earnings reaction diagnostic (mirror: react_ret <= -5%, vol >= 2x)
        if react_ret <= -0.05 and vol_mult >= 2.0 and c_in <= c_d * 1.02:
            entry_d = dates[d_idx + 1]
            ex20_d = dates[d_idx + 21]
            ex20_px = float(closes[d_idx + 21])
            neg_react_20d.append(EquityTrade(
                ticker=sym, signal_date=d_date, entry_date=entry_d, exit_date=ex20_d,
                entry_price=c_in, exit_price=ex20_px,
                net_return=(ex20_px / c_in - 1.0) - RT_COST,
                bench_return=spy.window_return(entry_d, ex20_d),
                exit_reason="horizon_20d_neg_mirror", stratum="broad",
            ))

    # Negative Control (Step 6 residualization): non-earnings +5% / 2x vol jumps on same universe
    non_earn_20d: List[EquityTrade] = []
    for sym in syms:
        if sym not in book:
            continue
        dates = book._dates[sym]
        closes = book._close[sym]
        vols = book._vol[sym]
        last_used = -999
        for i in range(25, closes.size - 22):
            if i <= last_used + 20:
                continue
            d_date = dates[i]
            if d_date < "2020-01-22" or d_date > "2026-06-30":
                continue
            if (sym, d_date) in earn_reaction_keys:
                continue
            c_prev = float(closes[i - 1])
            c_d = float(closes[i])
            c_in = float(closes[i + 1])
            if c_prev < 10.0 or c_d <= 0.0 or c_in <= 0.0:
                continue
            react_ret = c_d / c_prev - 1.0
            if react_ret < 0.05 or c_in < c_d * 0.98:
                continue
            med_vol = float(np.median(vols[i - 21 : i]))
            med_dv = float(np.median(closes[i - 21 : i] * vols[i - 21 : i]))
            if med_vol <= 0.0 or med_dv < 20_000_000.0 or (float(vols[i]) / med_vol) < 2.0:
                continue
            entry_d = dates[i + 1]
            ex20_d = dates[i + 21]
            ex20_px = float(closes[i + 21])
            non_earn_20d.append(EquityTrade(
                ticker=sym, signal_date=d_date, entry_date=entry_d, exit_date=ex20_d,
                entry_price=c_in, exit_price=ex20_px,
                net_return=(ex20_px / c_in - 1.0) - RT_COST,
                bench_return=spy.window_return(entry_d, ex20_d),
                exit_reason="horizon_20d_non_earn_ctrl", stratum="broad",
            ))
            last_used = i

    payload = {
        "prereg": "docs/PREREG_PEAD_AND_REVISIONS_20261124.md",
        "primary_pead_20d": _eval_cohort(trades_20d, "pead_pos_20d"),
        "secondary_pead_40d": _eval_cohort(trades_40d, "pead_pos_40d"),
        "diagnostic_neg_earnings_20d": _eval_cohort(neg_react_20d, "pead_neg_20d"),
        "control_non_earnings_jump_20d": _eval_cohort(non_earn_20d, "non_earn_pos_20d"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for k in ("primary_pead_20d", "secondary_pead_40d", "diagnostic_neg_earnings_20d", "control_non_earnings_jump_20d"):
        r = payload[k]
        f = r["full"]
        print(
            f"\n[{r['label']}] VERDICT={r['verdict']} alloc={r['alloc_promotion']}"
            f"\n  Full : n={f['n_trades']} mo={f['n_clusters']} mean={f['mean_ret']:+.4%} CI={f['mean_ret_ci']} PF={f['pf']:.3f} CI={f['pf_ci']} ex={f['mean_excess']:+.4%} CI={f['mean_excess_ci']}"
            f"\n  H1/H2: H1 ex={r['half1_2020_2023H1']['mean_excess']} | H2 ex={r['half2_2023H2_2026']['mean_excess']}"
        )


if __name__ == "__main__":
    main()
