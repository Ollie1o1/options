#!/usr/bin/env python3
"""Preregistered estimator for Equity Playbook 3: Path-Managed Squeeze Momentum (Shares).

Frozen registration: docs/PREREG_SQUEEZE_PATH_EQUITY_20261124.md
Reuses survivorship-free data/squeeze_backtest.db, data/squeeze_shares.db,
and data/squeeze_prices.db directly with exact +8d FINRA lag + 1d entry delay.
"""
from __future__ import annotations

import argparse
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
    load_equity_ohlcv,
    moving_block_bootstrap_stats,
    simulate_capacity_book,
)
from src.squeeze.backtest import DEFAULT_DB, PRICES_DB, SHARES_DB
from src.squeeze.backtest.panel import MAX_ENTRY_GAP_DAYS, PUB_LAG_DAYS, PriceBook
from src.squeeze.backtest.shares import SharesLookup
from src.squeeze.backtest.universe import study_symbols

SEED = 20261124
ALPHA_BONF3 = 0.05 / 3.0
SI_SCALE = 1.25          # 80% float assumption matching SQUEEZE_BACKTEST.md
RT_COST = 0.0040         # 40 bps round-trip (small/mid-cap realistic)
DELIST_PENALTY = 0.20    # -20% liquidation penalty if ticker ends before exit
MAX_HOLD = 42
MEME_START = "2021-01-01"
MEME_END = "2021-03-31"
HOLDOUT_START = "2023-01-01"


def _simulate_path_arm(
    dates: List[str],
    closes: np.ndarray,
    entry_idx: int,
    arm: str,
    rt_cost: float,
) -> Tuple[str, float, float, str]:
    """Execute frozen exit rules (S1, S2, S3, or HOLD42 control) with next-day close fills.

    Returns (exit_date, exit_price_eff, net_return, exit_reason).
    """
    n = closes.size
    entry_px = float(closes[entry_idx])
    peak = entry_px

    if arm == "HOLD42":
        end_i = entry_idx + MAX_HOLD
        if end_i >= n:
            last_px = float(closes[-1])
            net_r = (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost
            return dates[-1], last_px, net_r, "ticker_ended_penalized"
        px = float(closes[end_i])
        return dates[end_i], px, (px / entry_px - 1.0) - rt_cost, "time_42d"

    if arm == "S1":
        # -12% initial stop / -12% trailing stop from running peak close
        for k in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, n)):
            ck = float(closes[k])
            if ck > peak:
                peak = ck
            hit_stop = (ck / entry_px - 1.0 <= -0.12)
            hit_trail = (ck / peak - 1.0 <= -0.12)
            if hit_stop or hit_trail:
                ex_i = k + 1
                if ex_i >= n:
                    last_px = float(closes[-1])
                    return dates[-1], last_px, (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost, "ticker_ended_penalized"
                px = float(closes[ex_i])
                return dates[ex_i], px, (px / entry_px - 1.0) - rt_cost, ("stop_12" if hit_stop else "trail_12")
        ex_i = entry_idx + MAX_HOLD
        if ex_i >= n:
            last_px = float(closes[-1])
            return dates[-1], last_px, (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost, "ticker_ended_penalized"
        px = float(closes[ex_i])
        return dates[ex_i], px, (px / entry_px - 1.0) - rt_cost, "time_42d"

    if arm == "S3":
        # -12% stop / +25% full take-profit
        for k in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, n)):
            ck = float(closes[k])
            hit_tp = (ck / entry_px - 1.0 >= 0.25)
            hit_stop = (ck / entry_px - 1.0 <= -0.12)
            if hit_tp or hit_stop:
                ex_i = k + 1
                if ex_i >= n:
                    last_px = float(closes[-1])
                    return dates[-1], last_px, (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost, "ticker_ended_penalized"
                px = float(closes[ex_i])
                return dates[ex_i], px, (px / entry_px - 1.0) - rt_cost, ("tp_25" if hit_tp else "stop_12")
        ex_i = entry_idx + MAX_HOLD
        if ex_i >= n:
            last_px = float(closes[-1])
            return dates[-1], last_px, (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost, "ticker_ended_penalized"
        px = float(closes[ex_i])
        return dates[ex_i], px, (px / entry_px - 1.0) - rt_cost, "time_42d"

    if arm == "S2":
        # Primary: -12% initial stop, sell 50% on first close >= +20%, remainder rides -15% trail
        half_sold = False
        half_ret = 0.0
        for k in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, n)):
            ck = float(closes[k])
            if ck > peak:
                peak = ck
            if not half_sold and (ck / entry_px - 1.0 >= 0.20):
                ex_half_i = k + 1
                if ex_half_i >= n:
                    last_px = float(closes[-1])
                    return dates[-1], last_px, (last_px / entry_px - 1.0) - DELIST_PENALTY - rt_cost, "ticker_ended_penalized"
                half_ret = float(closes[ex_half_i]) / entry_px - 1.0
                half_sold = True
                continue
            hit_stop = (not half_sold) and (ck / entry_px - 1.0 <= -0.12)
            hit_trail = (ck / peak - 1.0 <= -0.15)
            if hit_stop or hit_trail:
                ex_i = k + 1
                if ex_i >= n:
                    last_px = float(closes[-1])
                    rem_r = (last_px / entry_px - 1.0) - DELIST_PENALTY
                    tot_r = (0.5 * half_ret + 0.5 * rem_r) if half_sold else rem_r
                    return dates[-1], last_px, tot_r - rt_cost, "ticker_ended_penalized"
                px = float(closes[ex_i])
                rem_r = px / entry_px - 1.0
                tot_r = (0.5 * half_ret + 0.5 * rem_r) if half_sold else rem_r
                reason = "scaled_trail_15" if half_sold else ("stop_12" if hit_stop else "trail_15")
                return dates[ex_i], px, tot_r - rt_cost, reason
        ex_i = entry_idx + MAX_HOLD
        if ex_i >= n:
            last_px = float(closes[-1])
            rem_r = (last_px / entry_px - 1.0) - DELIST_PENALTY
            tot_r = (0.5 * half_ret + 0.5 * rem_r) if half_sold else rem_r
            return dates[-1], last_px, tot_r - rt_cost, "ticker_ended_penalized"
        px = float(closes[ex_i])
        rem_r = px / entry_px - 1.0
        tot_r = (0.5 * half_ret + 0.5 * rem_r) if half_sold else rem_r
        return dates[ex_i], px, tot_r - rt_cost, ("scaled_time_42d" if half_sold else "time_42d")

    raise ValueError(f"Unknown arm: {arm}")


def evaluate_arm(trades: Sequence[EquityTrade], spy: BenchmarkSeries, arm_name: str, n_trials: int = 3) -> dict:
    full = list(trades)
    ex_meme = [t for t in full if not (MEME_START <= t.entry_date <= MEME_END)]
    holdout = [t for t in full if t.entry_date >= HOLDOUT_START]

    s_full = moving_block_bootstrap_stats(
        full, lambda t: t.cluster_override or t.signal_date,
        block_len=4, n_boot=10_000, seed=SEED, alpha=ALPHA_BONF3, min_clusters=50,
    )
    s_exm = moving_block_bootstrap_stats(
        ex_meme, lambda t: t.cluster_override or t.signal_date,
        block_len=4, n_boot=10_000, seed=SEED, alpha=ALPHA_BONF3, min_clusters=50,
    )
    s_hold = moving_block_bootstrap_stats(
        holdout, lambda t: t.cluster_override or t.signal_date,
        block_len=4, n_boot=10_000, seed=SEED, alpha=0.05, min_clusters=50,
    )
    s_exm_95 = moving_block_bootstrap_stats(
        ex_meme, lambda t: t.cluster_override or t.signal_date,
        block_len=4, n_boot=10_000, seed=SEED, alpha=0.05, min_clusters=50,
    )

    # Drop top-3 highest-contributing tickers check (on ex-meme)
    pnl_by_tk: Dict[str, float] = collections.defaultdict(float)
    for t in ex_meme:
        pnl_by_tk[t.ticker] += t.net_return
    top3 = {tk for tk, _ in sorted(pnl_by_tk.items(), key=lambda kv: kv[1], reverse=True)[:3]}
    no_top3 = [t for t in ex_meme if t.ticker not in top3]
    no_top3_mean = float(np.mean([t.net_return for t in no_top3])) if no_top3 else 0.0
    no_top3_r = np.array([t.net_return for t in no_top3], dtype=float)
    g3 = float(np.sum(no_top3_r[no_top3_r > 0]))
    l3 = float(np.abs(np.sum(no_top3_r[no_top3_r < 0])))
    no_top3_pf = g3 / l3 if l3 > 0 else 0.0

    # Binding 4-part Pass Rule
    if s_exm["insufficient"] or s_hold["insufficient"]:
        verdict = "INSUFFICIENT"
    else:
        c1 = (s_exm["mean_ret_ci"][0] > 0.0 and s_exm["pf_ci"][0] > 1.0)
        c2 = (s_hold["mean_ret"] > 0.0 and s_hold["pf"] > 1.10 and s_hold["mean_ret_ci"][0] > 0.0)
        c3 = (s_exm_95["mean_excess_ci"] is not None and s_exm_95["mean_excess_ci"][0] > 0.0)
        c4 = (no_top3_mean > 0.0 and no_top3_pf > 1.0)
        verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"

    alloc_sum = summarise(ex_meme, n_trials=n_trials) if len(ex_meme) >= 3 else {"n": len(ex_meme), "insufficient": True}
    cap_book = simulate_capacity_book(ex_meme, max_concurrent=4, slot_weight=0.10, spy_series=spy)
    delisted_n = sum(1 for t in full if "ticker_ended" in t.exit_reason)

    return {
        "arm": arm_name,
        "verdict": verdict,
        "alloc_promotion": promotion_verdict(alloc_sum),
        "delisted_count": delisted_n,
        "full": s_full,
        "ex_meme_bonf": s_exm,
        "ex_meme_95": s_exm_95,
        "holdout_2023_2026": s_hold,
        "drop_top3_check": {
            "dropped_tickers": sorted(top3),
            "mean_ret": round(no_top3_mean, 5),
            "pf": round(no_top3_pf, 3),
        },
        "capacity_4slot_ex_meme": cap_book,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest Equity Playbook 3 (Path-Managed Squeeze Momentum)")
    p.add_argument("--out", default="reports/backtest_squeeze_equity_20261124.json")
    args = p.parse_args()

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db", ["SPY"])
    spy = BenchmarkSeries(ohlcv["SPY"][0], ohlcv["SPY"][1])

    print("Loading survivorship-free squeeze panel inputs...")
    syms = study_symbols(DEFAULT_DB)
    book = PriceBook(PRICES_DB, syms)
    shares = SharesLookup(SHARES_DB)

    con = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT settlement_date, symbol, shares_short FROM si WHERE shares_short >= 250000 ORDER BY settlement_date"
    ).fetchall()
    con.close()

    by_settle: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for sdate, sym, ss in rows:
        if sym not in book:
            continue
        so = shares.get(sym, sdate)
        if not so or so <= 0:
            continue
        si_ratio = (ss / so) * SI_SCALE
        if not (0.0 < si_ratio < 1.5 * SI_SCALE):
            continue
        by_settle[sdate].append((sym, si_ratio))

    arms = ("S1", "S2", "S3", "HOLD42")
    arm_trades: Dict[str, List[EquityTrade]] = {a: [] for a in arms}
    arm_trades_100bps: List[EquityTrade] = []
    arm_trades_20m: List[EquityTrade] = []
    in_trade_until: Dict[Tuple[str, str], str] = {}

    for sdate in sorted(by_settle.keys()):
        items = by_settle[sdate]
        if len(items) < 20:
            continue
        ratios = np.array([r for _, r in items], dtype=float)
        cutoff = float(np.percentile(ratios, 95.0))
        top5 = [(sym, r) for sym, r in items if r >= cutoff]

        for sym, si_float in top5:
            i0 = book._idx_asof(sym, sdate)
            if i0 < 0:
                continue
            dissem_i = i0 + PUB_LAG_DAYS
            closes = book._close[sym]
            vols = book._vol[sym]
            dates = book._dates[sym]
            if dissem_i + 2 >= closes.size or dissem_i < 25:
                continue
            dissem_d = dates[dissem_i]
            if (dt.date.fromisoformat(dissem_d) - dt.date.fromisoformat(sdate)).days > MAX_ENTRY_GAP_DAYS:
                continue

            spot = float(closes[dissem_i])
            if spot < 5.0 or not np.isfinite(spot):
                continue
            dv_win = closes[dissem_i - 19 : dissem_i + 1] * vols[dissem_i - 19 : dissem_i + 1]
            med_dv = float(np.median(dv_win))
            if med_dv < 5_000_000.0:
                continue
            c5 = float(closes[dissem_i - 5])
            if c5 <= 0.0:
                continue
            ret5d = spot / c5 - 1.0
            if ret5d < 0.10:
                continue

            # Enter 1 session AFTER dissemination bar close
            entry_i = dissem_i + 1
            entry_d = dates[entry_i]
            entry_px = float(closes[entry_i])
            if entry_px <= 0.0 or not np.isfinite(entry_px):
                continue

            for arm in arms:
                if in_trade_until.get((arm, sym), "") >= entry_d:
                    continue
                ex_d, ex_px, net_r, reason = _simulate_path_arm(dates, closes, entry_i, arm, RT_COST)
                bench_r = spy.window_return(entry_d, ex_d)
                tr = EquityTrade(
                    ticker=sym,
                    signal_date=dissem_d,
                    entry_date=entry_d,
                    exit_date=ex_d,
                    entry_price=entry_px,
                    exit_price=ex_px,
                    net_return=net_r,
                    bench_return=bench_r,
                    exit_reason=reason,
                    stratum="broad",
                    score_tiebreak=-si_float,
                    cluster_override=sdate,
                    extra={"si_float": si_float, "ret5d": ret5d, "med_dv": med_dv},
                )
                arm_trades[arm].append(tr)
                in_trade_until[(arm, sym)] = ex_d
                if arm == "S2":
                    arm_trades_100bps.append(EquityTrade(**{**tr.__dict__, "net_return": net_r - 0.0060}))
                    if med_dv >= 20_000_000.0:
                        arm_trades_20m.append(tr)

    results = {arm: evaluate_arm(arm_trades[arm], spy, arm, n_trials=3) for arm in arms}
    results["S2_sensitivity_100bps"] = evaluate_arm(arm_trades_100bps, spy, "S2_100bps", n_trials=3)
    results["S2_sensitivity_20M_dv"] = evaluate_arm(arm_trades_20m, spy, "S2_20M_dv", n_trials=3)

    payload = {
        "prereg": "docs/PREREG_SQUEEZE_PATH_EQUITY_20261124.md",
        "arms": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for arm in ("S1", "S2", "S3", "HOLD42", "S2_sensitivity_100bps", "S2_sensitivity_20M_dv"):
        r = results[arm]
        e = r["ex_meme_bonf"]
        h = r["holdout_2023_2026"]
        print(
            f"\n[{arm}] VERDICT={r['verdict']} alloc={r['alloc_promotion']} delisted={r['delisted_count']}"
            f"\n  Ex-Meme (98.3%): n={e['n_trades']} dates={e['n_clusters']} mean={e['mean_ret']:+.4%} CI={e['mean_ret_ci']} PF={e['pf']:.3f} CI={e['pf_ci']}"
            f"\n  Holdout (23-26): n={h['n_trades']} dates={h['n_clusters']} mean={h['mean_ret']:+.4%} CI={h['mean_ret_ci']} PF={h['pf']:.3f}"
            f"\n  DropTop3: {r['drop_top3_check']}  |  4-slot ex-meme: CAGR={r['capacity_4slot_ex_meme']['cagr']:+.2%} MaxDD={r['capacity_4slot_ex_meme']['max_dd']:+.2%}"
        )


if __name__ == "__main__":
    main()
