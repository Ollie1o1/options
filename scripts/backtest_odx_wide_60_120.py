#!/usr/bin/env python3
"""Test O1: 60-120 DTE Wider (5%-of-Spot) SPY Credit Spreads, Delta-Adjusted (2010-2023).

Frozen registration: docs/PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md
Data: data/optionsdx.db (odx_chain, SPY 2010-01-04 .. 2023-12-29)
Output: reports/backtest_odx_wide_60_120_20261124.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.alloc.validate import effective_n

DB_PATH = "data/optionsdx.db"
OUT_PATH = "reports/backtest_odx_wide_60_120_20261124.json"
SEED = 20261124
N_BOOT = 10_000
BLOCK_LEN_MONTHS = 3
ALPHA_BONF = 0.05 / 2.0  # 2 exit arms -> 97.5% two-sided CI
MIN_EFF_N = 40.0


@dataclass
class SpreadTrade:
    side: str  # "bull_put" or "bear_call"
    arm: str   # "O1-H" or "O1-M"
    fill_policy: str  # "crossed" | "limit" | "mid"
    entry_date: str
    exit_date: str
    expiration: str
    k_short: float
    k_long: float
    dte_entry: float
    spot_entry: float
    spot_exit: float
    credit_entry: float
    close_cost: float
    car: float
    pnl: float
    roc: float
    delta_net_entry: float
    hedge_entry: float
    hedge_dyn: float
    excess_entry: float
    excess_dyn: float
    friction_to_credit: float
    exit_reason: str


def _fill_entry_credit(s_bid: float, s_ask: float, l_bid: float, l_ask: float, policy: str, k: float = 0.35) -> float:
    s_mid = 0.5 * (s_bid + s_ask)
    l_mid = 0.5 * (l_bid + l_ask)
    if policy == "mid":
        return s_mid - l_mid
    if policy == "limit":
        s_px = s_mid - k * (s_mid - s_bid)
        l_px = l_mid + k * (l_ask - l_mid)
        return s_px - l_px
    # crossed
    return s_bid - l_ask


def _fill_close_debit(s_bid: float, s_ask: float, l_bid: float, l_ask: float, policy: str, k: float = 0.35) -> float:
    s_mid = 0.5 * (s_bid + s_ask)
    l_mid = 0.5 * (l_bid + l_ask)
    if policy == "mid":
        return s_mid - l_mid
    if policy == "limit":
        s_px = s_mid + k * (s_ask - s_mid)
        l_px = l_mid - k * (l_mid - l_bid)
        return s_px - l_px
    # crossed
    return s_ask - l_bid


def _intrinsic_spread(spot: float, k_short: float, k_long: float, opt_type: str) -> float:
    if opt_type == "put":
        return max(k_short - spot, 0.0) - max(k_long - spot, 0.0)
    return max(spot - k_short, 0.0) - max(spot - k_long, 0.0)


def _moving_block_month_boot(
    trades: List[SpreadTrade],
    block_len: int = BLOCK_LEN_MONTHS,
    alpha: float = ALPHA_BONF,
) -> Dict[str, object]:
    if not trades:
        return {"n": 0, "n_months": 0, "insufficient": True}
    by_month: Dict[str, List[SpreadTrade]] = {}
    for t in trades:
        by_month.setdefault(t.entry_date[:7], []).append(t)
    months = sorted(by_month.keys())
    k = len(months)
    roc_all = np.array([t.roc for t in trades], dtype=float)
    x_all = np.array([t.excess_entry for t in trades], dtype=float)
    xd_all = np.array([t.excess_dyn for t in trades], dtype=float)

    def _pf(arr: np.ndarray) -> float:
        neg = float(-np.sum(arr[arr < 0.0]))
        pos = float(np.sum(arr[arr > 0.0]))
        return pos / neg if neg > 0 else float("inf")

    m_roc = [np.array([t.roc for t in by_month[m]], dtype=float) for m in months]
    m_x = [np.array([t.excess_entry for t in by_month[m]], dtype=float) for m in months]
    m_xd = [np.array([t.excess_dyn for t in by_month[m]], dtype=float) for m in months]

    rng = np.random.default_rng(SEED)
    b = max(1, min(block_len, k))
    n_blocks = int(np.ceil(k / b))
    max_start = max(1, k - b + 1)

    b_roc = np.empty(N_BOOT, dtype=float)
    b_pf = np.empty(N_BOOT, dtype=float)
    b_x = np.empty(N_BOOT, dtype=float)
    b_xd = np.empty(N_BOOT, dtype=float)

    for i in range(N_BOOT):
        starts = rng.integers(0, max_start, size=n_blocks)
        idxs: List[int] = []
        for s in starts:
            idxs.extend(range(int(s), min(int(s) + b, k)))
        idxs = idxs[:k]
        sr = np.concatenate([m_roc[j] for j in idxs])
        sx = np.concatenate([m_x[j] for j in idxs])
        sxd = np.concatenate([m_xd[j] for j in idxs])
        b_roc[i] = float(np.mean(sr))
        b_pf[i] = _pf(sr)
        b_x[i] = float(np.mean(sx))
        b_xd[i] = float(np.mean(sxd))

    lo_p, hi_p = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    pf_method = "nearest" if np.any(np.isinf(b_pf)) else "linear"

    # Compute effective_n on trade entry/exit dates
    eff_n = float(effective_n([t.entry_date for t in trades], [t.exit_date for t in trades]))
    hold_days = [
        (dt.date.fromisoformat(t.exit_date) - dt.date.fromisoformat(t.entry_date)).days
        for t in trades
    ]

    mean_x = float(np.mean(x_all))
    mean_xd = float(np.mean(xd_all))
    pf_val = _pf(roc_all)
    x_ci = [float(np.percentile(b_x, lo_p)), float(np.percentile(b_x, hi_p))]
    pf_ci = [
        float(np.percentile(b_pf, lo_p, method=pf_method)),
        float(np.percentile(b_pf, hi_p, method=pf_method)),
    ]
    roc_ci = [float(np.percentile(b_roc, lo_p)), float(np.percentile(b_roc, hi_p))]
    xd_ci = [float(np.percentile(b_xd, lo_p)), float(np.percentile(b_xd, hi_p))]

    passed = bool(
        eff_n >= MIN_EFF_N
        and x_ci[0] > 0.0
        and pf_ci[0] > 1.0
        and mean_xd > 0.0
    )

    by_year: Dict[str, Dict[str, float]] = {}
    for yr in sorted({t.entry_date[:4] for t in trades}):
        yt = [t for t in trades if t.entry_date[:4] == yr]
        yr_r = np.array([t.roc for t in yt], dtype=float)
        yr_x = np.array([t.excess_entry for t in yt], dtype=float)
        yr_xd = np.array([t.excess_dyn for t in yt], dtype=float)
        by_year[yr] = {
            "n": len(yt),
            "win_rate": round(float(np.mean(yr_r > 0)), 4),
            "mean_roc": round(float(np.mean(yr_r)), 5),
            "mean_excess_entry": round(float(np.mean(yr_x)), 5),
            "mean_excess_dyn": round(float(np.mean(yr_xd)), 5),
            "pf": round(_pf(yr_r), 3),
        }

    return {
        "n": len(trades),
        "n_months": k,
        "effective_n": round(eff_n, 2),
        "block_len_months": b,
        "alpha": round(alpha, 5),
        "median_hold_days": int(np.median(hold_days)),
        "mean_friction_to_credit": round(float(np.mean([t.friction_to_credit for t in trades])), 4),
        "median_friction_to_credit": round(float(np.median([t.friction_to_credit for t in trades])), 4),
        "win_rate": round(float(np.mean(roc_all > 0)), 4),
        "mean_roc": round(float(np.mean(roc_all)), 5),
        "roc_ci": [round(roc_ci[0], 5), round(roc_ci[1], 5)],
        "pf": round(pf_val, 4),
        "pf_ci": [round(pf_ci[0], 4), round(pf_ci[1], 4)],
        "mean_excess_entry": round(mean_x, 5),
        "excess_entry_ci": [round(x_ci[0], 5), round(x_ci[1], 5)],
        "mean_excess_dyn": round(mean_xd, 5),
        "excess_dyn_ci": [round(xd_ci[0], 5), round(xd_ci[1], 5)],
        "verdict": "PASS" if passed else "FAIL",
        "by_year": by_year,
    }


def _select_weekly_entries(all_dates: List[str]) -> List[str]:
    """Select the last trading day (usually Friday) of each ISO week."""
    by_week: Dict[Tuple[int, int], str] = {}
    for d in all_dates:
        iso = dt.date.fromisoformat(d).isocalendar()
        by_week[(iso.year, iso.week)] = d
    return [by_week[k] for k in sorted(by_week.keys())]


def run_side(
    con: sqlite3.Connection,
    all_dates: List[str],
    date_to_idx: Dict[str, int],
    spot_by_date: Dict[str, float],
    weekly_dates: List[str],
    side: str,
) -> Dict[str, List[SpreadTrade]]:
    """Simulate O1-H and O1-M across crossed/limit/mid policies for bull_put or bear_call."""
    opt_type = "put" if side == "bull_put" else "call"
    target_delta = -0.25 if opt_type == "put" else 0.25
    cur = con.cursor()

    # Step 1: Select weekly candidate spreads (same geometry across policies)
    candidates: List[Dict[str, object]] = []
    skipped: Dict[str, int] = {}

    for edate in weekly_dates:
        rows = cur.execute(
            """
            SELECT expiration, strike, bid, ask, mid, delta, underlying, dte
            FROM odx_chain
            WHERE symbol = 'SPY' AND date = ? AND type = ? AND dte BETWEEN 60 AND 120
            """,
            (edate, opt_type),
        ).fetchall()
        if not rows:
            skipped["no_chain"] = skipped.get("no_chain", 0) + 1
            continue

        # Nearest expiration to 90 DTE
        exp_dtes: Dict[str, float] = {}
        for r in rows:
            exp_dtes[r[0]] = float(r[7])
        chosen_exp = min(exp_dtes.keys(), key=lambda e: (abs(exp_dtes[e] - 90.0), e))
        exp_rows = [r for r in rows if r[0] == chosen_exp]

        # Short leg nearest target_delta within +/-0.08
        valid_short = [
            r for r in exp_rows
            if r[5] is not None and abs(float(r[5]) - target_delta) <= 0.08
            and r[2] is not None and r[3] is not None and float(r[2]) > 0 and float(r[3]) >= float(r[2])
        ]
        if not valid_short:
            skipped["no_short_delta"] = skipped.get("no_short_delta", 0) + 1
            continue
        short_leg = min(valid_short, key=lambda r: abs(float(r[5]) - target_delta))
        k_short = float(short_leg[1])
        spot = float(short_leg[6])
        dte_entry = float(short_leg[7])

        # Long leg nearest 5.0% of spot away
        if opt_type == "put":
            target_k_long = k_short - 0.05 * spot
            valid_long = [
                r for r in exp_rows
                if float(r[1]) < k_short
                and r[2] is not None and r[3] is not None and float(r[2]) > 0 and float(r[3]) >= float(r[2])
            ]
        else:
            target_k_long = k_short + 0.05 * spot
            valid_long = [
                r for r in exp_rows
                if float(r[1]) > k_short
                and r[2] is not None and r[3] is not None and float(r[2]) > 0 and float(r[3]) >= float(r[2])
            ]
        if not valid_long:
            skipped["no_long_leg"] = skipped.get("no_long_leg", 0) + 1
            continue
        long_leg = min(valid_long, key=lambda r: abs(float(r[1]) - target_k_long))
        k_long = float(long_leg[1])
        width = abs(k_short - k_long)
        if width <= 0:
            skipped["zero_width"] = skipped.get("zero_width", 0) + 1
            continue

        s_bid, s_ask, s_delta = float(short_leg[2]), float(short_leg[3]), float(short_leg[5])
        l_bid, l_ask = float(long_leg[2]), float(long_leg[3])
        l_delta = float(long_leg[5]) if long_leg[5] is not None else 0.0
        c_cross = s_bid - l_ask
        if c_cross <= 0 or c_cross >= width:
            skipped["non_pos_credit"] = skipped.get("non_pos_credit", 0) + 1
            continue
        rt_friction = (s_ask - s_bid) + (l_ask - l_bid)
        fric_ratio = rt_friction / c_cross
        if fric_ratio > 0.25:
            skipped["friction_gt_25pct"] = skipped.get("friction_gt_25pct", 0) + 1
            continue

        # Must have expiry within dataset range so trade closes cleanly
        if chosen_exp > all_dates[-1]:
            skipped["exp_after_sample"] = skipped.get("exp_after_sample", 0) + 1
            continue

        candidates.append(
            {
                "entry_date": edate,
                "expiration": chosen_exp,
                "k_short": k_short,
                "k_long": k_long,
                "width": width,
                "dte_entry": dte_entry,
                "spot_entry": spot,
                "s_bid": s_bid,
                "s_ask": s_ask,
                "l_bid": l_bid,
                "l_ask": l_ask,
                "delta_net_entry": (-s_delta) + l_delta,
                "fric_ratio": fric_ratio,
            }
        )

    # Step 2: Walk daily path per candidate using fast PK lookups
    results: Dict[str, List[SpreadTrade]] = {
        f"{arm}_{pol}": []
        for arm in ("O1-H", "O1-M")
        for pol in ("crossed", "limit", "mid")
    }

    for cand in candidates:
        edate = str(cand["entry_date"])
        exp = str(cand["expiration"])
        k_short = float(cand["k_short"])
        k_long = float(cand["k_long"])
        width = float(cand["width"])
        spot_entry = float(cand["spot_entry"])
        d_net_0 = float(cand["delta_net_entry"])

        i_start = date_to_idx[edate]
        # Find last trading date <= exp
        i_end = i_start
        while i_end + 1 < len(all_dates) and all_dates[i_end + 1] <= exp:
            i_end += 1
        if i_end <= i_start:
            continue

        # Collect daily marks along [i_start + 1 .. i_end]
        daily_marks: List[Tuple[str, float, float, Optional[Tuple[float, float, float, float, float, float]]]] = []
        # Pre-fetch via single range query on PK prefix? Better: PK lookups for this (exp, k_short/k_long) per date
        # Actually ~65 trading days -> 130 PK lookups per trade, super fast (~1ms/trade -> <0.8s total!)
        for j in range(i_start + 1, i_end + 1):
            d = all_dates[j]
            s_spot = spot_by_date[d]
            r_s = cur.execute(
                "SELECT bid, ask, delta, dte FROM odx_chain "
                "WHERE symbol='SPY' AND date=? AND expiration=? AND strike=? AND type=?",
                (d, exp, k_short, opt_type),
            ).fetchone()
            r_l = cur.execute(
                "SELECT bid, ask, delta, dte FROM odx_chain "
                "WHERE symbol='SPY' AND date=? AND expiration=? AND strike=? AND type=?",
                (d, exp, k_long, opt_type),
            ).fetchone()
            if r_s and r_l and r_s[0] is not None and r_s[1] is not None and r_l[0] is not None and r_l[1] is not None:
                sd = float(r_s[2]) if r_s[2] is not None else np.nan
                ld = float(r_l[2]) if r_l[2] is not None else np.nan
                dte_j = float(r_s[3]) if r_s[3] is not None else float((dt.date.fromisoformat(exp) - dt.date.fromisoformat(d)).days)
                daily_marks.append(
                    (d, s_spot, dte_j, (float(r_s[0]), float(r_s[1]), float(r_l[0]), float(r_l[1]), sd, ld))
                )
            else:
                dte_j = float((dt.date.fromisoformat(exp) - dt.date.fromisoformat(d)).days)
                daily_marks.append((d, s_spot, dte_j, None))

        # Precompute cumulative daily re-hedged delta P&L per share*100 for each step
        cum_hdyn: List[float] = []
        cur_delta = d_net_0
        prev_spot = spot_entry
        running = 0.0
        for _d, s_spot, _dte, q in daily_marks:
            running += cur_delta * 100.0 * (s_spot - prev_spot)
            cum_hdyn.append(running)
            prev_spot = s_spot
            if q is not None and np.isfinite(q[4]) and np.isfinite(q[5]):
                cur_delta = (-q[4]) + q[5]

        # Evaluate each fill policy for both O1-H and O1-M
        for pol in ("crossed", "limit", "mid"):
            c_in = _fill_entry_credit(
                float(cand["s_bid"]),
                float(cand["s_ask"]),
                float(cand["l_bid"]),
                float(cand["l_ask"]),
                pol,
            )
            car = 100.0 * (width - c_in)
            if c_in <= 0 or car <= 0:
                continue

            # O1-H: hold to expiry (intrinsic settlement, zero exit crossing)
            exp_date_actual, spot_exp, _, _ = daily_marks[-1]
            settle = _intrinsic_spread(spot_exp, k_short, k_long, opt_type)
            pnl_h = 100.0 * (c_in - settle)
            h_entry = d_net_0 * 100.0 * (spot_exp - spot_entry)
            h_dyn = cum_hdyn[-1]
            results[f"O1-H_{pol}"].append(
                SpreadTrade(
                    side=side,
                    arm="O1-H",
                    fill_policy=pol,
                    entry_date=edate,
                    exit_date=exp_date_actual,
                    expiration=exp,
                    k_short=k_short,
                    k_long=k_long,
                    dte_entry=float(cand["dte_entry"]),
                    spot_entry=spot_entry,
                    spot_exit=spot_exp,
                    credit_entry=c_in,
                    close_cost=settle,
                    car=car,
                    pnl=pnl_h,
                    roc=pnl_h / car,
                    delta_net_entry=d_net_0,
                    hedge_entry=h_entry,
                    hedge_dyn=h_dyn,
                    excess_entry=(pnl_h - h_entry) / car,
                    excess_dyn=(pnl_h - h_dyn) / car,
                    friction_to_credit=float(cand["fric_ratio"]),
                    exit_reason="expiry",
                )
            )

            # O1-M: TP 50% of credit OR DTE <= 21, no stop loss
            exited = False
            for idx_m, (d_m, s_m, dte_m, q_m) in enumerate(daily_marks):
                if q_m is not None:
                    sb, sa, lb, la, _, _ = q_m
                    if sa >= sb and la >= lb and sa > 0:
                        close_m = _fill_close_debit(sb, sa, lb, la, pol)
                        if (c_in - close_m) >= 0.50 * c_in:
                            pnl_m = 100.0 * (c_in - close_m)
                            h_e = d_net_0 * 100.0 * (s_m - spot_entry)
                            h_d = cum_hdyn[idx_m]
                            results[f"O1-M_{pol}"].append(
                                SpreadTrade(
                                    side=side,
                                    arm="O1-M",
                                    fill_policy=pol,
                                    entry_date=edate,
                                    exit_date=d_m,
                                    expiration=exp,
                                    k_short=k_short,
                                    k_long=k_long,
                                    dte_entry=float(cand["dte_entry"]),
                                    spot_entry=spot_entry,
                                    spot_exit=s_m,
                                    credit_entry=c_in,
                                    close_cost=close_m,
                                    car=car,
                                    pnl=pnl_m,
                                    roc=pnl_m / car,
                                    delta_net_entry=d_net_0,
                                    hedge_entry=h_e,
                                    hedge_dyn=h_d,
                                    excess_entry=(pnl_m - h_e) / car,
                                    excess_dyn=(pnl_m - h_d) / car,
                                    friction_to_credit=float(cand["fric_ratio"]),
                                    exit_reason="tp50",
                                )
                            )
                            exited = True
                            break
                        if dte_m <= 21.0:
                            pnl_m = 100.0 * (c_in - close_m)
                            h_e = d_net_0 * 100.0 * (s_m - spot_entry)
                            h_d = cum_hdyn[idx_m]
                            results[f"O1-M_{pol}"].append(
                                SpreadTrade(
                                    side=side,
                                    arm="O1-M",
                                    fill_policy=pol,
                                    entry_date=edate,
                                    exit_date=d_m,
                                    expiration=exp,
                                    k_short=k_short,
                                    k_long=k_long,
                                    dte_entry=float(cand["dte_entry"]),
                                    spot_entry=spot_entry,
                                    spot_exit=s_m,
                                    credit_entry=c_in,
                                    close_cost=close_m,
                                    car=car,
                                    pnl=pnl_m,
                                    roc=pnl_m / car,
                                    delta_net_entry=d_net_0,
                                    hedge_entry=h_e,
                                    hedge_dyn=h_d,
                                    excess_entry=(pnl_m - h_e) / car,
                                    excess_dyn=(pnl_m - h_d) / car,
                                    friction_to_credit=float(cand["fric_ratio"]),
                                    exit_reason="dte21",
                                )
                            )
                            exited = True
                            break
            if not exited:
                # Fallback to intrinsic expiry if quotes missing around 21 DTE
                pnl_m = 100.0 * (c_in - settle)
                results[f"O1-M_{pol}"].append(
                    SpreadTrade(
                        side=side,
                        arm="O1-M",
                        fill_policy=pol,
                        entry_date=edate,
                        exit_date=exp_date_actual,
                        expiration=exp,
                        k_short=k_short,
                        k_long=k_long,
                        dte_entry=float(cand["dte_entry"]),
                        spot_entry=spot_entry,
                        spot_exit=spot_exp,
                        credit_entry=c_in,
                        close_cost=settle,
                        car=car,
                        pnl=pnl_m,
                        roc=pnl_m / car,
                        delta_net_entry=d_net_0,
                        hedge_entry=h_entry,
                        hedge_dyn=h_dyn,
                        excess_entry=(pnl_m - h_entry) / car,
                        excess_dyn=(pnl_m - h_dyn) / car,
                        friction_to_credit=float(cand["fric_ratio"]),
                        exit_reason="expiry_fallback",
                    )
                )

    print(f"[{side}] weekly_dates={len(weekly_dates)} selected={len(candidates)} skipped={skipped}")
    return results


def main() -> int:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT date, underlying FROM odx_chain WHERE symbol='SPY' GROUP BY date ORDER BY date"
        ).fetchall()
        all_dates = [r[0] for r in rows if r[1] is not None]
        spot_by_date = {r[0]: float(r[1]) for r in rows if r[1] is not None}
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        weekly_dates = _select_weekly_entries(all_dates)
        print(f"Loaded {len(all_dates)} SPY trading dates ({all_dates[0]}..{all_dates[-1]}), {len(weekly_dates)} weekly entries.")

        bp_trades = run_side(con, all_dates, date_to_idx, spot_by_date, weekly_dates, "bull_put")
        bc_trades = run_side(con, all_dates, date_to_idx, spot_by_date, weekly_dates, "bear_call")
    finally:
        con.close()

    report: Dict[str, object] = {
        "prereg": "docs/PREREG_OPTIONS_DTE_AND_PRE_EARNINGS_20261124.md",
        "window": [all_dates[0], all_dates[-1]],
        "bull_put": {},
        "bear_call_sensitivity": {},
    }

    print("\n=== TEST O1 PRIMARY: SPY 60-120 DTE 5%-WIDE BULL PUT ===")
    for key in ("O1-H_crossed", "O1-M_crossed", "O1-H_limit", "O1-M_limit", "O1-H_mid", "O1-M_mid"):
        s = _moving_block_month_boot(bp_trades[key])
        report["bull_put"][key] = s  # type: ignore[index]
        print(
            f"  {key:14s}: n={s['n']:3d} effN={s['effective_n']:5.1f} fric={s['median_friction_to_credit']*100:4.1f}% "
            f"RoC={s['mean_roc']*100:+6.2f}% PF={s['pf']:.3f} {s['pf_ci']} "
            f"x_entry={s['mean_excess_entry']*100:+6.2f}% {s['excess_entry_ci']} "
            f"x_dyn={s['mean_excess_dyn']*100:+6.2f}% -> {s['verdict']}"
        )

    print("\n=== TEST O1 SENSITIVITY: SPY 60-120 DTE 5%-WIDE BEAR CALL ===")
    for key in ("O1-H_crossed", "O1-M_crossed", "O1-H_limit", "O1-M_limit", "O1-H_mid", "O1-M_mid"):
        s = _moving_block_month_boot(bc_trades[key])
        report["bear_call_sensitivity"][key] = s  # type: ignore[index]
        print(
            f"  {key:14s}: n={s['n']:3d} effN={s['effective_n']:5.1f} fric={s['median_friction_to_credit']*100:4.1f}% "
            f"RoC={s['mean_roc']*100:+6.2f}% PF={s['pf']:.3f} {s['pf_ci']} "
            f"x_entry={s['mean_excess_entry']*100:+6.2f}% {s['excess_entry_ci']} "
            f"x_dyn={s['mean_excess_dyn']*100:+6.2f}% -> {s['verdict']}"
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
