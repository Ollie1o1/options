#!/usr/bin/env python3
"""Preregistered estimator for Study 2E: Rebalance Execution Clock & Overnight/Intraday Decomposition.

Frozen registration: docs/PREREG_REBALANCE_TIMING_AND_OVERNIGHT_20261127.md
Uses local Alpaca 5-minute bars (`data/intraday_5m/`, 16 VM_OUTLOOK_TOP3 ETFs, 2016-2026).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.backtest_vol_managed_outlook import (
    MAX_SCALE,
    MIN_SCALE,
    STEP,
    TARGET_VOL,
    TURNOVER_COST,
    _align_matrix,
    _metrics_from_monthly,
    _trailing_basket_rv,
)
from scripts.sleeve_vol_forecast_test import grid
from src.equity_playbooks.backtest import load_equity_ohlcv
from src.intraday.alpaca_bars import fetch_calendar, load_symbol
from src.intraday.sessions import full_sessions, with_clock
from src.intraday.stats import block_bootstrap_means
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE

REPO = Path(__file__).resolve().parents[1]
DATA_5M = REPO / "data" / "intraday_5m"
SEED = 20261127
DRAWS = 10_000
# 5-minute bars are timestamped by bar START:
#   09:30 bar closes at 09:35; 09:55 closes at 10:00; 11:55 closes at 12:00;
#   13:55 closes at 14:00; 15:55 bar closes at 16:00 (end-of-session close).
CLOCKS_MIN = {
    "09:35": 9 * 60 + 30,
    "10:00": 9 * 60 + 55,
    "12:00": 11 * 60 + 55,
    "14:00": 13 * 60 + 55,
    "15:55": 15 * 60 + 55,
}
OPEN_MIN = 9 * 60 + 30
CLOSE_MIN = 15 * 60 + 55


def _extract_clock_prices(df: pd.DataFrame, valid_sessions: set[dt.date]) -> pd.DataFrame:
    """For each full session, extract open (09:30), 09:35, 10:00, 12:00, 14:00, 15:55(close) prices."""
    if df.empty:
        return pd.DataFrame()
    w = with_clock(df)
    w = w[w["date"].isin(valid_sessions)].copy()
    if w.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, Any]] = []
    for day, grp in w.groupby("date", sort=True):
        rth = grp[(grp["mod"] >= OPEN_MIN) & (grp["mod"] <= CLOSE_MIN)]
        if len(rth) < 10:
            continue
        rec: Dict[str, Any] = {"day": day, "open_0930": float(rth.iloc[0]["o"])}
        mins = rth["mod"].to_numpy()
        closes = rth["c"].to_numpy()
        for label, tgt in CLOCKS_MIN.items():
            idx = np.searchsorted(mins, tgt, side="right") - 1
            if idx >= 0:
                rec[label] = float(closes[idx])
            else:
                rec[label] = float(rth.iloc[0]["c"])
        out_rows.append(rec)
    return pd.DataFrame(out_rows).set_index("day") if out_rows else pd.DataFrame()


def run() -> Dict[str, Any]:
    ohlcv = load_equity_ohlcv(str(REPO / "data" / "equity_ohlcv.db"), [BENCH] + list(DEFAULT_UNIVERSE))
    dates, cols = _align_matrix(ohlcv, DEFAULT_UNIVERSE, strict_common=False, min_date="2005-01-01")
    months = grid(dates, cols)

    cal = fetch_calendar()
    raw_bars = {tk: load_symbol(tk, DATA_5M) for tk in DEFAULT_UNIVERSE}
    fs_series = full_sessions(with_clock(raw_bars["QQQ"]), cal)
    ref_sess = set(fs_series[fs_series].index)
    px_map: Dict[str, pd.DataFrame] = {
        tk: _extract_clock_prices(df, ref_sess) for tk, df in raw_bars.items() if not df.empty
    }

    # Filter months to 2016+ where intraday bars exist for exec_in & exec_out
    win_months = [m for m in months if m["exec_date"] >= "2016-02-01"]

    # ── E1: Rebalance Execution Clock Comparison ─────────────────────────────
    clock_rets: Dict[str, List[float]] = {k: [] for k in CLOCKS_MIN}
    clock_dates: List[str] = []
    prev_w: Dict[str, float] = {}
    month_weights: List[Tuple[str, str, Dict[str, float]]] = []

    for m in win_months:
        w_a = m["w_a"]
        rv = _trailing_basket_rv(cols, w_a, m["t"], window=20) if w_a else None
        scale = float(np.clip(TARGET_VOL / rv, MIN_SCALE, MAX_SCALE)) if (rv and rv > 0 and w_a) else 1.0
        w = {tk: v * scale for tk, v in w_a.items()}

        d_in = dt.date.fromisoformat(m["exec_date"])
        d_out = dt.date.fromisoformat(dates[m["exec_out"]])
        # Check all held tickers have intraday clock prices on both exec_in and exec_out
        if w and not all(tk in px_map and d_in in px_map[tk].index and d_out in px_map[tk].index for tk in w):
            prev_w = w
            continue

        to = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in set(w) | set(prev_w))
        cost = to * TURNOVER_COST
        for clk in CLOCKS_MIN:
            gross = sum(
                wt * (float(px_map[tk].loc[d_out, clk]) / float(px_map[tk].loc[d_in, clk]) - 1.0)
                for tk, wt in w.items()
            )
            clock_rets[clk].append((1.0 - cost) * (1.0 + gross) - 1.0)
        clock_dates.append(m["exec_date"])
        month_weights.append((m["exec_date"], dates[m["exec_out"]], w))
        prev_w = w

    years_m = np.array([int(d[:4]) for d in clock_dates])
    base_1555 = np.asarray(clock_rets["15:55"], dtype=float)
    e1_results: Dict[str, Any] = {
        "n_months": int(len(clock_dates)),
        "window": [clock_dates[0], clock_dates[-1]] if clock_dates else [],
        "baseline_1555": _metrics_from_monthly(base_1555),
        "clocks": {},
    }
    alpha_e1 = 0.05 / 4.0  # Bonferroni across 4 non-close clocks -> 98.75% CI
    for clk in ("09:35", "10:00", "12:00", "14:00"):
        arr = np.asarray(clock_rets[clk], dtype=float)
        diff_bps = (arr - base_1555) * 10_000.0
        means = block_bootstrap_means(diff_bps, block=3, draws=DRAWS, seed=SEED, circular=True)
        lo, hi = np.percentile(means, [100 * alpha_e1 / 2.0, 100 * (1.0 - alpha_e1 / 2.0)])
        h1 = float(diff_bps[years_m <= 2020].mean())
        h2 = float(diff_bps[years_m >= 2021].mean())
        verdict = "PASS" if (lo > 0.0 and h1 > 0.0 and h2 > 0.0) else "FAIL"
        e1_results["clocks"][clk] = {
            "metrics": _metrics_from_monthly(arr),
            "mean_diff_bps_per_rebal": round(float(diff_bps.mean()), 2),
            "ci9875_bps": [round(float(lo), 2), round(float(hi), 2)],
            "halves_bps": {"2016-2020": round(h1, 2), "2021-2026": round(h2, 2)},
            "verdict": verdict,
        }

    # ── E2a: Overnight vs Intraday Attribution of VM_OUTLOOK_TOP3 ────────────
    ovn_daily: List[float] = []
    int_daily: List[float] = []
    for d_in_s, d_out_s, w in month_weights:
        d_in = dt.date.fromisoformat(d_in_s)
        d_out = dt.date.fromisoformat(d_out_s)
        if not w:
            continue
        # Iterate common full sessions strictly inside (d_in, d_out]
        common_days = sorted(
            set.intersection(*(set(px_map[tk].index) for tk in w))
        )
        for i in range(1, len(common_days)):
            d_cur = common_days[i]
            d_prv = common_days[i - 1]
            if not (d_in < d_cur <= d_out):
                continue
            r_ovn = sum(
                wt * (float(px_map[tk].loc[d_cur, "open_0930"]) / float(px_map[tk].loc[d_prv, "15:55"]) - 1.0)
                for tk, wt in w.items()
            )
            r_int = sum(
                wt * (float(px_map[tk].loc[d_cur, "15:55"]) / float(px_map[tk].loc[d_cur, "open_0930"]) - 1.0)
                for tk, wt in w.items()
            )
            ovn_daily.append(r_ovn)
            int_daily.append(r_int)

    ovn_arr = np.asarray(ovn_daily, dtype=float)
    int_arr = np.asarray(int_daily, dtype=float)
    e2a = {
        "n_sessions": int(len(ovn_arr)),
        "overnight_ann_mean_pct": round(float(ovn_arr.mean() * 252 * 100), 2) if ovn_arr.size else None,
        "overnight_ann_vol_pct": round(float(ovn_arr.std(ddof=1) * math.sqrt(252) * 100), 2) if ovn_arr.size > 1 else None,
        "overnight_sharpe": round(float(ovn_arr.mean() / ovn_arr.std(ddof=1) * math.sqrt(252)), 3) if ovn_arr.size > 1 else None,
        "intraday_ann_mean_pct": round(float(int_arr.mean() * 252 * 100), 2) if int_arr.size else None,
        "intraday_ann_vol_pct": round(float(int_arr.std(ddof=1) * math.sqrt(252) * 100), 2) if int_arr.size > 1 else None,
        "intraday_sharpe": round(float(int_arr.mean() / int_arr.std(ddof=1) * math.sqrt(252)), 3) if int_arr.size > 1 else None,
    }

    # ── E2b: Intraday Selloff vs Overnight Gap-Down 3-Day Reversion (Uptrends) ─
    by_day_int_sell: Dict[dt.date, List[float]] = {}
    by_day_gap_dn: Dict[dt.date, List[float]] = {}
    cost_rt = 0.0006  # 6 bps round-trip shares

    for tk, px in px_map.items():
        if len(px) < 210:
            continue
        days_list = list(px.index)
        c1555 = px["15:55"].to_numpy(dtype=float)
        o0930 = px["open_0930"].to_numpy(dtype=float)
        sma200 = pd.Series(c1555).rolling(200).mean().to_numpy()
        for i in range(200, len(days_list) - 3):
            if not np.isfinite(sma200[i]) or c1555[i] <= sma200[i]:
                continue
            r_ovn = o0930[i] / c1555[i - 1] - 1.0
            r_int = c1555[i] / o0930[i] - 1.0
            fwd_3d = c1555[i + 3] / c1555[i] - 1.0 - cost_rt
            if r_int <= -0.015 and r_ovn > -0.005:
                by_day_int_sell.setdefault(days_list[i], []).append(fwd_3d)
            elif r_ovn <= -0.015:
                by_day_gap_dn.setdefault(days_list[i], []).append(fwd_3d)

    def _day_stats(bucket: Dict[dt.date, List[float]]) -> Dict[str, Any]:
        ds = sorted(bucket.keys())
        vals = np.array([float(np.mean(bucket[d])) for d in ds], dtype=float)
        yrs = np.array([d.year for d in ds])
        means = block_bootstrap_means(vals * 10_000.0, block=10, draws=DRAWS, seed=SEED, circular=True)
        lo, hi = np.percentile(means, [1.25, 98.75])  # 97.5% Bonferroni across 2 arms
        h1 = float((vals[yrs <= 2020] * 10_000.0).mean()) if np.any(yrs <= 2020) else float("nan")
        h2 = float((vals[yrs >= 2021] * 10_000.0).mean()) if np.any(yrs >= 2021) else float("nan")
        verdict = "PASS" if (lo > 0.0 and h1 > 0.0 and h2 > 0.0) else "FAIL"
        return {
            "n_days": int(len(ds)),
            "n_events": int(sum(len(v) for v in bucket.values())),
            "mean_3d_bps": round(float(vals.mean() * 10_000.0), 1),
            "ci975_bps": [round(float(lo), 1), round(float(hi), 1)],
            "halves_bps": {"2016-2020": round(h1, 1), "2021-2026": round(h2, 1)},
            "verdict": verdict,
        }

    def _diff_stats(b1: Dict[dt.date, List[float]], b2: Dict[dt.date, List[float]]) -> Dict[str, Any]:
        v1 = np.array([float(np.mean(b1[d])) for d in sorted(b1)], dtype=float) * 10_000.0
        v2 = np.array([float(np.mean(b2[d])) for d in sorted(b2)], dtype=float) * 10_000.0
        m1 = block_bootstrap_means(v1, block=10, draws=DRAWS, seed=SEED, circular=True)
        m2 = block_bootstrap_means(v2, block=10, draws=DRAWS, seed=SEED + 1, circular=True)
        diff = m1 - m2
        lo, hi = np.percentile(diff, [1.25, 98.75])
        return {
            "mean_diff_bps": round(float(v1.mean() - v2.mean()), 1),
            "ci975_bps": [round(float(lo), 1), round(float(hi), 1)],
        }

    e2b = {
        "intraday_selloff": _day_stats(by_day_int_sell),
        "overnight_gap_down": _day_stats(by_day_gap_dn),
        "diff_intraday_minus_gap": _diff_stats(by_day_int_sell, by_day_gap_dn),
    }

    return {
        "registered": "docs/PREREG_REBALANCE_TIMING_AND_OVERNIGHT_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "E1_rebalance_clock": e1_results,
        "E2a_overnight_vs_intraday_sleeve": e2a,
        "E2b_uptrend_pullback_decomposition": e2b,
    }


def main() -> int:
    r = run()
    print("=== E1: VM_OUTLOOK_TOP3 Rebalance Clock ===")
    e1 = r["E1_rebalance_clock"]
    print(f"Window {e1['window']} (n={e1['n_months']}), 15:55 close Sharpe={e1['baseline_1555']['sharpe']:.3f} CAGR={e1['baseline_1555']['cagr']:+.2%}")
    for clk, st in e1["clocks"].items():
        print(
            f"  {clk} vs 15:55: {st['mean_diff_bps_per_rebal']:+.2f} bps/mo CI98.75 {st['ci9875_bps']} "
            f"halves={st['halves_bps']} Sharpe={st['metrics']['sharpe']:.3f} | {st['verdict']}"
        )
    print("\n=== E2a: Sleeve Overnight vs Intraday Attribution (descriptive) ===")
    print(" ", r["E2a_overnight_vs_intraday_sleeve"])
    print("\n=== E2b: Uptrend 3d Reversion (Intraday Selloff vs Overnight Gap-Down, net 6bps) ===")
    for k in ("intraday_selloff", "overnight_gap_down"):
        st = r["E2b_uptrend_pullback_decomposition"][k]
        print(
            f"  {k}: days={st['n_days']} events={st['n_events']} mean={st['mean_3d_bps']:+.1f} bps "
            f"CI97.5 {st['ci975_bps']} halves={st['halves_bps']} | {st['verdict']}"
        )
    df_st = r["E2b_uptrend_pullback_decomposition"]["diff_intraday_minus_gap"]
    print(f"  intraday - gap_down: {df_st['mean_diff_bps']:+.1f} bps CI97.5 {df_st['ci975_bps']}")

    out = REPO / "reports" / "intraday" / f"rebalance_timing_overnight_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
