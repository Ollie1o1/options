#!/usr/bin/env python3
"""Preregistered estimator for Study 2B: Upgrading the VALIDATED VM_OUTLOOK_TOP3 sleeve.

Frozen registration: docs/PREREG_VM_OUTLOOK_V2_20261127.md
Zero parameter tuning. Reproduces baseline B0 to < 1e-12 parity before evaluating Arms A, B, C.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.backtest_vol_managed_outlook import (
    MAX_SCALE,
    MIN_SCALE,
    STEP,
    TARGET_VOL,
    TURNOVER_COST,
    _align_matrix,
    _metrics_from_monthly,
    _paired_block_bootstrap,
    _simulate_arms,
    _trailing_basket_rv,
)
from src.equity_playbooks.backtest import load_equity_ohlcv
from src.intraday import realized as rz
from src.intraday.alpaca_bars import fetch_calendar, load_symbol
from src.intraday.sessions import full_sessions, with_clock
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE, _features
from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG, rank_universe

REPO = Path(__file__).resolve().parents[1]
DATA_5M = REPO / "data" / "intraday_5m"
EXTRA_DIVERSIFIERS = ("DBC", "UUP", "IEF", "EFA", "EEM", "HYG", "SLV")
EXPANDED_UNIVERSE = tuple(DEFAULT_UNIVERSE) + EXTRA_DIVERSIFIERS
SEED = 20261127
TRIGGER_MULT = 1.50
RV5_WINDOW = 5


def _rebal_indices(dates: List[str], cols: Dict[str, np.ndarray], universe: Sequence[str]) -> List[int]:
    n = len(dates)
    out: List[int] = []
    for t in range(252, n - STEP - 1, STEP):
        avail = [tk for tk in universe if tk in cols and np.all(np.isfinite(cols[tk][t - 252 : t + 1]))]
        if len(avail) >= 6:
            out.append(t)
    return out


def _single_rv20(cols: Dict[str, np.ndarray], tk: str, t: int, window: int = 20) -> Optional[float]:
    if t < window or tk not in cols:
        return None
    seg = cols[tk][t - window : t + 1]
    if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
        return None
    sd = float(np.std(np.diff(seg) / seg[:-1], ddof=1))
    return sd * math.sqrt(252.0) if sd > 0 else None


def _inv_vol_weights(cols: Dict[str, np.ndarray], pos_top: List[str], t: int, top_k: int = 3) -> Dict[str, float]:
    """Weight positive top-K ETFs inversely to trailing 20d vol while preserving active-share `len/top_k`."""
    if not pos_top:
        return {}
    active_share = len(pos_top) / float(top_k)
    if len(pos_top) == 1:
        return {pos_top[0]: active_share}
    inv: Dict[str, float] = {}
    for tk in pos_top:
        rv = _single_rv20(cols, tk, t)
        if rv is None or rv <= 0:
            return {k: 1.0 / float(top_k) for k in pos_top}
        inv[tk] = 1.0 / rv
    s = sum(inv.values())
    raw = {k: v / s for k, v in inv.items()}
    # Clip per-slot share of the active basket to [0.15, 0.55] when 3 names, then renormalize
    if len(pos_top) == 3:
        clipped = {k: float(np.clip(v, 0.15, 0.55)) for k, v in raw.items()}
        cs = sum(clipped.values())
        raw = {k: v / cs for k, v in clipped.items()}
    return {k: v * active_share for k, v in raw.items()}


def simulate_sleeve(
    dates: List[str],
    cols: Dict[str, np.ndarray],
    universe: Sequence[str],
    *,
    inv_vol: bool = False,
    turnover_cost: float = TURNOVER_COST,
    intra_rets: Optional[Dict[str, Any]] = None,
    trigger_mult: Optional[float] = None,
    cash_daily_rate: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Simulate VM_OUTLOOK_TOP3 or a preregistered variant on the exact rebalance grid."""
    bench = cols[BENCH]
    n = len(dates)
    rebal = _rebal_indices(dates, cols, universe)

    prev_w: Dict[str, float] = {}
    period_rets: List[float] = []
    period_dates: List[str] = []
    daily_rets: List[float] = []
    exposures: List[float] = []
    turnovers: List[float] = []
    triggers_fired = 0

    for k, t in enumerate(rebal):
        exec_in = t + 1
        exec_out = (rebal[k + 1] + 1) if (k + 1 < len(rebal)) else (n - 1)
        if exec_out <= exec_in:
            continue
        avail = [
            tk
            for tk in universe
            if tk in cols
            and np.all(np.isfinite(cols[tk][t - 252 : t + 1]))
            and np.all(np.isfinite(cols[tk][exec_in : exec_out + 1]))
        ]
        if len(avail) < 3:
            continue

        bench_list = bench[: t + 1].tolist()
        feats = {tk: _features(cols[tk][: t + 1].tolist(), bench_list, t) for tk in avail}
        ranked = rank_universe(feats, DEFAULT_OUTLOOK_CONFIG)
        pos_top = [str(r["ticker"]) for r in ranked[:3] if float(r["score"]) > 0.0]

        w_base = _inv_vol_weights(cols, pos_top, t) if inv_vol else {tk: 1.0 / 3.0 for tk in pos_top}
        rv_entry = _trailing_basket_rv(cols, w_base, t, window=20)
        scale_entry = float(np.clip(TARGET_VOL / rv_entry, MIN_SCALE, MAX_SCALE)) if (rv_entry and rv_entry > 0 and w_base) else 1.0
        w_cur = {tk: v * scale_entry for tk, v in w_base.items()}

        to = sum(abs(w_cur.get(tk, 0.0) - prev_w.get(tk, 0.0)) for tk in set(w_cur) | set(prev_w))
        cost_entry = to * turnover_cost
        eq = 1.0 - cost_entry
        month_to = to

        # Optional Arm C: precompute held basket daily intraday RV series over [exec_in..exec_out]
        brv_series = None
        if trigger_mult is not None and intra_rets is not None and w_base and rv_entry and rv_entry > 0:
            if all(tk in intra_rets for tk in w_base):
                brv_series = rz.basket_daily_rv(intra_rets, w_base)

        fired_this_month = False
        pending_cut_w: Optional[Dict[str, float]] = None

        for d_i in range(exec_in + 1, exec_out + 1):
            day_r = 0.0
            for tk, wt in w_cur.items():
                day_r += wt * (cols[tk][d_i] / cols[tk][d_i - 1] - 1.0)
            if cash_daily_rate:
                cash_w = max(0.0, 1.0 - sum(w_cur.values()))
                day_r += cash_w * cash_daily_rate.get(dates[d_i], 0.0)

            step_cost = cost_entry if d_i == exec_in + 1 else 0.0

            # Apply any trigger decided at close of d_i - 1 at close of d_i (zero look-ahead)
            if pending_cut_w is not None:
                cut_to = sum(abs(pending_cut_w.get(tk, 0.0) - w_cur.get(tk, 0.0)) for tk in set(pending_cut_w) | set(w_cur))
                cut_cost = cut_to * turnover_cost
                month_to += cut_to
                eq *= (1.0 + day_r) * (1.0 - cut_cost)
                daily_rets.append((1.0 - step_cost) * (1.0 + day_r) * (1.0 - cut_cost) - 1.0)
                w_cur = pending_cut_w
                pending_cut_w = None
                continue

            eq *= 1.0 + day_r
            daily_rets.append((1.0 - step_cost) * (1.0 + day_r) - 1.0)

            # Check trigger after close of day d_i for next-day close fill (d_i + 1)
            if (
                brv_series is not None
                and trigger_mult is not None
                and not fired_this_month
                and d_i <= exec_out - 1
            ):
                d_date = dt.date.fromisoformat(dates[d_i])
                sub = brv_series.loc[:d_date].tail(RV5_WINDOW)
                if len(sub) == RV5_WINDOW:
                    rv5_ann = math.sqrt(float(rz.ANN * sub.mean()))
                    if rv5_ann >= trigger_mult * float(rv_entry):
                        new_scale = float(np.clip(TARGET_VOL / rv5_ann, MIN_SCALE, scale_entry))
                        if new_scale < scale_entry - 1e-6:
                            pending_cut_w = {tk: v * new_scale for tk, v in w_base.items()}
                            fired_this_month = True
                            triggers_fired += 1

        period_dates.append(dates[exec_in])
        period_rets.append(eq - 1.0)
        exposures.append(sum(w_cur.values()))
        turnovers.append(month_to)
        prev_w = w_cur

    return {
        "period_dates": period_dates,
        "period_rets": np.asarray(period_rets, dtype=float),
        "daily_rets": np.asarray(daily_rets, dtype=float),
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
        "ann_turnover": float(np.mean(turnovers) * 12.0) if turnovers else 0.0,
        "triggers_fired": triggers_fired,
    }


def _evaluate_arm(
    arm_rets: np.ndarray,
    base_rets: np.ndarray,
    dates: List[str],
    *,
    split_year: int,
    arm_stress: np.ndarray,
    base_stress: np.ndarray,
) -> Dict[str, Any]:
    boot = _paired_block_bootstrap(arm_rets, base_rets, block_len=3, n_boot=10_000, seed=SEED)
    yrs = np.array([int(d[:4]) for d in dates])
    h1 = yrs < split_year
    h2 = yrs >= split_year
    sh_h1 = _metrics_from_monthly(arm_rets[h1])["sharpe"] - _metrics_from_monthly(base_rets[h1])["sharpe"]
    sh_h2 = _metrics_from_monthly(arm_rets[h2])["sharpe"] - _metrics_from_monthly(base_rets[h2])["sharpe"]

    m_arm = _metrics_from_monthly(arm_rets)
    m_base = _metrics_from_monthly(base_rets)
    m_arm_s = _metrics_from_monthly(arm_stress)
    m_base_s = _metrics_from_monthly(base_stress)

    sh_pass = boot["delta_sharpe_ci9833"][0] > 0.0 and sh_h1 > 0.0 and sh_h2 > 0.0
    # Need 98.33% CI for delta_max_dd too (compute directly for exact Bonferroni bar)
    n = arm_rets.size
    rng = np.random.default_rng(SEED)
    n_blocks = int(math.ceil(n / 3))
    starts = rng.integers(0, n, size=(10_000, n_blocks))
    idx = ((starts[:, :, None] + np.arange(3)[None, None, :]) % n).reshape(10_000, n_blocks * 3)[:, :n]
    eq_a = np.cumprod(1.0 + arm_rets[idx], axis=1)
    pk_a = np.maximum.accumulate(np.concatenate([np.ones((10_000, 1)), eq_a], axis=1), axis=1)[:, 1:]
    dd_a = np.min(eq_a / pk_a - 1.0, axis=1)
    eq_b = np.cumprod(1.0 + base_rets[idx], axis=1)
    pk_b = np.maximum.accumulate(np.concatenate([np.ones((10_000, 1)), eq_b], axis=1), axis=1)[:, 1:]
    dd_b = np.min(eq_b / pk_b - 1.0, axis=1)
    d_dd = dd_a - dd_b
    lo98, hi98 = np.percentile(d_dd, [100 * (0.05 / 3.0) / 2.0, 100 * (1.0 - (0.05 / 3.0) / 2.0)])
    dd_pass = float(lo98) > 0.0 and m_arm["sharpe"] >= m_base["sharpe"]
    stress_ok = m_arm_s["sharpe"] >= m_base_s["sharpe"] and m_arm_s["max_dd"] >= m_base_s["max_dd"]
    verdict = "PASS" if ((sh_pass or dd_pass) and stress_ok) else "FAIL"

    return {
        "arm_metrics": m_arm,
        "base_metrics": m_base,
        "stress_10bps_arm": m_arm_s,
        "stress_10bps_base": m_base_s,
        "delta_sharpe": boot["delta_sharpe"],
        "delta_sharpe_ci9833": boot["delta_sharpe_ci9833"],
        "delta_sharpe_halves": {"first": round(float(sh_h1), 4), "second": round(float(sh_h2), 4)},
        "delta_max_dd": boot["delta_max_dd"],
        "delta_max_dd_ci9833": [round(float(lo98), 4), round(float(hi98), 4)],
        "stress_ok": bool(stress_ok),
        "verdict": verdict,
    }


def _year_returns(dates: List[str], rets: np.ndarray, year: int) -> Optional[float]:
    mask = np.array([d.startswith(f"{year}-") for d in dates])
    if not np.any(mask):
        return None
    return round(float(np.prod(1.0 + rets[mask]) - 1.0), 4)


def run() -> Dict[str, Any]:
    all_syms = sorted(set([BENCH] + list(EXPANDED_UNIVERSE)))
    ohlcv = load_equity_ohlcv(str(REPO / "data" / "equity_ohlcv.db"), all_syms)
    dates, cols = _align_matrix(ohlcv, EXPANDED_UNIVERSE, strict_common=False, min_date="2005-01-01")

    # 0. Strict parity check against validated scripts/backtest_vol_managed_outlook.py::_simulate_arms
    ref = _simulate_arms(dates, cols, DEFAULT_UNIVERSE)
    b0 = simulate_sleeve(dates, cols, DEFAULT_UNIVERSE, inv_vol=False, turnover_cost=TURNOVER_COST)
    parity_diff = float(np.max(np.abs(b0["period_rets"] - ref["period_rets"]["VM_OUTLOOK_TOP3"])))
    if parity_diff > 1e-12:
        raise RuntimeError(f"Parity check failed vs validated Arm B: max |diff|={parity_diff}")

    b0_s10 = simulate_sleeve(dates, cols, DEFAULT_UNIVERSE, inv_vol=False, turnover_cost=0.0010)

    # Arm A: Expanded 23-ETF crisis-diversifier universe
    arm_a = simulate_sleeve(dates, cols, EXPANDED_UNIVERSE, inv_vol=False, turnover_cost=TURNOVER_COST)
    arm_a_s10 = simulate_sleeve(dates, cols, EXPANDED_UNIVERSE, inv_vol=False, turnover_cost=0.0010)
    # Align Arm A to B0 common dates (identical since BASE_16 >= 6 starts at same date)
    common_ab = sorted(set(b0["period_dates"]) & set(arm_a["period_dates"]))
    b0_map = dict(zip(b0["period_dates"], b0["period_rets"]))
    b0_s_map = dict(zip(b0_s10["period_dates"], b0_s10["period_rets"]))
    a_map = dict(zip(arm_a["period_dates"], arm_a["period_rets"]))
    a_s_map = dict(zip(arm_a_s10["period_dates"], arm_a_s10["period_rets"]))
    ev_a = _evaluate_arm(
        np.array([a_map[d] for d in common_ab]),
        np.array([b0_map[d] for d in common_ab]),
        common_ab,
        split_year=2016,
        arm_stress=np.array([a_s_map[d] for d in common_ab]),
        base_stress=np.array([b0_s_map[d] for d in common_ab]),
    )
    ev_a["ann_turnover"] = round(arm_a["ann_turnover"], 3)
    ev_a["avg_exposure"] = round(arm_a["avg_exposure"], 4)
    ev_a["key_years"] = {
        str(y): {"arm": _year_returns(common_ab, np.array([a_map[d] for d in common_ab]), y),
                 "base": _year_returns(common_ab, np.array([b0_map[d] for d in common_ab]), y)}
        for y in (2008, 2018, 2020, 2022, 2025)
    }

    # Arm B: Intra-basket inverse-vol weighting on BASE_16
    arm_b = simulate_sleeve(dates, cols, DEFAULT_UNIVERSE, inv_vol=True, turnover_cost=TURNOVER_COST)
    arm_b_s10 = simulate_sleeve(dates, cols, DEFAULT_UNIVERSE, inv_vol=True, turnover_cost=0.0010)
    ev_b = _evaluate_arm(
        arm_b["period_rets"],
        b0["period_rets"],
        b0["period_dates"],
        split_year=2016,
        arm_stress=arm_b_s10["period_rets"],
        base_stress=b0_s10["period_rets"],
    )
    ev_b["ann_turnover"] = round(arm_b["ann_turnover"], 3)
    ev_b["avg_exposure"] = round(arm_b["avg_exposure"], 4)

    # Arm C: Intra-month Alpaca 5m drift-band vol trigger (2016-2026)
    cal = fetch_calendar()
    bars = {tk: load_symbol(tk, DATA_5M) for tk in DEFAULT_UNIVERSE}
    ref_b = with_clock(bars["QQQ"])
    sess = full_sessions(ref_b, cal)
    intra_rets = {tk: rz.interval_returns(b, sess) for tk, b in bars.items() if not b.empty}

    arm_c_full = simulate_sleeve(
        dates, cols, DEFAULT_UNIVERSE, inv_vol=False, turnover_cost=TURNOVER_COST,
        intra_rets=intra_rets, trigger_mult=TRIGGER_MULT,
    )
    arm_c_s10 = simulate_sleeve(
        dates, cols, DEFAULT_UNIVERSE, inv_vol=False, turnover_cost=0.0010,
        intra_rets=intra_rets, trigger_mult=TRIGGER_MULT,
    )
    mask_16 = np.array([d >= "2016-02-01" for d in b0["period_dates"]])
    d16 = [d for d, ok in zip(b0["period_dates"], mask_16) if ok]
    ev_c = _evaluate_arm(
        arm_c_full["period_rets"][mask_16],
        b0["period_rets"][mask_16],
        d16,
        split_year=2021,
        arm_stress=arm_c_s10["period_rets"][mask_16],
        base_stress=b0_s10["period_rets"][mask_16],
    )
    ev_c["triggers_fired"] = arm_c_full["triggers_fired"]
    ev_c["ann_turnover"] = round(arm_c_full["ann_turnover"], 3)

    # Diagnostic (no authority): Arm A+B combined
    arm_ab = simulate_sleeve(dates, cols, EXPANDED_UNIVERSE, inv_vol=True, turnover_cost=TURNOVER_COST)
    diag_ab = _metrics_from_monthly(arm_ab["period_rets"])

    return {
        "registered": "docs/PREREG_VM_OUTLOOK_V2_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "parity_max_abs_diff": parity_diff,
        "window_full": [b0["period_dates"][0], b0["period_dates"][-1]],
        "window_intraday": [d16[0], d16[-1]],
        "baseline_B0_full": _metrics_from_monthly(b0["period_rets"]) | {
            "avg_exposure": round(b0["avg_exposure"], 4),
            "ann_turnover": round(b0["ann_turnover"], 3),
        },
        "arms": {
            "A_EXPANDED_23": ev_a,
            "B_INV_VOL_WEIGHTS": ev_b,
            "C_INTRA_MONTH_TRIGGER": ev_c,
        },
        "diagnostic_A_plus_B": diag_ab,
    }


def main() -> int:
    r = run()
    print(f"Parity vs validated VM_OUTLOOK_TOP3: max |diff| = {r['parity_max_abs_diff']:.2e}")
    b = r["baseline_B0_full"]
    print(
        f"Baseline B0 ({r['window_full'][0]}..{r['window_full'][1]}, n={b['n_periods']}): "
        f"CAGR {b['cagr']:+.2%} Vol {b['ann_vol']:.2%} Sharpe {b['sharpe']:.3f} MaxDD {b['max_dd']:+.2%} Calmar {b['calmar']:.3f}"
    )
    for name, ev in r["arms"].items():
        am = ev["arm_metrics"]
        bm = ev["base_metrics"]
        print(
            f"\n[{name}] VERDICT={ev['verdict']}\n"
            f"  Arm : CAGR {am['cagr']:+.2%} Vol {am['ann_vol']:.2%} Sharpe {am['sharpe']:.3f} MaxDD {am['max_dd']:+.2%} Calmar {am['calmar']:.3f}\n"
            f"  Base: CAGR {bm['cagr']:+.2%} Vol {bm['ann_vol']:.2%} Sharpe {bm['sharpe']:.3f} MaxDD {bm['max_dd']:+.2%} Calmar {bm['calmar']:.3f}\n"
            f"  dSharpe {ev['delta_sharpe']:+.4f} CI98.33 {ev['delta_sharpe_ci9833']} halves={ev['delta_sharpe_halves']}\n"
            f"  dMaxDD  {ev['delta_max_dd']:+.4f} CI98.33 {ev['delta_max_dd_ci9833']} stress_10bps_ok={ev['stress_ok']}"
        )
        if "key_years" in ev:
            print("  Key years:", ev["key_years"])
        if "triggers_fired" in ev:
            print("  Intra-month triggers fired:", ev["triggers_fired"])
    print("\nDiagnostic A+B (no authority):", r["diagnostic_A_plus_B"])

    out = REPO / "reports" / f"backtest_vm_outlook_v2_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
