#!/usr/bin/env python3
"""Preregistered estimator for Equity Playbook 2: Volatility-Managed Sector/Asset ETF Rotation.

Frozen registration: docs/PREREG_VOL_MANAGED_OUTLOOK_20261124.md
Zero parameter tuning. Reuses src/outlook/engine.py and src/outlook/factors.py verbatim.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.equity_playbooks.backtest import load_equity_ohlcv
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE, _features
from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG, rank_universe

TARGET_VOL = 0.14
MIN_SCALE = 0.25
MAX_SCALE = 1.00
TURNOVER_COST = 0.0003   # 3 bps one-way
STEP = 21                # monthly rebalance (21 trading days)
SEED = 20261124
ALPHA_BONF3 = 0.05 / 3.0  # 98.33% two-sided CI


def _load_vix_csv(path: str = "data/discovery/VIX_History.csv") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not Path(path).exists():
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw_d = row.get("DATE", "").strip()
            raw_c = row.get("CLOSE", "").strip()
            if not raw_d or not raw_c:
                continue
            try:
                m, d, y = raw_d.split("/")
                iso = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                out[iso] = float(raw_c)
            except Exception:
                continue
    return out


def _align_matrix(
    ohlcv: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    tickers: Sequence[str],
    *,
    strict_common: bool,
    min_date: str = "2006-01-01",
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build date-aligned close arrays (NaN before launch when strict_common=False)."""
    spy_dates, spy_closes, _ = ohlcv[BENCH]
    base_idx = [i for i, d in enumerate(spy_dates) if d >= min_date]
    dates = [spy_dates[i] for i in base_idx]
    date_to_pos = {d: i for i, d in enumerate(dates)}

    cols: Dict[str, np.ndarray] = {BENCH: spy_closes[base_idx].astype(float)}
    for tk in tickers:
        if tk not in ohlcv:
            continue
        ds, cs, _ = ohlcv[tk]
        arr = np.full(len(dates), np.nan, dtype=float)
        for d, c in zip(ds, cs):
            p = date_to_pos.get(d)
            if p is not None:
                arr[p] = float(c)
        cols[tk] = arr

    if strict_common:
        valid = np.ones(len(dates), dtype=bool)
        for arr in cols.values():
            valid &= np.isfinite(arr) & (arr > 0.0)
        dates = [d for d, ok in zip(dates, valid) if ok]
        cols = {tk: arr[valid] for tk, arr in cols.items()}
    return dates, cols


def _trailing_basket_rv(
    cols: Dict[str, np.ndarray],
    weights: Dict[str, float],
    t: int,
    window: int = 20,
) -> Optional[float]:
    """Annualized trailing 20d realized volatility of the target basket weights."""
    if t < window or not weights:
        return None
    w_sum = sum(weights.values())
    if w_sum <= 0:
        return None
    norm_w = {k: v / w_sum for k, v in weights.items()}
    port_rets = np.zeros(window, dtype=float)
    for tk, w in norm_w.items():
        seg = cols[tk][t - window : t + 1]
        if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
            return None
        r = np.diff(seg) / seg[:-1]
        port_rets += w * r
    sd = float(np.std(port_rets, ddof=1))
    return sd * math.sqrt(252.0) if sd > 0 else None


def _simulate_arms(
    dates: List[str],
    cols: Dict[str, np.ndarray],
    universe: Sequence[str],
    *,
    turnover_cost: float = TURNOVER_COST,
    top_k: int = 3,
    vix_map: Optional[Dict[str, float]] = None,
    vix_cap: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Simulate B0 (SPY_BH), B1 (EW_16), Arm A (OUTLOOK_TOP3), Arm B (VM_OUTLOOK_TOP3), Arm C (VM_SPY)."""
    bench = cols[BENCH]
    n = len(dates)
    cfg = DEFAULT_OUTLOOK_CONFIG

    # Precompute 60d VIX MA if vix_cap sensitivity is requested
    vix_arr = np.full(n, np.nan)
    if vix_map:
        for i, d in enumerate(dates):
            if d in vix_map:
                vix_arr[i] = vix_map[d]

    arm_names = ("SPY_BH", "EW_16", "OUTLOOK_TOP3", "VM_OUTLOOK_TOP3", "VM_SPY")
    prev_w: Dict[str, Dict[str, float]] = {a: {} for a in arm_names}
    period_rets: Dict[str, List[float]] = {a: [] for a in arm_names}
    period_dates: List[str] = []
    daily_rets: Dict[str, List[float]] = {a: [] for a in arm_names}
    daily_dates: List[str] = []
    exposures: Dict[str, List[float]] = {a: [] for a in arm_names}
    turnovers: Dict[str, List[float]] = {a: [] for a in arm_names}

    # First rebalance index where SPY + at least 6 ETFs have >=252 trailing closes
    rebal_indices: List[int] = []
    for t in range(252, n - STEP - 1, STEP):
        avail = [
            tk for tk in universe
            if tk in cols and np.all(np.isfinite(cols[tk][t - 252 : t + 1]))
        ]
        if len(avail) >= 6:
            rebal_indices.append(t)

    for idx_pos, t in enumerate(rebal_indices):
        exec_in = t + 1
        exec_out = (rebal_indices[idx_pos + 1] + 1) if (idx_pos + 1 < len(rebal_indices)) else (n - 1)
        if exec_out <= exec_in:
            continue

        avail = [
            tk for tk in universe
            if tk in cols
            and np.all(np.isfinite(cols[tk][t - 252 : t + 1]))
            and np.all(np.isfinite(cols[tk][exec_in : exec_out + 1]))
        ]
        if len(avail) < top_k:
            continue

        bench_list = bench[: t + 1].tolist()
        feats = {tk: _features(cols[tk][: t + 1].tolist(), bench_list, t) for tk in avail}
        ranked = rank_universe(feats, cfg)

        # Arm A: top_k with score > 0, weight 1/top_k each (remaining in cash)
        pos_top = [r["ticker"] for r in ranked[:top_k] if r["score"] > 0.0]
        w_a = {tk: 1.0 / float(top_k) for tk in pos_top}

        # Arm B: Vol-managed Arm A
        rv_a = _trailing_basket_rv(cols, w_a, t, window=20)
        if rv_a and rv_a > 0 and w_a:
            scale_b = float(np.clip(TARGET_VOL / rv_a, MIN_SCALE, MAX_SCALE))
        else:
            scale_b = 1.0
        if vix_cap and t >= 60 and np.isfinite(vix_arr[t]):
            vix_ma = float(np.nanmean(vix_arr[t - 60 : t]))
            if np.isfinite(vix_ma) and vix_arr[t] > vix_ma * 1.15:
                scale_b = min(scale_b, 0.50)
        w_b = {tk: v * scale_b for tk, v in w_a.items()}

        # Arm C: Vol-managed SPY
        rv_spy = _trailing_basket_rv(cols, {BENCH: 1.0}, t, window=20)
        scale_c = float(np.clip(TARGET_VOL / rv_spy, MIN_SCALE, MAX_SCALE)) if (rv_spy and rv_spy > 0) else 1.0
        if vix_cap and t >= 60 and np.isfinite(vix_arr[t]):
            vix_ma = float(np.nanmean(vix_arr[t - 60 : t]))
            if np.isfinite(vix_ma) and vix_arr[t] > vix_ma * 1.15:
                scale_c = min(scale_c, 0.50)
        w_c = {BENCH: scale_c}

        w_b0 = {BENCH: 1.0}
        w_b1 = {tk: 1.0 / float(len(avail)) for tk in avail}

        targets = {
            "SPY_BH": w_b0,
            "EW_16": w_b1,
            "OUTLOOK_TOP3": w_a,
            "VM_OUTLOOK_TOP3": w_b,
            "VM_SPY": w_c,
        }

        period_dates.append(dates[exec_in])
        for d_i in range(exec_in + 1, exec_out + 1):
            daily_dates.append(dates[d_i])

        for arm in arm_names:
            w_new = targets[arm]
            w_old = prev_w[arm]
            all_keys = set(w_new) | set(w_old)
            to = sum(abs(w_new.get(k, 0.0) - w_old.get(k, 0.0)) for k in all_keys)
            cost = to * turnover_cost
            turnovers[arm].append(to)
            exposures[arm].append(sum(w_new.values()))

            # Compound daily returns across the holding interval for honest drawdown & daily Sharpe
            eq = 1.0 - cost
            cur_w = dict(w_new)
            for d_i in range(exec_in + 1, exec_out + 1):
                day_r = 0.0
                for tk, wt in cur_w.items():
                    day_r += wt * (cols[tk][d_i] / cols[tk][d_i - 1] - 1.0)
                if d_i == exec_in + 1:
                    daily_rets[arm].append((1.0 - cost) * (1.0 + day_r) - 1.0)
                else:
                    daily_rets[arm].append(day_r)
                eq *= (1.0 + day_r)
            period_rets[arm].append(eq - 1.0)
            prev_w[arm] = w_new

    return {
        "period_dates": period_dates,
        "daily_dates": daily_dates,
        "period_rets": {a: np.asarray(period_rets[a], dtype=float) for a in arm_names},
        "daily_rets": {a: np.asarray(daily_rets[a], dtype=float) for a in arm_names},
        "exposures": {a: float(np.mean(exposures[a])) if exposures[a] else 0.0 for a in arm_names},
        "ann_turnover": {a: float(np.mean(turnovers[a]) * 12.0) if turnovers[a] else 0.0 for a in arm_names},
    }


def _metrics_from_monthly(rets: np.ndarray) -> dict:
    if rets.size == 0:
        return {}
    eq = np.cumprod(1.0 + rets)
    years = rets.size / 12.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if eq[-1] > 0 else -1.0
    vol = float(np.std(rets, ddof=1) * math.sqrt(12.0)) if rets.size > 1 else 0.0
    sh = float((np.mean(rets) / np.std(rets, ddof=1)) * math.sqrt(12.0)) if (rets.size > 1 and np.std(rets, ddof=1) > 0) else 0.0
    down = rets[rets < 0]
    down_sd = float(np.sqrt(np.mean(down ** 2)) * math.sqrt(12.0)) if down.size else 0.0
    sortino = float((np.mean(rets) * 12.0) / down_sd) if down_sd > 0 else 0.0
    peak = np.maximum.accumulate(np.insert(eq, 0, 1.0))[1:]
    max_dd = float(np.min(eq / peak - 1.0))
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "n_periods": int(rets.size),
        "cagr": round(cagr, 5),
        "ann_vol": round(vol, 5),
        "sharpe": round(sh, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 5),
        "calmar": round(calmar, 4),
    }


def _paired_block_bootstrap(
    r_arm: np.ndarray,
    r_bench: np.ndarray,
    *,
    block_len: int = 3,
    n_boot: int = 10_000,
    seed: int = SEED,
) -> dict:
    n = r_arm.size
    n_blocks = int(math.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(block_len)[None, None, :]
    idx = ((starts[:, :, None] + offs) % n).reshape(n_boot, n_blocks * block_len)[:, :n]

    a = r_arm[idx]
    b = r_bench[idx]
    sh_a = (np.mean(a, axis=1) / np.std(a, axis=1, ddof=1)) * math.sqrt(12.0)
    sh_b = (np.mean(b, axis=1) / np.std(b, axis=1, ddof=1)) * math.sqrt(12.0)
    d_sh = sh_a - sh_b

    eq_a = np.cumprod(1.0 + a, axis=1)
    pk_a = np.maximum.accumulate(np.concatenate([np.ones((n_boot, 1)), eq_a], axis=1), axis=1)[:, 1:]
    dd_a = np.min(eq_a / pk_a - 1.0, axis=1)

    eq_b = np.cumprod(1.0 + b, axis=1)
    pk_b = np.maximum.accumulate(np.concatenate([np.ones((n_boot, 1)), eq_b], axis=1), axis=1)[:, 1:]
    dd_b = np.min(eq_b / pk_b - 1.0, axis=1)
    d_dd = dd_a - dd_b  # positive = arm has shallower drawdown than benchmark

    lo98 = 100.0 * (ALPHA_BONF3 / 2.0)
    hi98 = 100.0 * (1.0 - ALPHA_BONF3 / 2.0)
    return {
        "delta_sharpe": round(float(_metrics_from_monthly(r_arm)["sharpe"] - _metrics_from_monthly(r_bench)["sharpe"]), 4),
        "delta_sharpe_ci95": [round(float(np.percentile(d_sh, 2.5)), 4), round(float(np.percentile(d_sh, 97.5)), 4)],
        "delta_sharpe_ci9833": [round(float(np.percentile(d_sh, lo98)), 4), round(float(np.percentile(d_sh, hi98)), 4)],
        "delta_max_dd": round(float(_metrics_from_monthly(r_arm)["max_dd"] - _metrics_from_monthly(r_bench)["max_dd"]), 4),
        "delta_max_dd_ci95": [round(float(np.percentile(d_dd, 2.5)), 4), round(float(np.percentile(d_dd, 97.5)), 4)],
    }


def summarize_run(sim: dict, label: str) -> dict:
    prets = sim["period_rets"]
    pdates = sim["period_dates"]
    stats = {}
    for arm, arr in prets.items():
        m = _metrics_from_monthly(arr)
        m["avg_exposure"] = round(sim["exposures"][arm], 4)
        m["ann_turnover"] = round(sim["ann_turnover"][arm], 4)
        m["cost_drag_bps_yr"] = round(sim["ann_turnover"][arm] * TURNOVER_COST * 10_000, 2)
        stats[arm] = m

    comps = {}
    for arm in ("OUTLOOK_TOP3", "VM_OUTLOOK_TOP3", "VM_SPY"):
        comps[f"{arm}_vs_SPY"] = _paired_block_bootstrap(prets[arm], prets["SPY_BH"])
        comps[f"{arm}_vs_EW16"] = _paired_block_bootstrap(prets[arm], prets["EW_16"])
    comps["VM_OUTLOOK_vs_VM_SPY"] = _paired_block_bootstrap(prets["VM_OUTLOOK_TOP3"], prets["VM_SPY"])

    # Sub-period Sharpe & MaxDD check (pre-2020 vs 2020-2026 + key regimes)
    regimes = {
        "2007_2011": ("2007-01-01", "2011-12-31"),
        "2012_2019": ("2012-01-01", "2019-12-31"),
        "pre_2020": ("2006-01-01", "2019-12-31"),
        "2020_2026": ("2020-01-01", "2026-12-31"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
        "2023_2026": ("2023-01-01", "2026-12-31"),
    }
    sub = {}
    for rname, (lo, hi) in regimes.items():
        mask = np.array([lo <= d <= hi for d in pdates], dtype=bool)
        if int(mask.sum()) >= 6:
            sub[rname] = {
                arm: {
                    "cagr": _metrics_from_monthly(prets[arm][mask])["cagr"],
                    "sharpe": _metrics_from_monthly(prets[arm][mask])["sharpe"],
                    "max_dd": _metrics_from_monthly(prets[arm][mask])["max_dd"],
                }
                for arm in ("SPY_BH", "EW_16", "OUTLOOK_TOP3", "VM_OUTLOOK_TOP3", "VM_SPY")
            }

    # Binding Pass Rule on Arm B (VM_OUTLOOK_TOP3) & Arm C (VM_SPY)
    def _eval_arm(arm: str) -> str:
        c = comps[f"{arm}_vs_SPY"]
        s_arm = stats[arm]
        s_spy = stats["SPY_BH"]
        if s_arm["n_periods"] < 60:
            return "INSUFFICIENT"
        sharpe_sig = c["delta_sharpe_ci9833"][0] > 0.0
        dd_cut = (
            s_arm["sharpe"] >= s_spy["sharpe"] - 0.05
            and abs(s_arm["max_dd"]) <= abs(s_spy["max_dd"]) * 0.75
            and s_arm["cagr"] > 0.05
        )
        # Sub-period sign replication if both pre_2020 and 2020_2026 exist
        rep_ok = True
        if "pre_2020" in sub and "2020_2026" in sub:
            for reg in ("pre_2020", "2020_2026"):
                sa = sub[reg][arm]
                sb = sub[reg]["SPY_BH"]
                if not (sa["sharpe"] >= sb["sharpe"] - 0.05 or abs(sa["max_dd"]) < abs(sb["max_dd"])):
                    rep_ok = False
        return "PASS" if ((sharpe_sig or dd_cut) and rep_ok) else "FAIL"

    return {
        "label": label,
        "window": [pdates[0], pdates[-1]] if pdates else [],
        "verdict_VM_OUTLOOK_TOP3": _eval_arm("VM_OUTLOOK_TOP3"),
        "verdict_VM_SPY": _eval_arm("VM_SPY"),
        "verdict_OUTLOOK_TOP3": _eval_arm("OUTLOOK_TOP3"),
        "stats": stats,
        "comparisons": comps,
        "subperiods": sub,
    }


def main():
    p = argparse.ArgumentParser(description="Backtest Equity Playbook 2 (Vol-Managed Outlook Rotation)")
    p.add_argument("--out", default="reports/backtest_vol_managed_outlook_20261124.json")
    args = p.parse_args()

    ohlcv = load_equity_ohlcv("data/equity_ohlcv.db", [BENCH] + list(DEFAULT_UNIVERSE))
    vix_map = _load_vix_csv()

    # 1. Aligned 16-ETF window (2018-06-19 .. 2026-11-24)
    d_aligned, c_aligned = _align_matrix(ohlcv, DEFAULT_UNIVERSE, strict_common=True, min_date="2018-01-01")
    sim_aligned = _simulate_arms(d_aligned, c_aligned, DEFAULT_UNIVERSE, vix_map=vix_map)
    res_aligned = summarize_run(sim_aligned, "aligned_16etf_2018_2026")

    # 2. Extended dynamic-universe window (2006 .. 2026, covers 2008 GFC + 2011 + 2015 + 2018 + 2020 + 2022)
    d_ext, c_ext = _align_matrix(ohlcv, DEFAULT_UNIVERSE, strict_common=False, min_date="2005-01-01")
    sim_ext = _simulate_arms(d_ext, c_ext, DEFAULT_UNIVERSE, vix_map=vix_map)
    res_ext = summarize_run(sim_ext, "extended_dynamic_2006_2026")

    # Sensitivities (no authority)
    sim_vix = _simulate_arms(d_ext, c_ext, DEFAULT_UNIVERSE, vix_map=vix_map, vix_cap=True)
    res_vix = summarize_run(sim_vix, "sensitivity_vix_cap_2006_2026")

    sim_10bps = _simulate_arms(d_ext, c_ext, DEFAULT_UNIVERSE, turnover_cost=0.0010, vix_map=vix_map)
    res_10bps = summarize_run(sim_10bps, "sensitivity_10bps_2006_2026")

    sim_top4 = _simulate_arms(d_ext, c_ext, DEFAULT_UNIVERSE, top_k=4, vix_map=vix_map)
    res_top4 = summarize_run(sim_top4, "sensitivity_top4_2006_2026")

    payload = {
        "prereg": "docs/PREREG_VOL_MANAGED_OUTLOOK_20261124.md",
        "aligned_2018_2026": res_aligned,
        "extended_2006_2026": res_ext,
        "sensitivity_vix_cap": res_vix,
        "sensitivity_10bps": res_10bps,
        "sensitivity_top4": res_top4,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for key in ("aligned_2018_2026", "extended_2006_2026", "sensitivity_vix_cap"):
        r = payload[key]
        print(f"\n[{r['label']}] window={r['window']}  VERDICTS: VM_OUTLOOK={r['verdict_VM_OUTLOOK_TOP3']} VM_SPY={r['verdict_VM_SPY']} OUTLOOK={r['verdict_OUTLOOK_TOP3']}")
        for arm in ("SPY_BH", "EW_16", "OUTLOOK_TOP3", "VM_OUTLOOK_TOP3", "VM_SPY"):
            s = r["stats"][arm]
            print(f"  {arm:<16} CAGR={s['cagr']:+.2%} Vol={s['ann_vol']:.2%} Sh={s['sharpe']:.3f} MaxDD={s['max_dd']:+.2%} Exp={s['avg_exposure']:.0%} TO={s['ann_turnover']:.1f}x")
        c = r["comparisons"]["VM_OUTLOOK_TOP3_vs_SPY"]
        print(f"  VM_OUTLOOK vs SPY: dSharpe={c['delta_sharpe']:+.3f} CI98.33={c['delta_sharpe_ci9833']} dMaxDD={c['delta_max_dd']:+.2%} CI95={c['delta_max_dd_ci95']}")


if __name__ == "__main__":
    main()
