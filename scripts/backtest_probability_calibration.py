#!/usr/bin/env python3
"""Preregistered estimator for Study 4B: Walk-Forward Empirical Probability Calibration.

Frozen registration: docs/PREREG_PROBABILITY_CALIBRATION_20261127.md

Runs strictly out-of-sample walk-forward calibration across:
  C1: VM_OUTLOOK_TOP3 (2006-2026, n=250 months) by basket vol regime (CALM vs ELEVATED)
  C2: equity_pullback (2016-2026, 375 symbols, 10 bps) with walk-forward logistic + SPY>SMA200 regimes
  C3: Options DTE Curve (0DTE..240D, 2024-2026 OPRA k=0.35 limit fills) + pop_calibration ship_check
  C4: Option-Implied Delta vs Realized Finish (30d & 90d puts across 144 OPRA Wednesdays)

Writes:
  * configs/calibrated_evidence.json  (committed machine-readable calibration table for [6] DISCOVER)
  * reports/probability_calibration_<ts>.json (full audit artifact)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.backtest_equity_pullbacks import load_earnings_calendar, simulate_symbol_pullbacks
from scripts.backtest_opra_o1_and_calendars import ETFS, weekly_wednesdays
from scripts.backtest_vol_managed_outlook import _align_matrix, _simulate_arms, _trailing_basket_rv
from scripts.sleeve_vol_forecast_test import grid
from src.databento_opra import DEFAULT_DB
from src.equity_playbooks.backtest import BenchmarkSeries, load_equity_ohlcv
from src.equity_playbooks.indicators import sma, trailing_rv, wilder_rsi
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE
from src.paths import repo_path
from src.pop_calibration import fit as pop_fit
from src.pop_calibration import predict as pop_predict
from src.pop_calibration import reliability, ship_check, wilson
from src.radar.wide_spread_rule import quotes_from_rows
from src.raw_closes import official_raw_closes

REPO = Path(__file__).resolve().parents[1]


def _brier_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    if y.size == 0:
        return {"n": 0, "brier": None, "brier_base": None, "bss": None, "ece": None}
    yf = y.astype(float)
    pf = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    base = float(np.mean(yf))
    brier = float(np.mean((pf - yf) ** 2))
    brier_base = float(np.mean((base - yf) ** 2))
    bss = (1.0 - brier / brier_base) if brier_base > 1e-12 else 0.0
    # 5-bin expected calibration error
    bins = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    for i in range(5):
        mask = (pf >= bins[i]) & (pf <= bins[i + 1] if i == 4 else pf < bins[i + 1])
        if np.any(mask):
            ece += (float(np.sum(mask)) / yf.size) * abs(float(np.mean(pf[mask])) - float(np.mean(yf[mask])))
    return {
        "n": int(yf.size),
        "base_rate": round(base, 4),
        "brier": round(brier, 5),
        "brier_base": round(brier_base, 5),
        "bss": round(bss, 4),
        "ece": round(ece, 4),
    }


def _rate_block(rets: np.ndarray, *, label: str = "") -> Dict[str, Any]:
    if rets.size == 0:
        return {"n": 0}
    wins = int(np.sum(rets > 0.0))
    lo, hi = wilson(wins, int(rets.size))
    w = rets[rets > 0.0]
    ls = rets[rets <= 0.0]
    neg = float(-np.sum(ls))
    pos = float(np.sum(w))
    pf = (pos / neg) if neg > 0 else float("inf")
    return {
        "label": label,
        "n": int(rets.size),
        "p_win": round(wins / rets.size, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "mean_ret": round(float(np.mean(rets)), 5),
        "avg_win": round(float(np.mean(w)), 5) if w.size else 0.0,
        "avg_loss": round(float(np.mean(ls)), 5) if ls.size else 0.0,
        "worst_10pct": round(float(np.percentile(rets, 10)), 5),
        "pf": round(pf, 3) if math.isfinite(pf) else 99.0,
    }


# ── C1: VM_OUTLOOK_TOP3 Core Sleeve Calibration ──────────────────────────────


def calibrate_vm_outlook() -> Dict[str, Any]:
    ohlcv = load_equity_ohlcv(repo_path("data/equity_ohlcv.db"), [BENCH] + list(DEFAULT_UNIVERSE))
    dates, cols = _align_matrix(ohlcv, DEFAULT_UNIVERSE, strict_common=False, min_date="2005-01-01")
    sim = _simulate_arms(dates, cols, DEFAULT_UNIVERSE)
    r_vm = np.asarray(sim["period_rets"]["VM_OUTLOOK_TOP3"], dtype=float)
    r_spy = np.asarray(sim["period_rets"]["SPY_BH"], dtype=float)
    p_dates = list(sim["period_dates"])
    n = int(r_vm.size)

    # Exact entry-date basket RV20 regime (CALM <= 14% vs ELEVATED > 14%)
    months_g = grid(dates, cols)
    rv_by_exec = {}
    for m in months_g:
        rv = _trailing_basket_rv(cols, m["w_a"], m["t"], window=20) if m["w_a"] else None
        rv_by_exec[m["exec_date"]] = rv
    is_calm = np.array([(rv_by_exec.get(d) is None) or (float(rv_by_exec[d]) <= 0.14) for d in p_dates], dtype=bool)
    r_3m = np.array([float(np.prod(1.0 + r_vm[i : i + 3]) - 1.0) for i in range(n - 2)], dtype=float)
    r_spy_3m = np.array([float(np.prod(1.0 + r_spy[i : i + 3]) - 1.0) for i in range(n - 2)], dtype=float)
    r_12m = np.array([float(np.prod(1.0 + r_vm[i : i + 12]) - 1.0) for i in range(n - 11)], dtype=float)

    # Expanding walk-forward (min_train=60m) for 1m positive and 3m positive conditioned on CALM vs ELEVATED
    y1_oos: List[int] = []
    p1_oos: List[float] = []
    for t in range(60, n):
        reg_mask = is_calm[:t] == is_calm[t]
        p_hat = float(np.mean(r_vm[:t][reg_mask] > 0.0)) if np.sum(reg_mask) >= 12 else float(np.mean(r_vm[:t] > 0.0))
        y1_oos.append(1 if r_vm[t] > 0.0 else 0)
        p1_oos.append(p_hat)

    y3_oos: List[int] = []
    p3_oos: List[float] = []
    for t in range(60, len(r_3m)):
        # Strict non-overlap purge: only use training windows whose 3m end <= t
        tr_end = t - 2
        if tr_end < 36:
            continue
        reg_mask = is_calm[:tr_end] == is_calm[t]
        p_hat = float(np.mean(r_3m[:tr_end][reg_mask] > 0.0)) if np.sum(reg_mask) >= 12 else float(np.mean(r_3m[:tr_end] > 0.0))
        y3_oos.append(1 if r_3m[t] > 0.0 else 0)
        p3_oos.append(p_hat)

    wf_1m = _brier_metrics(np.asarray(y1_oos), np.asarray(p1_oos))
    wf_3m = _brier_metrics(np.asarray(y3_oos), np.asarray(p3_oos))
    ece_ok = (wf_1m.get("ece") or 1.0) <= 0.06 and (wf_3m.get("ece") or 1.0) <= 0.06

    return {
        "track": "C1_VM_OUTLOOK_TOP3",
        "window": [p_dates[0], p_dates[-1]],
        "n_months": n,
        "calibrated": bool(ece_ok),
        "walk_forward_1m": wf_1m,
        "walk_forward_3m": wf_3m,
        "horizons": {
            "1m_all": _rate_block(r_vm, label="1m return > 0 (all regimes)"),
            "1m_calm": _rate_block(r_vm[is_calm], label="1m return > 0 (CALM RV20 <= 14%)"),
            "1m_elevated": _rate_block(r_vm[~is_calm], label="1m return > 0 (ELEVATED RV20 > 14%)"),
            "3m_all": _rate_block(r_3m, label="3m return > 0 (all regimes)"),
            "3m_calm": _rate_block(r_3m[is_calm[: len(r_3m)]], label="3m return > 0 (CALM)"),
            "3m_elevated": _rate_block(r_3m[~is_calm[: len(r_3m)]], label="3m return > 0 (ELEVATED)"),
            "12m_all": _rate_block(r_12m, label="12m return > 0 (all regimes)"),
            "3m_beat_spy": _rate_block(r_3m - r_spy_3m, label="3m beats SPY"),
        },
    }


# ── C2: equity_pullback Shares Satellite Calibration ─────────────────────────


def calibrate_equity_pullback() -> Dict[str, Any]:
    db = repo_path("data/equity_ohlcv.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        syms = [str(r[0]) for r in con.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker").fetchall()]
    finally:
        con.close()
    ohlcv = load_equity_ohlcv(db, syms)
    spy_ds, spy_cs, _ = ohlcv["SPY"]
    spy_bench = BenchmarkSeries(spy_ds, spy_cs)
    spy_sma200 = sma(spy_cs, 200)
    spy_above_map = {
        d: bool(math.isfinite(spy_sma200[i]) and spy_cs[i] > spy_sma200[i])
        for i, d in enumerate(spy_ds)
    }
    earn_map = load_earnings_calendar(repo_path("data/dolt_options.db"))

    rows: List[Dict[str, Any]] = []
    for sym, (ds, cs, vs) in ohlcv.items():
        if sym == "SPY" or len(ds) < 260:
            continue
        trades, _ = simulate_symbol_pullbacks(
            sym, ds, cs, vs, spy_bench, earn_map, stratum="liquid", rt_cost=0.0010, next_day_fill=True
        )
        if not trades:
            continue
        idx_map = {d: i for i, d in enumerate(ds)}
        rsi2 = wilder_rsi(cs, 2)
        rv20 = trailing_rv(cs, 20)
        for tr in trades:
            if tr.signal_date < "2016-01-01":
                continue
            i = idx_map.get(tr.signal_date)
            if i is None or i < 20:
                continue
            r3 = float(cs[i] / cs[i - 3] - 1.0) if cs[i - 3] > 0 else -0.03
            rows.append(
                {
                    "entry_date": tr.entry_date,
                    "symbol": sym,
                    "spy_above_200d": 1.0 if spy_above_map.get(tr.signal_date, True) else 0.0,
                    "rsi2": float(rsi2[i]) if math.isfinite(rsi2[i]) else 5.0,
                    "ret_3d": r3,
                    "rv20": float(rv20[i]) if math.isfinite(rv20[i]) else 0.25,
                    "strategy": "equity_pullback",
                    "is_short": 0.0,
                    "won": 1 if tr.net_return > 0.0 else 0,
                    "net_ret": float(tr.net_return),
                }
            )

    df = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    # Expanding yearly walk-forward (train < yr, predict == yr, starting 2018+)
    df["year"] = df["entry_date"].str[:4].astype(int)
    feats = ["spy_above_200d", "rsi2", "ret_3d", "rv20"]
    oos_parts: List[pd.DataFrame] = []
    for yr in sorted(df["year"].unique()):
        tr_df = df[df["year"] < yr]
        te_df = df[df["year"] == yr]
        if len(tr_df) < 100 or len(te_df) == 0:
            continue
        mod = pop_fit(tr_df, features=feats, ridge=2.0)
        preds = pop_predict(mod, te_df)
        oos_parts.append(pd.DataFrame({"predicted": preds, "won": te_df["won"].to_numpy()}))

    oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=["predicted", "won"])
    wf_m = _brier_metrics(oos_df["won"].to_numpy(), oos_df["predicted"].to_numpy()) if len(oos_df) else {}
    rel_df = reliability(oos_df, bucket_width=0.05, min_n=25)
    ship_ok, ship_msg = ship_check(rel_df, min_buckets=3)
    feature_calibrated = bool(ship_ok and (wf_m.get("bss") or -1.0) > 0.0)

    bull_rets = df[df["spy_above_200d"] > 0.5]["net_ret"].to_numpy(dtype=float)
    bear_rets = df[df["spy_above_200d"] <= 0.5]["net_ret"].to_numpy(dtype=float)
    all_rets = df["net_ret"].to_numpy(dtype=float)

    return {
        "track": "C2_EQUITY_PULLBACK",
        "window": [str(df["entry_date"].min()), str(df["entry_date"].max())] if len(df) else [],
        "n_trades": int(len(df)),
        "feature_model_calibrated": feature_calibrated,
        "ship_check": {"passed": ship_ok, "reason": ship_msg},
        "walk_forward": wf_m,
        "regimes": {
            "all": _rate_block(all_rets, label="All uptrend pullbacks (10 bps)"),
            "spy_above_200d": _rate_block(bull_rets, label="SPY > SMA200 (bull regime, 10 bps)"),
            "spy_below_200d": _rate_block(bear_rets, label="SPY <= SMA200 (bear regime — AVOID, 10 bps)"),
        },
        "reliability_bins": rel_df.to_dict("records"),
    }


# ── C3: Options DTE Curve Walk-Forward Calibration ───────────────────────────


def calibrate_options_dte(trades_path: Path, dte_report: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.loads(trades_path.read_text(encoding="utf-8"))
    all_rows: List[Dict[str, Any]] = []
    for bname, trs in raw.items():
        for t in trs:
            all_rows.append(
                {
                    "entry_date": t["entry_date"],
                    "bucket": bname,
                    "strategy": bname,
                    "is_short": 1.0,
                    "abs_delta": float(t["abs_delta"]),
                    "dte": float(t["dte"]),
                    "credit_to_width": float(t["credit_to_width"]),
                    "friction": float(t["friction"]),
                    "won": 1 if float(t["roc"]) > 0.0 else 0,
                    "roc": float(t["roc"]),
                }
            )
    df = pd.DataFrame(all_rows).sort_values("entry_date").reset_index(drop=True)
    df["ym"] = df["entry_date"].str[:7]
    months = sorted(df["ym"].unique())
    feats = ["abs_delta", "dte", "credit_to_width", "friction"]

    oos_parts: List[pd.DataFrame] = []
    for mi in range(6, len(months)):
        m_cur = months[mi]
        # Purge overlap: train on entries at least 2 months prior
        tr_df = df[df["ym"] <= months[mi - 2]]
        te_df = df[df["ym"] == m_cur]
        if len(tr_df) < 100 or len(te_df) == 0:
            continue
        mod = pop_fit(tr_df, features=feats, ridge=2.0)
        preds = pop_predict(mod, te_df)
        oos_parts.append(pd.DataFrame({"predicted": preds, "won": te_df["won"].to_numpy()}))

    oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=["predicted", "won"])
    wf_m = _brier_metrics(oos_df["won"].to_numpy(), oos_df["predicted"].to_numpy()) if len(oos_df) else {}
    rel_df = reliability(oos_df, bucket_width=0.10, min_n=20)
    ship_ok, ship_msg = ship_check(rel_df, min_buckets=3)
    model_calibrated = bool(ship_ok and (wf_m.get("bss") or -1.0) > 0.0)

    buckets_out: Dict[str, Any] = {}
    for bname, binfo in (dte_report.get("buckets") or {}).items():
        lim = binfo.get("limit_k35") or {}
        crs = binfo.get("crossed_k100") or {}
        buckets_out[bname] = {
            "n": lim.get("n", 0),
            "eff_n": lim.get("eff_n", 0),
            "p_win": lim.get("win_rate"),
            "ci95": lim.get("win_rate_ci95"),
            "mean_roc": lim.get("mean_roc"),
            "avg_win": lim.get("avg_win_roc"),
            "avg_loss": lim.get("avg_loss_roc"),
            "worst_5pct": lim.get("worst_5pct_roc"),
            "pf": lim.get("pf"),
            "pf_ci": lim.get("pf_ci"),
            "excess_entry_limit": lim.get("excess_entry"),
            "excess_ci_limit": lim.get("excess_ci"),
            "excess_entry_crossed": crs.get("excess_entry"),
            "med_friction": lim.get("med_friction"),
            "verdict": "LEAD_REFERENCE" if bname == "60_120D" else lim.get("verdict", "CLOSED"),
        }

    return {
        "track": "C3_OPTIONS_DTE_CURVE",
        "n_total_trades": int(len(df)),
        "contract_model_calibrated": model_calibrated,
        "ship_check": {"passed": ship_ok, "reason": ship_msg},
        "walk_forward": wf_m,
        "reliability_bins": rel_df.to_dict("records"),
        "buckets": buckets_out,
    }


# ── C4: Option-Implied Delta vs Realized Finish Calibration ──────────────────


def calibrate_implied_vs_realized(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    dates_w = weekly_wednesdays()
    series_map = official_raw_closes(list(ETFS), start="2023-12-15", end="2026-10-15")
    spots: Dict[Tuple[str, str], float] = {}
    for sym, ser in series_map.items():
        for d_s, px in ser.items():
            if math.isfinite(float(px)) and float(px) > 0:
                spots[(sym, str(d_s)[:10])] = float(px)

    con = sqlite3.connect(f"file:{repo_path(db_path)}?mode=ro", uri=True)
    try:
        ph = ",".join("?" * len(dates_w))
        raw = con.execute(
            f"SELECT symbol, date, expiration, strike, bid, ask FROM opra_chain "
            f"WHERE clock_et = '15:45' AND type = 'put' AND date IN ({ph}) AND bid > 0 AND ask >= bid",
            dates_w,
        ).fetchall()
    finally:
        con.close()

    by_sd: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for sym, d, exp, k, b, a in raw:
        if sym in ETFS:
            by_sd.setdefault((str(sym), str(d)), {}).setdefault(str(exp), []).append(
                {"strike": float(k), "type": "put", "bid": float(b), "ask": float(a)}
            )

    bins_def = [(0.10, 0.20, "10-20d"), (0.20, 0.30, "20-30d"), (0.30, 0.40, "30-40d"), (0.40, 0.50, "40-50d")]
    out: Dict[str, Any] = {}
    for h_lbl, d_lo, d_hi, tgt in (("30d", 20, 45, 30), ("90d", 60, 120, 90)):
        recs: Dict[str, List[Tuple[float, int]]] = {b[2]: [] for b in bins_def}
        for (sym, d_in), exps in by_sd.items():
            if d_in > "2026-06-24":
                continue
            sp_in = spots.get((sym, d_in))
            if not sp_in:
                continue
            d_obj = dt.date.fromisoformat(d_in)
            cands = [(abs((dt.date.fromisoformat(e) - d_obj).days - tgt), e, (dt.date.fromisoformat(e) - d_obj).days)
                     for e in exps if d_lo <= (dt.date.fromisoformat(e) - d_obj).days <= d_hi]
            if not cands:
                continue
            cands.sort()
            _, exp_s, dte = cands[0]
            exp_obj = dt.date.fromisoformat(exp_s)
            sp_exp: Optional[float] = None
            for off in range(4):
                v = series_map[sym].get(exp_obj - dt.timedelta(days=off))
                if v is not None and math.isfinite(float(v)) and float(v) > 0:
                    sp_exp = float(v)
                    break
            if sp_exp is None:
                continue
            near = [r for r in exps[exp_s] if 0.75 * sp_in <= r["strike"] <= 1.03 * sp_in]
            qs = quotes_from_rows(near, spot=sp_in, dte=float(dte), opt_type="put", r=0.045)
            for lo_d, hi_d, blbl in bins_def:
                in_bin = [q for q in qs if q.delta is not None and lo_d <= abs(float(q.delta)) < hi_d]
                if not in_bin:
                    continue
                q_mid = min(in_bin, key=lambda q: abs(abs(float(q.delta or 0)) - 0.5 * (lo_d + hi_d)))
                imp_otm = 1.0 - abs(float(q_mid.delta or 0.0))
                realized_otm = 1 if sp_exp > q_mid.strike else 0
                recs[blbl].append((imp_otm, realized_otm))
        h_tbl: Dict[str, Any] = {}
        for _, _, blbl in bins_def:
            pts = recs[blbl]
            if not pts:
                continue
            n = len(pts)
            imp = float(np.mean([x[0] for x in pts]))
            wins = sum(x[1] for x in pts)
            real = wins / n
            wlo, whi = wilson(wins, n)
            h_tbl[blbl] = {
                "n": n,
                "implied_otm_prob": round(imp, 4),
                "realized_otm_prob": round(real, 4),
                "realized_ci95": [round(wlo, 4), round(whi, 4)],
                "vrp_bias_pp": round((real - imp) * 100.0, 2),
            }
        out[h_lbl] = h_tbl
    return {"track": "C4_IMPLIED_VS_REALIZED", "horizons": out}


def main() -> int:
    trades_path = REPO / "reports" / "dte_curve_trades_latest.json"
    dte_files = sorted((REPO / "reports").glob("backtest_dte_curve_opra_*.json"))
    if not trades_path.exists() or not dte_files:
        raise FileNotFoundError("Run `python -m scripts.backtest_dte_curve_opra --download` first.")
    dte_report = json.loads(dte_files[-1].read_text(encoding="utf-8"))

    print("[4B] Running C1: VM_OUTLOOK_TOP3 calibration...", flush=True)
    c1 = calibrate_vm_outlook()
    print("[4B] Running C2: equity_pullback calibration...", flush=True)
    c2 = calibrate_equity_pullback()
    print("[4B] Running C3: Options DTE curve calibration...", flush=True)
    c3 = calibrate_options_dte(trades_path, dte_report)
    print("[4B] Running C4: Implied vs Realized calibration...", flush=True)
    c4 = calibrate_implied_vs_realized()

    full_res = {
        "registered": "docs/PREREG_PROBABILITY_CALIBRATION_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "C1_VM_OUTLOOK_TOP3": c1,
        "C2_EQUITY_PULLBACK": c2,
        "C3_OPTIONS_DTE_CURVE": c3,
        "C4_IMPLIED_VS_REALIZED": c4,
    }

    # Write committed compact evidence artifact for offline UI use (`configs/calibrated_evidence.json`)
    cfg_out = REPO / "configs" / "calibrated_evidence.json"
    cfg_payload = {
        "schema_version": 1,
        "generated_at": full_res["generated"][:10],
        "prereg": "docs/PREREG_PROBABILITY_CALIBRATION_20261127.md",
        "vm_outlook_top3": c1,
        "equity_pullback": {k: v for k, v in c2.items() if k != "reliability_bins"},
        "options_dte": {k: v for k, v in c3.items() if k != "reliability_bins"},
        "implied_vs_realized": c4["horizons"],
    }
    cfg_out.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    rep_out = REPO / "reports" / f"probability_calibration_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    rep_out.write_text(json.dumps(full_res, indent=2), encoding="utf-8")

    print("\n=== Study 4B Summary ===")
    print("C1 VM_OUTLOOK_TOP3:", "CALIBRATED" if c1["calibrated"] else "UNCONFIRMED",
          "| 3m:", c1["horizons"]["3m_all"]["p_win"], c1["horizons"]["3m_all"]["ci95"],
          "| 12m:", c1["horizons"]["12m_all"]["p_win"], c1["horizons"]["12m_all"]["ci95"])
    print("C2 equity_pullback: n=", c2["n_trades"], "feature_calibrated=", c2["feature_model_calibrated"],
          "BSS=", c2["walk_forward"].get("bss"),
          "| SPY>200:", c2["regimes"]["spy_above_200d"]["p_win"], c2["regimes"]["spy_above_200d"]["ci95"],
          "| SPY<=200:", c2["regimes"]["spy_below_200d"]["p_win"])
    print("C3 Options DTE: n=", c3["n_total_trades"], "contract_model_calibrated=", c3["contract_model_calibrated"],
          "ship=", c3["ship_check"], "BSS=", c3["walk_forward"].get("bss"))
    print("C4 Implied vs Realized:", json.dumps(c4["horizons"]))
    print(f"Wrote {cfg_out} and {rep_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
