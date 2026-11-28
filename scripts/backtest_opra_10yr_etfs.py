#!/usr/bin/env python3
"""Preregistered estimator for Study 5A: 11-Year Institutional OPRA Test (2015-2026) of TLT, QQQ, IWM 60-120 DTE.

Frozen registration: docs/PREREG_OPRA_10YR_TLT_QQQ_IWM_20261127.md
Defaults to `--estimate` ($0 dry-run). Pass `--download --max-usd 9.0` to fetch 2015-2023 Wednesdays & evaluate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.backtest_opra_o1_and_calendars import (
    FEE_RT_PER_SHARE,
    _fill_credit,
    _fill_debit_close,
)
from src.alloc.validate import effective_n
from src.databento_opra import DEFAULT_DB, fetch_opra_snapshots
from src.paths import load_project_env, repo_path
from src.radar.wide_spread_rule import choose_expiration, quotes_from_rows, select_legs
from src.raw_closes import official_raw_closes

REPO = Path(__file__).resolve().parents[1]
TARGET_ETFS: Tuple[str, ...] = ("TLT", "QQQ", "IWM")
CLOCK_ET = "15:45"
START_DATE = "2015-01-07"
ENTRY_END = "2026-06-24"
TRACK_END = "2026-09-30"
SPLIT_DATE = "2021-01-01"  # H1: 2015-2020 vs H2: 2021-2026
SEED = 20261127
DRAWS = 10_000
ALPHA_BONF = 0.05 / 6.0  # 6 arms: {TLT, QQQ, IWM} x {bull_put, bear_call} -> 99.17% CI


def trading_wednesdays_11yr() -> List[str]:
    """Every US trading Wednesday 2015-01-07..2026-09-30 derived from official SPY daily trading days."""
    ser_map = official_raw_closes(["SPY"], start="2015-01-01", end="2026-10-05")
    spy_ser = ser_map.get("SPY")
    if spy_ser is None or len(spy_ser) == 0:
        raise RuntimeError("Could not load SPY trading calendar from official_raw_closes")
    trading_days = {dt.date.fromisoformat(str(d)[:10]) for d in spy_ser.index}
    # Exclude known US half-day (13:00 ET close) dates: day before July 4th (Jul 3), Black Friday, Dec 24
    half_days = set()
    for d in trading_days:
        if (d.month == 7 and d.day == 3) or (d.month == 12 and d.day == 24):
            half_days.add(d)
        if d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29:
            half_days.add(d)
    full_days = trading_days - half_days

    out: List[str] = []
    d = dt.date.fromisoformat(START_DATE)
    end_d = dt.date.fromisoformat(TRACK_END)
    while d <= end_d:
        chosen: Optional[dt.date] = None
        for off in (0, 1, 2, -1):
            cand = d + dt.timedelta(days=off)
            if cand in full_days:
                chosen = cand
                break
        if chosen is not None:
            iso = chosen.isoformat()
            if not out or out[-1] != iso:
                out.append(iso)
        d += dt.timedelta(days=7)
    return out


def download_or_estimate(
    *,
    download: bool = False,
    max_usd: float = 9.0,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    dates = trading_wednesdays_11yr()
    existing: set[str] = set()
    full = repo_path(db_path)
    if Path(full).exists():
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for (d,) in con.execute(
                "SELECT date FROM opra_chain WHERE clock_et = ? AND symbol IN ('TLT','QQQ','IWM') "
                "GROUP BY date HAVING COUNT(DISTINCT symbol) = 3",
                (CLOCK_ET,),
            ).fetchall():
                existing.add(str(d))
        except sqlite3.Error:
            pass
        finally:
            con.close()

    todo = [d for d in dates if d not in existing]
    if not todo:
        return {
            "etfs": list(TARGET_ETFS),
            "dates_total": len(dates),
            "dates_cached": len(dates),
            "dates_needed": 0,
            "total_est_usd": 0.0,
            "downloaded": False,
            "status": "all_cached",
        }

    # Group missing dates by year (newest -> oldest) so progress streams every ~52 dates
    by_year: Dict[str, List[str]] = {}
    for d in todo:
        by_year.setdefault(d[:4], []).append(d)

    total_est = 0.0
    total_rows = 0
    prior_spend = 0.0
    year_plans: List[Tuple[str, List[Dict[str, Any]]]] = []

    for yr in sorted(by_year.keys(), reverse=True):
        yr_dates = by_year[yr]
        est = fetch_opra_snapshots(
            TARGET_ETFS,
            yr_dates,
            hhmm_et=CLOCK_ET,
            minutes=1,
            max_usd=max_usd,
            dry_run=True,
            db_path=db_path,
            max_workers=6,
        )
        yr_usd = float(est["total_est_usd"])
        prior_spend = float(est.get("prior_spend_usd") or prior_spend)
        total_est += yr_usd
        year_plans.append((yr, list(est["plan"])))
        print(
            f"[5A estimate {yr}] {len(yr_dates)} dates = ${yr_usd:.4f} (running est=${total_est:.4f} / cap=${max_usd:.2f})",
            flush=True,
        )
        if total_est > max_usd:
            raise RuntimeError(f"Refused Study 5A download: running est ${total_est:.4f} > max_usd ${max_usd:.2f}")

    summary: Dict[str, Any] = {
        "etfs": list(TARGET_ETFS),
        "dates_total": len(dates),
        "dates_cached": len(dates) - len(todo),
        "dates_needed": len(todo),
        "total_est_usd": round(total_est, 4),
        "prior_spend_usd": prior_spend,
        "max_usd": max_usd,
        "downloaded": False,
    }
    if not download:
        return summary

    for yr, plan in year_plans:
        for i in range(0, len(plan), 18):
            sub_plan = plan[i : i + 18]
            sub_dates = [p["date"] for p in sub_plan]
            r = fetch_opra_snapshots(
                TARGET_ETFS,
                sub_dates,
                hhmm_et=CLOCK_ET,
                minutes=1,
                max_usd=max_usd,
                dry_run=False,
                db_path=db_path,
                max_workers=6,
                precomputed_plan=sub_plan,
            )
            total_rows += int(r["rows_written"])
            print(
                f"[5A download {yr}] {i + len(sub_plan)}/{len(plan)} "
                f"+{r['rows_written']:,} rows cum_spend=${r.get('new_cumulative_spend_usd', 0):.4f}",
                flush=True,
            )
    summary["downloaded"] = True
    summary["rows_written"] = total_rows
    return summary


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _boot_arm(trades: List[Dict[str, Any]], *, alpha: float = ALPHA_BONF) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "verdict": "INSUFFICIENT"}
    by_m: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_m.setdefault(t["entry_date"][:7], []).append(t)
    months = sorted(by_m)
    k = len(months)
    xs = np.array([float(t["excess_entry"]) for t in trades], dtype=float)
    rocs = np.array([float(t["roc"]) for t in trades], dtype=float)

    def _pf(a: np.ndarray) -> float:
        neg = float(-np.sum(a[a < 0.0]))
        pos = float(np.sum(a[a > 0.0]))
        return pos / neg if neg > 0 else float("inf")

    m_x = [np.array([float(t["excess_entry"]) for t in by_m[m]], dtype=float) for m in months]
    m_r = [np.array([float(t["roc"]) for t in by_m[m]], dtype=float) for m in months]

    rng = np.random.default_rng(SEED)
    b = max(1, min(3, k))
    n_blocks = int(math.ceil(k / b))
    max_start = max(1, k - b + 1)
    bx = np.empty(DRAWS, dtype=float)
    bpf = np.empty(DRAWS, dtype=float)
    for i in range(DRAWS):
        starts = rng.integers(0, max_start, size=n_blocks)
        idxs: List[int] = []
        for s in starts:
            idxs.extend(range(int(s), min(int(s) + b, k)))
        idxs = idxs[:k]
        bx[i] = float(np.mean(np.concatenate([m_x[j] for j in idxs])))
        bpf[i] = _pf(np.concatenate([m_r[j] for j in idxs]))

    lo_p, hi_p = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    pf_meth = "nearest" if np.any(np.isinf(bpf)) else "linear"
    x_lo, x_hi = float(np.percentile(bx, lo_p)), float(np.percentile(bx, hi_p))
    pf_lo, pf_hi = float(np.percentile(bpf, lo_p, method=pf_meth)), float(np.percentile(bpf, hi_p, method=pf_meth))

    h1 = [float(t["excess_entry"]) for t in trades if t["entry_date"] < SPLIT_DATE]
    h2 = [float(t["excess_entry"]) for t in trades if t["entry_date"] >= SPLIT_DATE]
    h1_m = float(np.mean(h1)) if h1 else None
    h2_m = float(np.mean(h2)) if h2 else None
    eff = float(effective_n([t["entry_date"] for t in trades], [t["exit_date"] for t in trades]))
    wins = int(np.sum(rocs > 0))
    w_lo, w_hi = wilson_ci(wins, len(trades))
    neg_rocs = rocs[rocs <= 0]
    pos_rocs = rocs[rocs > 0]

    # Yearly breakdown
    by_yr: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_yr.setdefault(t["entry_date"][:4], []).append(t)
    yr_tbl = {
        y: {
            "n": len(ts),
            "win_rate": round(float(np.mean([x["roc"] > 0 for x in ts])), 3),
            "mean_roc": round(float(np.mean([x["roc"] for x in ts])), 4),
            "mean_x": round(float(np.mean([x["excess_entry"] for x in ts])), 4),
        }
        for y, ts in sorted(by_yr.items())
    }

    passed = (
        eff >= 30.0
        and x_lo > 0.0
        and pf_lo > 1.0
        and (h1_m is not None and h1_m > 0.0)
        and (h2_m is not None and h2_m > 0.0)
    )
    verdict = "PASS" if passed else ("UNDERPOWERED" if eff < 30.0 else "CLOSED")

    return {
        "n": len(trades),
        "n_months": k,
        "eff_n": round(eff, 1),
        "win_rate": round(wins / len(trades), 4),
        "win_ci95": [round(w_lo, 4), round(w_hi, 4)],
        "mean_roc": round(float(np.mean(rocs)), 5),
        "avg_win": round(float(np.mean(pos_rocs)), 5) if pos_rocs.size else 0.0,
        "avg_loss": round(float(np.mean(neg_rocs)), 5) if neg_rocs.size else 0.0,
        "worst_5pct": round(float(np.percentile(rocs, 5)), 5),
        "pf": round(_pf(rocs), 3),
        "pf_ci9917": [round(pf_lo, 3), round(pf_hi, 3)],
        "excess_entry": round(float(np.mean(xs)), 5),
        "excess_ci9917": [round(x_lo, 5), round(x_hi, 5)],
        "halves": {
            "2015_2020": round(h1_m, 5) if h1_m is not None else None,
            "2021_2026": round(h2_m, 5) if h2_m is not None else None,
        },
        "med_friction": round(float(np.median([t["friction"] for t in trades])), 4),
        "verdict": verdict,
        "by_year": yr_tbl,
    }


def evaluate(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    dates = trading_wednesdays_11yr()
    spots: Dict[Tuple[str, str], float] = {}
    series_map = official_raw_closes(list(TARGET_ETFS), start="2015-01-01", end="2026-10-15")
    for sym, ser in series_map.items():
        for d_s, px in ser.items():
            if math.isfinite(float(px)) and float(px) > 0:
                spots[(sym, str(d_s)[:10])] = float(px)

    # Stream per-symbol to keep memory super clean across 11 years
    full = repo_path(db_path)
    results: Dict[str, Any] = {}
    monthly_roc_series: Dict[str, Dict[str, float]] = {}

    for sym in TARGET_ETFS:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            raw = con.execute(
                "SELECT date, expiration, strike, type, bid, ask, bid_size, ask_size "
                "FROM opra_chain WHERE symbol = ? AND clock_et = ? AND date >= ? AND date <= ?",
                (sym, CLOCK_ET, START_DATE, TRACK_END),
            ).fetchall()
        finally:
            con.close()

        chains: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        quote_idx: Dict[Tuple[str, str, str, float], Tuple[float, float]] = {}
        for d, exp, k, ot, b, a, bsz, asz in raw:
            ds = str(d)
            exps = str(exp)
            kf = float(k)
            ots = str(ot)
            chains.setdefault(ds, {}).setdefault(exps, []).append(
                {"strike": kf, "type": ots, "bid": b, "ask": a, "bid_size": bsz, "ask_size": asz}
            )
            if b is not None and a is not None and float(b) >= 0 and float(a) >= float(b):
                quote_idx[(ds, exps, ots, kf)] = (float(b), float(a))

        avail_dates = sorted(chains.keys())
        entry_dates = [d for d in avail_dates if d <= ENTRY_END]
        print(f"[5A eval] {sym}: {len(avail_dates)} OPRA Wednesdays ({avail_dates[0] if avail_dates else '?'}..{avail_dates[-1] if avail_dates else '?'})", flush=True)

        for side in ("bull_put", "bear_call"):
            opt_type = "put" if side == "bull_put" else "call"
            trades_by_pol: Dict[str, List[Dict[str, Any]]] = {"limit": [], "crossed": [], "mid": []}
            for d_in in entry_dates:
                by_exp = chains.get(d_in)
                spot_in = spots.get((sym, d_in))
                if not by_exp or not spot_in:
                    continue
                d_in_obj = dt.date.fromisoformat(d_in)
                exp_dtes = {e: float((dt.date.fromisoformat(e) - d_in_obj).days) for e in by_exp}
                o1_exp = choose_expiration(exp_dtes)
                if o1_exp is None:
                    continue
                dte_in = exp_dtes[o1_exp]
                lo_k = 0.65 * spot_in if opt_type == "put" else 0.95 * spot_in
                hi_k = 1.05 * spot_in if opt_type == "put" else 1.40 * spot_in
                rows_side = [
                    r for r in by_exp[o1_exp]
                    if r["type"] == opt_type and lo_k <= float(r["strike"]) <= hi_k
                ]
                qs = quotes_from_rows(rows_side, spot=spot_in, dte=dte_in, opt_type=opt_type, r=0.04)
                legs, _reason = select_legs(
                    qs, spot=spot_in, side=side, expiration=o1_exp, dte=dte_in, max_friction_to_credit=0.25
                )
                if legs is None:
                    continue
                ks, kl = float(legs.short.strike), float(legs.long.strike)
                sb_in, sa_in = float(legs.short.bid or 0), float(legs.short.ask or 0)
                lb_in, la_in = float(legs.long.bid or 0), float(legs.long.ask or 0)
                net_delta = float(legs.net_delta or 0.0)
                width = abs(ks - kl)
                if width <= 0:
                    continue
                exp_obj = dt.date.fromisoformat(o1_exp)
                future_dates = [d for d in avail_dates if d > d_in]

                for pol in ("limit", "crossed", "mid"):
                    cr_in = _fill_credit(sb_in, sa_in, lb_in, la_in, pol)
                    if cr_in <= 0.02 or cr_in >= width:
                        continue
                    car = width - cr_in
                    exit_d: Optional[str] = None
                    close_deb: Optional[float] = None
                    reason_out = "21dte"
                    for df in future_dates:
                        df_obj = dt.date.fromisoformat(df)
                        dte_now = (exp_obj - df_obj).days
                        if dte_now <= 0:
                            sp_ex = spots.get((sym, df)) or spot_in
                            if opt_type == "put":
                                close_deb = max(ks - sp_ex, 0.0) - max(kl - sp_ex, 0.0)
                            else:
                                close_deb = max(sp_ex - ks, 0.0) - max(sp_ex - kl, 0.0)
                            exit_d = df
                            reason_out = "expiry"
                            break
                        sq = quote_idx.get((df, o1_exp, opt_type, ks))
                        lq = quote_idx.get((df, o1_exp, opt_type, kl))
                        if sq and lq:
                            deb = _fill_debit_close(sq[0], sq[1], lq[0], lq[1], pol)
                            if deb <= 0.50 * cr_in:
                                close_deb = max(0.0, deb)
                                exit_d = df
                                reason_out = "tp50"
                                break
                            if dte_now <= 21:
                                close_deb = max(0.0, deb)
                                exit_d = df
                                reason_out = "21dte"
                                break
                    if exit_d is None or close_deb is None:
                        continue
                    sp_ex = spots.get((sym, exit_d)) or spot_in
                    pnl = cr_in - close_deb - FEE_RT_PER_SHARE
                    hedge = net_delta * (sp_ex - spot_in)
                    trades_by_pol[pol].append(
                        {
                            "symbol": sym,
                            "side": side,
                            "entry_date": d_in,
                            "exit_date": exit_d,
                            "expiration": o1_exp,
                            "roc": pnl / car,
                            "excess_entry": (pnl - hedge) / car,
                            "friction": legs.friction_ratio,
                            "exit_reason": reason_out,
                        }
                    )

            arm_key = f"{sym}_{side}"
            lim_res = _boot_arm(trades_by_pol["limit"])
            crs_res = _boot_arm(trades_by_pol["crossed"])
            mid_res = _boot_arm(trades_by_pol["mid"])
            results[arm_key] = {
                "symbol": sym,
                "side": side,
                "limit_k035": lim_res,
                "crossed_k100": {
                    k: crs_res.get(k)
                    for k in ("n", "mean_roc", "pf", "excess_entry", "excess_ci9917", "halves", "verdict")
                },
                "mid_k000": {
                    k: mid_res.get(k)
                    for k in ("n", "mean_roc", "pf", "excess_entry")
                },
            }
            if side == "bull_put":
                by_m: Dict[str, List[float]] = {}
                for t in trades_by_pol["limit"]:
                    by_m.setdefault(t["entry_date"][:7], []).append(float(t["roc"]))
                monthly_roc_series[sym] = {m: float(np.mean(vs)) for m, vs in by_m.items()}

    # Pairwise monthly return correlations across Bull Put sleeves
    corrs: Dict[str, Optional[float]] = {}
    syms = list(TARGET_ETFS)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            common = sorted(set(monthly_roc_series.get(a, {})) & set(monthly_roc_series.get(b, {})))
            if len(common) >= 12:
                va = np.array([monthly_roc_series[a][m] for m in common])
                vb = np.array([monthly_roc_series[b][m] for m in common])
                corrs[f"{a}_{b}"] = round(float(np.corrcoef(va, vb)[0, 1]), 3)
            else:
                corrs[f"{a}_{b}"] = None

    return {
        "registered": "docs/PREREG_OPRA_10YR_TLT_QQQ_IWM_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "window": [START_DATE, TRACK_END],
        "arms": results,
        "bull_put_monthly_corr": corrs,
    }


def main() -> None:
    load_project_env()
    ap = argparse.ArgumentParser(description="Study 5A: 11-Year Institutional OPRA Test of TLT/QQQ/IWM 60-120 DTE")
    ap.add_argument("--download", action="store_true", help="Download incremental 2015-2023 OPRA snapshots")
    ap.add_argument("--max-usd", type=float, default=9.0, help="Hard USD spend cap")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate cached OPRA data without downloading")
    args = ap.parse_args()

    dl = None
    if not args.eval_only:
        dl = download_or_estimate(download=args.download, max_usd=args.max_usd)
        if not args.download:
            print(json.dumps(dl, indent=2))
            return

    out = evaluate()
    if dl is not None:
        out["download"] = dl
    reports = REPO / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_p = reports / f"backtest_opra_10yr_etfs_{ts}.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_p}\n")
    print("Arm               n  effN    Win%    RoC%     PF   AvgLoss   x_lim [99.17% CI]            H1(15-20)  H2(21-26)   x_crs  Verdict")
    print("-" * 128)
    for key, info in out["arms"].items():
        lim = info["limit_k035"]
        crs = info["crossed_k100"]
        if lim.get("n", 0) == 0:
            print(f"{key:15s}  0")
            continue
        ci = lim["excess_ci9917"]
        h1 = lim["halves"]["2015_2020"]
        h2 = lim["halves"]["2021_2026"]
        xc = crs.get("excess_entry")
        print(
            f"{key:15s} {lim['n']:4d} {lim['eff_n']:5.1f} "
            f"{lim['win_rate']*100:6.1f}% {lim['mean_roc']*100:+6.2f}% {lim['pf']:6.2f} "
            f"{lim['avg_loss']*100:+7.1f}%  {lim['excess_entry']*100:+6.2f}% [{ci[0]*100:+6.2f},{ci[1]*100:+6.2f}]%  "
            f"{(f'{h1*100:+6.2f}%' if h1 is not None else '   n/a'):>9s} "
            f"{(f'{h2*100:+6.2f}%' if h2 is not None else '   n/a'):>10s} "
            f"{(f'{xc*100:+6.2f}%' if xc is not None else '   n/a'):>7s}  {lim['verdict']}"
        )
    print("\nBull Put Monthly Return Correlations:", out["bull_put_monthly_corr"])


if __name__ == "__main__":
    main()
