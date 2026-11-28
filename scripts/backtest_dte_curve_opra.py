#!/usr/bin/env python3
"""Preregistered estimator for Study 4A: Complete Options DTE Curve (0DTE through 240 DTE) on OPRA `cbbo-1m`.

Frozen registration: docs/PREREG_DTE_CURVE_AND_0DTE_OPRA_20261127.md
Defaults to `--estimate` ($0 dry-run for incremental 10:30 ET 0DTE SPY/QQQ pulls).
Pass `--download --max-usd 2.5` to fetch incremental 10:30 ET 0DTE snapshots & evaluate all 7 buckets.
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
    ETFS,
    FEE_RT_PER_SHARE,
    _fill_credit,
    _fill_debit_close,
    weekly_wednesdays,
)
from src.alloc.validate import effective_n
from src.databento_opra import DEFAULT_DB, fetch_opra_snapshots
from src.intraday.alpaca_bars import load_symbol
from src.intraday.sessions import with_clock
from src.paths import load_project_env, repo_path
from src.radar.wide_spread_rule import LegQuote, quotes_from_rows
from src.raw_closes import official_raw_closes

REPO = Path(__file__).resolve().parents[1]
DATA_5M = REPO / "data" / "intraday_5m"
ZERO_DTE_ETFS: Tuple[str, ...] = ("SPY", "QQQ")
ENTRY_END = "2026-06-24"
ENTRY_END_LONG = "2026-03-31"
SEED = 20261127
DRAWS = 10_000


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def target_0dte_dates() -> List[str]:
    """80 deterministic full-session Wednesdays 2024-01..2026-06 (includes all 24 H4 dates + 56 evenly spaced)."""
    all_w = [d for d in weekly_wednesdays() if d <= ENTRY_END]
    if len(all_w) <= 80:
        return all_w
    # Keep first Wednesday of each month 2024-2025 (already cached at 10:30 from H4) plus evenly spaced others
    h4_months: Dict[str, str] = {}
    for d in all_w:
        ym = d[:7]
        if ym < "2026-01" and ym not in h4_months:
            h4_months[ym] = d
    chosen = set(h4_months.values())
    rem = [d for d in all_w if d not in chosen]
    need = max(0, 80 - len(chosen))
    if need > 0 and rem:
        idxs = np.linspace(0, len(rem) - 1, num=min(need, len(rem)), dtype=int)
        for i in idxs:
            chosen.add(rem[int(i)])
    return sorted(chosen)


def download_or_estimate_0dte(
    *,
    download: bool = False,
    max_usd: float = 2.5,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    dates = target_0dte_dates()
    existing: set[str] = set()
    full = repo_path(db_path)
    if Path(full).exists():
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for (d,) in con.execute(
                "SELECT date FROM opra_chain WHERE clock_et = '10:30' AND symbol IN ('SPY','QQQ') "
                "GROUP BY date HAVING COUNT(DISTINCT symbol) = 2"
            ).fetchall():
                existing.add(str(d))
        except sqlite3.Error:
            pass
        finally:
            con.close()

    todo = [d for d in dates if d not in existing]
    if not todo:
        return {
            "etfs": list(ZERO_DTE_ETFS),
            "dates_total": len(dates),
            "dates_cached": len(dates),
            "dates_needed": 0,
            "total_est_usd": 0.0,
            "downloaded": False,
            "status": "all_cached",
        }

    est = fetch_opra_snapshots(
        ZERO_DTE_ETFS, todo, hhmm_et="10:30", minutes=1, max_usd=max_usd, dry_run=True, db_path=db_path, max_workers=4
    )
    print(f"[4A 0DTE estimate] {len(todo)} dates (SPY+QQQ @ 10:30) = ${est['total_est_usd']:.4f}", flush=True)
    summary: Dict[str, Any] = {
        "etfs": list(ZERO_DTE_ETFS),
        "dates_total": len(dates),
        "dates_cached": len(dates) - len(todo),
        "dates_needed": len(todo),
        "total_est_usd": round(float(est["total_est_usd"]), 4),
        "prior_spend_usd": est.get("prior_spend_usd"),
        "max_usd": max_usd,
        "downloaded": False,
    }
    if not download:
        return summary
    if float(est["total_est_usd"]) > max_usd:
        raise RuntimeError(f"Refused 4A 0DTE download: est ${est['total_est_usd']:.4f} > max_usd ${max_usd:.2f}")

    plan = list(est["plan"])
    total_rows = 0
    for i in range(0, len(plan), 14):
        sub_plan = plan[i : i + 14]
        sub_dates = [p["date"] for p in sub_plan]
        r = fetch_opra_snapshots(
            ZERO_DTE_ETFS,
            sub_dates,
            hhmm_et="10:30",
            minutes=1,
            max_usd=max_usd,
            dry_run=False,
            db_path=db_path,
            max_workers=4,
            precomputed_plan=sub_plan,
        )
        total_rows += int(r["rows_written"])
        print(
            f"[4A 0DTE download] batch {i + len(sub_plan)}/{len(plan)} "
            f"+{r['rows_written']:,} rows cum_spend=${r.get('new_cumulative_spend_usd', 0):.4f}",
            flush=True,
        )
    summary["downloaded"] = True
    summary["rows_written"] = total_rows
    return summary


def _select_vertical(
    quotes: Sequence[LegQuote],
    *,
    spot: float,
    opt_type: str,
    target_abs_delta: float,
    delta_tol: float,
    wing_dollars: float,
    max_friction: float = 0.25,
) -> Optional[Tuple[LegQuote, LegQuote, float]]:
    """Select short (~target_abs_delta) & long (~wing_dollars OTM) two-sided spread."""
    is_put = opt_type == "put"
    cands = [
        q for q in quotes
        if q.two_sided and q.delta is not None and abs(abs(float(q.delta)) - target_abs_delta) <= delta_tol
        and ((is_put and q.strike <= spot) or ((not is_put) and q.strike >= spot))
    ]
    if not cands:
        return None
    short = min(cands, key=lambda q: abs(abs(float(q.delta or 0.0)) - target_abs_delta))
    tgt_k = (short.strike - wing_dollars) if is_put else (short.strike + wing_dollars)
    long_cands = [
        q for q in quotes
        if q.two_sided and ((is_put and q.strike < short.strike) or ((not is_put) and q.strike > short.strike))
    ]
    if not long_cands:
        return None
    long = min(long_cands, key=lambda q: abs(q.strike - tgt_k))
    width = abs(short.strike - long.strike)
    if width <= 0:
        return None
    cr_mid = float(short.mid or 0.0) - float(long.mid or 0.0)
    cr_cross = float(short.bid or 0.0) - float(long.ask or 0.0)
    if cr_mid <= 0.05 or cr_cross <= 0.0 or cr_mid >= width:
        return None
    rt_spr = (float(short.ask or 0.0) - float(short.bid or 0.0)) + (float(long.ask or 0.0) - float(long.bid or 0.0))
    fric = rt_spr / cr_mid
    if fric > max_friction:
        return None
    return short, long, fric


def _intraday_spot_ratios(symbols: Sequence[str]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Map `(symbol, date) -> (ratio_1030_to_close, ratio_1545_to_close)` from local 5m Alpaca bars."""
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for sym in symbols:
        df = load_symbol(sym, DATA_5M)
        if df.empty:
            continue
        w = with_clock(df)
        for d_obj, grp in w.groupby("date", sort=True):
            rth = grp[(grp["mod"] >= 570) & (grp["mod"] <= 955)]
            if len(rth) < 15:
                continue
            mins = rth["mod"].to_numpy()
            cs = rth["c"].to_numpy(dtype=float)
            c_eod = float(cs[-1])
            if c_eod <= 0:
                continue
            i_1030 = int(np.searchsorted(mins, 10 * 60 + 25, side="right")) - 1
            i_1545 = int(np.searchsorted(mins, 15 * 60 + 40, side="right")) - 1
            if i_1030 >= 0 and i_1545 >= 0:
                out[(sym, d_obj.isoformat())] = (float(cs[i_1030]) / c_eod, float(cs[i_1545]) / c_eod)
    return out


def _summarize_bucket(trades: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "verdict": "INSUFFICIENT"}
    by_m: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_m.setdefault(t["entry_date"][:7], []).append(t)
    months = sorted(by_m)
    k = len(months)
    rocs = np.array([float(t["roc"]) for t in trades], dtype=float)
    xs = np.array([float(t["excess_entry"]) for t in trades], dtype=float)
    wins = int(np.sum(rocs > 0.0))
    w_lo, w_hi = wilson_ci(wins, len(trades))
    win_arr = rocs[rocs > 0.0]
    loss_arr = rocs[rocs <= 0.0]

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
    h1 = [float(t["excess_entry"]) for t in trades if t["entry_date"] < "2025-01-01"]
    h2 = [float(t["excess_entry"]) for t in trades if t["entry_date"] >= "2025-01-01"]
    h1_m = float(np.mean(h1)) if h1 else math.nan
    h2_m = float(np.mean(h2)) if h2 else math.nan
    med_fric = float(np.median([float(t["friction"]) for t in trades]))
    eff = float(effective_n([t["entry_date"] for t in trades], [t["exit_date"] for t in trades]))

    passed = bool(x_lo > 0.0 and pf_lo > 1.0 and h1_m > 0.0 and h2_m > 0.0 and med_fric <= 0.15)
    return {
        "n": len(trades),
        "n_months": k,
        "eff_n": round(eff, 1),
        "win_rate": round(wins / len(trades), 4),
        "win_rate_ci95": [round(w_lo, 4), round(w_hi, 4)],
        "mean_roc": round(float(np.mean(rocs)), 5),
        "avg_win_roc": round(float(np.mean(win_arr)), 5) if win_arr.size else 0.0,
        "avg_loss_roc": round(float(np.mean(loss_arr)), 5) if loss_arr.size else 0.0,
        "worst_5pct_roc": round(float(np.percentile(rocs, 5)), 5),
        "pf": round(_pf(rocs), 3),
        "pf_ci": [round(pf_lo, 3), round(pf_hi, 3)],
        "excess_entry": round(float(np.mean(xs)), 5),
        "excess_ci": [round(x_lo, 5), round(x_hi, 5)],
        "halves_excess": {"2024": round(h1_m, 5) if math.isfinite(h1_m) else None, "2025-2026": round(h2_m, 5) if math.isfinite(h2_m) else None},
        "med_friction": round(med_fric, 4),
        "verdict": "PASS" if passed else "CLOSED",
    }


def evaluate(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    dates_w = weekly_wednesdays()
    series_map = official_raw_closes(list(ETFS), start="2023-12-15", end="2026-10-15")
    spots: Dict[Tuple[str, str], float] = {}
    for sym, ser in series_map.items():
        for d_s, px in ser.items():
            if math.isfinite(float(px)) and float(px) > 0:
                spots[(sym, str(d_s)[:10])] = float(px)

    ratios_5m = _intraday_spot_ratios(ZERO_DTE_ETFS)

    full = repo_path(db_path)
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        # 1. Load 15:45 ET chains across all 144 Wednesdays (filter two-sided bid>=0 for fast lookup)
        ph = ",".join("?" * len(dates_w))
        raw_1545 = con.execute(
            f"SELECT symbol, date, expiration, strike, type, bid, ask FROM opra_chain "
            f"WHERE clock_et = '15:45' AND date IN ({ph}) AND bid IS NOT NULL AND ask IS NOT NULL AND ask >= bid",
            dates_w,
        ).fetchall()
        # 2. Load 10:30 ET 0DTE rows for SPY & QQQ (expiration == date)
        raw_0dte_1030 = con.execute(
            "SELECT symbol, date, expiration, strike, type, bid, ask FROM opra_chain "
            "WHERE clock_et = '10:30' AND symbol IN ('SPY','QQQ') AND expiration = date "
            "AND bid IS NOT NULL AND ask IS NOT NULL AND ask >= bid"
        ).fetchall()
    finally:
        con.close()

    chains_1545: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    q_1545: Dict[Tuple[str, str, str, str, float], Tuple[float, float]] = {}
    for sym, d, exp, k, ot, b, a in raw_1545:
        if sym not in ETFS:
            continue
        bf, af, kf = float(b), float(a), float(k)
        q_1545[(str(sym), str(d), str(exp), str(ot), kf)] = (bf, af)
        if bf > 0:
            chains_1545.setdefault((str(sym), str(d)), {}).setdefault(str(exp), []).append(
                {"strike": kf, "type": str(ot), "bid": bf, "ask": af}
            )

    chains_0dte: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for sym, d, _exp, k, ot, b, a in raw_0dte_1030:
        if float(b) > 0:
            chains_0dte.setdefault((str(sym), str(d)), []).append(
                {"strike": float(k), "type": str(ot), "bid": float(b), "ask": float(a)}
            )

    avail_dates = sorted({d for (_s, d) in chains_1545})
    entry_dates = [d for d in avail_dates if d <= ENTRY_END]

    buckets_spec = {
        "1_7D": {"dte_lo": 1, "dte_hi": 7, "tgt": 3, "wing_pct": 0.02, "delta_tol": 0.10, "exit_dte": 0, "hold_expiry": True, "end_entry": ENTRY_END},
        "8_30D": {"dte_lo": 8, "dte_hi": 30, "tgt": 21, "wing_pct": 0.03, "delta_tol": 0.08, "exit_dte": 5, "hold_expiry": False, "end_entry": ENTRY_END},
        "31_60D": {"dte_lo": 31, "dte_hi": 60, "tgt": 45, "wing_pct": 0.05, "delta_tol": 0.08, "exit_dte": 14, "hold_expiry": False, "end_entry": ENTRY_END},
        "60_120D": {"dte_lo": 60, "dte_hi": 120, "tgt": 90, "wing_pct": 0.05, "delta_tol": 0.08, "exit_dte": 21, "hold_expiry": False, "end_entry": ENTRY_END},
        "120_240D": {"dte_lo": 120, "dte_hi": 240, "tgt": 180, "wing_pct": 0.05, "delta_tol": 0.08, "exit_dte": 45, "hold_expiry": False, "end_entry": ENTRY_END_LONG},
    }

    trades_by_bucket: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        b: {"limit": [], "crossed": []} for b in ("0DTE_BP", "0DTE_IC", *buckets_spec.keys())
    }

    # ── Part 1: 0DTE Bull Put & Iron Condor (SPY, QQQ: 10:30 ET -> 15:45 ET) ──
    for (sym, d), rows in sorted(chains_0dte.items()):
        if d > ENTRY_END:
            continue
        eod_spot = spots.get((sym, d))
        rat = ratios_5m.get((sym, d))
        if not eod_spot or not rat:
            continue
        spot_1030 = eod_spot * rat[0]
        spot_1545 = eod_spot * rat[1]
        # 5.25 hours left from 10:30 to 15:45 -> ~0.22 trading-day fraction for BS delta
        dte_frac = 0.25
        puts_near = [r for r in rows if r["type"] == "put" and 0.94 * spot_1030 <= r["strike"] <= 1.01 * spot_1030]
        calls_near = [r for r in rows if r["type"] == "call" and 0.99 * spot_1030 <= r["strike"] <= 1.06 * spot_1030]
        p_qs = quotes_from_rows(puts_near, spot=spot_1030, dte=dte_frac, opt_type="put", r=0.045)
        c_qs = quotes_from_rows(calls_near, spot=spot_1030, dte=dte_frac, opt_type="call", r=0.045)

        bp = _select_vertical(p_qs, spot=spot_1030, opt_type="put", target_abs_delta=0.20, delta_tol=0.08, wing_dollars=5.0, max_friction=0.25)
        if bp is not None:
            sp, lp, fric = bp
            sq_out = q_1545.get((sym, d, d, "put", sp.strike))
            lq_out = q_1545.get((sym, d, d, "put", lp.strike))
            if sq_out and lq_out:
                width = sp.strike - lp.strike
                net_d = (-float(sp.delta or 0.0)) + float(lp.delta or 0.0)
                for pol in ("limit", "crossed"):
                    cr_in = _fill_credit(float(sp.bid or 0), float(sp.ask or 0), float(lp.bid or 0), float(lp.ask or 0), pol)
                    if cr_in <= 0.05 or cr_in >= width:
                        continue
                    deb_out = max(0.0, _fill_debit_close(sq_out[0], sq_out[1], lq_out[0], lq_out[1], pol))
                    car = width - cr_in
                    pnl = cr_in - deb_out - FEE_RT_PER_SHARE
                    hedge = net_d * (spot_1545 - spot_1030)
                    trades_by_bucket["0DTE_BP"][pol].append(
                        {
                            "bucket": "0DTE_BP", "symbol": sym, "entry_date": d, "exit_date": d,
                            "dte": 0.25, "abs_delta": abs(float(sp.delta or 0.20)),
                            "credit_to_width": cr_in / width, "friction": fric,
                            "roc": pnl / car, "excess_entry": (pnl - hedge) / car,
                        }
                    )

        bc = _select_vertical(c_qs, spot=spot_1030, opt_type="call", target_abs_delta=0.20, delta_tol=0.08, wing_dollars=5.0, max_friction=0.25)
        if bp is not None and bc is not None:
            sp, lp, _ = bp
            sc, lc, _ = bc
            sq_p = q_1545.get((sym, d, d, "put", sp.strike))
            lq_p = q_1545.get((sym, d, d, "put", lp.strike))
            sq_c = q_1545.get((sym, d, d, "call", sc.strike))
            lq_c = q_1545.get((sym, d, d, "call", lc.strike))
            if sq_p and lq_p and sq_c and lq_c:
                cr_mid_ic = (float(sp.mid or 0) - float(lp.mid or 0)) + (float(sc.mid or 0) - float(lc.mid or 0))
                rt_ic = (
                    (float(sp.ask or 0) - float(sp.bid or 0)) + (float(lp.ask or 0) - float(lp.bid or 0))
                    + (float(sc.ask or 0) - float(sc.bid or 0)) + (float(lc.ask or 0) - float(lc.bid or 0))
                )
                fric_ic = rt_ic / cr_mid_ic if cr_mid_ic > 0 else 9.0
                width_ic = max(sp.strike - lp.strike, lc.strike - sc.strike)
                if fric_ic <= 0.25 and width_ic > 0:
                    net_d = (-float(sp.delta or 0.0) + float(lp.delta or 0.0)) + (-float(sc.delta or 0.0) + float(lc.delta or 0.0))
                    for pol in ("limit", "crossed"):
                        cr_in = _fill_credit(float(sp.bid or 0), float(sp.ask or 0), float(lp.bid or 0), float(lp.ask or 0), pol) + _fill_credit(float(sc.bid or 0), float(sc.ask or 0), float(lc.bid or 0), float(lc.ask or 0), pol)
                        if cr_in <= 0.10 or cr_in >= width_ic:
                            continue
                        deb_out = max(0.0, _fill_debit_close(sq_p[0], sq_p[1], lq_p[0], lq_p[1], pol)) + max(0.0, _fill_debit_close(sq_c[0], sq_c[1], lq_c[0], lq_c[1], pol))
                        car = width_ic - cr_in
                        pnl = cr_in - deb_out - 2.0 * FEE_RT_PER_SHARE
                        hedge = net_d * (spot_1545 - spot_1030)
                        trades_by_bucket["0DTE_IC"][pol].append(
                            {
                                "bucket": "0DTE_IC", "symbol": sym, "entry_date": d, "exit_date": d,
                                "dte": 0.25, "abs_delta": 0.20,
                                "credit_to_width": cr_in / width_ic, "friction": fric_ic,
                                "roc": pnl / car, "excess_entry": (pnl - hedge) / car,
                            }
                        )

    # ── Part 2: Multi-DTE Term Structure (1-7D through 120-240D, 6 ETFs) ─────
    for sym in ETFS:
        ser_sym = series_map[sym]
        for d_in in entry_dates:
            by_exp = chains_1545.get((sym, d_in))
            spot_in = spots.get((sym, d_in))
            if not by_exp or not spot_in:
                continue
            d_in_obj = dt.date.fromisoformat(d_in)
            exp_dtes = {e: (dt.date.fromisoformat(e) - d_in_obj).days for e in by_exp}
            future_dates = [d for d in avail_dates if d > d_in]

            for bname, cfg in buckets_spec.items():
                if d_in > str(cfg["end_entry"]):
                    continue
                cands_exp = [
                    (abs(dte - int(cfg["tgt"])), e, dte)
                    for e, dte in exp_dtes.items()
                    if int(cfg["dte_lo"]) <= dte <= int(cfg["dte_hi"])
                ]
                if not cands_exp:
                    continue
                cands_exp.sort()
                _, exp_s, dte_in = cands_exp[0]
                lo_k = (0.85 if dte_in <= 7 else (0.75 if dte_in <= 30 else 0.65)) * spot_in
                puts = [r for r in by_exp[exp_s] if r["type"] == "put" and lo_k <= r["strike"] <= 1.03 * spot_in]
                if len(puts) < 4:
                    continue
                qs = quotes_from_rows(puts, spot=spot_in, dte=float(dte_in), opt_type="put", r=0.045)
                sel = _select_vertical(
                    qs,
                    spot=spot_in,
                    opt_type="put",
                    target_abs_delta=0.25,
                    delta_tol=float(cfg["delta_tol"]),
                    wing_dollars=max(1.0, spot_in * float(cfg["wing_pct"])),
                    max_friction=0.25,
                )
                if sel is None:
                    continue
                sp, lp, fric = sel
                width = sp.strike - lp.strike
                net_d = (-float(sp.delta or 0.0)) + float(lp.delta or 0.0)
                exp_obj = dt.date.fromisoformat(exp_s)

                for pol in ("limit", "crossed"):
                    cr_in = _fill_credit(float(sp.bid or 0), float(sp.ask or 0), float(lp.bid or 0), float(lp.ask or 0), pol)
                    if cr_in <= 0.05 or cr_in >= width:
                        continue
                    car = width - cr_in
                    exit_d: Optional[str] = None
                    close_deb: Optional[float] = None
                    fee = 0.5 * FEE_RT_PER_SHARE if cfg["hold_expiry"] else FEE_RT_PER_SHARE

                    if cfg["hold_expiry"]:
                        # Settle at official close on expiration (or latest trading day on/before expiry)
                        sp_ex: Optional[float] = None
                        for off in range(4):
                            cd = exp_obj - dt.timedelta(days=off)
                            v = ser_sym.get(cd)
                            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                                sp_ex = float(v)
                                exit_d = cd.isoformat()
                                break
                        if sp_ex is None or exit_d is None:
                            continue
                        close_deb = max(sp.strike - sp_ex, 0.0) - max(lp.strike - sp_ex, 0.0)
                    else:
                        sp_ex = None
                        for df in future_dates:
                            dte_now = (exp_obj - dt.date.fromisoformat(df)).days
                            if dte_now <= 0:
                                sp_ex = spots.get((sym, df)) or spot_in
                                close_deb = max(sp.strike - sp_ex, 0.0) - max(lp.strike - sp_ex, 0.0)
                                exit_d = df
                                fee = 0.5 * FEE_RT_PER_SHARE
                                break
                            sq = q_1545.get((sym, df, exp_s, "put", sp.strike))
                            lq = q_1545.get((sym, df, exp_s, "put", lp.strike))
                            if sq and lq:
                                deb = _fill_debit_close(sq[0], sq[1], lq[0], lq[1], pol)
                                if deb <= 0.50 * cr_in or dte_now <= int(cfg["exit_dte"]):
                                    close_deb = max(0.0, deb)
                                    exit_d = df
                                    sp_ex = spots.get((sym, df)) or spot_in
                                    break
                        if exit_d is None or close_deb is None or sp_ex is None:
                            continue

                    pnl = cr_in - close_deb - fee
                    hedge = net_d * (sp_ex - spot_in)
                    trades_by_bucket[bname][pol].append(
                        {
                            "bucket": bname, "symbol": sym, "entry_date": d_in, "exit_date": exit_d,
                            "dte": float(dte_in), "abs_delta": abs(float(sp.delta or 0.25)),
                            "credit_to_width": cr_in / width, "friction": fric,
                            "roc": pnl / car, "excess_entry": (pnl - hedge) / car,
                        }
                    )

    alpha_bonf = 0.05 / 6.0  # Bonferroni across 6 non-reference buckets -> 99.17% CI
    summary: Dict[str, Any] = {}
    for bname, pol_map in trades_by_bucket.items():
        summary[bname] = {
            "limit_k35": _summarize_bucket(pol_map["limit"], alpha=alpha_bonf),
            "crossed_k100": _summarize_bucket(pol_map["crossed"], alpha=alpha_bonf),
        }

    # Persist per-trade limit records for Study 4B walk-forward probability calibration
    trades_out = REPO / "reports" / "dte_curve_trades_latest.json"
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    trades_out.write_text(
        json.dumps({b: m["limit"] for b, m in trades_by_bucket.items()}, indent=2),
        encoding="utf-8",
    )

    return {
        "registered": "docs/PREREG_DTE_CURVE_AND_0DTE_OPRA_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "alpha_bonf": round(alpha_bonf, 5),
        "buckets": summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Study 4A: Complete Options DTE Curve (0DTE to 240D) on OPRA")
    ap.add_argument("--download", action="store_true", help="Download incremental 10:30 ET 0DTE SPY/QQQ snapshots")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate on local databento_opra.db without downloading")
    ap.add_argument("--max-usd", type=float, default=2.5)
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(list(argv) if argv is not None else None)

    if not a.evaluate:
        dl = download_or_estimate_0dte(download=a.download, max_usd=a.max_usd, db_path=a.db)
        print(json.dumps(dl, indent=2))
        if not a.download:
            return 0

    res = evaluate(db_path=a.db)
    print("\n=== Study 4A: Complete Options DTE Curve (k=0.35 Limit vs Crossed) ===")
    for bname, st in res["buckets"].items():
        lim = st["limit_k35"]
        cr = st["crossed_k100"]
        if lim.get("n", 0) == 0:
            print(f"  {bname:10s}: n=0")
            continue
        print(
            f"  {bname:10s}: n={lim['n']:4d} (effN={lim['eff_n']:4.1f}) "
            f"Win={lim['win_rate']*100:5.1f}% CI[{lim['win_rate_ci95'][0]*100:4.1f},{lim['win_rate_ci95'][1]*100:4.1f}] "
            f"RoC={lim['mean_roc']*100:+5.2f}% Loss={lim['avg_loss_roc']*100:+6.2f}% PF={lim['pf']:4.2f} "
            f"x_lim={lim['excess_entry']*100:+5.2f}% CI{lim['excess_ci']} "
            f"x_crs={cr.get('excess_entry', 0)*100:+5.2f}% fric={lim['med_friction']*100:4.1f}% | {lim['verdict']}"
        )

    out = REPO / "reports" / f"backtest_dte_curve_opra_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
