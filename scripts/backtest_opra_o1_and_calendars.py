#!/usr/bin/env python3
"""Preregistered estimator for Study 3A: 2024-2026 OOS Multi-ETF Test O1 & ATM 1M/3M Calendars on OPRA `cbbo-1m`.

Frozen registration: docs/PREREG_MULTI_ETF_O1_AND_CALENDARS_OPRA_20261127.md
Defaults to `--estimate` ($0 dry-run). Pass `--download --max-usd 6.5` to pull incremental Wednesdays & evaluate.
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

from src.alloc.validate import effective_n
from src.databento_opra import DEFAULT_DB, fetch_opra_snapshots
from src.intraday.alpaca_bars import fetch_calendar
from src.paths import load_project_env, repo_path
from src.radar.wide_spread_rule import choose_expiration, quotes_from_rows, select_legs
from src.raw_closes import official_raw_closes

REPO = Path(__file__).resolve().parents[1]
ETFS: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLE", "GLD", "TLT")
CLOCK_ET = "15:45"
ENTRY_END = "2026-06-24"
TRACK_END = "2026-09-30"
FEE_RT_PER_SHARE = 0.026  # $0.65/contract * 4 legs (2 open + 2 close) / 100 shares
SEED = 20261127
DRAWS = 10_000


def weekly_wednesdays() -> List[str]:
    """Every full-session (16:00 ET close) Wednesday 2024-01-03..2026-09-30 (roll Thu/Tue if holiday/early close)."""
    cal = fetch_calendar()
    full_days = set(cal[cal["close"] == "16:00"]["date"].tolist())
    out: List[str] = []
    d = dt.date(2024, 1, 3)
    end_d = dt.date.fromisoformat(TRACK_END)
    while d <= end_d:
        cur = d
        # Roll forward up to 2 days (Thu/Fri), else back 1 day (Tue) to stay inside the same ISO week
        chosen: Optional[dt.date] = None
        for off in (0, 1, 2, -1):
            cand = cur + dt.timedelta(days=off)
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
    max_usd: float = 6.5,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    dates = weekly_wednesdays()
    existing: set[str] = set()
    full = repo_path(db_path)
    if Path(full).exists():
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for (d,) in con.execute("SELECT DISTINCT date FROM opra_chain WHERE clock_et = ?", (CLOCK_ET,)).fetchall():
                existing.add(str(d))
        except sqlite3.Error:
            pass
        finally:
            con.close()

    todo = [d for d in dates if d not in existing]
    if not todo:
        return {"dates_total": len(dates), "dates_needed": 0, "total_est_usd": 0.0, "downloaded": False, "status": "all_cached"}

    est = fetch_opra_snapshots(
        ETFS, todo, hhmm_et=CLOCK_ET, minutes=1, max_usd=max_usd, dry_run=True, db_path=db_path, max_workers=4
    )
    print(f"[3A estimate] {len(todo)} dates = ${est['total_est_usd']:.4f} (prior=${est.get('prior_spend_usd', 0):.4f})", flush=True)
    summary: Dict[str, Any] = {
        "etfs": list(ETFS),
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
        raise RuntimeError(f"Refused Study 3A download: est ${est['total_est_usd']:.4f} > max_usd ${max_usd:.2f}")

    # Download in batches of 10 dates (max_workers=4) so SQLite commits frequently and progress is visible
    plan = list(est["plan"])
    total_rows = 0
    for i in range(0, len(plan), 10):
        sub_plan = plan[i : i + 10]
        sub_dates = [p["date"] for p in sub_plan]
        r = fetch_opra_snapshots(
            ETFS,
            sub_dates,
            hhmm_et=CLOCK_ET,
            minutes=1,
            max_usd=max_usd,
            dry_run=False,
            db_path=db_path,
            max_workers=4,
            precomputed_plan=sub_plan,
        )
        total_rows += int(r["rows_written"])
        print(
            f"[3A download] batch {i + len(sub_plan)}/{len(plan)} "
            f"+{r['rows_written']:,} rows cum_spend=${r.get('new_cumulative_spend_usd', 0):.4f}",
            flush=True,
        )
    summary["downloaded"] = True
    summary["rows_written"] = total_rows
    return summary


def _fill_credit(s_bid: float, s_ask: float, l_bid: float, l_ask: float, policy: str, k: float = 0.35) -> float:
    sm, lm = 0.5 * (s_bid + s_ask), 0.5 * (l_bid + l_ask)
    if policy == "mid":
        return sm - lm
    if policy == "limit":
        return (sm - k * (sm - s_bid)) - (lm + k * (l_ask - lm))
    return s_bid - l_ask


def _fill_debit_close(s_bid: float, s_ask: float, l_bid: float, l_ask: float, policy: str, k: float = 0.35) -> float:
    sm, lm = 0.5 * (s_bid + s_ask), 0.5 * (l_bid + l_ask)
    if policy == "mid":
        return sm - lm
    if policy == "limit":
        return (sm + k * (s_ask - sm)) - (lm - k * (lm - l_bid))
    return s_ask - l_bid


def _boot_trades(trades: List[Dict[str, Any]], metric_key: str, *, alpha: float = 0.025) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "verdict": "INSUFFICIENT"}
    by_m: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_m.setdefault(t["entry_date"][:7], []).append(t)
    months = sorted(by_m)
    k = len(months)
    vals = np.array([float(t[metric_key]) for t in trades], dtype=float)
    rocs = np.array([float(t["roc"]) for t in trades], dtype=float)

    def _pf(a: np.ndarray) -> float:
        neg = float(-np.sum(a[a < 0.0]))
        pos = float(np.sum(a[a > 0.0]))
        return pos / neg if neg > 0 else float("inf")

    m_v = [np.array([float(t[metric_key]) for t in by_m[m]], dtype=float) for m in months]
    m_r = [np.array([float(t["roc"]) for t in by_m[m]], dtype=float) for m in months]

    rng = np.random.default_rng(SEED)
    b = max(1, min(3, k))
    n_blocks = int(math.ceil(k / b))
    max_start = max(1, k - b + 1)
    bv = np.empty(DRAWS, dtype=float)
    bpf = np.empty(DRAWS, dtype=float)
    for i in range(DRAWS):
        starts = rng.integers(0, max_start, size=n_blocks)
        idxs: List[int] = []
        for s in starts:
            idxs.extend(range(int(s), min(int(s) + b, k)))
        idxs = idxs[:k]
        bv[i] = float(np.mean(np.concatenate([m_v[j] for j in idxs])))
        bpf[i] = _pf(np.concatenate([m_r[j] for j in idxs]))

    lo_p, hi_p = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    pf_meth = "nearest" if np.any(np.isinf(bpf)) else "linear"
    h1 = [float(t[metric_key]) for t in trades if t["entry_date"] < "2025-01-01"]
    h2 = [float(t[metric_key]) for t in trades if t["entry_date"] >= "2025-01-01"]
    eff = float(effective_n([t["entry_date"] for t in trades], [t["exit_date"] for t in trades]))
    lo, hi = float(np.percentile(bv, lo_p)), float(np.percentile(bv, hi_p))
    pf_lo, pf_hi = float(np.percentile(bpf, lo_p, method=pf_meth)), float(np.percentile(bpf, hi_p, method=pf_meth))
    return {
        "n": len(trades),
        "n_months": k,
        "eff_n": round(eff, 1),
        "mean_metric": round(float(np.mean(vals)), 5),
        "ci975": [round(lo, 5), round(hi, 5)],
        "mean_roc": round(float(np.mean(rocs)), 5),
        "win_rate": round(float(np.mean(rocs > 0)), 3),
        "pf": round(_pf(rocs), 3),
        "pf_ci975": [round(pf_lo, 3), round(pf_hi, 3)],
        "halves": {
            "2024": round(float(np.mean(h1)), 5) if h1 else None,
            "2025-2026": round(float(np.mean(h2)), 5) if h2 else None,
        },
        "med_friction": round(float(np.median([t["friction"] for t in trades])), 4),
    }


def evaluate(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    dates = weekly_wednesdays()
    # Load official split-unadjusted spot closes
    spots: Dict[Tuple[str, str], float] = {}
    series_map = official_raw_closes(list(ETFS), start="2023-12-15", end="2026-10-15")
    for sym, ser in series_map.items():
        for d_s, px in ser.items():
            if math.isfinite(float(px)) and float(px) > 0:
                spots[(sym, str(d_s)[:10])] = float(px)

    # Load all 15:45 ET OPRA rows indexed by (symbol, date) -> exp -> (type, strike) -> (bid, ask)
    full = repo_path(db_path)
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        ph = ",".join("?" * len(dates))
        raw = con.execute(
            f"SELECT symbol, date, expiration, strike, type, bid, ask, bid_size, ask_size "
            f"FROM opra_chain WHERE clock_et = ? AND date IN ({ph})",
            [CLOCK_ET] + dates,
        ).fetchall()
    finally:
        con.close()

    chains: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    quote_idx: Dict[Tuple[str, str, str, str, float], Tuple[float, float]] = {}
    for sym, d, exp, k, ot, b, a, bsz, asz in raw:
        if sym not in ETFS:
            continue
        chains.setdefault((str(sym), str(d)), {}).setdefault(str(exp), []).append(
            {"strike": float(k), "type": str(ot), "bid": b, "ask": a, "bid_size": bsz, "ask_size": asz}
        )
        if b is not None and a is not None and float(b) >= 0 and float(a) >= float(b):
            quote_idx[(str(sym), str(d), str(exp), str(ot), float(k))] = (float(b), float(a))

    avail_dates = sorted({d for (_s, d) in chains})
    entry_dates = [d for d in avail_dates if d <= ENTRY_END]

    # ── Arm 3A-1: Multi-ETF Test O1-M Bull Put (2024-2026 OOS) ───────────────
    o1_trades: Dict[str, List[Dict[str, Any]]] = {"limit": [], "crossed": [], "mid": []}
    for sym in ETFS:
        for ei, d_in in enumerate(entry_dates):
            by_exp = chains.get((sym, d_in))
            spot_in = spots.get((sym, d_in))
            if not by_exp or not spot_in:
                continue
            d_in_obj = dt.date.fromisoformat(d_in)
            exp_dtes = {e: float((dt.date.fromisoformat(e) - d_in_obj).days) for e in by_exp}
            o1_exp = choose_expiration(exp_dtes)
            if o1_exp is None:
                continue
            dte_in = exp_dtes[o1_exp]
            put_rows = [
                r for r in by_exp[o1_exp]
                if r["type"] == "put" and 0.65 * spot_in <= float(r["strike"]) <= 1.05 * spot_in
            ]
            qs = quotes_from_rows(put_rows, spot=spot_in, dte=dte_in, opt_type="put", r=0.045)
            legs, _reason = select_legs(qs, spot=spot_in, side="bull_put", expiration=o1_exp, dte=dte_in, max_friction_to_credit=0.25)
            if legs is None:
                continue
            ks, kl = float(legs.short.strike), float(legs.long.strike)
            sb_in, sa_in = float(legs.short.bid or 0), float(legs.short.ask or 0)
            lb_in, la_in = float(legs.long.bid or 0), float(legs.long.ask or 0)
            net_delta = float(legs.net_delta or 0.0)
            width = ks - kl
            exp_obj = dt.date.fromisoformat(o1_exp)

            # Subsequent dates after entry
            future_dates = [d for d in avail_dates if d > d_in]
            for policy in ("limit", "crossed", "mid"):
                cr_in = _fill_credit(sb_in, sa_in, lb_in, la_in, policy)
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
                        close_deb = max(ks - sp_ex, 0.0) - max(kl - sp_ex, 0.0)
                        exit_d = df
                        reason_out = "expiry"
                        break
                    sq = quote_idx.get((sym, df, o1_exp, "put", ks))
                    lq = quote_idx.get((sym, df, o1_exp, "put", kl))
                    if sq and lq:
                        deb = _fill_debit_close(sq[0], sq[1], lq[0], lq[1], policy)
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
                o1_trades[policy].append(
                    {
                        "symbol": sym,
                        "entry_date": d_in,
                        "exit_date": exit_d,
                        "expiration": o1_exp,
                        "roc": pnl / car,
                        "excess_entry": (pnl - hedge) / car,
                        "friction": legs.friction_ratio,
                        "exit_reason": reason_out,
                    }
                )

    # ── Arm 3A-2: ATM 1M/3M Call Calendar Spreads ────────────────────────────
    cal_trades: Dict[str, List[Dict[str, Any]]] = {"limit": [], "crossed": [], "mid": []}
    for sym in ETFS:
        for d_in in entry_dates:
            by_exp = chains.get((sym, d_in))
            spot_in = spots.get((sym, d_in))
            if not by_exp or not spot_in:
                continue
            d_in_obj = dt.date.fromisoformat(d_in)
            exp_dtes = {e: (dt.date.fromisoformat(e) - d_in_obj).days for e in by_exp}
            fronts = [(abs(v - 35), e, v) for e, v in exp_dtes.items() if 25 <= v <= 45]
            backs = [(abs(v - 90), e, v) for e, v in exp_dtes.items() if 75 <= v <= 125]
            if not fronts or not backs:
                continue
            fronts.sort()
            backs.sort()
            _, f_exp, _ = fronts[0]
            _, b_exp, _ = backs[0]
            f_calls = {
                r["strike"]: r for r in by_exp[f_exp]
                if r["type"] == "call" and r.get("bid") and r.get("ask") and float(r["bid"]) > 0 and float(r["ask"]) >= float(r["bid"])
            }
            b_calls = {
                r["strike"]: r for r in by_exp[b_exp]
                if r["type"] == "call" and r.get("bid") and r.get("ask") and float(r["bid"]) > 0 and float(r["ask"]) >= float(r["bid"])
            }
            common_k = [k for k in f_calls if k in b_calls and abs(k / spot_in - 1.0) <= 0.03]
            if not common_k:
                continue
            k_atm = min(common_k, key=lambda k: abs(k - spot_in))
            fc, bc = f_calls[k_atm], b_calls[k_atm]
            fb, fa = float(fc["bid"]), float(fc["ask"])
            bb, ba = float(bc["bid"]), float(bc["ask"])
            deb_mid = 0.5 * (bb + ba) - 0.5 * (fb + fa)
            deb_cross = ba - fb
            rt_spr = (fa - fb) + (ba - bb)
            if deb_mid <= 0.10 or deb_cross <= 0 or (rt_spr / deb_mid) > 0.25:
                continue
            fric_cal = rt_spr / deb_mid
            f_exp_obj = dt.date.fromisoformat(f_exp)
            future_dates = [d for d in avail_dates if d > d_in]

            for policy in ("limit", "crossed", "mid"):
                # Enter calendar = BUY back call, SELL front call -> pay debit
                deb_in = _fill_debit_close(bb, ba, fb, fa, policy)  # buys back, sells front
                if deb_in <= 0.05:
                    continue
                exit_d = None
                cr_out = None
                reason_out = "7dte"
                for df in future_dates:
                    dte_f = (f_exp_obj - dt.date.fromisoformat(df)).days
                    fq = quote_idx.get((sym, df, f_exp, "call", k_atm))
                    bq = quote_idx.get((sym, df, b_exp, "call", k_atm))
                    if fq and bq:
                        # Close calendar = SELL back call, BUY front call -> receive credit
                        val = _fill_credit(bq[0], bq[1], fq[0], fq[1], policy)
                        if val >= 1.25 * deb_in:
                            cr_out, exit_d, reason_out = val, df, "tp25"
                            break
                        if val <= 0.75 * deb_in:
                            cr_out, exit_d, reason_out = val, df, "sl25"
                            break
                        if dte_f <= 7:
                            cr_out, exit_d, reason_out = val, df, "7dte"
                            break
                    elif dte_f <= 0:
                        break
                if exit_d is None or cr_out is None:
                    continue
                pnl = cr_out - deb_in - FEE_RT_PER_SHARE
                cal_trades[policy].append(
                    {
                        "symbol": sym,
                        "entry_date": d_in,
                        "exit_date": exit_d,
                        "front_exp": f_exp,
                        "back_exp": b_exp,
                        "roc": pnl / deb_in,
                        "friction": fric_cal,
                        "exit_reason": reason_out,
                    }
                )

    # Summarize results
    o1_summary: Dict[str, Any] = {}
    for pol in ("limit", "crossed", "mid"):
        trs = o1_trades[pol]
        o1_summary[pol] = {
            "pooled_all_etfs": _boot_trades(trs, "excess_entry", alpha=0.025),
            "by_etf": {
                sym: _boot_trades([t for t in trs if t["symbol"] == sym], "excess_entry", alpha=0.025)
                for sym in ETFS
            },
        }
    cal_summary: Dict[str, Any] = {}
    for pol in ("limit", "crossed", "mid"):
        trs = cal_trades[pol]
        cal_summary[pol] = {
            "pooled_all_etfs": _boot_trades(trs, "roc", alpha=0.025),
            "by_etf": {
                sym: _boot_trades([t for t in trs if t["symbol"] == sym], "roc", alpha=0.025)
                for sym in ETFS
            },
        }

    # Apply frozen verdicts
    o1_lim = o1_summary["limit"]["pooled_all_etfs"]
    if o1_lim.get("n", 0) == 0:
        o1_verdict = "INSUFFICIENT"
    elif o1_lim["ci975"][0] > 0 and o1_lim["pf"] > 1.0 and (o1_lim["halves"]["2024"] or 0) > 0 and (o1_lim["halves"]["2025-2026"] or 0) > 0:
        o1_verdict = "PASS_OOS"
    elif o1_lim["mean_metric"] > 0:
        o1_verdict = "UNDERPOWERED_CONSISTENT"
    else:
        o1_verdict = "FAIL_OOS"

    cal_lim = cal_summary["limit"]["pooled_all_etfs"]
    if cal_lim.get("n", 0) == 0:
        cal_verdict = "INSUFFICIENT"
    elif cal_lim["ci975"][0] > 0 and cal_lim["pf_ci975"][0] > 1.0 and (cal_lim["halves"]["2024"] or 0) > 0 and (cal_lim["halves"]["2025-2026"] or 0) > 0:
        cal_verdict = "PASS"
    else:
        cal_verdict = "CLOSED"

    return {
        "registered": "docs/PREREG_MULTI_ETF_O1_AND_CALENDARS_OPRA_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "available_wednesdays": len(avail_dates),
        "entry_window": [entry_dates[0], entry_dates[-1]] if entry_dates else [],
        "O1_MULTI_ETF_OOS": {"verdict_limit_k35": o1_verdict, **o1_summary},
        "ATM_CALENDAR_1M3M": {"verdict_limit_k35": cal_verdict, **cal_summary},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Study 3A: 2024-2026 OOS Multi-ETF Test O1 & Calendars on OPRA")
    ap.add_argument("--download", action="store_true", help="Download incremental Wednesdays if under --max-usd")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate on local databento_opra.db without downloading")
    ap.add_argument("--max-usd", type=float, default=6.5)
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(list(argv) if argv is not None else None)

    if not a.evaluate:
        dl = download_or_estimate(download=a.download, max_usd=a.max_usd, db_path=a.db)
        print(json.dumps(dl, indent=2))
        if not a.download:
            return 0

    res = evaluate(db_path=a.db)
    print("\n=== Study 3A: O1 Multi-ETF OOS (2024-2026) ===")
    print("Verdict (k=0.35):", res["O1_MULTI_ETF_OOS"]["verdict_limit_k35"])
    for pol in ("limit", "crossed", "mid"):
        print(f"  [{pol}] pooled:", res["O1_MULTI_ETF_OOS"][pol]["pooled_all_etfs"])
    print("\n=== Study 3A: ATM 1M/3M Call Calendars ===")
    print("Verdict (k=0.35):", res["ATM_CALENDAR_1M3M"]["verdict_limit_k35"])
    for pol in ("limit", "crossed", "mid"):
        print(f"  [{pol}] pooled:", res["ATM_CALENDAR_1M3M"][pol]["pooled_all_etfs"])

    out = REPO / "reports" / f"backtest_opra_o1_calendars_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
