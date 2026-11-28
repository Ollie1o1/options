#!/usr/bin/env python3
"""Preregistered estimator for Study 5B: Intraday Selloff vs. Overnight Gap-Down Filter on `equity_pullback`.

Frozen registration: docs/PREREG_PULLBACK_INTRADAY_VS_GAP_20261127.md
Exact parity with `scripts/backtest_equity_pullbacks.py` on `data/equity_ohlcv.db` (375 liquid stocks, 10 bps).
Also verifies with free Alpaca split-adjusted daily Open/Close bars (`--with-alpaca-open`).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from scripts.backtest_equity_pullbacks import (
    HOLDOUT_START,
    MIN_CLUSTERS,
    TRAIN_END,
    TRAIN_START,
    load_earnings_calendar,
    simulate_symbol_pullbacks,
)
from src.equity_playbooks.backtest import (
    BenchmarkSeries,
    EquityTrade,
    cluster_bootstrap_stats,
    iso_week_key,
    load_equity_ohlcv,
)
from src.paths import load_project_env, repo_path

REPO = Path(__file__).resolve().parents[1]
SEED = 20261127
DRAWS = 10_000
ALPHA_BONF = 0.025  # 2 binding hypotheses -> 97.5% CI


def load_hlc_index(db_path: str = "data/equity_ohlcv.db") -> Dict[str, Dict[str, Tuple[float, float, float, Optional[float]]]]:
    """Return `sym -> date -> (high, low, close, prev_close)` from `data/equity_ohlcv.db`."""
    full = repo_path(db_path)
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    out: Dict[str, Dict[str, Tuple[float, float, float, Optional[float]]]] = {}
    try:
        cur = con.execute("SELECT ticker, date, high, low, close FROM ohlcv ORDER BY ticker, date")
        prev_sym: Optional[str] = None
        prev_c: Optional[float] = None
        for tk, d, hi, lo, cl in cur:
            s = str(tk).upper()
            if s != prev_sym:
                prev_sym = s
                prev_c = None
            if hi is not None and lo is not None and cl is not None:
                out.setdefault(s, {})[str(d)[:10]] = (float(hi), float(lo), float(cl), prev_c)
                prev_c = float(cl)
    finally:
        con.close()
    return out


def fetch_alpaca_daily_open_close(
    symbols: Sequence[str],
    *,
    start: str = "2018-09-15",
    end: str = "2026-11-26",
) -> Dict[str, Dict[str, Tuple[float, float, Optional[float]]]]:
    """Fetch free Alpaca split-adjusted daily (open, close, prev_close) in batches of 40 symbols."""
    load_project_env()
    key = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or "").strip()
    sec = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or "").strip()
    if not key or not sec:
        return {}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    url = "https://data.alpaca.markets/v2/stocks/bars"
    raw_bars: Dict[str, List[Tuple[str, float, float]]] = collections.defaultdict(list)
    sym_list = sorted({s.upper() for s in symbols})

    for i in range(0, len(sym_list), 40):
        batch = sym_list[i : i + 40]
        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "symbols": ",".join(batch),
                "timeframe": "1Day",
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z",
                "adjustment": "all",
                "limit": 10000,
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code != 200:
                break
            js = r.json()
            bars = js.get("bars") or {}
            for sym, rows in bars.items():
                for b in rows:
                    d_s = str(b.get("t", ""))[:10]
                    op = b.get("o")
                    cl = b.get("c")
                    if d_s and op and cl and float(op) > 0 and float(cl) > 0:
                        raw_bars[str(sym).upper()].append((d_s, float(op), float(cl)))
            page_token = js.get("next_page_token")
            if not page_token:
                break

    out: Dict[str, Dict[str, Tuple[float, float, Optional[float]]]] = {}
    for sym, rows in raw_bars.items():
        rows.sort(key=lambda x: x[0])
        prev_c: Optional[float] = None
        m: Dict[str, Tuple[float, float, Optional[float]]] = {}
        for d_s, op, cl in rows:
            m[d_s] = (op, cl, prev_c)
            prev_c = cl
        out[sym] = m
    return out


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def diff_cluster_bootstrap(
    group_a: Sequence[EquityTrade],
    group_b: Sequence[EquityTrade],
    *,
    alpha: float = ALPHA_BONF,
) -> Dict[str, Any]:
    """ISO-week cluster bootstrap for difference in mean net return: mean(A) - mean(B)."""
    if not group_a or not group_b:
        return {"insufficient": True}
    wa: Dict[str, List[float]] = collections.defaultdict(list)
    wb: Dict[str, List[float]] = collections.defaultdict(list)
    for t in group_a:
        wa[iso_week_key(t.entry_date)].append(float(t.net_return))
    for t in group_b:
        wb[iso_week_key(t.entry_date)].append(float(t.net_return))
    weeks = sorted(set(wa) | set(wb))
    k = len(weeks)
    if len(wa) < 10 or len(wb) < 10:
        return {"insufficient": True, "n_a": len(group_a), "n_b": len(group_b)}

    arr_a = [np.array(wa.get(w, []), dtype=float) for w in weeks]
    arr_b = [np.array(wb.get(w, []), dtype=float) for w in weeks]
    rng = np.random.default_rng(SEED)
    diffs = np.empty(DRAWS, dtype=float)
    for i in range(DRAWS):
        idxs = rng.integers(0, k, size=k)
        sa = [arr_a[j] for j in idxs if arr_a[j].size]
        sb = [arr_b[j] for j in idxs if arr_b[j].size]
        if not sa or not sb:
            diffs[i] = np.nan
            continue
        diffs[i] = float(np.mean(np.concatenate(sa))) - float(np.mean(np.concatenate(sb)))
    diffs = diffs[np.isfinite(diffs)]
    lo_p, hi_p = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    pt = float(np.mean([t.net_return for t in group_a])) - float(np.mean([t.net_return for t in group_b]))
    return {
        "insufficient": False,
        "n_a": len(group_a),
        "n_b": len(group_b),
        "weeks_a": len(wa),
        "weeks_b": len(wb),
        "diff_mean": round(pt, 5),
        "ci975": [round(float(np.percentile(diffs, lo_p)), 5), round(float(np.percentile(diffs, hi_p)), 5)],
    }


def summarize_bucket(trades: Sequence[EquityTrade], *, alpha: float = ALPHA_BONF) -> Dict[str, Any]:
    if not trades:
        return {"n": 0}
    rs = np.array([t.net_return for t in trades], dtype=float)
    wins = int(np.sum(rs > 0))
    w_lo, w_hi = wilson_ci(wins, len(trades))
    neg = float(-np.sum(rs[rs < 0]))
    pos = float(np.sum(rs[rs > 0]))
    pf = pos / neg if neg > 0 else float("inf")
    st = cluster_bootstrap_stats(
        trades, lambda t: iso_week_key(t.entry_date), n_boot=DRAWS, seed=SEED, min_clusters=MIN_CLUSTERS, alpha=alpha
    )
    return {
        "n": len(trades),
        "clusters": st.get("n_clusters"),
        "win_rate": round(wins / len(trades), 4),
        "win_ci95": [round(w_lo, 4), round(w_hi, 4)],
        "mean_ret": round(float(np.mean(rs)), 5),
        "mean_ci975": [round(float(st["mean_ret_ci"][0]), 5), round(float(st["mean_ret_ci"][1]), 5)]
        if st.get("mean_ret_ci")
        else None,
        "pf": round(pf, 3),
        "pf_ci975": [round(float(st["pf_ci"][0]), 3), round(float(st["pf_ci"][1]), 3)]
        if st.get("pf_ci")
        else None,
        "avg_win": round(float(np.mean(rs[rs > 0])), 5) if np.any(rs > 0) else 0.0,
        "avg_loss": round(float(np.mean(rs[rs <= 0])), 5) if np.any(rs <= 0) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Study 5B: Intraday Selloff vs Overnight Gap-Down on equity_pullback")
    ap.add_argument("--with-alpaca-open", action="store_true", help="Also verify exact Open/Close via Alpaca 1Day bars")
    args = ap.parse_args()

    book = load_equity_ohlcv("data/equity_ohlcv.db")
    spy_ds, spy_cs, _ = book["SPY"]
    spy = BenchmarkSeries(spy_ds, spy_cs)
    earns = load_earnings_calendar()
    hlc = load_hlc_index("data/equity_ohlcv.db")

    all_trades: List[EquityTrade] = []
    for sym, (ds, cs, vs) in sorted(book.items()):
        ts, _sj = simulate_symbol_pullbacks(sym, ds, cs, vs, spy, earns, stratum="liquid", rt_cost=0.0010, next_day_fill=True)
        all_trades.extend(ts)

    # Classify OHLCV-native buckets on signal day t
    buckets_ohlcv: Dict[str, List[EquityTrade]] = {
        "ALL": [],
        "TRUE_GAP_DOWN": [],
        "NON_GAP": [],
        "INTRADAY_FLUSH": [],
        "MILD_DRIFT": [],
    }
    for tr in all_trades:
        rec = (hlc.get(tr.ticker) or {}).get(tr.signal_date)
        if not rec or rec[3] is None or rec[3] <= 0 or rec[0] <= 0:
            continue
        hi, _lo, cl, prev_c = rec
        buckets_ohlcv["ALL"].append(tr)
        if hi < prev_c:
            buckets_ohlcv["TRUE_GAP_DOWN"].append(tr)
        else:
            buckets_ohlcv["NON_GAP"].append(tr)
            if (cl / hi - 1.0) <= -0.025:
                buckets_ohlcv["INTRADAY_FLUSH"].append(tr)
            else:
                buckets_ohlcv["MILD_DRIFT"].append(tr)

    def _split(ts: Sequence[EquityTrade]) -> Tuple[List[EquityTrade], List[EquityTrade]]:
        tr_l = [t for t in ts if TRAIN_START <= t.entry_date <= TRAIN_END]
        ho_l = [t for t in ts if t.entry_date >= HOLDOUT_START]
        return tr_l, ho_l

    ohlcv_report: Dict[str, Any] = {}
    for name, ts in buckets_ohlcv.items():
        tr_l, ho_l = _split(ts)
        ohlcv_report[name] = {
            "full": summarize_bucket(ts),
            "train": summarize_bucket(tr_l),
            "holdout": summarize_bucket(ho_l),
        }

    # Binding hypotheses H5B-1 & H5B-2
    ng_tr, ng_ho = _split(buckets_ohlcv["NON_GAP"])
    gd_tr, gd_ho = _split(buckets_ohlcv["TRUE_GAP_DOWN"])
    if_tr, if_ho = _split(buckets_ohlcv["INTRADAY_FLUSH"])

    h1_ho = diff_cluster_bootstrap(ng_ho, gd_ho)
    h1_tr = diff_cluster_bootstrap(ng_tr, gd_tr)
    h1_pass = (
        not h1_ho.get("insufficient")
        and h1_ho["ci975"][0] > 0.0
        and not h1_tr.get("insufficient")
        and h1_tr["diff_mean"] > 0.0
    )

    h2_ho = diff_cluster_bootstrap(if_ho, gd_ho)
    h2_tr = diff_cluster_bootstrap(if_tr, gd_tr)
    if_ho_sum = ohlcv_report["INTRADAY_FLUSH"]["holdout"]
    h2_pass = (
        not h2_ho.get("insufficient")
        and h2_ho["ci975"][0] > 0.0
        and not h2_tr.get("insufficient")
        and h2_tr["diff_mean"] > 0.0
        and bool(if_ho_sum.get("mean_ci975"))
        and if_ho_sum["mean_ci975"][0] > 0.0
        and if_ho_sum["pf"] > 1.0
    )

    alpaca_report: Dict[str, Any] = {}
    if args.with_alpaca_open:
        syms_used = sorted({t.ticker for t in all_trades})
        print(f"[5B] Fetching free Alpaca 1Day Open/Close for {len(syms_used)} tickers...", flush=True)
        op_map = fetch_alpaca_daily_open_close(syms_used)
        b_open: Dict[str, List[EquityTrade]] = {
            "OVERNIGHT_GAP_TRAP": [],
            "NO_BIG_GAP": [],
            "INTRADAY_CLEAN": [],
        }
        matched = 0
        for tr in all_trades:
            rec_o = (op_map.get(tr.ticker) or {}).get(tr.signal_date)
            if not rec_o or rec_o[2] is None or rec_o[2] <= 0 or rec_o[0] <= 0:
                continue
            op, cl, pc = rec_o
            matched += 1
            gap = op / pc - 1.0
            intra = cl / op - 1.0
            if gap <= -0.015:
                b_open["OVERNIGHT_GAP_TRAP"].append(tr)
            else:
                b_open["NO_BIG_GAP"].append(tr)
                if intra <= -0.015:
                    b_open["INTRADAY_CLEAN"].append(tr)
        print(f"[5B] Matched {matched}/{len(all_trades)} pullback signals to Alpaca Open/Close", flush=True)
        for name, ts in b_open.items():
            tr_l, ho_l = _split(ts)
            alpaca_report[name] = {
                "full": summarize_bucket(ts),
                "train": summarize_bucket(tr_l),
                "holdout": summarize_bucket(ho_l),
            }
        gt_tr, gt_ho = _split(b_open["OVERNIGHT_GAP_TRAP"])
        ng2_tr, ng2_ho = _split(b_open["NO_BIG_GAP"])
        ic_tr, ic_ho = _split(b_open["INTRADAY_CLEAN"])
        alpaca_report["H5B_1_OPEN_NO_GAP_VS_TRAP"] = {
            "holdout": diff_cluster_bootstrap(ng2_ho, gt_ho),
            "train": diff_cluster_bootstrap(ng2_tr, gt_tr),
        }
        alpaca_report["H5B_2_OPEN_INTRADAY_VS_TRAP"] = {
            "holdout": diff_cluster_bootstrap(ic_ho, gt_ho),
            "train": diff_cluster_bootstrap(ic_tr, gt_tr),
        }

    out = {
        "registered": "docs/PREREG_PULLBACK_INTRADAY_VS_GAP_20261127.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "ohlcv_buckets": ohlcv_report,
        "H5B_1_NON_GAP_VS_GAP": {"verdict": "PASS" if h1_pass else "FAIL", "holdout": h1_ho, "train": h1_tr},
        "H5B_2_FLUSH_VS_GAP": {"verdict": "PASS" if h2_pass else "FAIL", "holdout": h2_ho, "train": h2_tr},
        "alpaca_open_verification": alpaca_report,
    }

    reports = REPO / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts_s = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_p = reports / f"backtest_pullback_intraday_vs_gap_{ts_s}.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_p}\n")
    print("OHLCV-Native Bucket   Split      n    Win%     Mean%    PF   AvgLoss")
    print("-" * 72)
    for name in ("ALL", "TRUE_GAP_DOWN", "NON_GAP", "INTRADAY_FLUSH", "MILD_DRIFT"):
        for sp in ("train", "holdout", "full"):
            d = ohlcv_report[name][sp]
            if not d.get("n"):
                continue
            print(f"{name:20s} {sp:8s} {d['n']:5d} {d['win_rate']*100:6.1f}% {d['mean_ret']*100:+7.2f}% {d['pf']:6.2f} {d['avg_loss']*100:+7.2f}%")
    print(f"\nH5B-1 (NON_GAP vs TRUE_GAP_DOWN): {out['H5B_1_NON_GAP_VS_GAP']['verdict']} | holdout={h1_ho} | train={h1_tr}")
    print(f"H5B-2 (INTRADAY_FLUSH vs TRUE_GAP_DOWN): {out['H5B_2_FLUSH_VS_GAP']['verdict']} | holdout={h2_ho} | train={h2_tr}")
    if alpaca_report:
        print("\nAlpaca Open/Close Verification:")
        for name in ("OVERNIGHT_GAP_TRAP", "NO_BIG_GAP", "INTRADAY_CLEAN"):
            for sp in ("train", "holdout", "full"):
                d = alpaca_report[name][sp]
                if not d.get("n"):
                    continue
                print(f"  {name:20s} {sp:8s} {d['n']:5d} {d['win_rate']*100:6.1f}% {d['mean_ret']*100:+7.2f}% {d['pf']:6.2f} {d['avg_loss']*100:+7.2f}%")
        print("  H5B-1-OPEN:", alpaca_report["H5B_1_OPEN_NO_GAP_VS_TRAP"])
        print("  H5B-2-OPEN:", alpaca_report["H5B_2_OPEN_INTRADAY_VS_TRAP"])


if __name__ == "__main__":
    main()
