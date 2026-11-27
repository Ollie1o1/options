"""Audit: how wrong is the inferred underlying price in the DoltHub studies?

Plan (frozen before measuring): docs/AUDIT_DOLT_SPOT_PLAN_20261127.md (incl. amendment A1).

    python -m scripts.audit_dolt_spot                 # measurements 1-3 (fast)
    python -m scripts.audit_dolt_spot --rerun          # + measurement 4 (re-runs the history study twice)

Truth = official close on the strikes' own (unadjusted) basis: yfinance `Close`
(auto_adjust=False, split-adjusted only) multiplied back by every later split.
Alpaca raw daily closes cross-check that un-splitting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.selector_study import dolt_history as dh

REPO = Path(__file__).resolve().parents[1]
SYMBOLS = ["AAPL", "AMZN", "DIA", "GOOGL", "JPM", "META", "MSFT", "NVDA", "SPY", "XLE"]
DOLT_DB = REPO / "data" / "dolt_options.db"
STUDY_DIR = REPO / "data" / "selector_study_dolt"


# ------------------------------------------------------------------ truth
from src.raw_closes import official_raw_closes, on_or_before, unsplit  # noqa: E402,F401
from src.raw_closes import settlement_close as settlement_truth  # noqa: E402


def err_stats(errs: Sequence[float]) -> Dict[str, Any]:
    a = np.abs(np.asarray([e for e in errs if e is not None and np.isfinite(e)], dtype=float))
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median_pct": float(np.median(a) * 100), "p95_pct": float(np.quantile(a, .95) * 100),
            "max_pct": float(a.max() * 100), "over_0p25pct": int((a > 0.0025).sum())}


# ------------------------------------------------------------------ measurements
def cross_check_alpaca(truth: Dict[str, pd.Series]) -> Dict[str, Any]:
    from src.intraday.alpaca_bars import fetch_daily
    a = fetch_daily(list(truth), "2019-01-01", "2026-06-30", adjustment="raw")
    out = {}
    for s, tr in truth.items():
        m = a[a.symbol == s].set_index("date")["c"]
        j = pd.concat([m, tr], axis=1, keys=["alpaca", "truth"]).dropna()
        e = (j.alpaca / j.truth - 1)
        post = e[[d.year >= 2021 for d in e.index]]
        out[s] = {"days": int(len(j)), "agree_0p05pct": float((e.abs() <= 0.0005).mean()),
                  "agree_0p05pct_2021on": float((post.abs() <= 0.0005).mean()) if len(post) else None,
                  "median_abs_pct": float(e.abs().median() * 100), "splits": truth[s].attrs.get("splits", {})}
        # every split bracketed: ratio of Alpaca raw to truth must not jump across a split date
        jumps = []
        for sd in truth[s].attrs.get("splits", {}):
            sdd = dt.date.fromisoformat(sd[:10])
            before = e[[d < sdd for d in e.index]].tail(5)
            after = e[[d >= sdd for d in e.index]].head(5)
            if len(before) and len(after):
                jumps.append({"split": sd[:10], "median_err_before_pct": float(before.median() * 100),
                              "median_err_after_pct": float(after.median() * 100)})
        out[s]["split_brackets"] = jumps
    return out


def entry_spot_errors(truth: Dict[str, pd.Series]) -> Dict[str, Any]:
    con = sqlite3.connect(f"file:{DOLT_DB}?mode=ro", uri=True)
    df = pd.read_sql("SELECT symbol, date, expiration, strike, type, bid, ask FROM dolt_chain", con)
    con.close()
    per_sym: Dict[str, List[float]] = {s: [] for s in SYMBOLS}
    rows = []
    for (sym, day), g in df.groupby(["symbol", "date"]):
        if sym not in truth:
            continue
        spot = dh.parity_spot(g.to_dict("records"), day)
        tc = on_or_before(truth[sym], dt.date.fromisoformat(day))
        if spot is None or tc is None:
            continue
        e = spot / tc - 1
        per_sym[sym].append(e)
        rows.append((sym, day, spot, tc, e))
    pooled = [r[4] for r in rows]
    signed = np.asarray(pooled)
    return {"pooled": err_stats(pooled), "mean_signed_pct": float(signed.mean() * 100) if signed.size else None,
            "per_symbol": {s: err_stats(v) for s, v in per_sym.items()},
            "worst": sorted(rows, key=lambda r: -abs(r[4]))[:5]}


def settlement_errors(truth: Dict[str, pd.Series], study_db: Path) -> Dict[str, Any]:
    con = sqlite3.connect(f"file:{study_db}?mode=ro", uri=True)
    q = ("SELECT e.entry_id, e.symbol, e.session, e.expiration, e.short_strike, e.long_strike, e.width, e.credit_cross, "
         "m.underlying_close FROM entries e JOIN marks m ON m.entry_id = e.entry_id "
         "WHERE m.status='settled' AND m.underlying_close IS NOT NULL")
    df = pd.read_sql(q, con)
    con.close()
    errs, flips, pnl_diff = [], 0, []
    for r in df.itertuples():
        tc = settlement_truth(truth[r.symbol], r.session, r.expiration)
        if tc is None:
            continue
        errs.append(r.underlying_close / tc - 1)

        def value(s: float) -> float:   # put-spread intrinsic per share
            return max(r.short_strike - s, 0.0) - max(r.long_strike - s, 0.0)
        if (r.underlying_close < r.short_strike) != (tc < r.short_strike):
            flips += 1
        risk = r.width - (r.credit_cross or 0.0)
        if risk > 0:
            pnl_diff.append((value(tc) - value(r.underlying_close)) / risk)
    pd_arr = np.abs(np.asarray(pnl_diff))
    return {"settled": int(len(errs)), "error": err_stats(errs), "short_strike_itm_flips": int(flips),
            "abs_pnl_change_on_risk": {"mean_pct": float(pd_arr.mean() * 100) if pd_arr.size else None,
                                       "p99_pct": float(np.quantile(pd_arr, .99) * 100) if pd_arr.size else None,
                                       "changed_over_1pct": int((pd_arr > 0.01).sum())}}


# ------------------------------------------------------------------ re-run
def _primary(report: Dict[str, Any]) -> Dict[str, Any]:
    p = report["paired"]["cross"]
    return {k: p.get(k) for k in ("resolved_pairs", "weeks", "mean_diff_weekly", "ci95",
                                  "mean_ror_selector", "mean_ror_baseline", "identical_resolved")} | {
        "verdict": report["verdict"]}


def rerun(truth: Dict[str, pd.Series], work: Path) -> Dict[str, Any]:
    from src.selector_study import readout
    from src.selector_study.db import connect, default_settings
    today = "2026-11-26"   # the interim's own read date
    out: Dict[str, Any] = {}
    # control: original method, fresh directory -> must reproduce the recorded interim
    ctrl = work / "control"
    shutil.rmtree(ctrl, ignore_errors=True)
    out["control"] = _primary(dh.run(out_dir=str(ctrl), today=today)["readout"])

    tdir = work / "true_close"
    shutil.rmtree(tdir, ignore_errors=True)
    conv = dh.convert(out_dir=str(tdir))
    with sqlite3.connect(conv["chain_db"]) as con:
        pairs = con.execute("SELECT DISTINCT symbol, snap_date FROM chain_snapshots").fetchall()
        changed = 0
        for sym, day in pairs:
            tc = on_or_before(truth.get(sym, pd.Series(dtype=float)), dt.date.fromisoformat(str(day)))
            if tc is not None:
                con.execute("UPDATE chain_snapshots SET spot=? WHERE symbol=? AND snap_date=?", (tc, sym, day))
                changed += 1
        con.commit()
    res = dh.run(out_dir=str(tdir), today=today, reconvert=False)
    study_db = tdir / "selector_study_dolt.db"
    conn = connect(str(study_db), settings=dh.settings(default_settings()))
    try:
        rows = conn.execute("SELECT m.entry_id, e.symbol, e.session, e.expiration FROM marks m JOIN entries e "
                            "ON e.entry_id=m.entry_id WHERE m.status='settled'").fetchall()
        for eid, sym, sess, exp in rows:
            tc = settlement_truth(truth[sym], sess, exp)
            conn.execute("UPDATE marks SET underlying_close=?, source=? WHERE entry_id=? AND status='settled'",
                         (tc, "official close (audit)", eid))
        conn.commit()
        rep = readout.build(conn, today=today)
    finally:
        conn.close()
    out["true_close"] = _primary(rep) | {"spot_rows_replaced": changed, "settled": res["settle"]}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerun", action="store_true", help="also re-run the history study (slow)")
    ap.add_argument("--work", default=str(REPO / "data" / "audit_dolt_spot"))
    a = ap.parse_args(argv)
    truth = official_raw_closes(SYMBOLS)
    res: Dict[str, Any] = {"plan": "docs/AUDIT_DOLT_SPOT_PLAN_20261127.md",
                           "generated": dt.datetime.now().isoformat(timespec="seconds")}
    res["m0_cross_check_alpaca"] = cross_check_alpaca(truth)
    res["m1_entry_spot"] = entry_spot_errors(truth)
    res["m2_settlement"] = settlement_errors(truth, STUDY_DIR / "selector_study_dolt.db")
    if a.rerun:
        res["m4_rerun"] = rerun(truth, Path(a.work))
    out = REPO / "reports" / "audit_dolt_spot" / f"audit_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
