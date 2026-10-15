"""Index vs single-name Bull Puts on capital at risk — the registered estimator.

Registration: docs/PREREG_INDEX_VS_SINGLE_BULL_PUT_20261015.md (frozen before
any outcome was queried). Change nothing here without amending that document.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.index_vs_single_bull_put
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

DB = "data/candidates.db"
INDEX = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI"}
OTHER_ETF = {"GLD", "SLV", "TLT", "IEF", "EEM"}
SCAN_CUTOFF = "2026-10-08"
WEEKEND_ARTIFACTS = ("2026-08-29", "2026-08-30")
N_BOOT = 10_000
SEED = 20261015
R_RATE = 0.04


def group_of(sym: str) -> str:
    if sym in INDEX:
        return "index"
    if sym in OTHER_ETF or sym.startswith("XL"):
        return "etf"
    return "single"


def _put(S, K, s, T):
    d1 = (np.log(S / K) + (R_RATE + s * s / 2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    return K * np.exp(-R_RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def skew_penalty(f: dict, ts: str, expiration: str) -> Optional[float]:
    """(flat-IV value - market credit) / width, both legs at the short leg's IV."""
    try:
        Ks, Kl, iv = f["short_strike"], f["long_strike"], f["entry_iv"]
        ps = (f["short_bid"] + f["short_ask"]) / 2
        pl = (f["long_bid"] + f["long_ask"]) / 2
        T = ((pd.Timestamp(expiration) + pd.Timedelta(hours=20))
             - pd.Timestamp(ts).tz_localize(None)).total_seconds() / (365 * 86400)
        if T <= 0 or not iv or iv <= 0.02 or Ks <= Kl:
            return None
        S = brentq(lambda x: _put(x, Ks, iv, T) - ps, Ks * 0.5, Ks * 2)
        flat = _put(S, Ks, iv, T) - _put(S, Kl, iv, T)
        return float((flat - (ps - pl)) / (Ks - Kl))
    except Exception:
        return None


def load(db: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql(
        """SELECT c.scan_id, c.ts, c.symbol, c.expiration, c.gate_passed,
                  c.refused_by, c.features_json, p.entry_price, p.exit_date,
                  p.pnl_pct
           FROM candidates c JOIN candidate_positions p
             ON p.scan_id=c.scan_id AND p.board=c.board
            AND p.contract_key=c.contract_key
           WHERE c.strategy_name='Bull Put' AND c.board='CREDIT SPREADS'
             AND (c.gate_passed=1 OR c.refused_by='negative_ev')
             AND p.status='CLOSED' AND p.pnl_pct IS NOT NULL""", con)
    con.close()
    df["scan_date"] = df.scan_id.str[:10]
    df["arm"] = np.where(df.gate_passed == 1, "passed", "refused")
    df["grp"] = df.symbol.map(group_of)
    feats = df.features_json.map(json.loads)
    df["w"] = feats.map(lambda f: f.get("spread_width"))
    e, w = df.entry_price.astype(float), pd.to_numeric(df.w, errors="coerce")
    ok = (e > 0) & (w > 0) & (e < w)
    df = df[ok].copy()
    e, w = e[ok], w[ok]
    raw = df.pnl_pct * e / (w - e)
    hi = e / (w - e)
    df["R"] = raw.clip(lower=-1.0, upper=hi)
    df["clipped"] = (raw < -1.0) | (raw > hi)
    df["skew_pen"] = [skew_penalty(f, t, x) for f, t, x in
                      zip(feats[ok], df.ts, df.expiration)]
    d = pd.to_datetime(df.scan_date)
    iso = d.dt.isocalendar()
    df["week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return df.drop(columns=["features_json"])


def arm_day(df: pd.DataFrame, symbol_equal: bool = True) -> pd.Series:
    """Mean R per scan_date; symbols weighted equally within the day."""
    if symbol_equal:
        return df.groupby(["scan_date", "symbol"]).R.mean().groupby("scan_date").mean()
    return df.groupby("scan_date").R.mean()


def _week_of(dates: pd.Index) -> np.ndarray:
    iso = pd.to_datetime(dates).isocalendar()
    return (iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)).values


def week_boot(values: pd.Series, stat: Callable[[np.ndarray], float] = np.mean) -> Dict:
    """values indexed by scan_date; resample whole weeks."""
    v = values.dropna()
    if v.empty:
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "days": 0, "weeks": 0}
    weeks = _week_of(v.index)
    uw = np.unique(weeks)
    by = {k: v.values[weeks == k] for k in uw}
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        pick = rng.choice(uw, size=len(uw), replace=True)
        boots[i] = stat(np.concatenate([by[k] for k in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": float(stat(v.values)), "lo": float(lo), "hi": float(hi),
            "days": int(len(v)), "weeks": int(len(uw))}


def paired(a: pd.DataFrame, b: pd.DataFrame, symbol_equal: bool = True) -> Dict:
    da, db_ = arm_day(a, symbol_equal), arm_day(b, symbol_equal)
    both = da.index.intersection(db_.index)
    out = week_boot((da[both] - db_[both]))
    out["syms_a"], out["syms_b"] = a.symbol.nunique(), b.symbol.nunique()
    out["rows_a"], out["rows_b"] = len(a), len(b)
    return out


def verdict(r: Dict, min_weeks: int = 4) -> str:
    if r["weeks"] < min_weeks or np.isnan(r["est"]):
        return "insufficient"
    if r["lo"] > 0:
        return "misdirected"
    if r["hi"] < 0:
        return "gate right"
    return "inconclusive"


def fmt(r: Dict) -> str:
    return (f"{r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
            f"days={r['days']} weeks={r['weeks']}")


def primary_population(df: pd.DataFrame, cutoff: str = SCAN_CUTOFF,
                       drop_weekend: bool = True) -> pd.DataFrame:
    out = df[df.scan_date <= cutoff]
    if drop_weekend:
        out = out[~out.exit_date.isin(WEEKEND_ARTIFACTS)]
    return out


def contrast(df: pd.DataFrame, g1="index", a1="refused", g2="single",
             a2="passed", symbol_equal=True) -> Dict:
    return paired(df[(df.grp == g1) & (df.arm == a1)],
                  df[(df.grp == g2) & (df.arm == a2)], symbol_equal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    with open(args.db, "rb") as fh:
        fp = hashlib.sha256(fh.read()).hexdigest()[:16]
    full = load(args.db)
    pop = primary_population(full)
    print(f"DB sha256 {fp} | rows loaded {len(full):,} | primary rows {len(pop):,}"
          f" | clipped {int(pop.clipped.sum())}")

    print("\n== PRIMARY: index refused − single passed (day-paired, week bootstrap)")
    p = contrast(pop)
    print(f"D = {fmt(p)}  syms {p['syms_a']}/{p['syms_b']}  rows {p['rows_a']}/{p['rows_b']}")
    print(f"VERDICT: {verdict(p)}")

    print("\n== S3 levels (mean of symbol-equal day means, week bootstrap)")
    for g in ("index", "single", "etf"):
        for a in ("refused", "passed"):
            sub = pop[(pop.grp == g) & (pop.arm == a)]
            if len(sub):
                print(f"  {g:6s} {a:7s} rows={len(sub):6d} syms={sub.symbol.nunique():3d}  "
                      f"{fmt(week_boot(arm_day(sub)))}")

    print("\n== S2 gate within index: index passed − index refused")
    s2 = contrast(pop, "index", "passed", "index", "refused")
    print(f"  {fmt(s2)}  -> {verdict(s2)}")

    print("\n== S1 skew penalty quintiles (edges on pooled index+single primary)")
    core = pop[pop.grp.isin(["index", "single"]) & pop.skew_pen.notna()].copy()
    edges = np.quantile(core.skew_pen, [0.2, 0.4, 0.6, 0.8])
    core["q"] = np.searchsorted(edges, core.skew_pen, side="right") + 1
    print("  edges", np.round(edges, 4))
    for label, sub in (("refused, index+single", core[core.arm == "refused"]),
                       ("refused, single only", core[(core.arm == "refused") & (core.grp == "single")]),
                       ("passed, index+single", core[core.arm == "passed"])):
        print(f"  [{label}]")
        for q in range(1, 6):
            s = sub[sub.q == q]
            if len(s):
                print(f"    Q{q} rows={len(s):6d} idx_share={np.mean(s.grp == 'index'):.2f}  "
                      f"{fmt(week_boot(arm_day(s)))}")
        r15 = paired(sub[sub.q == 5], sub[sub.q == 1])
        print(f"    Q5 − Q1: {fmt(r15)}")

    print("\n== Sensitivities (primary contrast)")
    sens = {
        "drop exit 10-12": contrast(pop[pop.exit_date != "2026-10-12"]),
        "row-weighted": contrast(pop, symbol_equal=False),
        "include 08-29/30": contrast(primary_population(full, drop_weekend=False)),
        "other-ETF refused vs single passed": contrast(pop, "etf", "refused"),
        "mature scans <= 10-01": contrast(primary_population(full, cutoff="2026-10-01")),
    }
    for k, r in sens.items():
        print(f"  {k:36s} {fmt(r)}  -> {verdict(r)}")


if __name__ == "__main__":
    main()
