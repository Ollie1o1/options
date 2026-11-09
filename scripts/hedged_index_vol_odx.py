"""Delta-hedged SPY premium on optionsDX — the registered estimator.

Registration: docs/PREREG_HEDGED_INDEX_VOL_20261109.md (frozen before running).
One trade per monthly expiry, entered at the first day with DTE <= 35, held to
expiry, optionally delta-hedged daily with SPY shares.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.hedged_index_vol_odx
"""
from __future__ import annotations

import argparse
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DB = "data/optionsdx.db"
SYM = "SPY"
SEED = 20261109
N_BOOT = 10_000
STRESS = (2011, 2015, 2018, 2020, 2022)


def load_spot(con: sqlite3.Connection) -> pd.Series:
    s = pd.read_sql("SELECT date, MAX(underlying) AS s FROM odx_chain "
                    "WHERE symbol=? GROUP BY date ORDER BY date", con, params=(SYM,))
    return s.set_index("date")["s"]


def monthly_entries(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT date, expiration, MIN(dte) AS dte FROM odx_chain "
                     "WHERE symbol=? AND dte BETWEEN 25 AND 35 "
                     "GROUP BY date, expiration", con, params=(SYM,))
    e = pd.to_datetime(df.expiration)
    # third-Friday monthlies; pre-2015 some are listed on the Saturday after
    monthly = ((e.dt.weekday == 4) & e.dt.day.between(15, 21)) | \
              ((e.dt.weekday == 5) & e.dt.day.between(16, 22))
    df = df[monthly]
    return (df.sort_values("date").groupby("expiration", as_index=False).first()
            .sort_values("date").reset_index(drop=True))


def nearest(rows: pd.DataFrame, col: str, target: float) -> Optional[pd.Series]:
    r = rows.dropna(subset=[col])
    if r.empty:
        return None
    return r.loc[(r[col] - target).abs().idxmin()]


def pick_legs(day: pd.DataFrame, spot: float, short_d: Optional[float],
              wing_d: float) -> Optional[Dict[str, float]]:
    """Return strikes {sc, sp, lc, lp}. short_d None = ATM fly."""
    c = day[(day.type == "call") & (day.bid > 0)]
    p = day[(day.type == "put") & (day.bid > 0)]
    if c.empty or p.empty:
        return None
    if short_d is None:
        k = c.loc[(c.strike - spot).abs().idxmin(), "strike"]
        if k not in set(p.strike):
            return None
        sc = sp = float(k)
    else:
        a, b = nearest(c, "delta", short_d), nearest(p, "delta", -short_d)
        if a is None or b is None:
            return None
        sc, sp = float(a.strike), float(b.strike)
    lc_c = day[(day.type == "call") & (day.strike > sc) & (day.ask > 0)]
    lp_c = day[(day.type == "put") & (day.strike < sp) & (day.ask > 0)]
    a, b = nearest(lc_c, "delta", wing_d), nearest(lp_c, "delta", -wing_d)
    if a is None or b is None:
        return None
    return {"sc": sc, "sp": sp, "lc": float(a.strike), "lp": float(b.strike)}


LEGS = (("sc", "call", -1), ("sp", "put", -1), ("lc", "call", +1), ("lp", "put", +1))


def replay(con, spot: pd.Series, ent: pd.Series, short_d, wing_d=0.10,
           hedge=True, share_cost=0.01, mid=False) -> Optional[Dict]:
    exp, d0 = ent.expiration, ent.date
    rows = pd.read_sql(
        "SELECT date, strike, type, bid, ask, mid, iv, delta FROM odx_chain "
        "WHERE symbol=? AND expiration=? AND date BETWEEN ? AND ?",
        con, params=(SYM, exp, d0, exp))
    day0 = rows[rows.date == d0]
    s0 = float(spot[d0])
    legs = pick_legs(day0, s0, short_d, wing_d)
    if legs is None:
        return None
    q0 = day0.set_index(["strike", "type"])
    credit = 0.0
    for k, t, sgn in LEGS:
        row = q0.loc[(legs[k], t)]
        px = row["mid"] if mid else (row["ask"] if sgn > 0 else row["bid"])
        credit -= sgn * float(px)
    width = max(legs["lc"] - legs["sc"], legs["sp"] - legs["lp"])
    car = 100.0 * (width - credit)
    if car <= 0 or credit <= 0:
        return None

    path = spot[(spot.index >= d0) & (spot.index <= exp)]
    st = float(path.iloc[-1])
    opt = 100.0 * credit
    for k, t, sgn in LEGS:
        intr = max(st - legs[k], 0) if t == "call" else max(legs[k] - st, 0)
        opt += sgn * 100.0 * intr

    hpnl = hcost = 0.0
    if hedge:
        dl = rows.pivot_table(index="date", columns=["strike", "type"],
                              values="delta", aggfunc="first")
        cols = [(legs[k], t) for k, t, _ in LEGS]
        dl = dl.reindex(index=path.index, columns=cols).ffill()
        pos_d = pd.Series(0.0, index=dl.index)
        for k, t, sgn in LEGS:
            pos_d = pos_d + sgn * 100.0 * dl[(legs[k], t)]
        prev = 0.0
        dates = list(path.index)
        for i in range(len(dates) - 1):
            sh = -float(pos_d.iloc[i]) if np.isfinite(pos_d.iloc[i]) else prev
            hcost += share_cost * abs(sh - prev)
            hpnl += sh * (path.iloc[i + 1] - path.iloc[i])
            prev = sh
        hcost += share_cost * abs(prev)

    atm = day0.loc[(day0.strike - s0).abs() == (day0.strike - s0).abs().min(), "iv"]
    hist = np.log(spot[spot.index <= d0]).diff().dropna().iloc[-21:]
    rv = float(hist.std() * np.sqrt(252)) if len(hist) == 21 else np.nan
    pnl = opt + hpnl - hcost
    return {"entry": d0, "expiration": exp, "year": int(d0[:4]), "s0": s0,
            "credit": credit, "width": width, "car": car, "opt_pnl": opt,
            "hedge_pnl": hpnl, "hedge_cost": hcost, "pnl": pnl, "r": pnl / car,
            "atm_iv": float(atm.mean()), "rv21": rv,
            "vrp": float(atm.mean()) - rv, **legs}


def pf(r: np.ndarray) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else np.inf


def boot(r: np.ndarray, f, lvl=0.95) -> Dict[str, float]:
    rng = np.random.default_rng(SEED)
    n = len(r)
    vals = np.array([f(r[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    a = (1 - lvl) / 2
    return {"est": f(r), "lo": float(np.quantile(vals, a)),
            "hi": float(np.quantile(vals, 1 - a))}


def judge(label: str, df: pd.DataFrame, claim: bool = True) -> Dict:
    r = df.r.values
    m95, p95 = boot(r, np.mean), boot(r, pf)
    m98, p98 = boot(r, np.mean, 1 - 0.05 / 3), boot(r, pf, 1 - 0.05 / 3)
    ok = m98["lo"] > 0 and p98["lo"] > 1
    print(f"  {label}: n={len(r)} win {np.mean(r > 0):.2f}")
    print(f"    mean r {m95['est']:+.4f} 95% [{m95['lo']:+.4f},{m95['hi']:+.4f}] "
          f"98.33% lo {m98['lo']:+.4f}")
    print(f"    PF     {p95['est']:.3f} 95% [{p95['lo']:.3f},{p95['hi']:.3f}] "
          f"98.33% lo {p98['lo']:.3f}")
    if claim:
        print(f"    VERDICT: {'PASS' if ok else 'fail'}")
    print(f"    worst trade {r.min():+.3f} | sum r {r.sum():+.2f} | "
          f"options {df.opt_pnl.sum():+,.0f} hedge {df.hedge_pnl.sum():+,.0f} "
          f"hedge cost {-df.hedge_cost.sum():+,.0f} ($, 1 contract)")
    return {"pass": ok, "mean": m95, "pf": p95}


def years(df: pd.DataFrame) -> None:
    y = df.groupby("year").agg(n=("r", "size"), mean_r=("r", "mean"),
                               sum_r=("r", "sum"), worst=("r", "min"))
    print(y.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("    stress:", ", ".join(f"{s} {y.loc[s, 'sum_r']:+.2f}"
                                   for s in STRESS if s in y.index))


def run_all(con, spot, ents, **kw) -> pd.DataFrame:
    out: List[Dict] = []
    for _, e in ents.iterrows():
        t = replay(con, spot, e, **kw)
        if t:
            out.append(t)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()
    con = sqlite3.connect(DB)
    spot = load_spot(con)
    ents = monthly_entries(con)
    print(f"optionsDX {SYM} {spot.index[0]}..{spot.index[-1]} | "
          f"{len(ents)} monthly expiries with a 25-35 DTE entry")

    print("\n== T1 hedged iron condor 25/10 delta (PRIMARY)")
    t1 = run_all(con, spot, ents, short_d=0.25)
    if a.csv:
        t1.to_csv(a.csv, index=False)
    print(f"  median credit/width {np.median(t1.credit / t1.width):.3f} | "
          f"median CAR ${t1.car.median():,.0f}")
    j1 = judge("T1", t1)
    years(t1)

    print("\n== T2 VRP-timed T1")
    t = t1.sort_values("entry").reset_index(drop=True)
    med = [np.nanmedian(t.vrp.iloc[:i]) if i >= 12 else np.nan for i in range(len(t))]
    t["prior_med"] = med
    eligible = t.dropna(subset=["prior_med", "vrp"])
    hi = eligible[eligible.vrp > eligible.prior_med]
    lo = eligible[eligible.vrp <= eligible.prior_med]
    j2 = judge("T2 (VRP above prior median)", hi)
    judge("  complement (VRP at/below median)", lo, claim=False)
    rng = np.random.default_rng(SEED)
    diffs = [rng.choice(hi.r.values, len(hi)).mean() - rng.choice(lo.r.values, len(lo)).mean()
             for _ in range(N_BOOT)]
    print(f"    above-minus-below mean r {hi.r.mean() - lo.r.mean():+.4f} "
          f"95% [{np.quantile(diffs, .025):+.4f},{np.quantile(diffs, .975):+.4f}]")
    print(f"    VRP at entry: median {t.vrp.median():+.4f}, "
          f"share > 0 {np.mean(t.vrp > 0):.2f}")

    print("\n== T3 hedged ATM iron fly, 10-delta wings")
    t3 = run_all(con, spot, ents, short_d=None)
    j3 = judge("T3", t3)
    years(t3)

    print("\n== Sensitivities (no authority)")
    judge("T1 unhedged", run_all(con, spot, ents, short_d=0.25, hedge=False), claim=False)
    judge("T1 mid fills", run_all(con, spot, ents, short_d=0.25, mid=True), claim=False)
    judge("T1 hedge cost $0.02/sh", run_all(con, spot, ents, short_d=0.25,
                                            share_cost=0.02), claim=False)
    judge("T1 short 0.16 delta", run_all(con, spot, ents, short_d=0.16), claim=False)
    judge("T3 mid fills", run_all(con, spot, ents, short_d=None, mid=True), claim=False)

    n_pass = sum(j["pass"] for j in (j1, j2, j3))
    print(f"\nFAMILY: {n_pass}/3 claimants pass at the Bonferroni level")


if __name__ == "__main__":
    main()
