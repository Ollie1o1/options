"""Does the multi-leg composite rank outcomes?

`spread_scoring` recomputes `quality_score` for spreads and condors from
`credit_spread_weights` / `iron_condor_weights` — a different composite from
the 27-component single-leg one, and one that **never touches the ~20-constant
adjustment stack**. So unlike the single-leg case there is no residual to
infer: the stored score IS the composite, and it can be measured directly.

Research only: writes nothing, opens the ledger read-only.

Returns are reported two ways. `pnl_pct` is computed against the MID credit,
which for a credit structure is the number the execution-truth work showed to
be wrong — you receive less than the mid when you cross. Rows carrying
`entry_price_mid` and `entry_price_cross` are restated at the crossed credit,
paying the slip once on entry and once to close.

Medians accompany every mean: these are ratios over a credit base that can be
arbitrarily small, so the means have heavy tails and are not robust on their own.
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

STRATEGIES = ("Bull Put", "Bear Call", "Iron Condor")
CUTS = ["2026-05-15", "2026-06-01", "2026-06-15", "2026-07-01", "2026-07-15"]


def load(db: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in STRATEGIES)
        df = pd.read_sql(
            "SELECT * FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
            f"AND strategy_name IN ({placeholders})", con, params=list(STRATEGIES))
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def _ic(x, y):
    r = stats.spearmanr(x, y)
    return r.statistic, r.pvalue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="paper_trades.db")
    args = ap.parse_args()

    df = load(args.db)
    print(f"n = {len(df)} closed multi-leg rows, "
          f"{df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    print("  " + "  ".join(f"{k} {v}" for k, v in
                           df.strategy_name.value_counts().items()))
    print(f"  at the MID: mean {df.pnl_pct.mean():+.4f}  "
          f"median {df.pnl_pct.median():+.4f}  win {(df.pnl_pct > 0).mean():.1%}")

    print("\n--- does the composite rank? (returns as recorded, at the mid) ---")
    print(f"{'cohort':<14}{'n':>5}{'rank IC':>10}{'p':>8}")
    for label in ("ALL",) + STRATEGIES:
        m = df if label == "ALL" else df[df.strategy_name == label]
        ic, p = _ic(m.quality_score, m.pnl_pct)
        print(f"  {label:<12}{len(m):>5}{ic:>10.4f}{p:>8.3f}")

    print("\n--- expanding walk-forward, all multi-leg ---")
    print(f"{'cut':<12}{'n_te':>6}{'rank IC':>10}{'p':>8}")
    ics = []
    for cut in CUTS:
        te = df.date >= pd.Timestamp(cut)
        if te.sum() < 30:
            continue
        ic, p = _ic(df.loc[te, "quality_score"], df.loc[te, "pnl_pct"])
        ics.append(ic)
        print(f"{cut:<12}{int(te.sum()):>6}{ic:>10.4f}{p:>8.3f}")
    if ics:
        print(f"  mean {np.mean(ics):+.4f}   negative in "
              f"{sum(1 for i in ics if i < 0)} of {len(ics)}")

    # ---- execution truth ----
    ex = df.dropna(subset=["entry_price_mid", "entry_price_cross"]).copy()
    if ex.empty:
        print("\n(no execution-truth rows)")
        return

    print(f"\n--- what crossing actually costs (n={len(ex)} rows with real prices) ---")
    n_never = int((ex.entry_price_mid <= 0).sum())
    n_vanish = int((ex.entry_price_cross <= 0).sum())
    print(f"  never a credit even at the mid: {n_never}")
    print(f"  CREDIT VANISHES once crossed:   {n_vanish}  "
          f"({n_vanish / len(ex):.0%}) — refused by candidate_verdict")

    g = ex[ex.entry_price_mid > 0].copy()
    slip = g.entry_price_mid - g.entry_price_cross
    g["friction"] = slip / g.entry_price_mid
    print(f"  entry crossing, share of mid credit: median {g.friction.median():.1%}"
          f"  mean {g.friction.mean():.1%}  p90 {g.friction.quantile(.9):.1%}")
    print(f"  round trip (x2):                     median {g.friction.median()*2:.1%}"
          f"  mean {g.friction.mean()*2:.1%}")
    print(f"  round trip exceeds the WHOLE credit on {(g.friction * 2 > 1).mean():.0%}"
          " of trades")

    close_cost = g.entry_price_mid * (1 - g.pnl_pct)
    g["pnl_net"] = (g.entry_price_cross - close_cost - slip) / g.entry_price_cross
    g = g[g.entry_price_cross > 0]

    print(f"\n--- return at the mid vs restated at the cross (n={len(g)}) ---")
    print(f"{'cohort':<14}{'n':>5}{'mid mean':>11}{'mid med':>10}{'mid win':>9}"
          f"{'net mean':>11}{'net med':>10}{'net win':>9}")
    for label in ("ALL",) + STRATEGIES:
        m = g if label == "ALL" else g[g.strategy_name == label]
        if len(m) < 5:
            continue
        print(f"  {label:<12}{len(m):>5}{m.pnl_pct.mean():>11.4f}"
              f"{m.pnl_pct.median():>10.4f}{(m.pnl_pct > 0).mean():>9.0%}"
              f"{m.pnl_net.mean():>11.4f}{m.pnl_net.median():>10.4f}"
              f"{(m.pnl_net > 0).mean():>9.0%}")

    print("\n--- rank IC on the execution-truth subset ---")
    print(f"{'cohort':<14}{'n':>5}{'mid IC':>10}{'p':>8}{'net IC':>10}{'p':>8}")
    for label in ("ALL",) + STRATEGIES:
        m = g if label == "ALL" else g[g.strategy_name == label]
        if len(m) < 15:
            print(f"  {label:<12}{len(m):>5}     (too few to read)")
            continue
        a, ap_ = _ic(m.quality_score, m.pnl_pct)
        b, bp = _ic(m.quality_score, m.pnl_net)
        print(f"  {label:<12}{len(m):>5}{a:>10.4f}{ap_:>8.2f}{b:>10.4f}{bp:>8.2f}")


if __name__ == "__main__":
    main()
