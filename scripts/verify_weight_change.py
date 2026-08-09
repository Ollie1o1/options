"""Old multi-leg composite against the new one, on the real closed ledger.

Written to check the 2026-08-09 weight deletions (docs/CALIBRATION_JOURNAL.md):
`iv_rank` 0.15 -> 0 on verticals, `pop` 0.30 -> 0 and `iv_rank` 0.12 -> 0 on
condors. Kept because the same question will be asked of the next weight
change, and answering it by hand each time is how a change ships unmeasured.

READ THE LIMIT BEFORE READING THE NUMBERS. This can FALSIFY a weight change
but it cannot confirm one: the component ICs that justify a deletion are
measured on this same ledger, so an improvement here is partly circular. A
DEGRADATION is the decisive outcome, and that is what it is looking for. It
is also the ledger whose own headline verdict is "NO SIGNIFICANT EDGE
(IC=-0.03, p=0.433)" — see docs/SCORE_AUDIT_20260807.md item 1.

Returns are reported twice: at the MID, and restated at the crossed credit
where both prices exist, paying the slip on entry and again to close. The
second is the one that matters; docs/EXECUTION_TRUTH.md is why.

Research only. Opens the ledger read-only and writes nothing.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/verify_weight_change.py
"""
import sqlite3

import pandas as pd
from scipy import stats

from src.spread_scoring import _weighted_score

OLD_SPREAD = {"pop": 0.25, "credit_to_width": 0.20, "iv_rank": 0.15,
              "return_on_risk": 0.10, "liquidity": 0.10, "theta": 0.08,
              "spread": 0.05, "momentum": 0.04, "catalyst": 0.03}
NEW_SPREAD = dict(OLD_SPREAD, iv_rank=0.0)

OLD_IRON = {"pop": 0.30, "credit_to_width": 0.20, "delta_neutral": 0.15,
            "iv_rank": 0.12, "liquidity": 0.10, "theta": 0.08, "spread": 0.05}
NEW_IRON = dict(OLD_IRON, iv_rank=0.0, pop=0.0)

SPREAD_COLS = {"pop": "pop_score", "credit_to_width": "credit_to_width_score",
               "iv_rank": "iv_rank_score", "return_on_risk": "return_on_risk_score",
               "liquidity": "liquidity_score", "theta": "theta_score",
               "spread": "spread_score", "momentum": "momentum_score",
               "catalyst": "catalyst_score"}
IRON_COLS = {"pop": "pop_score", "credit_to_width": "credit_to_width_score",
             "delta_neutral": "delta_neutral_score", "iv_rank": "iv_rank_score",
             "liquidity": "liquidity_score", "theta": "theta_score",
             "spread": "spread_score"}


def load(db="paper_trades.db"):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql(
            "SELECT * FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
            "AND duplicate_of IS NULL AND strategy_name IN "
            "('Bull Put','Bear Call','Iron Condor')", con)
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def rescore(df, weights, cols):
    out = []
    for _, r in df.iterrows():
        out.append(_weighted_score(r.to_dict(), cols, weights))
    return pd.Series(out, index=df.index, dtype=float)


def net_return(df):
    """Restated at the crossed credit where both prices exist, else pnl_pct."""
    r = df["pnl_pct"].astype(float).copy()
    if {"entry_price_mid", "entry_price_cross"} <= set(df.columns):
        mid = pd.to_numeric(df["entry_price_mid"], errors="coerce")
        cross = pd.to_numeric(df["entry_price_cross"], errors="coerce")
        ok = mid.notna() & cross.notna() & (mid > 0)
        slip = (mid - cross).abs()
        r.loc[ok] = r.loc[ok] - 2.0 * (slip.loc[ok] / mid.loc[ok])
    return r


def ic(x, y):
    m = pd.notna(x) & pd.notna(y)
    if m.sum() < 8 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan"), float("nan"), int(m.sum())
    res = stats.spearmanr(x[m], y[m])
    return float(res.statistic), float(res.pvalue), int(m.sum())


def report(name, sub, old_w, new_w, cols):
    if sub.empty:
        print(f"\n{name}: no rows")
        return None
    old = rescore(sub, old_w, cols)
    new = rescore(sub, new_w, cols)
    ret_mid = sub["pnl_pct"].astype(float)
    ret_net = net_return(sub)

    print(f"\n{name}  n={len(sub)}   "
          f"{sub.date.min():%Y-%m-%d} -> {sub.date.max():%Y-%m-%d}")
    both = old.notna() & new.notna()
    if both.sum() >= 3:
        rc = stats.spearmanr(old[both], new[both]).statistic
        moved = (old[both].rank() != new[both].rank()).mean()
        print(f"  scored {both.sum()}/{len(sub)} rows; rank corr(old,new) = {rc:.4f}"
              f"   ({moved:.0%} of the ordering moves)")
    else:
        print(f"  scored {both.sum()}/{len(sub)} rows — too few to compare orderings")
    for label, ret in (("at MID", ret_mid), ("net of cross", ret_net)):
        io, po, no = ic(old, ret)
        inw, pn, nn = ic(new, ret)
        flag = "BETTER" if inw > io else ("worse" if inw < io else "same")
        print(f"  {label:<14} old IC {io:+.4f} (p {po:.3f})   "
              f"new IC {inw:+.4f} (p {pn:.3f})   -> {flag}  n={nn}")
    return old, new, ret_net


def top_k(name, old, new, ret, ks=(5, 10, 15)):
    print(f"  top-K by score, median return net of cross:")
    for k in ks:
        if k > len(ret):
            continue
        o = ret.iloc[old.sort_values(ascending=False).index[:k]]
        n = ret.iloc[new.sort_values(ascending=False).index[:k]]
        print(f"    K={k:<3} old {o.median():+.4f} ({(o > 0).mean():.0%} win)"
              f"   new {n.median():+.4f} ({(n > 0).mean():.0%} win)")


def main():
    df = load()
    print(f"closed multi-leg rows: {len(df)}")
    print("  " + "  ".join(f"{k}={v}" for k, v in
                           df.strategy_name.value_counts().items()))

    verticals = df[df.strategy_name.isin(["Bull Put", "Bear Call"])].reset_index(drop=True)
    condors = df[df.strategy_name == "Iron Condor"].reset_index(drop=True)

    for name, sub, ow, nw, cols in (
            ("VERTICALS (iv_rank 0.15 -> 0)", verticals, OLD_SPREAD, NEW_SPREAD, SPREAD_COLS),
            ("CONDORS (pop 0.30 -> 0, iv_rank 0.12 -> 0)", condors, OLD_IRON, NEW_IRON, IRON_COLS)):
        got = report(name, sub, ow, nw, cols)
        if got:
            old, new, ret = got
            top_k(name, old, new, ret)

    # Walk-forward: the sign has to be stable, not just right once.
    print("\nwalk-forward, condors, IC net of cross by test window:")
    for cut in ("2026-05-15", "2026-06-01", "2026-06-15", "2026-07-01"):
        sub = condors[condors.date >= cut].reset_index(drop=True)
        if len(sub) < 12:
            print(f"  >= {cut}: n={len(sub)} too few")
            continue
        old = rescore(sub, OLD_IRON, IRON_COLS)
        new = rescore(sub, NEW_IRON, IRON_COLS)
        ret = net_return(sub)
        io, _, _ = ic(old, ret)
        inw, _, _ = ic(new, ret)
        print(f"  >= {cut}: n={len(sub):<4} old {io:+.4f}  new {inw:+.4f}  "
              f"{'BETTER' if inw > io else 'worse'}")


if __name__ == "__main__":
    main()
