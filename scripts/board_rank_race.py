"""Which key should order the board? Declared BEFORE looking.

POPULATION: the 912 closed trades in paper_trades.db. Deliberately NOT
`candidates.db` — that is the frozen pre-registration's cohort and racing
keys there is looking early at the 2026-11-19 test.

CONSEQUENCE OF THAT CHOICE, stated up front: `ev_net` and `round_trip_pct`
live only in candidates.db, so THE KEY THE BOARD CURRENTLY SORTS BY CANNOT
BE RACED HERE. This test cannot crown ev_net and cannot depose it. It can
only say whether anything available on the ledger orders outcomes.

The ledger is also a SELECTED population — trades that were taken — so an IC
measured here is an upper bound on what the same key does across a full
board, not an unbiased estimate of it.

KEYS (fixed now, no additions after seeing results):
    cal_pop          calibrated P(closes green), walk-forward, out-of-sample
    exp_return       p*W_s + (1-p)*L_s, walk-forward, out-of-sample
    quality_score    the legacy composite, as a benchmark expected to fail
    abs_delta        raw feature control
    dte              raw feature control

OUTCOME: ret_on_risk (return on CAPITAL AT RISK).

STATISTIC: Spearman rank IC of key vs outcome, ranks demeaned within
entry_date cells so a day's regime cannot masquerade as skill. 95% CI by
cluster bootstrap resampling entry_date, 10,000 draws, seed 20260824.

DECISION RULE, fixed now:
    A key may order the board only if its 95% CI lies entirely above zero.
    If several qualify, the highest point estimate wins.
    If none qualify, the board's ORDER DOES NOT CHANGE and its LABEL must
    stop implying that #1 is a best pick.

Every key is reported whichever way it lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import pop_calibration as pc, expected_return as er

SEED, N_BOOT = 20260824, 10000
KEYS = ["cal_pop", "exp_return", "quality_score", "abs_delta", "dte"]


def demeaned_ic(df, key, outcome="ret_on_risk", cell="entry_date"):
    """Spearman IC on ranks demeaned within each day."""
    d = df[[key, outcome, cell]].dropna()
    if len(d) < 30:
        return np.nan
    out = []
    for _, g in d.groupby(cell):
        if len(g) < 3:
            continue
        x = stats.rankdata(g[key]); y = stats.rankdata(g[outcome])
        out.append(pd.DataFrame({"x": x - x.mean(), "y": y - y.mean()}))
    if not out:
        return np.nan
    a = pd.concat(out)
    if a["x"].std() == 0 or a["y"].std() == 0:
        return np.nan
    return float(np.corrcoef(a["x"], a["y"])[0, 1])


def boot_ci(df, key):
    rng = np.random.default_rng(SEED)
    cells = df["entry_date"].unique()
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(cells, size=len(cells), replace=True)
        s = pd.concat([df[df.entry_date == c] for c in pick])
        v = demeaned_ic(s, key)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    df = pc.load_training_set("paper_trades.db").dropna(subset=["ret_on_risk"])

    # Out-of-sample only: a key scored by a model that has seen the row is not
    # a key, it is a memory.
    oos_p = pc.walk_forward(df, seed_n=300, step=50)
    oos_e = er.walk_forward(df, seed_n=300, step=50)
    tail = df.iloc[300:300 + len(oos_p)].reset_index(drop=True)
    assert (tail["entry_date"].to_numpy() == oos_p["entry_date"].to_numpy()).all()

    frame = tail.copy()
    frame["cal_pop"] = oos_p["predicted"].to_numpy()
    frame["exp_return"] = oos_e["predicted"].to_numpy()[:len(tail)]

    import sqlite3
    conn = sqlite3.connect("file:paper_trades.db?mode=ro", uri=True)
    q = pd.read_sql("select date as entry_date, quality_score from trades "
                    "where status='CLOSED' and pnl_usd is not null", conn)
    conn.close()
    q["entry_date"] = pd.to_datetime(q["entry_date"]).dt.strftime("%Y-%m-%d")
    q = q.sort_values("entry_date", kind="mergesort").reset_index(drop=True)
    frame["quality_score"] = q["quality_score"].to_numpy()[300:300 + len(tail)]

    print(__doc__)
    print(f"n = {len(frame)} out-of-sample closed trades, "
          f"{frame.entry_date.nunique()} days\n")
    print(f"{'key':16}{'rank IC':>10}{'95% CI':>22}   verdict")
    results = {}
    for k in KEYS:
        ic = demeaned_ic(frame, k)
        lo, hi = boot_ci(frame, k)
        results[k] = (ic, lo, hi)
        ok = (not np.isnan(lo)) and lo > 0
        print(f"{k:16}{ic:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]   "
              f"{'QUALIFIES' if ok else 'no'}")

    winners = {k: v for k, v in results.items()
               if not np.isnan(v[1]) and v[1] > 0}
    print()
    if winners:
        best = max(winners, key=lambda k: winners[k][0])
        print(f"DECISION: order the board by `{best}` "
              f"(IC {winners[best][0]:+.4f}, CI clears zero).")
    else:
        print("DECISION: no key qualifies. The board's ORDER does not change; "
              "its LABEL must stop implying #1 is a best pick.")


if __name__ == "__main__":
    main()
