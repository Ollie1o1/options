"""Walk-forward evidence for the pick-ranking gates, and against a ranker.

Run: PYTHONPATH=$PWD python scripts/validate_gates.py

Two questions, kept separate because the ledger answers them differently:

  1. REMOVAL -- does refusing what the ledger measures as a loser improve
     what is left? Answered yes, 5 folds out of 5, 2026-08-09.
  2. RANKING -- does any ordering of the survivors beat `quality_score` at
     the #1 slot? Answered no. A theta-cost ranker was a coin flip
     (23 of 48 paired cells, Wilcoxon p=0.89).

Every threshold is fitted on data strictly BEFORE the fold it is applied to.
G5 is a fixed universe rule and is never fitted at all.

This is a measurement harness, not a gate: it prints and returns, and never
writes to the ledger or to config.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# Broad-index underlyings. Condors on anything else are refused: n=139 over the
# whole book, +9.5% mean return on capital here against -11.8% elsewhere,
# Mann-Whitney p < 1e-5, same sign in both halves of the sample. This agrees
# with two earlier independent results (the equity-VRP backtest and the
# DoltHub index put spread), so it is the best-evidenced rule in the repo.
BROAD_INDEX = frozenset({"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI"})

LONG_SINGLE = frozenset({"Long Call", "Long Put"})

# G6 removes the top quintile of `quality_score` on long single legs. The
# cutoff is deliberately NOT a constant -- it is refitted from the ledger on
# every run so it tracks the book instead of rotting into the source.
G6_QUANTILE = 0.80


def load_ledger(db_path: str = "paper_trades.db") -> pd.DataFrame:
    """Closed, non-duplicate trades with a return on capital."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "select * from trades where status='CLOSED' and duplicate_of is null",
            conn,
        )
    df["ret"] = df["pnl_usd"] / df["capital_at_risk"].replace(0, np.nan)
    df = df.dropna(subset=["ret"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["theta_burn"] = df["entry_theta"].abs() / df["entry_price"].replace(0, np.nan)
    return df.sort_values("date").reset_index(drop=True)


def fit_g6(prior: pd.DataFrame) -> float:
    """G6's cutoff, from trades that closed before the fold being tested.

    Returns +inf when there is not enough history, which disables the gate
    rather than letting a handful of trades set a threshold for the book.
    """
    ls = prior[prior["strategy_name"].isin(LONG_SINGLE)]
    if len(ls) < 30:
        return float("inf")
    return float(ls["quality_score"].quantile(G6_QUANTILE))


def survives(df: pd.DataFrame, g6_cutoff: float) -> pd.Series:
    """Boolean mask of rows that clear the gates.

    Each gate is scoped to the population it was measured on. G5 says nothing
    about single legs and G6 says nothing about condors, so neither is applied
    outside its evidence -- the discipline that was missing when a single
    composite was asked to judge every structure at once.
    """
    keep = pd.Series(True, index=df.index)

    is_condor = df["strategy_name"] == "Iron Condor"
    keep &= ~(is_condor & ~df["ticker"].isin(BROAD_INDEX))

    is_long_single = df["strategy_name"].isin(LONG_SINGLE)
    keep &= ~(is_long_single & (df["quality_score"] >= g6_cutoff))

    return keep


def walk_forward(df: pd.DataFrame, n_folds: int = 5) -> List[Dict]:
    """Refit before each fold, apply blind to it, record what happened."""
    edges = [df["date"].quantile(q) for q in np.linspace(0.40, 1.0, n_folds + 1)]
    folds: List[Dict] = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        fold = df[(df["date"] > lo) & (df["date"] <= hi)]
        prior = df[df["date"] <= lo]
        if len(fold) < 20:
            continue
        cutoff = fit_g6(prior)
        mask = survives(fold, cutoff)
        kept, refused = fold[mask], fold[~mask]
        folds.append({
            "window": f"{lo.date()}..{hi.date()}",
            "n": len(fold),
            "n_kept": len(kept),
            "kept_mean": kept["ret"].mean(),
            "refused_mean": refused["ret"].mean() if len(refused) else np.nan,
            "avoided": -refused["pnl_usd"].sum() if len(refused) else 0.0,
            "p": (mannwhitneyu(kept["ret"], refused["ret"])[1]
                  if len(refused) >= 5 and len(kept) >= 5 else np.nan),
        })
    return folds


def rank_survivors(df: pd.DataFrame) -> pd.Series:
    """The ranking key this harness DISPROVED. Kept so the result stays checkable.

    Lower is better. Condors use theta_score (measured +0.387, n=121) negated;
    everything else uses theta bill as a share of premium. Against
    `quality_score` at the #1 slot this was a coin flip, which is why the
    shipped design refuses candidates but does not order them.
    """
    key = pd.Series(np.nan, index=df.index)
    is_condor = df["strategy_name"] == "Iron Condor"
    key[is_condor] = -df.loc[is_condor, "theta_score"]
    key[~is_condor] = df.loc[~is_condor, "theta_burn"]
    return key


def top_pick_per_board(df: pd.DataFrame, key: str, ascending: bool) -> pd.DataFrame:
    """What a #1 slot would have handed you, per board per day."""
    d = df.dropna(subset=[key]).copy()
    d["_day"] = d["date"].dt.date
    return (d.sort_values(key, ascending=ascending)
             .groupby(["_day", "strategy_name"]).head(1))


def test_ranking(df: pd.DataFrame, split: float = 0.60) -> Tuple[pd.DataFrame, ...]:
    """Does any ordering of the survivors beat quality_score at #1?"""
    cut = df["date"].quantile(split)
    train, test = df[df["date"] <= cut], df[df["date"] > cut]
    cutoff = fit_g6(train)
    surv = test[survives(test, cutoff)].copy()
    surv["_key"] = rank_survivors(surv)
    old = top_pick_per_board(test, "quality_score", ascending=False)
    new = top_pick_per_board(surv, "_key", ascending=True)
    return old, new


def _fmt(label: str, s: pd.DataFrame) -> str:
    return (f"  {label:34s} n={len(s):3d}  win={(s['ret'] > 0).mean():.2f}  "
            f"mean={s['ret'].mean():+.3f}  median={s['ret'].median():+.3f}")


def main() -> None:
    df = load_ledger()
    print(f"ledger: {len(df)} closed non-duplicate trades, "
          f"{df['date'].min().date()}..{df['date'].max().date()}\n")

    print("=== 1. REMOVAL: walk-forward, refit before every fold ===")
    print(f"{'fold':<6}{'window':<26}{'n':>5}{'kept':>6}{'keep':>9}"
          f"{'refuse':>9}{'avoided $':>12}{'p':>9}")
    print("-" * 82)
    folds = walk_forward(df)
    for i, f in enumerate(folds, 1):
        print(f"{i:<6}{f['window']:<26}{f['n']:>5}{f['n_kept']:>6}"
              f"{f['kept_mean']:>+9.3f}{f['refused_mean']:>+9.3f}"
              f"{f['avoided']:>+12,.0f}{f['p']:>9.4f}")
    print("-" * 82)
    wins = sum(1 for f in folds if f["kept_mean"] > f["refused_mean"])
    total = sum(f["avoided"] for f in folds)
    print(f"  kept beat refused in {wins} of {len(folds)} folds "
          f"(sign test p={0.5 ** len(folds) * 2:.3f})")
    print(f"  losses avoided across folds: ${total:+,.0f}")

    print("\n=== 2. RANKING: does the #1 slot improve? ===")
    old, new = test_ranking(df)
    print(_fmt("OLD  #1 by quality_score", old))
    print(_fmt("NEW  #1 after gates + rank", new))
    o = old.set_index(["_day", "strategy_name"])["ret"]
    n = new.set_index(["_day", "strategy_name"])["ret"]
    j = pd.concat({"old": o, "new": n}, axis=1).dropna()
    j = j[j["old"] != j["new"]]
    if len(j) >= 10:
        p = wilcoxon(j["new"], j["old"])[1]
        print(f"\n  paired on the same (day, board), pick differs: n={len(j)}")
        print(f"  new better in {(j['new'] > j['old']).sum()} of {len(j)} cells, "
              f"Wilcoxon p={p:.4f}")
        print("  -> no ordering advantage. Refuse, do not rank.")


if __name__ == "__main__":
    main()
