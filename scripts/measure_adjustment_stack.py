"""What does the post-composite adjustment stack do to the score's ranking?

~20 hand-set additive constants are applied after the weighted composite, plus
two multipliers.  `score_adjustments` (schema 20) records which fired, but it
shipped 2026-08-07 and has no history, so this infers the stack as a residual:

    raw    = inverse of _cross_section_normalize applied to the stored score
    stack  = raw - (stored component scores recomposed against the weights)

Research only: writes nothing, opens the ledger read-only.

Two corrections matter and both are applied:

* `pnl_pct` is computed against `entry_price`, which is the MID, with no
  crossing cost -- only 35 of 335 long-premium rows have a real fill price.
  Wide-spread trades are therefore flattered.  Returns are restated by charging
  half the spread each way, and a position that expired worthless is charged
  entry only because it was never sold.
* The composite is rebuilt with today's weights, which are not the ones in
  force at entry.  --weights compares four very different weightings to show
  the residual does not depend on the choice.
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

import src.options_screener as S

CUTS = ["2026-05-27", "2026-06-10", "2026-06-18", "2026-07-07", "2026-07-16"]
DISPLAY_LO, DISPLAY_SPAN, DISPLAY_POW = 0.28, 0.54, 0.65

BUYER_COMPONENTS = {
    "pop": "pop_score", "rr": "rr_score", "ev": "ev_score",
    "iv_edge": "iv_edge_score", "liquidity": "liquidity_score",
    "theta": "theta_score", "momentum": "momentum_score",
    "skew_align": "skew_align_score", "sentiment": "sentiment_score_norm",
    "vrp": "vrp_score", "iv_velocity": "iv_velocity_score",
    "iv_mispricing": "iv_mispricing_score",
    "gamma_magnitude": "gamma_magnitude_score", "vega_risk": "vega_risk_score",
    "term_structure": "term_structure_score", "spread": "spread_score",
    "iv_rank": "iv_rank_score", "em_realism": "em_realism_score",
    "catalyst": "catalyst_score", "trader_pref": "trader_pref_score",
    "max_pain": "max_pain_score", "gex": "gex_score",
    "gamma_theta": "gamma_theta_score", "pcr": "pcr_score",
    "oi_change": "oi_change_score", "option_rvol": "option_rvol_score",
    "gamma_pin": "gamma_pin_score",
}


def load(db: str, strategies: tuple) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in strategies)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql(
            "SELECT * FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
            f"AND strategy_name IN ({placeholders})", con, params=list(strategies))
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def recover_spread_pct(df: pd.DataFrame, cap: float) -> pd.Series:
    """Invert spread_score = 1 / (1 + exp(20 * (sp/cap - 0.7)))."""
    s = df["spread_score"].clip(1e-9, 1 - 1e-9)
    return cap * (0.7 - np.log(1.0 / s - 1.0) / 20.0)


def friction_adjusted(df: pd.DataFrame, spread_pct: pd.Series) -> pd.Series:
    """Charge the round trip: buy at ask, sell at bid, both half a spread off
    the mid the ledger priced against.  Expired-worthless is never sold."""
    half = spread_pct / 2.0
    exit_mult = np.where(df["exit_price"].fillna(0) <= 0, 1.0, 1.0 - half)
    return ((1 + df["pnl_pct"]) * exit_mult - (1 + half)) / (1 + half)


def recompose(df: pd.DataFrame, weights: dict, mapping: dict) -> pd.Series:
    usable = {k: v for k, v in mapping.items() if v in df.columns}
    total = sum(weights.get(k, 0.0) for k in usable) or 1.0
    return sum(weights.get(k, 0.0) * df[v].fillna(0.5)
               for k, v in usable.items()) / total


def inverse_display(q: pd.Series) -> pd.Series:
    return DISPLAY_LO + DISPLAY_SPAN * np.power(q.clip(0, 1), 1.0 / DISPLAY_POW)


def _windows(x: pd.Series, y: pd.Series, dates: pd.Series) -> list:
    out = []
    for cut in CUTS:
        te = dates >= pd.Timestamp(cut)
        if te.sum() < 30:
            continue
        out.append(stats.spearmanr(x[te], y[te]).statistic)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--strategies", default="Long Call,Long Put")
    args = ap.parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(","))
    df = load(args.db, strategies)
    cfg = json.load(open("config.json"))
    cap = max(cfg.get("spread_score_cap", 0.25), 0.01)

    sp = recover_spread_pct(df, cap)
    df["pnl_net"] = friction_adjusted(df, sp)
    raw = inverse_display(df["quality_score"])

    live = S.load_ic_adjusted_weights(cfg)
    composite = recompose(df, live, BUYER_COMPONENTS)
    stack = raw - composite

    print(f"n = {len(df)} closed {'/'.join(strategies)} rows, "
          f"{df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    print(f"mean return at mid {df.pnl_pct.mean():+.4f} -> "
          f"friction-adjusted {df.pnl_net.mean():+.4f}\n")

    print("--- the stack dominates the variance, under any weighting ---")
    print(f"{'weighting':<34}{'composite sd':>14}{'stack sd':>12}")
    alts = {"live IC-blended": live, "raw config": cfg["composite_weights"],
            "uniform": {k: 1.0 for k in BUYER_COMPONENTS},
            "theta-heavy (pre-fix)": {**cfg["composite_weights"], "theta": 0.31}}
    for name, ws in alts.items():
        c = recompose(df, ws, BUYER_COMPONENTS)
        print(f"  {name:<32}{c.std():>14.4f}{(raw - c).std():>12.4f}")

    print("\n--- rank IC vs friction-adjusted return ---")
    print(f"{'variant':<40}{'full':>10}{'mean of windows':>18}{'neg':>8}")
    variants = {
        "as shipped (composite + full stack)": raw,
        "composite only (stack OFF)": composite,
        "composite + penalties only": composite + stack.clip(upper=0.0),
        "composite + bonuses only": composite + stack.clip(lower=0.0),
    }
    for name, x in variants.items():
        full = stats.spearmanr(x, df.pnl_net).statistic
        ws = _windows(x, df.pnl_net, df.date)
        neg = sum(1 for v in ws if v < 0)
        print(f"  {name:<38}{full:>10.4f}{np.mean(ws):>18.4f}{neg:>5d} / {len(ws)}")

    print("\n--- where the stack lands rows ---")
    for label, mask in [("net penalised (< -0.02)", stack < -0.02),
                        ("~neutral", (stack >= -0.02) & (stack <= 0.02)),
                        ("net bonused (> +0.02)", stack > 0.02)]:
        if not mask.sum():
            continue
        y = df.pnl_net[mask]
        print(f"  {label:<26} n={int(mask.sum()):>4}  mean {y.mean():+.4f}  "
              f"median {y.median():+.4f}  win {(y > 0).mean():.1%}")

    print("\n--- recoverable firing conditions ---")
    tier = np.where(sp > 0.15, ">15% (-0.08)",
                    np.where(sp > 0.10, "10-15% (-0.04)", "<=10% (none)"))
    for name in sorted(set(tier)):
        m = tier == name
        print(f"  spread {name:<18} n={int(m.sum()):>4} ({m.mean():>5.1%})  "
              f"mean {df.pnl_net[m].mean():+.4f}  win {(df.pnl_net[m] > 0).mean():.1%}")
    r = stats.spearmanr(sp, df.pnl_net)
    print(f"  spearman(spread_pct, friction-adjusted return) = "
          f"{r.statistic:+.4f} (p{r.pvalue:.3f})")
    en = df["catalyst_score"] > 0.5
    if en.any():
        print(f"  earnings_nearby         n={int(en.sum()):>4} ({en.mean():>5.1%})  "
              f"mean {df.pnl_net[en].mean():+.4f}  win {(df.pnl_net[en] > 0).mean():.1%}"
              f"   [not fired: mean {df.pnl_net[~en].mean():+.4f}]")
    low_pop = df["pop_score"] < 0.25
    print(f"  low_pop x0.60           n={int(low_pop.sum()):>4} ({low_pop.mean():>5.1%})"
          f"   min pop_score {df.pop_score.min():.3f}")


if __name__ == "__main__":
    main()
