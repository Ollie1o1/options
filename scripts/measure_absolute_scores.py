"""Does replacing the within-chain rank on theta/vega with an absolute
mapping change out-of-sample IC?  Research only: writes nothing, changes
nothing, opens the ledger read-only.

Both components are within-chain ranks because calculate_scores runs
per-symbol.  The composite is then compared ACROSS symbols, so the rank
carries no cross-ticker level information.  This measures the swap.
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


def load(db: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql(
            "SELECT * FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
            "AND strategy_name IN ('Long Call','Long Put')", con)
    finally:
        con.close()
    df = df.dropna(subset=["theta_score", "vega_risk_score", "entry_theta",
                           "entry_vega", "entry_price", "quality_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def inverse_display(q: pd.Series) -> pd.Series:
    """Undo _cross_section_normalize.  Verified: stored scores span [0, 0.9999]
    and invert into [0.2800, 0.8199], the documented raw range."""
    return DISPLAY_LO + DISPLAY_SPAN * np.power(q.clip(0, 1), 1.0 / DISPLAY_POW)


def forward_display(raw: pd.Series) -> pd.Series:
    return np.power(((raw - DISPLAY_LO) / DISPLAY_SPAN).clip(0, 1), DISPLAY_POW)


def theta_pressure(df: pd.DataFrame) -> pd.Series:
    return df["entry_theta"].abs() / df["entry_price"].clip(lower=0.01)


def vega_dollar(df: pd.DataFrame) -> pd.Series:
    return df["entry_vega"].abs() * 100.0


def fit_logistic(train_vals: pd.Series) -> tuple:
    """Centre on the train-fold median of log10(x), scale so the train IQR
    spans roughly the central half of the sigmoid.  Calibrating on train only
    is what keeps the evaluation honest."""
    lg = np.log10(train_vals.clip(lower=1e-6))
    centre = float(lg.median())
    iqr = float(lg.quantile(0.75) - lg.quantile(0.25))
    scale = 2.197 / max(iqr, 1e-6)   # logit(0.75)*2 / IQR -> IQR spans 0.25..0.75
    return centre, scale


def apply_logistic(vals: pd.Series, centre: float, scale: float) -> pd.Series:
    lg = np.log10(vals.clip(lower=1e-6))
    return pd.Series(1.0 / (1.0 + np.exp(-scale * (lg - centre))), index=vals.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="paper_trades.db")
    args = ap.parse_args()

    df = load(args.db)
    cfg = json.load(open("config.json"))
    w = S.load_ic_adjusted_weights(cfg)
    w_sum = sum(w.values()) or 1.0
    w_theta, w_vega = w.get("theta", 0.0), w.get("vega_risk", 0.0)

    print(f"n = {len(df)} closed Long Call/Put rows, "
          f"{df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    print(f"live weights: theta {w_theta/w_sum:.2%}, vega_risk {w_vega/w_sum:.2%}\n")

    raw = inverse_display(df["quality_score"])
    tp, vd = theta_pressure(df), vega_dollar(df)

    rows = []
    last_const = None
    for cut in CUTS:
        cut_ts = pd.Timestamp(cut)
        tr, te = df["date"] < cut_ts, df["date"] >= cut_ts
        if tr.sum() < 30 or te.sum() < 30:
            print(f"  (skipped {cut}: n_tr={int(tr.sum())} n_te={int(te.sum())})")
            continue

        t_c, t_s = fit_logistic(tp[tr])
        v_c, v_s = fit_logistic(vd[tr])
        last_const = (t_c, t_s, v_c, v_s)

        # Buyers: fast decay is bad, so the score is 1 - pressure, matching
        # the sign convention at options_screener.py:1749.
        theta_new = 1.0 - apply_logistic(tp, t_c, t_s)
        # vega_risk_score = 1 - (vega_rank * iv_percentile).  The iv_percentile
        # factor is not stored, so hold it at the neutral 0.5 it defaults to,
        # applied identically to both arms -> the comparison stays like-for-like.
        vega_new = 1.0 - apply_logistic(vd, v_c, v_s) * 0.5

        delta = ((w_theta * (theta_new - df["theta_score"])
                  + w_vega * (vega_new - df["vega_risk_score"])) / w_sum)
        raw_new = (raw + delta).clip(0.0, 1.0)

        y = df.loc[te, "pnl_pct"]
        ic_old = stats.spearmanr(forward_display(raw)[te], y)
        ic_new = stats.spearmanr(forward_display(raw_new)[te], y)
        rows.append((cut, int(tr.sum()), int(te.sum()),
                     ic_old.statistic, ic_old.pvalue,
                     ic_new.statistic, ic_new.pvalue))

    print(f"{'cut':<12}{'n_tr':>6}{'n_te':>6}{'OOS rank IC old':>21}"
          f"{'OOS rank IC new':>21}{'delta':>9}")
    for c, ntr, nte, io, po, inw, pn in rows:
        print(f"{c:<12}{ntr:>6}{nte:>6}{io:>14.4f} (p{po:.2f})"
              f"{inw:>14.4f} (p{pn:.2f}){inw-io:>9.4f}")
    if rows:
        mo = float(np.mean([r[3] for r in rows]))
        mn = float(np.mean([r[5] for r in rows]))
        print(f"\nmean OOS rank IC:  old {mo:+.4f}   new {mn:+.4f}   "
              f"difference {mn-mo:+.4f}")
        t_c, t_s, v_c, v_s = last_const
        theta_new = 1.0 - apply_logistic(tp, t_c, t_s)
        vega_new = 1.0 - apply_logistic(vd, v_c, v_s) * 0.5
        delta = ((w_theta * (theta_new - df["theta_score"])
                  + w_vega * (vega_new - df["vega_risk_score"])) / w_sum)
        full = stats.spearmanr(forward_display(raw),
                               forward_display((raw + delta).clip(0.0, 1.0)))
        print(f"rank correlation between the two orderings: {full.statistic:.4f}")
        print(f"\nfinal-fold constants: THETA_LOG_CENTRE={t_c:.4f} "
              f"THETA_LOG_SCALE={t_s:.4f} VEGA_LOG_CENTRE={v_c:.4f} "
              f"VEGA_LOG_SCALE={v_s:.4f}")


if __name__ == "__main__":
    main()
