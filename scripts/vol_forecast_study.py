#!/usr/bin/env python3
"""Which realized-vol estimator actually predicts the horizon we trade?

PRE-REGISTERED 2026-08-17, before any result was looked at.

WHY
---
`calculate_metrics` prices every contract at Black-Scholes on `hv_252d` — a
252-DAY trailing realized vol — and the options being priced are 14-45 DTE.
That is a one-year backward window forecasting a one-month forward outcome.
`hv_30d` (a blend of 30d rolling, EWMA-20 and Parkinson) is horizon-matched and
already computed, but sits second in the fallback chain and is used only when
the long window is missing.

The long window is NOT obviously wrong. It was introduced deliberately on
2026-08-04 because the short window read 51.8% off a stale earnings gap and
turned a $5 edge into a reported +$4,664. Short estimators are horizon-matched
but unstable. So this is a genuine trade-off and the answer is empirical.

HYPOTHESES
----------
H1  `hv_252d` is NOT the most accurate predictor of forward realized vol over
    the 21-trading-day horizon these contracts are priced for.
H2  A horizon-matched estimator beats it on RMSE but is WORSE on tail error
    (the blow-up case that motivated the long window in the first place).

DESIGN
------
* Universe: liquid names the screener actually scans.
* For each symbol and each sample date t, every estimator is computed from
  data up to and including t — no lookahead — using THE REPO'S OWN functions,
  so this measures what the code does rather than a reimplementation.
* Realization: annualized close-to-close realized vol over the NEXT `HORIZON`
  trading days.
* Windows are NON-OVERLAPPING (`t` steps by HORIZON). Overlapping windows
  share returns and would make every error band look far tighter than it is.

METRICS (primary first, fixed in advance)
-----------------------------------------
1. RMSE against realized forward vol   <- decides H1
2. MAE
3. Spearman rank correlation           <- does it order periods correctly
4. Bias = mean(forecast - realized)    <- systematic over/under-statement
5. P(forecast > 1.5x realized)         <- the phantom-edge tail, decides H2

Nothing here changes the screener. It reports; a change is a separate,
argued decision.

Usage:  PYTHONPATH=$PWD python scripts/vol_forecast_study.py [--symbols N] [--years N]
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZON = 21          # trading days ~ 30 calendar days, the middle of 14-45 DTE
MIN_HISTORY = 260     # enough for the 252-day window to exist at all

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "GLD", "AAPL", "MSFT", "NVDA", "AMZN",
            "GOOGL", "META", "AVGO", "JPM", "XOM", "PFE", "F", "T", "KO",
            "WMT", "DIS"]


def _estimators(hist: pd.DataFrame) -> dict:
    """Every candidate, computed from the repo's own implementations."""
    from src.data_fetching import (calculate_ewma_volatility,
                                   calculate_historical_volatility,
                                   calculate_parkinson_volatility,
                                   long_window_volatility)

    roll30 = calculate_historical_volatility(hist, period=30)
    ewma20 = calculate_ewma_volatility(hist, span=20)
    park30 = calculate_parkinson_volatility(hist, period=30)
    hv252 = long_window_volatility(hist)

    # The exact blend `data_fetching` assembles as `hv_30d`, fallbacks included.
    if roll30 and ewma20 and park30:
        blend30 = 0.34 * roll30 + 0.33 * ewma20 + 0.33 * park30
    elif roll30 and ewma20:
        blend30 = 0.5 * roll30 + 0.5 * ewma20
    else:
        blend30 = roll30 or ewma20

    out = {
        "hv_252d (CURRENT)": hv252,
        "hv_30d blend": blend30,
        "rolling_30": roll30,
        "ewma_20": ewma20,
        "parkinson_30": park30,
        "rolling_60": calculate_historical_volatility(hist, period=60),
        "rolling_90": calculate_historical_volatility(hist, period=90),
    }
    # Compromise candidates: keep the long window's stability, add horizon match
    if hv252 and blend30:
        out["50/50 252d+30d"] = 0.5 * hv252 + 0.5 * blend30
        out["70/30 252d+30d"] = 0.7 * hv252 + 0.3 * blend30
    return out


def _realized_forward(hist: pd.DataFrame, start: int, horizon: int):
    """Annualized close-to-close realized vol over the NEXT `horizon` bars."""
    fwd = hist.iloc[start:start + horizon + 1]
    if len(fwd) < horizon:
        return None
    r = np.log(fwd["Close"] / fwd["Close"].shift(1)).dropna()
    if len(r) < 5:
        return None
    return float(r.std() * math.sqrt(252))


def collect(symbols, years):
    import yfinance as yf
    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            hist = yf.Ticker(sym).history(period=f"{years}y", auto_adjust=False)
        except Exception as exc:
            print(f"  [{i}/{len(symbols)}] {sym}: fetch failed ({exc})")
            continue
        if hist is None or len(hist) < MIN_HISTORY + HORIZON:
            print(f"  [{i}/{len(symbols)}] {sym}: too little history")
            continue
        n = 0
        # Non-overlapping: step by the horizon so no two windows share returns.
        for t in range(MIN_HISTORY, len(hist) - HORIZON, HORIZON):
            past = hist.iloc[:t + 1]
            realized = _realized_forward(hist, t, HORIZON)
            if realized is None or realized <= 0:
                continue
            for name, val in _estimators(past).items():
                if val is None or not np.isfinite(val) or val <= 0:
                    continue
                rows.append({"symbol": sym, "date": hist.index[t],
                             "estimator": name, "forecast": float(val),
                             "realized": realized})
            n += 1
        print(f"  [{i}/{len(symbols)}] {sym}: {n} non-overlapping windows")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in df.groupby("estimator"):
        err = g["forecast"] - g["realized"]
        out.append({
            "estimator": name,
            "n": len(g),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE": float(err.abs().mean()),
            "spearman": float(g["forecast"].corr(g["realized"], method="spearman")),
            "bias": float(err.mean()),
            "P(fc>1.5x real)": float((g["forecast"] > 1.5 * g["realized"]).mean()),
        })
    return pd.DataFrame(out).sort_values("RMSE").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=len(UNIVERSE))
    ap.add_argument("--years", type=int, default=6)
    a = ap.parse_args()

    syms = UNIVERSE[:a.symbols]
    print(f"Pre-registered vol-forecast study — horizon {HORIZON} trading days, "
          f"{len(syms)} symbols, {a.years}y, NON-OVERLAPPING windows\n")
    df = collect(syms, a.years)
    if df.empty:
        print("\nNo data collected.")
        return 1

    # Holdout: the ranking has to survive a split it was not chosen on.
    df = df.sort_values("date")
    mid = df["date"].quantile(0.5)
    early, late = df[df["date"] <= mid], df[df["date"] > mid]

    res = report(df)
    print(f"\n{'='*84}\nRESULTS  (lower RMSE better; bias>0 = the estimator "
          f"OVERSTATES vol)\n{'='*84}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    cur = res[res["estimator"] == "hv_252d (CURRENT)"]
    best = res.iloc[0]
    print(f"\nwindows per estimator: {int(res['n'].median())}")
    if not cur.empty:
        c = cur.iloc[0]
        print(f"\nCURRENT hv_252d : RMSE {c['RMSE']:.4f}  rank "
              f"{res.index[res['estimator'] == 'hv_252d (CURRENT)'][0] + 1} of {len(res)}")
        print(f"BEST            : {best['estimator']}  RMSE {best['RMSE']:.4f}"
              f"  ({100*(c['RMSE']-best['RMSE'])/c['RMSE']:+.1f}% vs current)")
        print(f"\nH2 check — tail blow-ups, P(forecast > 1.5x realized):")
        for _, r in res.sort_values("P(fc>1.5x real)").iterrows():
            print(f"   {r['estimator']:<22} {r['P(fc>1.5x real)']:.3f}")
    print(f"\n{'='*84}\nHOLDOUT — same ranking on two disjoint halves?\n{'='*84}")
    for label, part in (("EARLY half", early), ("LATE half", late)):
        r = report(part)
        span = f"{part['date'].min():%Y-%m} to {part['date'].max():%Y-%m}"
        print(f"\n{label}  ({span}, n={len(part)//9} windows)")
        for i, row in r.iterrows():
            mark = "  <-- CURRENT" if "CURRENT" in row["estimator"] else ""
            print(f"   {i+1}. {row['estimator']:<22} RMSE {row['RMSE']:.4f}"
                  f"  tail {row['P(fc>1.5x real)']:.3f}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
