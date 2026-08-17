#!/usr/bin/env python3
"""Does the IV term-structure nudge improve the vol forecast, or is it noise?

PRE-REGISTERED 2026-08-17, before any result was looked at.

WHAT IS BEING TESTED
--------------------
`calculate_metrics` multiplies the EV's vol basis by 1.05 / 0.95 / 1.0:

    exp_iv_mean = df.groupby("expiration")["impliedVolatility"].transform("mean")
    chain_iv_mean = df["impliedVolatility"].mean()
    ts_signal = where((exp_iv_mean > chain_iv_mean*1.02) & (dte > 20), 1.05,
                where((exp_iv_mean < chain_iv_mean*0.98) & (dte > 20), 0.95, 1.0))

The +/-5% is hand-set and has never been measured. This repo records what
hand-set constants have cost before ([[project_adjustment_stack_carries_the_negative]],
additive constants at IC -0.096), so it does not get the benefit of the doubt —
but it does get a fair test.

IT IS NOT OBVIOUSLY WORTHLESS. On the 1,180-window price study an ORACLE +/-5%
(sign always correct) cut RMSE 10.1%, a coin flip cost 1.0%, and always-wrong
cost 11.1%. The payoff is near-symmetric, so break-even is ~52% directional
accuracy. The question is purely whether the signal carries information.

Prior evidence does NOT settle it: `term_slope` died on the 2020-21 holdout
(IC +0.043 -> -0.057) and residualises to IC|ctl +0.043 (p=0.0855), but both
measured term slope against TRADE OUTCOMES. Predicting vol-forecast error is a
different target. [[project_holdout_20260809]] says explicitly: "Only retest if
optionsDX (all expirations) lands." It landed.

HYPOTHESES
----------
H1  The SHIPPED rule predicts the sign of (realized_vol - forecast_vol) at
    better than 52% - i.e. it clears its own break-even.
H2  A properly specified ATM term slope (far-tenor ATM IV minus near-tenor ATM
    IV) does better than the shipped rule, which compares an expiration's mean
    IV to the whole-chain mean and is therefore confounded by strike
    composition: an expiration carrying more OTM strikes reads "high IV" from
    skew alone, with no term-structure content.

DESIGN
------
* SPY, optionsDX, 2010-2023, one snapshot per date.
* NON-OVERLAPPING: sample every 21st trading day, so no two windows share a
  return. Overlapping windows would make every band look far tighter.
* Signal computed from the chain at t only. Forecast from prices up to t only.
  Realized vol measured over the NEXT 21 trading days. No lookahead anywhere.
* The scanner only ever sees 14-45 DTE and strikes inside `moneyness_band`
  0.30, so the shipped rule is reproduced on exactly that subset rather than
  on the full chain it never touches.

METRICS (fixed in advance)
--------------------------
1. directional accuracy vs the 52% break-even   <- decides H1
2. RMSE with the nudge applied vs without       <- the practical question
3. the same for the properly specified slope    <- decides H2

Usage: PYTHONPATH=$PWD python scripts/term_structure_study.py [--every N]
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys

import numpy as np
import pandas as pd

DB = "data/optionsdx.db"
HORIZON = 21          # trading days
DTE_LO, DTE_HI = 14, 45
MONEYNESS = 0.30      # config.json moneyness_band
BREAKEVEN = 0.52      # from the oracle/coin-flip payoffs, see docstring


def underlying_series(conn) -> pd.Series:
    rows = conn.execute(
        "SELECT date, MAX(underlying) FROM odx_chain WHERE symbol='SPY' "
        "GROUP BY date ORDER BY date").fetchall()
    s = pd.Series({d: float(u) for d, u in rows if u})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _ann_vol(px: pd.Series) -> float | None:
    r = np.log(px / px.shift(1)).dropna()
    if len(r) < 5:
        return None
    return float(r.std() * math.sqrt(252))


def forecast_at(px: pd.Series, i: int) -> float | None:
    """The shipped basis: 0.5*hv_252d + 0.5*hv_30d, close-to-close.

    optionsDX carries no High/Low, so the 30d leg uses the code's own
    no-Parkinson fallback (0.5*rolling30 + 0.5*ewma20) rather than a different
    estimator invented here.
    """
    hist = px.iloc[:i + 1]
    if len(hist) < 260:
        return None
    long_w = _ann_vol(hist.iloc[-253:])
    roll30 = _ann_vol(hist.iloc[-31:])
    r = np.log(hist / hist.shift(1)).dropna()
    ewma = float(np.sqrt((r ** 2).ewm(span=20, adjust=False).mean().iloc[-1] * 252))
    short_w = 0.5 * roll30 + 0.5 * ewma if roll30 else ewma
    if not long_w or not short_w:
        return None
    return 0.5 * long_w + 0.5 * short_w


def signals_at(conn, date_str: str) -> tuple:
    """(shipped_multiplier, atm_slope) from the chain on `date_str`."""
    q = ("SELECT expiration, strike, iv, underlying, "
         "  CAST(julianday(expiration)-julianday(date) AS INT) AS dte "
         "FROM odx_chain WHERE symbol='SPY' AND date=? AND iv > 0")
    df = pd.DataFrame(conn.execute(q, (date_str,)).fetchall(),
                      columns=["expiration", "strike", "iv", "underlying", "dte"])
    if df.empty:
        return None, None
    u = float(df["underlying"].iloc[0])
    band = df[(df["dte"].between(DTE_LO, DTE_HI))
              & ((df["strike"] - u).abs() / u <= MONEYNESS)]
    if band.empty or band["expiration"].nunique() < 2:
        return None, None

    # --- the SHIPPED rule, on the subset the scanner actually sees ---
    chain_mean = band["iv"].mean()
    exp_mean = band.groupby("expiration")["iv"].mean()
    exp_dte = band.groupby("expiration")["dte"].first()
    mult = []
    for e in exp_mean.index:
        if exp_dte[e] <= 20:
            mult.append(1.0)
        elif exp_mean[e] > chain_mean * 1.02:
            mult.append(1.05)
        elif exp_mean[e] < chain_mean * 0.98:
            mult.append(0.95)
        else:
            mult.append(1.0)
    shipped = float(np.mean(mult))       # the frame's average nudge

    # --- a properly specified ATM slope: far ATM IV - near ATM IV ---
    atm = df[((df["strike"] - u).abs() / u <= 0.02) & df["dte"].between(7, 120)]
    slope = None
    if not atm.empty:
        by = atm.groupby("expiration").agg(iv=("iv", "mean"), dte=("dte", "first"))
        by = by.sort_values("dte")
        near = by[by["dte"] <= 30]
        far = by[by["dte"] >= 45]
        if not near.empty and not far.empty:
            slope = float(far["iv"].iloc[0] - near["iv"].iloc[-1])
    return shipped, slope


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=HORIZON,
                    help="sample every Nth trading day (default = horizon, "
                         "which makes the windows non-overlapping)")
    a = ap.parse_args()

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    px = underlying_series(conn)
    print(f"SPY underlying: {len(px)} days, {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}")

    rows = []
    for i in range(260, len(px) - HORIZON, a.every):
        d = px.index[i]
        fc = forecast_at(px, i)
        if fc is None:
            continue
        realized = _ann_vol(px.iloc[i:i + HORIZON + 1])
        if realized is None or realized <= 0:
            continue
        shipped, slope = signals_at(conn, d.strftime("%Y-%m-%d"))
        if shipped is None:
            continue
        rows.append({"date": d, "forecast": fc, "realized": realized,
                     "shipped": shipped, "slope": slope})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no observations")
        return 1

    err = df["realized"] - df["forecast"]          # >0 => forecast too LOW
    rmse = lambda f: float(np.sqrt(((f - df["realized"]) ** 2).mean()))

    print(f"\n{len(df)} NON-OVERLAPPING windows, {df.date.min():%Y-%m} to {df.date.max():%Y-%m}")
    print(f"forecast too low on {100*(err>0).mean():.1f}% of them\n")

    # H1 -- the shipped rule
    up = df["shipped"] > 1.0
    dn = df["shipped"] < 1.0
    called = up | dn
    correct = ((up & (err > 0)) | (dn & (err < 0)))
    acc = correct[called].mean() if called.any() else float("nan")
    print("=== H1: the SHIPPED +/-5% rule ===")
    print(f"  fires on {called.mean()*100:.1f}% of windows "
          f"({int(up.sum())} up, {int(dn.sum())} down, {int((~called).sum())} neutral)")
    print(f"  directional accuracy when it fires : {acc*100:.1f}%   (break-even {BREAKEVEN*100:.0f}%)")
    print(f"  RMSE without the nudge             : {rmse(df['forecast']):.4f}")
    print(f"  RMSE with    the nudge             : {rmse(df['forecast']*df['shipped']):.4f}")

    # H2 -- a properly specified ATM slope
    sub = df.dropna(subset=["slope"])
    print("\n=== H2: a properly specified ATM term slope (far - near) ===")
    if len(sub) < 30:
        print(f"  only {len(sub)} windows carry both tenors — not testable")
    else:
        s_up, s_dn = sub["slope"] > 0, sub["slope"] < 0
        s_err = sub["realized"] - sub["forecast"]
        s_ok = ((s_up & (s_err > 0)) | (s_dn & (s_err < 0)))
        print(f"  n={len(sub)}  directional accuracy: {s_ok.mean()*100:.1f}%")
        nudged = sub["forecast"] * np.where(s_up, 1.05, np.where(s_dn, 0.95, 1.0))
        base = float(np.sqrt(((sub['forecast'] - sub['realized'])**2).mean()))
        alt = float(np.sqrt(((nudged - sub['realized'])**2).mean()))
        print(f"  RMSE without {base:.4f}  ->  with slope-nudge {alt:.4f} "
              f"({100*(base-alt)/base:+.2f}%)")
        print(f"  Spearman(slope, forecast error): "
              f"{sub['slope'].corr(s_err, method='spearman'):+.4f}")
    # --- base-rate control: is the "accuracy" above just a biased coin? ---
    always_down = (err < 0).mean()
    print("\n=== BASE-RATE CONTROL (the trap in the numbers above) ===")
    print(f"  a rule that ALWAYS said 'down' would be right: {always_down*100:.1f}%")
    print(f"  the shipped rule fires 'down' on {int(dn.sum())}/{len(df)} windows, "
          f"so its accuracy is mostly this base rate, not skill")

    # --- holdout: does the slope relationship survive a split? ---
    if len(sub) >= 60:
        mid = sub["date"].quantile(0.5)
        print("\n=== HOLDOUT on the ATM slope (Spearman vs forecast error) ===")
        for lab, part in (("EARLY", sub[sub.date <= mid]), ("LATE", sub[sub.date > mid])):
            e = part["realized"] - part["forecast"]
            print(f"  {lab:<6} n={len(part):3}  {part['date'].min():%Y-%m} to "
                  f"{part['date'].max():%Y-%m}  Spearman "
                  f"{part['slope'].corr(e, method='spearman'):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
