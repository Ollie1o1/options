"""Index put spread, one crossing, held to expiry — the registered estimator.

Registration: docs/PREREG_INDEX_HOLD_TO_EXPIRY_20261015.md. Reuses the task-4
estimator's replay, outcome frame and statistics so the two are comparable.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -u -m scripts.index_hold_to_expiry_odx
"""
from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd

from scripts.live_bull_put_odx_stress import (DB, END, START, STRESS_YEARS, SYM,
                                              drawdown, fmt, frame, month_boot,
                                              pf, run, year_table)
from src.optionsdx import available_dates, terminal_date

ENTRY = {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
         "width_pct_of_spot": 0.02, "max_friction_pct_of_credit": 0.25}
EXIT = {"hold_to_expiry": True}


def by(df: pd.DataFrame, cluster: str) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[cluster].str[:7]
    return out


def judge(df: pd.DataFrame) -> Dict:
    a = month_boot(df, lambda d: d.x.mean())
    b = month_boot(df, lambda d: pf(d.roc.values))
    return {"A": a, "B": b, "pass_A": a["lo"] > 0, "pass_B": b["lo"] > 1}


def show(label: str, df: pd.DataFrame) -> Dict:
    j = judge(df)
    roc = month_boot(df, lambda d: d.roc.mean())
    print(f"  {label}")
    print(f"    n={len(df)}  win {(df.pnl > 0).mean():.3f}  RoC {fmt(roc)}")
    print(f"    A' excess {fmt(j['A'])} -> {'PASS' if j['pass_A'] else 'fail'}")
    print(f"    B' PF     {fmt(j['B'], pct=False)} -> {'PASS' if j['pass_B'] else 'fail'}")
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--skip-sens", action="store_true")
    a = ap.parse_args()
    dates = available_dates(DB, SYM, START, END)
    last = terminal_date(DB, SYM)
    term = {SYM: last} if last else {}

    def replay(entry, exit_, label):
        tr, sp, st = run(entry, exit_, label, dates, term)
        return frame(tr, sp, with_gate=False), sp, st

    df0, spot, stats = replay(ENTRY, EXIT, "R")
    if a.csv:
        df0.to_csv(a.csv, index=False)
    print(f"optionsDX {SYM} {dates[0]}..{dates[-1]}  engine: "
          f"{ {k: v for k, v in stats.items() if v} }")
    hold = (pd.to_datetime(df0.exit) - pd.to_datetime(df0.entry)).dt.days
    print(f"trades {len(df0)} | expiry months {df0.expiration.str[:7].nunique()} | "
          f"median DTE {df0.dte.median():.0f} | median hold {hold.median():.0f}d | "
          f"short delta {df0.short_delta.median():.3f} | width/spot "
          f"{(df0.width / df0.s0).median():.4f} | credit/width "
          f"{(df0.credit / df0.width).median():.3f}")
    print("exit reasons:", df0.reason.value_counts().to_dict())

    df = by(df0, "expiration")
    print("\n== PRIMARY (clustered by expiration month)")
    j = show("rule R", df)
    print(f"  VERDICT: R {'PASSES' if j['pass_A'] and j['pass_B'] else 'FAILS'} "
          f"decision rule §3 step 2 (A' {'pass' if j['pass_A'] else 'fail'}, "
          f"B' {'pass' if j['pass_B'] else 'fail'})")
    print(f"  delta-equivalent mean {fmt(month_boot(df, lambda d: (d.H / d.car).mean()))}")

    dd = drawdown(df)
    print(f"  book (2% of $100k/trade, not compounded): total {100 * dd['final']:+.1f}% | "
          f"max DD {100 * dd['max_dd']:.1f}% (peak {dd['peak']}, trough {dd['trough']}, "
          f"recovered {dd['recovered']}, {dd['days']} days)")
    yt = year_table(df, spot)
    print("\n  per year:")
    print(yt.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("  stress years:", ", ".join(
        f"{y}: RoC {yt.loc[y, 'roc']:+.3f} x {yt.loc[y, 'x']:+.3f}"
        for y in STRESS_YEARS if y in yt.index))

    print("\n== Non-overlapping series (one trade per expiration month; parallel, no authority)")
    first = (df.sort_values("entry").assign(em=df.expiration.str[:7])
             .groupby("em", as_index=False).head(1))
    show("first entry per expiry month", first)

    print("\n== Sensitivities (no authority)")
    show("cluster by entry month", by(df0, "entry"))
    mid = df0.copy()
    mid["roc"] = (mid.pnl + 100.0 * mid.fric_entry) / mid.car
    mid["x"] = mid.roc - mid.H / mid.car
    mid["pnl"] = mid.pnl + 100.0 * mid.fric_entry
    show("mid fill at entry", by(mid.dropna(subset=["roc"]), "expiration"))
    if not a.skip_sens:
        for label, entry in (
                ("short delta 0.16", {**ENTRY, "short_delta": 0.16}),
                ("short delta 0.32", {**ENTRY, "short_delta": 0.32}),
                ("DTE [45,60]", {**ENTRY, "dte": [45, 60]}),
                ("width 1% of spot", {**ENTRY, "width_pct_of_spot": 0.01}),
                ("width 3% of spot", {**ENTRY, "width_pct_of_spot": 0.03}),
                ("credit floor 0.20", {**ENTRY, "min_credit_to_width": 0.20})):
            d, _, _ = replay(entry, EXIT, label)
            show(label, by(d, "expiration"))


if __name__ == "__main__":
    main()
