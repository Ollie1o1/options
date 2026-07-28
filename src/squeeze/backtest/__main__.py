"""Run the squeeze study and print the report.

    python -m src.squeeze.backtest --run
    python -m src.squeeze.backtest --run --si-scale 1.25 --ret5d as_intended
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
from typing import List, Optional

from src.squeeze.backtest import DEFAULT_DB
from src.squeeze.backtest import panel as _panel
from src.squeeze.backtest import study as _study

# Local rebuild cache only: this module is the sole writer, the file never leaves
# data/ and is never fetched from anywhere, so pickle carries no untrusted input.
# Delete it or pass --rebuild to regenerate from the source tables.
PANEL_CACHE = "data/squeeze_panel.pkl"
_RULE = "─" * 78


def _fmt_pct(x, nd=1):
    return "  n/a " if x is None or x != x else f"{100.0 * x:>5.{nd}f}%"


def _load_panel(db: str, prices_db: str, rebuild: bool = False) -> List[dict]:
    if not rebuild and os.path.exists(PANEL_CACHE):
        with open(PANEL_CACHE, "rb") as fh:
            return pickle.load(fh)
    rows = _panel.build(db, prices_db)
    os.makedirs(os.path.dirname(PANEL_CACHE) or ".", exist_ok=True)
    with open(PANEL_CACHE, "wb") as fh:
        pickle.dump(rows, fh)
    return rows


def _coverage(db: str) -> dict:
    conn = sqlite3.connect(db, timeout=60)
    dates = conn.execute(
        "SELECT COUNT(*), MIN(settlement_date), MAX(settlement_date) "
        "FROM si_dates WHERE fetched=1").fetchone()
    conn.close()
    return {"dates": dates[0], "first": dates[1], "last": dates[2]}


def report(rows: List[dict], horizons=(10, 21, 42), k: float = 2.0,
           si_scale: float = 1.0, ret5d: str = "as_written",
           n_boot: int = 2000, db: str = DEFAULT_DB) -> None:
    _panel.grade(rows, si_scale=si_scale, ret5d_scale=ret5d)
    cov = _coverage(db)

    print()
    print("SQUEEZE GRADER — POINT-IN-TIME BACKTEST")
    print(_RULE)
    print(f"  settlement dates : {cov['dates']}  ({cov['first']} → {cov['last']})")
    print(f"  panel rows       : {len(rows):,}")
    print(f"  symbols          : {len({r['symbol'] for r in rows}):,}")
    print(f"  float assumption : shares_out × {si_scale:g}"
          f"{'  (float = all shares outstanding)' if si_scale == 1.0 else ''}")
    print(f"  ret_5d rule      : {ret5d}")
    print("  iv_skew          : unavailable (no option history for these names)")

    counts = {g: sum(1 for r in rows if r.get("grade") == g) for g in _study.GRADES}
    total = sum(counts.values()) or 1
    print()
    print("  grade distribution (gradeable rows only)")
    for g in _study.GRADES:
        print(f"    {g:<6s} {counts[g]:>8,}  ({100.0 * counts[g] / total:>4.1f}%)")

    cb = _study.coverage_bias(rows, 21, k)
    if cb["lost_n"]:
        print()
        print("  coverage check — rows dropped for missing EDGAR shares-outstanding")
        print(f"    gradeable      {cb['kept_n']:>8,}  ({cb['kept_pct']:.1f}% of priced rows)")
        print(f"    ungradeable    {cb['lost_n']:>8,}")
        print(f"    P(+2σ) kept {_fmt_pct(cb['kept_up'])} vs dropped {_fmt_pct(cb['lost_up'])}"
              f"   gap {100 * cb['up_gap']:+.2f}pp")
        print(f"    P(-2σ) kept {_fmt_pct(cb['kept_down'])} vs dropped {_fmt_pct(cb['lost_down'])}")
        print("    (a small gap means the join is ~random w.r.t. outcome)")

    if counts["SETUP"] < 100:
        print()
        print(f"  ⚠  only {counts['SETUP']:,} SETUP observations — too few to test at this"
              f" float assumption. Try a larger --si-scale.")

    print()
    print(_RULE)
    print(f"  OUTCOME PROFILE BY GRADE   (tail = path max ≥ k·σ, σ = trailing 60d vol)")
    print(_RULE)
    hdr = (f"  {'H':>3} {'grade':<6} {'n':>8} {'P(+2σ)':>7} {'P(+3σ)':>7} {'P(-2σ)':>7} "
           f"{'asym':>7} {'med max':>8} {'med end':>8} {'P(+20%)':>8}")
    print(hdr)
    for h in horizons:
        for g in _study.GRADES:
            sel = [r for r in rows if r.get("grade") == g]
            d = _study.describe(sel, h)
            if not d.get("n"):
                continue
            print(f"  {h:>3} {g:<6} {d['n']:>8,} {_fmt_pct(d['p_2sig'])} {_fmt_pct(d['p_3sig'])} "
                  f"{_fmt_pct(d['p_2sig_down'])} {_fmt_pct(d['asymmetry'])} "
                  f"{_fmt_pct(d['median_max'])} {_fmt_pct(d['median_end'])} {_fmt_pct(d['p_up20'])}")
        print()

    print(_RULE)
    print(f"  LIFT: SETUP − NONE   (moving-block bootstrap over settlement dates,"
          f" {n_boot:,} draws)")
    print(_RULE)
    print("  'up' = upside tail rate.  'asym' = upside minus downside tail rate — the")
    print("  control for junk names simply having fat tails in both directions.")
    print()
    for metric, label in (("up", "upside tail"), ("asym", "asymmetry")):
        print(f"  {label}")
        print(f"    {'H':>3} {'SETUP':>7} {'NONE':>7} {'lift':>9} {'95% CI':>18} {'p(lift≤0)':>10}")
        for h in horizons:
            lb = _study.lift_bootstrap(rows, h, k, n_boot=n_boot, metric=metric)
            if not lb.get("treat_n"):
                continue
            ci = (f"[{100 * lb['ci_lo']:+5.2f}, {100 * lb['ci_hi']:+5.2f}]"
                  if "ci_lo" in lb else "n/a")
            print(f"    {h:>3} {_fmt_pct(lb['treat_rate'])} {_fmt_pct(lb['control_rate'])} "
                  f"{100 * lb['observed']:>+7.2f}pp {ci:>18} "
                  f"{lb.get('p_le_zero', float('nan')):>10.3f}")
        print()

    print()
    print(_RULE)
    print("  DOSE-RESPONSE: tail rate by evidence points (21d)")
    print(_RULE)
    mono = _study.monotonicity(rows, 21, k)
    if mono:
        print(f"    {'pts':>3} {'n':>8} {'P(+2σ)':>7} {'P(-2σ)':>7}")
        for m in mono:
            bar = "█" * int(round(m["rate"] * 200))
            print(f"    {m['points']:>3} {m['n']:>8,} {_fmt_pct(m['rate'])} "
                  f"{_fmt_pct(m['down'])}  {bar}")
    else:
        print("    (insufficient data)")

    sp = _study.spearman_clustered(rows, 21)
    if "rho" in sp:
        print()
        print(f"    Spearman(points, σ-normalised max move) = {sp['rho']:+.4f}"
              f"  95% CI [{sp['ci_lo']:+.4f}, {sp['ci_hi']:+.4f}]   n={sp['n']:,}")

    print()
    print(_RULE)
    print("  ROBUSTNESS — same lift under cuts that expose a one-episode result (21d)")
    print(_RULE)
    for name, lb in _study.robustness(rows, 21, k, n_boot=max(500, n_boot // 4)).items():
        if not lb.get("treat_n"):
            continue
        ci = (f"[{100 * lb['ci_lo']:+5.2f}, {100 * lb['ci_hi']:+5.2f}]"
              if "ci_lo" in lb else "n/a")
        print(f"    {name:<22} lift {100 * lb['observed']:>+6.2f}pp  CI {ci:>18}  "
              f"n_setup={lb['treat_n']:>6,}")
    print()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Squeeze grader point-in-time backtest")
    p.add_argument("--run", action="store_true")
    p.add_argument("--rebuild", action="store_true", help="rebuild the cached panel")
    p.add_argument("--si-scale", type=float, default=1.0)
    p.add_argument("--ret5d", choices=("as_written", "as_intended"), default="as_written")
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--prices-db", default="data/squeeze_prices.db")
    args = p.parse_args(argv)

    rows = _load_panel(args.db, args.prices_db, args.rebuild)
    if not rows:
        print("empty panel — run the finra/prices/shares backfills first")
        return 1
    report(rows, k=args.k, si_scale=args.si_scale, ret5d=args.ret5d,
           n_boot=args.boot, db=args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
