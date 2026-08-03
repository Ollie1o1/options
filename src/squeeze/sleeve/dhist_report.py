"""Compute D_hist and write it down.

D_hist is the payoff half of ``E = D_hist - P_live - F_live``. The two live
terms only ever SUBTRACT, so a D_hist at or below zero ends the question without
waiting three months for chains: nothing measured later can add to it.

The report says so on its face. A positive number here is not an edge — it is
permission to go and measure the pricing terms.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, Optional, Sequence

from src.squeeze.sleeve import dhist

HORIZONS = (21, 42)
VARIANTS = ("conservative", "central")
OUT_DIR = "reports/dhist"


def summarise(results: Dict[tuple, dict], stats: Dict[str, int]) -> Dict[str, Any]:
    cells = []
    for (horizon, variant), r in sorted(results.items()):
        cells.append({
            "horizon": horizon,
            "variant": variant,
            "observed": r.get("observed"),
            "ci_lo": r.get("ci_lo"),
            "ci_hi": r.get("ci_hi"),
            "n_dates": r.get("n_dates", 0),
            "treat_n": r.get("treat_n", 0),
            "control_n": r.get("control_n", 0),
            "flagged": len(r.get("flagged_dates") or []),
        })
    return {"generated": _dt.date.today().isoformat(),
            "cells": cells, "stats": dict(stats)}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100.0:.2f}%"


def render(payload: Dict[str, Any]) -> str:
    lines = ["# D_hist — the payoff term", "",
             f"_Generated {payload['generated']}_", "",
             "| horizon | variant | observed | 95% CI | dates | treated | control | flagged |",
             "|---:|---|---:|---|---:|---:|---:|---:|"]
    for c in payload["cells"]:
        ci = f"[{_pct(c['ci_lo'])}, {_pct(c['ci_hi'])}]"
        lines.append(
            f"| {c['horizon']}td | {c['variant']} | {_pct(c['observed'])} | {ci} "
            f"| {c['n_dates']} | {c['treat_n']} | {c['control_n']} | {c['flagged']} |")

    s = payload["stats"]
    lines += ["", "## Universe accounting", "",
              f"- Rows without short interest, never ranked: **{s.get('ungradeable', 0):,}**",
              f"- Rows whose price series ended inside the window: **{s.get('short_path', 0):,}**",
              f"- Treated: **{s.get('treated', 0):,}** · control: **{s.get('control', 0):,}** "
              f"· excluded as partially treated: **{s.get('excluded', 0):,}**",
              "", "## What this number is not", "",
              "D_hist is the payoff term alone. A positive value does **not** mean the",
              "strategy has an edge — it means the strategy is **not yet dead**, and the",
              "pricing terms have still to be subtracted. `P_live` (the implied-vol premium",
              "on heavily shorted names) and `F_live` (spread cost) only ever reduce it.",
              "Only the full expectation decides, and that needs the live half."]
    return "\n".join(lines) + "\n"


def run(prices_db: Optional[str] = None, db_path: Optional[str] = None,
        shares_db: Optional[str] = None) -> Dict[str, Any]:
    """Build the panel once, then compute every horizon and variant from it."""
    from src.squeeze.backtest import panel as _panel
    from src.squeeze.backtest import shares as _shares
    from src.squeeze.backtest.universe import study_symbols
    from src.squeeze.sleeve import panel_rows

    db_path = db_path or _panel.DEFAULT_DB
    prices_db = prices_db or _panel.PRICES_DB
    shares_db = shares_db or _panel.SHARES_DB

    # keep_ungraded=True so the adapter SEES the rows that lack short interest
    # and can count them. With False the panel drops them itself and the report
    # would state zero for a number that is not zero.
    records = _panel.build(db_path=db_path, prices_db=prices_db,
                           shares_db=shares_db, keep_ungraded=True,
                           verbose=True)
    book = _panel.PriceBook(prices_db, study_symbols(db_path))
    lookup = _shares.SharesLookup(shares_db)

    results: Dict[tuple, dict] = {}
    stats: Dict[str, int] = {}
    for horizon in HORIZONS:
        rows, st = panel_rows.build(records, book, lookup, horizon)
        # Every count except short_path is horizon-invariant; short_path grows
        # with the window, so the longest horizon's figures are the ones worth
        # quoting — they are the strictest.
        if horizon == max(HORIZONS):
            stats = st
        for variant in VARIANTS:
            results[(horizon, variant)] = dhist.compute(
                rows, horizon=horizon, variant=variant)
    return summarise(results, stats)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute and write D_hist.")
    ap.add_argument("--from", dest="src", default="",
                    help="re-render from an existing JSON sidecar, no compute")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    if args.src:
        with open(args.src) as fh:
            payload = json.load(fh)
    else:
        payload = run()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, payload["generated"])
    with open(stem + ".json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    text = render(payload)
    with open(stem + ".md", "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {stem}.md and {stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
