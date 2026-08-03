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
import math
import os
from typing import Any, Dict, Optional, Sequence

from src.squeeze.sleeve import dhist

HORIZONS = (21, 42)
VARIANTS = ("conservative", "central")
OUT_DIR = "reports/dhist"


def summarise(results: Dict[tuple, dict], stats: Dict[str, int]) -> Dict[str, Any]:
    cells = []
    # Matching does not depend on the payoff variant, so the selection is a
    # property of the horizon alone; keyed that way it prints once instead of
    # twice with identical numbers.
    selection: Dict[Any, Any] = {}
    for (horizon, variant), r in sorted(results.items()):
        n_dates = r.get("n_dates", 0)
        flagged = len(r.get("flagged_dates") or [])
        if r.get("selection") is not None:
            selection[str(horizon)] = r["selection"]
        cells.append({
            "horizon": horizon,
            "variant": variant,
            "observed": r.get("observed"),
            "ci_lo": r.get("ci_lo"),
            "ci_hi": r.get("ci_hi"),
            "n_dates": n_dates,
            "treat_n": r.get("treat_n", 0),
            "control_n": r.get("control_n", 0),
            "flagged": flagged,
            # The tripwire now reads IMBALANCE, not unmatchability: under the
            # matchable-subsample estimand a cycle is flagged only when the
            # units that did match fail covariate balance. A majority of
            # flagged cycles still means no verdict is quotable.
            "verdict": "INVALID" if flagged > n_dates else "VALID",
        })
    return {"generated": _dt.date.today().isoformat(),
            "cells": cells, "stats": dict(stats), "selection": selection}


def _pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100.0:.2f}%"


_COVARIATE_LABELS = (("rv", "realised vol"), ("ret_5d", "5-day return"),
                     ("log_mcap", "log market cap"), ("log_price", "log price"))


def _num(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):.3f}"


def _selection_section(selection: Dict[str, Any]) -> list:
    """What the matchable-subsample estimand covers, and which way it leans.

    A selected sample whose selection is not characterised is just a biased
    sample. So this prints the coverage AND the covariate means of the treated
    units that were dropped beside those that were kept — the reader can see
    the direction of the selection rather than being asked to trust it.
    """
    if not selection:
        return []
    lines = ["", "## Selection — what this estimand covers", "",
             "D_hist is measured on the **matchable subsample**: treated names with an",
             "in-caliper matched control. Names without one are dropped and counted here",
             "rather than invalidating the cycle (operator decision, 2026-08-03 — the",
             "original all-treated estimand is not measurable on this panel; see",
             "`status/DECISIONS.md`). Cycles are still invalidated by covariate",
             "IMBALANCE, which is what keeps the surviving comparison fair.", ""]
    for horizon in sorted(selection, key=lambda h: int(h)):
        sel = selection[horizon]
        cov = sel.get("coverage")
        cov_txt = "n/a" if cov is None or (isinstance(cov, float) and math.isnan(cov)) \
            else f"{cov * 100.0:.1f}%"
        lines += [
            f"### {horizon}td", "",
            f"- Treated units on balanced cycles: **{sel.get('treated_eligible', 0):,}** "
            f"· matched: **{sel.get('treated_matched', 0):,}** "
            f"· **coverage {cov_txt}**",
            f"- Median per-cycle drop rate: **{_num(sel.get('median_drop_rate'))}** "
            f"({sel.get('dates_over_drop_bar', 0)} cycles above the old 0.30 bar — "
            "reported, no longer invalidating)", "",
            "| covariate | matched treated | dropped treated |",
            "|---|---:|---:|"]
        matched = sel.get("matched_mean") or {}
        dropped = sel.get("dropped_mean") or {}
        for key, label in _COVARIATE_LABELS:
            lines.append(f"| {label} | {_num(matched.get(key))} "
                         f"| {_num(dropped.get(key))} |")
        lines += ["",
                  "Read the two columns against each other: where they differ is the",
                  "direction the estimand has been narrowed. They are means over treated",
                  "units on balanced cycles only.", ""]
    return lines


def render(payload: Dict[str, Any]) -> str:
    lines = ["# D_hist — the payoff term", "",
             f"_Generated {payload['generated']}_", ""]

    # The verdict prints BEFORE the table. A reader must not meet a percentage
    # computed from a matchability-selected subsample without first being told
    # the measurement failed its own committed validity bar.
    # Older JSON sidecars re-rendered via --from carry no verdict field;
    # derive it from the counts rather than let them render as clean.
    def _verdict(c: Dict[str, Any]) -> str:
        v = c.get("verdict")
        if v is not None:
            return v
        return "INVALID" if c.get("flagged", 0) > c.get("n_dates", 0) else "VALID"

    invalid = [c for c in payload["cells"] if _verdict(c) == "INVALID"]
    if invalid:
        lines += ["## VERDICT: INVALID — the measurement did not clear its own validity bar", ""]
        for c in invalid:
            total = c["flagged"] + c["n_dates"]
            lines.append(
                f"- **{c['horizon']}td {c['variant']}: INVALID** — "
                f"{c['flagged']} of {total} settlement dates failed the "
                f"covariate-balance check; only {c['n_dates']} survived.")
        lines += ["",
                  "A majority of flagged cycles means, by the design spec's non-tunable",
                  "tripwire, that **no verdict is quotable in either direction**. The point",
                  "estimates below are computed from the balanced minority only; they are",
                  "context, not a headline.",
                  "",
                  "Note what this is NOT saying. Since 2026-08-03 an unmatchable treated",
                  "name no longer flags its cycle — that is the selection documented below.",
                  "These cycles were flagged for **imbalance between the units that DID",
                  "match**, which the estimand change deliberately left as a hard bar. The",
                  "blocker here is balance, not matchability.", ""]
    else:
        lines += ["## Verdict: VALID — every cell cleared the majority-flagged tripwire", ""]

    lines += ["| horizon | variant | observed | 95% CI | dates | treated | control | flagged |",
              "|---:|---|---:|---|---:|---:|---:|---:|"]
    for c in payload["cells"]:
        ci = f"[{_pct(c['ci_lo'])}, {_pct(c['ci_hi'])}]"
        lines.append(
            f"| {c['horizon']}td | {c['variant']} | {_pct(c['observed'])} | {ci} "
            f"| {c['n_dates']} | {c['treat_n']} | {c['control_n']} | {c['flagged']} |")

    lines += _selection_section(payload.get("selection") or {})

    s = payload["stats"]
    lines += ["", "## Universe accounting", "",
              f"- Rows without short interest, never ranked: **{s.get('ungradeable', 0):,}**",
              f"- Rows whose price series ended inside the window: **{s.get('short_path', 0):,}** "
              f"(treated {s.get('short_path_treated', 0):,} · "
              f"control {s.get('short_path_control', 0):,})",
              f"- Treated: **{s.get('treated', 0):,}** · control: **{s.get('control', 0):,}** "
              f"· excluded as partially treated: **{s.get('excluded', 0):,}**",
              "", "## What this number is not", "",
              "D_hist is the payoff term alone. A positive value does **not** mean the",
              "strategy has an edge — it means the strategy is **not yet dead**, and the",
              "pricing terms have still to be subtracted. `P_live` (the implied-vol premium",
              "on heavily shorted names) and `F_live` (spread cost) only ever reduce it.",
              "Only the full expectation decides, and that needs the live half.",
              "",
              "It is also **not the effect on a uniformly-drawn treated name**. The",
              "estimand is the matchable subsample above, and matchability is not random:",
              "a treated name is matchable when the low-SI pool happens to contain",
              "something at its volatility, which is the bias the matched design was",
              "built to avoid and can now only disclose. Any GO built on this number",
              "authorises trading the matchable cohort, not the cohort the screener",
              "produces — and the two are the same only if the covariate table above",
              "shows the dropped names looking like the kept ones."]
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
