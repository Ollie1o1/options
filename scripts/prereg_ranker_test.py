"""The single look. Refuses to run early; refuses to recompute.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.prereg_ranker_test

Reads every threshold from docs/PREREG_RANKER_TEST.md. This script decides
nothing on its own — the decision rule was frozen before any outcome existed.

See docs/PREREG_RANKER_TEST_SPEC.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.prereg_ranker_power import DEFAULT_REGISTRATION, parse_field
from src import prereg_ranker as pk

_NOT_RUN_MARKER = "*Not yet run.*"


def decide(lo: Optional[float], hi: Optional[float]) -> str:
    """PASS / FAIL / INVERTED from the confidence interval.

    INVERTED is separated from FAIL because an `ev_net` that predicts backwards
    is real information — `quality_score` was exactly that shape, its top
    quintile the worst cell in the ledger. It does not PASS: reversing a sign
    on the strength of one look is how overfitting starts.
    """
    if lo is None or hi is None:
        return "FAIL"
    if lo > 0:
        return "PASS"
    if hi < 0:
        return "INVERTED"
    return "FAIL"


def already_run(text: str) -> bool:
    return _NOT_RUN_MARKER not in text


def run(registration_path: str, db_path: str,
        today: Optional[str] = None) -> Dict[str, Any]:
    """Perform the look if permitted, and stamp the registration."""
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    with open(registration_path) as fh:
        text = fh.read()

    if already_run(text):
        return {"status": "ALREADY_RUN",
                "decision": parse_field(text, "decision"),
                "rank_ic": parse_field(text, "rank_ic")}

    n_star = float(parse_field(text, "n_star_nominal") or 0)
    deadline = parse_field(text, "deadline") or "9999-12-31"
    n_boot = int(float(parse_field(text, "n_boot") or 10000))
    seed = int(float(parse_field(text, "seed") or 0))
    alpha = float(parse_field(text, "alpha") or 0.05)

    cohort = pk.load_cohort(db_path)
    n = len(cohort)

    if n < n_star and today < deadline:
        return {"status": "NOT_YET", "n": n, "n_star": n_star,
                "deadline": deadline}

    cells = ["entry_date", "strategy"]
    ic = pk.rank_ic(cohort, "ev_net", "pnl_pct", cells)
    lo: Optional[float] = None
    hi: Optional[float] = None
    if n < n_star:
        decision = "UNDERPOWERED"
    else:
        lo, hi = pk.cluster_bootstrap_ci(cohort, "ev_net", "pnl_pct", cells,
                                         "contract_key", n_boot=n_boot,
                                         alpha=alpha, seed=seed)
        decision = decide(lo, hi)

    control = pk.negative_control(cohort, "ev_net", "pnl_pct", cells,
                                  n_shuffles=200, seed=seed)
    first, second = pk.half_sample_ics(cohort, "ev_net", "pnl_pct", cells,
                                       "entry_date")
    de = pk.design_effect(cohort, "pnl_pct", "contract_key")

    secondary = {}
    for feature in ("quality_score", "carry", "delta"):
        secondary[feature] = pk.rank_ic(cohort, feature, "pnl_pct", cells)

    stamp = [
        "## Result",
        "",
        f"Run {today}. One look, as registered.",
        "",
        "```",
        f"decision: {decision}",
        f"n: {n}",
        f"rank_ic: {ic}",
        f"ci_low: {lo}",
        f"ci_high: {hi}",
        f"design_effect: {de}",
        f"effective_n: {pk.effective_n(n, de) if de else n}",
        f"negative_control_mean: {control['mean']}",
        f"negative_control_p95_abs: {control['p95_abs']}",
        f"half1_ic: {first}",
        f"half2_ic: {second}",
    ]
    for feature, value in secondary.items():
        stamp.append(f"secondary_{feature}_ic: {value}")
    stamp += ["```", ""]

    marker_block = f"## Result\n\n{_NOT_RUN_MARKER}"
    replacement = "\n".join(stamp)
    if marker_block in text:
        text = text.replace(marker_block, replacement)
    else:
        text = text.replace(_NOT_RUN_MARKER, replacement)
    with open(registration_path, "w") as fh:
        fh.write(text)

    return {"status": "RUN", "decision": decision, "n": n, "rank_ic": ic,
            "ci": (lo, hi), "secondary": secondary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registration", default=DEFAULT_REGISTRATION)
    ap.add_argument("--db", default="data/candidates.db")
    ap.add_argument("--today", default=None)
    args = ap.parse_args(argv)

    out = run(args.registration, args.db, today=args.today)
    if out["status"] == "NOT_YET":
        print(f"NOT YET — {out['n']} of {out['n_star']:.0f} closed survivor "
              f"positions, deadline {out['deadline']}. Nothing computed.")
        return 0
    if out["status"] == "ALREADY_RUN":
        print(f"ALREADY RUN — decision {out['decision']}, "
              f"rank IC {out['rank_ic']}. Not recomputed.")
        return 0
    print(f"DECISION: {out['decision']}  n={out['n']}  IC={out['rank_ic']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
