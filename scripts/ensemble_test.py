"""Run docs/PREREG_ENSEMBLE_20260905.md's frozen design once, per structure.

H-ENSEMBLE: does a ridge combination of the credit-richness-residualized
`ATTRIBUTION_FEATURES` (minus the two controls) beat the single best
residualized feature on the same holdout population, with
`|t_clustered| >= 3.0`?

Nothing here is tunable — the feature list, controls, alpha grid, CV block
count and both date windows are copied verbatim from the frozen doc and from
`src/alloc/__main__.py::ATTRIBUTION_FEATURES`. This script only assembles
the inputs the doc already specifies, fits on in-sample, and scores the
frozen model on holdout exactly once.

CLI:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/ensemble_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alloc.__main__ import ATTRIBUTION_FEATURES, DB, UNIVERSE, _spec  # noqa: E402
from src.alloc.__main__ import _trading_dates  # noqa: E402
from src.alloc.attribution import RESIDUAL_CONTROLS, residual_ic  # noqa: E402
from src.alloc.engine import SqliteChainSource, replay  # noqa: E402
from src.alloc.ensemble import ensemble_ic, fit_ensemble  # noqa: E402
from src.alloc.splits import detect_splits  # noqa: E402
from src.alloc.universe import (audit_coverage, load_universe,  # noqa: E402
                                symbol_stratum, terminal_dates,
                                usable_dates, usable_symbols)

IN_SAMPLE = ("2022-01-07", "2024-12-31")
HOLDOUT = ("2020-01-27", "2021-12-31")
MIN_TSTAT = 3.0
CANDIDATE_FEATURES = [f for f in ATTRIBUTION_FEATURES if f not in RESIDUAL_CONTROLS]
STRUCTURES = {
    "bull_put": {"dte": [25, 60], "short_delta": 0.25, "width": 5.0},
    "long_call": {"dte": [25, 60], "target_delta": 0.40},
}


def replay_structure(structure: str, entry: dict, start: str, end: str):
    universe = load_universe(UNIVERSE)
    audit = audit_coverage(DB, universe)
    usable = set(usable_symbols(audit))
    syms = sorted(usable)
    dates = usable_dates(audit, _trading_dates(start, end, weekly=False))
    source = SqliteChainSource(DB)
    term, strat = terminal_dates(audit), symbol_stratum(universe)
    splits = detect_splits(DB, symbols=syms)
    spec = _spec(structure, structure, entry, n_trials=1)
    trades, stats = replay(spec, syms, dates, source, terminal=term,
                           splits=splits, stratum_of=strat)
    return trades, stats


def best_single_residualized_ic(trades, features: List[str]) -> float:
    closed = [t for t in trades if t.exit_date and t.capital_at_risk]
    best = 0.0
    for f in features:
        r = residual_ic(closed, f)
        if r["ic"] is not None:
            best = max(best, abs(r["ic"]))
    return best


def run_one(structure: str, entry: dict) -> None:
    print("=" * 74)
    print(f"STRUCTURE: {structure}")
    print(f"Replaying in-sample {IN_SAMPLE}...")
    in_trades, in_stats = replay_structure(structure, entry, *IN_SAMPLE)
    print(f"  {in_stats}")
    print(f"Replaying holdout {HOLDOUT}...")
    hold_trades, hold_stats = replay_structure(structure, entry, *HOLDOUT)
    print(f"  {hold_stats}\n")

    in_closed = [t for t in in_trades if t.exit_date and t.capital_at_risk]
    hold_closed = [t for t in hold_trades if t.exit_date and t.capital_at_risk]
    print(f"n closed: in-sample={len(in_closed)}, holdout={len(hold_closed)}")

    model = fit_ensemble(in_closed, CANDIDATE_FEATURES)
    if model is None:
        print("VERDICT: UNDERPOWERED — could not fit an ensemble on the "
             "in-sample window (too few trades or too few measurable "
             "features)\n")
        return

    print(f"\nFrozen model: {len(model.features)} features used "
         f"(of {len(CANDIDATE_FEATURES)} candidates), alpha={model.alpha}, "
         f"n_fit={model.n_fit}")
    print(f"  features: {model.features}")
    print(f"  coef:     {tuple(round(c, 4) for c in model.coef)}")
    print(f"  CV IC by alpha: {model.cv_ic_by_alpha}")

    in_ens = ensemble_ic(in_closed, model)
    hold_ens = ensemble_ic(hold_closed, model)
    print(f"\nEnsemble IC — in-sample: {in_ens}")
    print(f"Ensemble IC — holdout:   {hold_ens}")

    best_hold = best_single_residualized_ic(hold_closed, model.features)
    print(f"\nBest single residualized |IC| on the SAME holdout population: "
         f"{best_hold:.4f}")

    t = hold_ens["t_clustered"]
    ic = hold_ens["ic"]
    if ic is None or hold_ens["n"] < 8:
        verdict = "UNDERPOWERED — too few holdout trades to measure"
    elif abs(t) < MIN_TSTAT:
        verdict = f"NULL — |t_clustered|={abs(t):.3f} < {MIN_TSTAT}"
    elif abs(ic) <= best_hold:
        verdict = (f"NOT A DISCOVERY — |t_clustered|={abs(t):.3f} >= "
                  f"{MIN_TSTAT} but |ic|={abs(ic):.4f} does not beat the "
                  f"best single feature ({best_hold:.4f})")
    else:
        verdict = (f"REAL — |t_clustered|={abs(t):.3f} >= {MIN_TSTAT} AND "
                  f"|ic|={abs(ic):.4f} beats the best single feature "
                  f"({best_hold:.4f})")
    print(f"\nVERDICT (docs/PREREG_ENSEMBLE_20260905.md decision rule): {verdict}\n")


def main() -> int:
    for structure, entry in STRUCTURES.items():
        run_one(structure, entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
