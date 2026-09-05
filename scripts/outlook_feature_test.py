"""Run docs/PREREG_OUTLOOK_FEATURE_20260905.md's frozen design once.

H-OUTLOOK: does `outlook_composite` (src/outlook/cross_sectional.py, the
already-validated src/outlook construction transferred point-in-time onto
single names) have a residualized, day-clustered IC against bull_put return
on capital, on the 2020-21 holdout, that clears Harvey's |t| >= 3.0 hurdle
and matches the predicted (positive) sign?

Nothing here is tunable — every date boundary, feature list and control set
is copied verbatim from the frozen doc. This script only assembles the
inputs the doc already specifies and reports what comes out.

CLI:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/outlook_feature_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alloc.__main__ import DB, UNIVERSE, _spec, _trading_dates  # noqa: E402
from src.alloc.attribution import (RESIDUAL_CONTROLS, _clustered_t,  # noqa: E402
                                   _residual_values, feature_ic)
from src.alloc.engine import SqliteChainSource, replay  # noqa: E402
from src.alloc.splits import detect_splits  # noqa: E402
from src.alloc.universe import (audit_coverage, load_universe,  # noqa: E402
                                symbol_stratum, terminal_dates,
                                usable_dates, usable_symbols)
from src.dolt_stocks import DEFAULT_CACHE, close_history  # noqa: E402
from src.outlook.cross_sectional import composite_lookup  # noqa: E402

IN_SAMPLE = ("2022-01-07", "2024-12-31")
HOLDOUT = ("2020-01-27", "2021-12-31")
MIN_TSTAT = 3.0
FEATURE = "outlook_composite"


def _already_fetched(symbols: List[str], db_path: str = DEFAULT_CACHE) -> set:
    """Symbols whose closes are already cached — never trigger a live fetch
    from this script. `data/` is gitignored and machine-local; a symbol
    outside this set is reported missing, same as any other coverage gap."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT symbol FROM stocks_fetched WHERE symbol IN (%s)" %
            ",".join("?" * len(symbols)), symbols).fetchall()
    return {r[0] for r in rows}


def build_outlook_lookup(dates: List[str]) -> Dict[Tuple[str, str], float]:
    universe = load_universe(UNIVERSE)
    syms = sorted({s for v in universe.values() for s in v} | {"SPY"})
    fetched = _already_fetched(syms)
    skipped = [s for s in syms if s not in fetched]
    if skipped:
        print(f"  skipping {len(skipped)} symbols with no cached closes "
              f"(no live fetch from this script): {skipped}")
    closes = {s: close_history(s) for s in syms if s in fetched}
    closes = {s: c for s, c in closes.items() if c}
    return composite_lookup(closes, dates)


def replay_bull_put(start: str, end: str, outlook_lookup):
    universe = load_universe(UNIVERSE)
    audit = audit_coverage(DB, universe)
    usable = set(usable_symbols(audit))
    syms = sorted(usable)
    dates = usable_dates(audit, _trading_dates(start, end, weekly=False))
    source = SqliteChainSource(DB)
    term, strat = terminal_dates(audit), symbol_stratum(universe)
    splits = detect_splits(DB, symbols=syms)
    entry = {"dte": [25, 60], "short_delta": 0.25, "width": 5.0}
    spec = _spec("bull_put", "bull_put", entry, n_trials=1)
    return replay(spec, syms, dates, source, terminal=term, splits=splits,
                 stratum_of=strat, outlook_lookup=outlook_lookup)


def residualized_clustered_t(trades) -> Dict[str, object]:
    """The statistic the frozen doc's decision rule actually needs:
    `residual_ic` reports a naive (non-clustered) p-value, not a t-stat —
    reusing its own `_residual_values` machinery to compute the day-clustered
    t on the residual, matching every other decision rule in this repo."""
    closed = [t for t in trades if t.exit_date and t.capital_at_risk]
    r = _residual_values(closed, FEATURE, RESIDUAL_CONTROLS)
    if r is None:
        return {"n": 0, "ic": None, "t_clustered": None, "controls": []}
    import numpy as np
    from scipy import stats as sps
    ic, _p = sps.spearmanr(r["resid"], r["ys"])
    if ic != ic:
        return {"n": len(r["keep"]), "ic": None, "t_clustered": None,
                "controls": r["controls"]}
    t_clustered = _clustered_t(r["keep"], np.asarray(r["resid"]), r["ys"])
    return {"n": len(r["keep"]), "ic": round(float(ic), 4),
           "t_clustered": round(float(t_clustered), 3),
           "controls": r["controls"]}


def main() -> int:
    all_dates = sorted(set(_trading_dates(*IN_SAMPLE, weekly=False))
                       | set(_trading_dates(*HOLDOUT, weekly=False)))
    print(f"Building outlook_composite lookup over {len(all_dates)} dates...")
    lookup = build_outlook_lookup(all_dates)
    print(f"  {len(lookup)} (symbol, date) composite scores computed\n")

    print(f"Replaying bull_put, in-sample {IN_SAMPLE}...")
    in_trades, in_stats = replay_bull_put(*IN_SAMPLE, lookup)
    print(f"  {in_stats}")
    print(f"Replaying bull_put, holdout {HOLDOUT}...")
    hold_trades, hold_stats = replay_bull_put(*HOLDOUT, lookup)
    print(f"  {hold_stats}\n")

    print("=" * 74)
    print("RAW feature_ic (uncontrolled, for reference only):")
    print(f"  in-sample: {feature_ic(in_trades, FEATURE)}")
    print(f"  holdout:   {feature_ic(hold_trades, FEATURE)}")

    print("\nRESIDUALIZED (credit_pct_width, atm_iv), day-clustered t "
         "— this is the decision statistic:")
    in_r = residualized_clustered_t(in_trades)
    hold_r = residualized_clustered_t(hold_trades)
    print(f"  in-sample: {in_r}")
    print(f"  holdout:   {hold_r}")

    print("\n" + "=" * 74)
    t = hold_r["t_clustered"]
    ic = hold_r["ic"]
    if t is None or hold_r["n"] < 8:
        verdict = "UNDERPOWERED — too few holdout trades to measure"
    elif abs(t) < MIN_TSTAT:
        verdict = f"NULL — |t_clustered|={abs(t):.3f} < {MIN_TSTAT}"
    elif ic is not None and ic > 0:
        verdict = f"REAL — |t_clustered|={abs(t):.3f} >= {MIN_TSTAT}, sign matches prediction (positive)"
    else:
        verdict = f"INVERTED — |t_clustered|={abs(t):.3f} >= {MIN_TSTAT}, but sign is negative (wrong direction)"
    print(f"VERDICT (docs/PREREG_OUTLOOK_FEATURE_20260905.md decision rule): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
