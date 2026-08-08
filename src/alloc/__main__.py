"""Allocation backtester CLI.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.alloc --audit
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.alloc --signals
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.alloc --structures

Every run prints the trial count it deflated by. That number is the whole point:
a Sharpe ratio without it is a story, not a measurement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Dict, List, Optional

from src.alloc.engine import SqliteChainSource, replay
from src.alloc.report import format_summary, summarise
from src.alloc.attribution import (feature_ic, format_ranking, rank_features,
                                   split_by_time)
from src.alloc.splits import detect_splits
from src.alloc.universe import (audit_coverage, load_universe, symbol_stratum,
                                terminal_dates, usable_dates, usable_symbols)
from src.strategies.spec import StrategySpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "data", "dolt_options.db")
UNIVERSE = os.path.join(ROOT, "data", "backtest_universe.json")

# Everything knowable at entry. The list length IS the search size, and
# rank_features carries it so results can be deflated by it.
ATTRIBUTION_FEATURES = ["iv_rank", "atm_iv", "rv", "iv_minus_rv", "trend",
                        "ret_4w", "dte", "width", "credit_pct_width",
                        "short_delta", "long_delta", "moneyness",
                        "friction_pct_credit", "capital_at_risk"]

# Tight-spread names: 3.6% median bid-ask against 16-21% elsewhere. The largest
# single driver of results found anywhere in this study. Defined in
# src.dolt_options so the backfill and the backtest cannot drift apart.
from src.dolt_options import MEGA  # noqa: E402  (re-exported for callers)


def _trading_dates(start: str, end: str, weekly: bool) -> List[str]:
    d, out = dt.date.fromisoformat(start), []
    stop = dt.date.fromisoformat(end)
    while d <= stop:
        if (d.weekday() == 4) if weekly else (d.weekday() < 5):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _spec(name: str, structure: str, entry: Dict, n_trials: int,
          hold: bool = True, max_concurrent: int = 3) -> StrategySpec:
    return StrategySpec(
        id=name, version=1, structure=structure, universe={}, entry=entry,
        exit={"hold_to_expiry": True} if hold else
             {"profit_target": 0.5, "stop": 2.0, "hold_to_expiry": False},
        sizing={"max_capital_at_risk": 4000, "max_concurrent": max_concurrent},
        created=dt.date.today().isoformat(), trial_count=n_trials)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Allocation backtester")
    ap.add_argument("--audit", action="store_true",
                    help="coverage audit; can veto everything else")
    ap.add_argument("--signals", action="store_true",
                    help="signal conditions against the unconditional baseline")
    ap.add_argument("--structures", action="store_true",
                    help="compare structures on the tight-spread universe")
    ap.add_argument("--attribute", action="store_true",
                    help="rank entry features by IC against realized return, "
                         "with clustered t and a chronological OOS split")
    ap.add_argument("--mega", action="store_true", default=True,
                    help="restrict to tight-spread names (default)")
    ap.add_argument("--all-names", action="store_true",
                    help="use every usable symbol instead")
    ap.add_argument("--weekly", action="store_true",
                    help="sample Fridays only rather than every weekday")
    ap.add_argument("--trials", type=int, default=34,
                    help="configurations tried, for deflation")
    ap.add_argument("--max-concurrent", type=int, default=3,
                    help="positions the ENGINE may hold open at once. The "
                         "default 3 is an account-level constraint and it "
                         "starves the per-trade sample: across 117 symbols it "
                         "yields ~36 trades/yr no matter how wide the "
                         "universe. Raise it to measure a per-trade effect "
                         "(the account-level line stays honest — report.py "
                         "applies its own capacity cap independently).")
    ap.add_argument("--start", default="2022-01-07")
    ap.add_argument("--end", default="2026-06-12")
    args = ap.parse_args(argv)

    universe = load_universe(UNIVERSE)
    audit = audit_coverage(DB, universe)

    if audit.get("read_error"):
        # "could not read" and "read it, and it is empty" are different claims,
        # and only one of them means your data is gone.
        print(f"COULD NOT READ THE CACHE: {audit['read_error']}")
        print(f"  {DB}")
        print("  Coverage is UNKNOWN, not absent — this is not a verdict on "
              "the data.\n  A backfill holding the write lock is the usual "
              "cause; re-run when it finishes.")
        return 1

    if args.audit or not (args.signals or args.structures or args.attribute):
        print("COVERAGE AUDIT")
        for stratum, v in audit["summary"].items():
            print(f"  {stratum:<8} total={v['total']:>3} usable={v['usable']:>3} "
                  f"sparse={v['sparse']:>2} absent={v['absent']:>2}")
        print(f"  viable={audit['viable']}  "
              f"dead dates={len(audit['dead_dates'])}")
        if not (args.signals or args.structures or args.attribute):
            return 0 if audit["viable"] else 1

    if not audit["viable"]:
        print("Universe is not viable — refusing to backtest on it.")
        return 1

    usable = set(usable_symbols(audit))
    syms = (sorted(usable) if args.all_names
            else [s for s in MEGA if s in usable])
    dates = usable_dates(audit, _trading_dates(args.start, args.end,
                                               args.weekly))
    src = SqliteChainSource(DB)
    term, strat = terminal_dates(audit), symbol_stratum(universe)
    # Splits MUST be passed. Without them the engine's split branch is dead
    # code: this dataset is not split-adjusted, so a short put struck at 2200
    # books a catastrophic loss when GOOG's data-spot drops to 108, and every
    # trend signal reads the same jump as a 95% crash for a year afterwards.
    splits = detect_splits(DB, symbols=syms)
    n_events = sum(len(v) for v in splits.values())
    print(f"\n{len(syms)} symbols x {len(dates)} dates  "
          f"({n_events} split events across {len(splits)} symbols; "
          f"deflating by {args.trials} configurations)\n")

    def run(label: str, structure: str, **entry):
        e = {"dte": [25, 60], "short_delta": 0.25, "width": 5.0}
        e.update(entry)
        trades, _ = replay(_spec(label, structure, e, args.trials,
                                max_concurrent=args.max_concurrent), syms,
                           dates, src, terminal=term, splits=splits,
                           stratum_of=strat)
        print(format_summary(label, summarise(trades, args.trials)))

    if args.structures:
        run("bull_put", "bull_put")
        run("bear_call", "bear_call")
        run("iron_condor", "iron_condor", short_delta=0.16)
        run("long_call [CONTROL]", "long_call", target_delta=0.40)

    if args.attribute:
        for label, extra in (("bull_put", {}), ("bear_call", {}),
                             ("iron_condor", {"short_delta": 0.16}),
                             ("long_call", {"target_delta": 0.40})):
            e = {"dte": [25, 60], "short_delta": 0.25, "width": 5.0}
            e.update(extra)
            trades, _ = replay(_spec(label, label, e, args.trials,
                                    max_concurrent=args.max_concurrent), syms,
                               dates, src, terminal=term, splits=splits,
                               stratum_of=strat)
            closed = [t for t in trades if t.exit_date and t.capital_at_risk]
            print("=" * 74)
            print(format_summary(label, summarise(trades, args.trials)))
            if len(closed) < 20:
                print("  too few closed trades to attribute\n")
                continue
            rows = rank_features(trades, ATTRIBUTION_FEATURES)
            print(format_ranking(rows, f"  entry-feature IC vs RoC "
                                       f"({len(ATTRIBUTION_FEATURES)}-way search):"))
            train, test = split_by_time(closed, 0.7)
            print(f"  OOS (train {len(train)} -> test {len(test)}):")
            for r in rows[:5]:
                if r["ic"] is None:
                    continue
                a = feature_ic(train, r["feature"])["ic"]
                b = feature_ic(test, r["feature"])["ic"]
                held = "" if (a is None or b is None) else (
                    " HOLDS" if a * b > 0 else " FLIPS")
                print(f"    {r['feature']:<22} train={a}  test={b}{held}")
            print("  NOTE: credit_pct_width, friction_pct_credit and moneyness "
                  "are largely\n        accounting identities for a "
                  "held-to-expiry structure — a high IC there\n        is "
                  "arithmetic, not a timing signal. Judge on bucket "
                  "monotonicity.\n")

    if args.signals:
        run("BASELINE no signal", "bull_put")
        run("IV rank >= 50", "bull_put", iv_rank_min=50)
        run("IV rank >= 70", "bull_put", iv_rank_min=70)
        run("IV rank <= 30", "bull_put", iv_rank_max=30)
        run("uptrend", "bull_put", trend_min=0)
        run("downtrend [AVOID]", "bull_put", trend_max=0)
        run("after 4w drop [AVOID]", "bull_put", ret_4w_max=-5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
