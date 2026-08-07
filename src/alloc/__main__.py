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
from src.alloc.universe import (audit_coverage, load_universe, symbol_stratum,
                                terminal_dates, usable_dates, usable_symbols)
from src.strategies.spec import StrategySpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "data", "dolt_options.db")
UNIVERSE = os.path.join(ROOT, "data", "backtest_universe.json")

# Tight-spread names: 3.6% median bid-ask against 16-21% elsewhere. The largest
# single driver of results found anywhere in this study.
MEGA = ["SPY", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOG", "AMZN",
        "XLI", "XLV", "XLY", "XME", "XPH", "XHE"]


def _trading_dates(start: str, end: str, weekly: bool) -> List[str]:
    d, out = dt.date.fromisoformat(start), []
    stop = dt.date.fromisoformat(end)
    while d <= stop:
        if (d.weekday() == 4) if weekly else (d.weekday() < 5):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _spec(name: str, structure: str, entry: Dict, n_trials: int,
          hold: bool = True) -> StrategySpec:
    return StrategySpec(
        id=name, version=1, structure=structure, universe={}, entry=entry,
        exit={"hold_to_expiry": True} if hold else
             {"profit_target": 0.5, "stop": 2.0, "hold_to_expiry": False},
        sizing={"max_capital_at_risk": 4000, "max_concurrent": 3},
        created=dt.date.today().isoformat(), trial_count=n_trials)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Allocation backtester")
    ap.add_argument("--audit", action="store_true",
                    help="coverage audit; can veto everything else")
    ap.add_argument("--signals", action="store_true",
                    help="signal conditions against the unconditional baseline")
    ap.add_argument("--structures", action="store_true",
                    help="compare structures on the tight-spread universe")
    ap.add_argument("--mega", action="store_true", default=True,
                    help="restrict to tight-spread names (default)")
    ap.add_argument("--all-names", action="store_true",
                    help="use every usable symbol instead")
    ap.add_argument("--weekly", action="store_true",
                    help="sample Fridays only rather than every weekday")
    ap.add_argument("--trials", type=int, default=34,
                    help="configurations tried, for deflation")
    ap.add_argument("--start", default="2022-01-07")
    ap.add_argument("--end", default="2026-06-12")
    args = ap.parse_args(argv)

    universe = load_universe(UNIVERSE)
    audit = audit_coverage(DB, universe)

    if args.audit or not (args.signals or args.structures):
        print("COVERAGE AUDIT")
        for stratum, v in audit["summary"].items():
            print(f"  {stratum:<8} total={v['total']:>3} usable={v['usable']:>3} "
                  f"sparse={v['sparse']:>2} absent={v['absent']:>2}")
        print(f"  viable={audit['viable']}  "
              f"dead dates={len(audit['dead_dates'])}")
        if not args.signals and not args.structures:
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
    print(f"\n{len(syms)} symbols x {len(dates)} dates  "
          f"(deflating by {args.trials} configurations)\n")

    def run(label: str, structure: str, **entry):
        e = {"dte": [25, 45], "short_delta": 0.25, "width": 5.0}
        e.update(entry)
        trades, _ = replay(_spec(label, structure, e, args.trials), syms,
                           dates, src, terminal=term, stratum_of=strat)
        print(format_summary(label, summarise(trades, args.trials)))

    if args.structures:
        run("bull_put", "bull_put")
        run("bear_call", "bear_call")
        run("iron_condor", "iron_condor", short_delta=0.16)
        run("long_call [CONTROL]", "long_call", target_delta=0.40)

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
