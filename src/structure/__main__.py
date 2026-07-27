"""CLI: python -m src.structure [SYMBOL] [--composite X] [--capital USD]

Display-only. Reads paper_trades.db for the league table; never writes to it.
"""
import argparse
from datetime import datetime

from .chain import fetch_candidates
from .express import express, load_costs
from .margins import (DEFAULT_HISTORY, apply_states, compute_league_table,
                      load_history)
from .report import render
from .view import build_view


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.structure")
    ap.add_argument("symbol", nargs="?", default="SPY")
    ap.add_argument("--composite", type=float, default=0.0,
                    help="directional composite in [-1,+1]; 0 = no view")
    ap.add_argument("--capital", type=float, default=511.0,
                    help="capital in USD (default 511 = 700 CAD)")
    ap.add_argument("--snapshot", action="store_true",
                    help="append this week's league table to history")
    ap.add_argument("--no-chain", action="store_true",
                    help="skip the live chain fetch (league table only)")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    table = compute_league_table()
    table = apply_states(table, load_history(DEFAULT_HISTORY), today)

    symbol = args.symbol.upper()
    cands, err = ({}, None) if args.no_chain else fetch_candidates(symbol, capital_usd=args.capital)
    if err:
        print("  note: {}".format(err))

    view = build_view(symbol, composite=args.composite)
    commission, slippage = load_costs()
    exprs, rej = express(view, table, args.capital, cands,
                         commission=commission, slippage=slippage)
    print(render(view, exprs, rej, table, args.capital))

    if args.snapshot:
        from .margins import append_snapshot
        append_snapshot(DEFAULT_HISTORY, table, today)
        print("  snapshot appended to {}".format(DEFAULT_HISTORY))


if __name__ == "__main__":
    main()
