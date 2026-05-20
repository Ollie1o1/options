"""Unified crypto entry point: python -m src.crypto {scan,log,exits,pnl,backtest}"""
from __future__ import annotations
import argparse, sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.crypto")
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("scan", help="Run the crypto options screener")
    sub.add_parser("log", help="Auto-log the top crypto pick")
    sub.add_parser("exits", help="Enforce TP/SL/time exits")
    sub.add_parser("pnl", help="Show crypto portfolio P&L")
    sub.add_parser("backtest", help="Run the crypto backtester")
    args, rest = p.parse_known_args(argv)
    if args.verb == "scan":
        from src.crypto.screener import main as m
        return int(m() or 0)
    if args.verb == "log":
        from src.crypto.auto_logger import main as m
        return int(m(rest) or 0)
    if args.verb == "exits":
        from src.crypto.exit_enforcer import main as m
        try:
            return int(m(rest) or 0)
        except TypeError:
            return int(m() or 0)
    if args.verb == "pnl":
        from src.crypto.check_pnl import main as m
        return int(m() or 0)
    if args.verb == "backtest":
        try:
            from src.crypto.backtester import main as m
        except ImportError:
            print("error: src.crypto.backtester has no standalone main()", file=sys.stderr)
            return 1
        try:
            return int(m(rest) or 0)
        except TypeError:
            return int(m() or 0)
    return 2


if __name__ == "__main__":
    sys.exit(main())
