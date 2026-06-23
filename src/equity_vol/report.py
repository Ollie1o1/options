"""Render the equity-vol backtest: per-symbol breakdown, aggregate, and the
out-of-sample verdict. Decision-support / research — no raw ANSI (source-scanned)."""
from __future__ import annotations
from typing import List
from collections import defaultdict
from src import ui
from src.equity_vol import metrics as M

W = 78
_COLS = [{"h": "Symbol", "w": 8, "align": "left"},
         {"h": "n", "w": 5, "align": "right"},
         {"h": "MeanR", "w": 8, "align": "right"},
         {"h": "Sharpe", "w": 8, "align": "right"},
         {"h": "Hit%", "w": 7, "align": "right"}]


def summarize(results: List) -> dict:
    rets = [r.ret for r in results]
    dated = [(r.date, r.ret) for r in results]
    return {"n": len(results), "mean_ret": (sum(rets) / len(rets) if rets else None),
            "sharpe": M.sharpe(rets), "hit": M.hit_rate(rets),
            "pf": M.profit_factor(rets), "nw_t": M.newey_west_t(rets),
            "oos": M.split_oos(dated)}


def _num(x, fmt="{:+.2f}"):
    return fmt.format(x) if x is not None else "-"


def render(results: List) -> str:
    if not results:
        return "  no trades — run with a populated data/dolt_options.db"
    by_sym = defaultdict(list)
    for r in results:
        by_sym[r.symbol].append(r)
    rows = []
    for sym in sorted(by_sym):
        rs = [t.ret for t in by_sym[sym]]
        rows.append([sym, str(len(rs)), _num(sum(rs) / len(rs)),
                     _num(M.sharpe(rs)), f"{M.hit_rate(rs) * 100:.0f}%"])
    s = summarize(results)
    tr, te = s["oos"]["train"], s["oos"]["test"]
    verdict = ("EDGE survives OOS" if (te["sharpe"] or -9) > 0 and (s["nw_t"] or 0) > 2
               else "NO edge after cost / OOS")
    lines = [ui.rule(W, "EQUITY VRP - delta-hedged short straddle"),
             ui.table(_COLS, rows), "",
             f"  aggregate: n={s['n']}  meanR={_num(s['mean_ret'])}  "
             f"Sharpe={_num(s['sharpe'])}  NW-t={_num(s['nw_t'])}  "
             f"hit={s['hit'] * 100:.0f}%  PF={_num(s['pf'], '{:.2f}')}",
             f"  OOS: train n={tr['n']} Sharpe={_num(tr['sharpe'])} | "
             f"test n={te['n']} Sharpe={_num(te['sharpe'])}",
             f"  verdict: {verdict}",
             "  research-only: real dolt fills, time-sparse entries (~weekly), "
             "BS-repriced hedge w/ entry IV, ~8 mega-caps (survivorship). Not deployable as-is."]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    from src.equity_vol.engine import run_backtest
    p = argparse.ArgumentParser(description="Equity VRP (delta-hedged straddle) backtest")
    p.add_argument("--db", default="data/dolt_options.db")
    p.add_argument("--symbol", action="append", default=None)
    p.add_argument("--target-dte", type=int, default=30)
    p.add_argument("--freq-days", type=int, default=28)
    args = p.parse_args(argv)
    syms = args.symbol or ["AAPL", "AMD", "NVDA", "GOOG", "SPY", "MSFT", "AMZN", "META"]
    results = run_backtest(args.db, syms, target_dte=args.target_dte, freq_days=args.freq_days)
    print(render(results))
