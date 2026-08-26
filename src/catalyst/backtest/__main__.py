"""Catalyst backtest CLI.

    python -m src.catalyst.backtest --write-prereg   write the hypothesis file
    python -m src.catalyst.backtest                  run the study

The study REFUSES to print results unless the prereg file matches.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List, Optional, Sequence

from src.catalyst import pit_cache
from src.catalyst.backtest import outcomes as O
from src.catalyst.backtest import panel as P
from src.catalyst.backtest import prereg, report
from src.catalyst.backtest import study as S

BENCHMARK = "XBI"


def _prices(ticker: str, start: str, end: str) -> Dict[str, float]:
    try:
        import warnings

        import yfinance as yf
        warnings.filterwarnings("ignore")
        hist = yf.Ticker(ticker).history(start=start, end=end)
        return {d.date().isoformat(): float(c)
                for d, c in zip(hist.index, hist["Close"])}
    except Exception:
        return {}


def _sweep_ncts(start: str, end: str) -> List[str]:
    """Trial ids worth reconstructing, pre-filtered by sponsor and cap band.

    WHY PRE-FILTER. board_as_of reconstructs every id it is handed at every
    vintage, and reconstruction is the expensive part — one call per trial for
    the version list, more for the versions themselves. Measured 2026-08-25:
    120 trials took 31s for a single vintage, so the unfiltered 7,701-trial
    sweep would have taken about 6.6 hours across 12 vintages, and ~95% of
    that work is spent on trials whose sponsor never resolves to a ticker.

    THE COMPROMISE, STATED. The pre-filter uses the sponsor name as it stands
    TODAY, not as of each vintage. Sponsor names change rarely (mostly through
    acquisition), and this is a universe definition rather than a feature under
    test — the same category as market cap. It is not zero lookahead, and the
    report says so.
    """
    from src.catalyst import ctgov, resolve, universe

    trials = ctgov.sweep(start, end)
    index, aliases = resolve.name_index(), resolve.load_aliases()
    by_ticker: Dict[str, List[str]] = {}
    for t in trials:
        ticker = resolve.resolve(t.sponsor_name, index, aliases)
        if ticker:
            by_ticker.setdefault(ticker, []).append(t.nct_id)
    caps = universe.market_caps(sorted(by_ticker))
    return [nct for ticker, ncts in by_ticker.items()
            if universe.in_band(caps.get(ticker)) for nct in ncts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.catalyst.backtest")
    parser.add_argument("--write-prereg", action="store_true")
    parser.add_argument("--prereg", default=prereg.DEFAULT_PATH)
    parser.add_argument("--db", default=pit_cache.DEFAULT_DB)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-10-01")
    parser.add_argument("--today", default=dt.date.today().isoformat())
    parser.add_argument("--limit", type=int, default=0,
                        help="cap trials swept (0 = no cap); for smoke runs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.write_prereg:
        print("hash:", prereg.write(args.prereg))
        return 0

    if not prereg.verify(args.prereg):
        print(report.render([], {}, 0, prereg_ok=False))
        return 2

    conn = pit_cache.connect(args.db)
    try:
        ncts = _sweep_ncts(args.start, "2027-12-31")
        if args.limit:
            ncts = ncts[:args.limit]
        bench = _prices(BENCHMARK, args.start, args.today)

        rows: List[P.PanelRow] = []
        for vintage in P.vintages(args.start, args.end):
            built, _ = P.build(vintage, ncts, conn)
            rows.extend(built)

        # Each hypothesis is (key, label, horizon, splitter). The splitter
        # returns True/False for the two arms, or None to exclude the row —
        # tri-state throughout, so "unknown" never silently joins an arm.
        splits = (
            ("H1", "FUNDED THROUGH vs RAISE BEFORE", 6,
             lambda r: r.funded_through),
            ("H2", "ENDPOINT AMENDED vs NOT", 6,
             lambda r: r.amended),
            ("H3", "PHASE 3 vs PHASE 2", 6,
             lambda r: (None if r.phase not in ("PHASE2", "PHASE3")
                        else r.phase == "PHASE3")),
        )
        arms: Dict[str, List[List[float]]] = {k: [[], []] for k, _, _, _ in splits}

        by_ticker: Dict[str, Dict[str, float]] = {}
        counts: Dict[int, int] = {}
        for row in rows:
            if all(fn(row) is None for _, _, _, fn in splits):
                continue
            if row.ticker not in by_ticker:
                by_ticker[row.ticker] = _prices(row.ticker, args.start,
                                                args.today)
            outs = O.outcomes_for(row.ticker, row.vintage, args.today,
                                  by_ticker[row.ticker], bench)
            for o in outs:
                counts[o.months] = counts.get(o.months, 0) + 1
                if o.relative is None:
                    continue
                for key, _, horizon, fn in splits:
                    if o.months != horizon:
                        continue
                    side = fn(row)
                    if side is None:
                        continue
                    arms[key][0 if side else 1].append(o.relative)

        results = [
            S.compare(arms[k][0], arms[k][1], key=k,
                      label=f"{lbl} ({h}mo, XBI-rel)")
            for k, lbl, h, _ in splits
        ]
        # H4 needs the implied move AS OF each vintage, which needs historical
        # option chains. Verified 2026-08-26: data/chain_archive.db holds only
        # mega-cap tech (AAPL, NVDA, META...) and no biotech at all, and no
        # free source backfills small-cap chains. Declared, not runnable.
        results.append(S.not_computable(
            "H4", "IMPLIED vs REALISED MOVE (3mo)",
            "no historical option chains exist for these names"))

        dropped = sum(1 for t in by_ticker if not by_ticker[t])
        print(report.render(results, counts, dropped, prereg_ok=True))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
