"""Catalyst backtest CLI.

    python -m src.catalyst.backtest --write-prereg   write the hypothesis file
    python -m src.catalyst.backtest                  run the study

The study REFUSES to print results unless the prereg file matches.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.catalyst import pit_cache
from src.catalyst.backtest import outcomes as O
from src.catalyst.backtest import panel as P
from src.catalyst.backtest import prereg, report
from src.catalyst.backtest import study as S

BENCHMARK = "XBI"


def _fetch_prices(ticker: str, start: str, end: str) -> Dict[str, float]:
    """Close series from yfinance. {} on any failure."""
    try:
        import warnings

        import yfinance as yf
        warnings.filterwarnings("ignore")
        hist = yf.Ticker(ticker).history(start=start, end=end)
        return {d.date().isoformat(): float(c)
                for d, c in zip(hist.index, hist["Close"])}
    except Exception:
        return {}


def _prices(ticker: str, start: str, end: str, conn: Optional[Any] = None,
            today: Optional[str] = None) -> Dict[str, float]:
    """Close series, cached with a freshness rule that matches the data.

    This was the run's dominant cost: one uncached, serial yfinance call per
    ticker, roughly 270 of them. Caching it is only safe because the validity
    rule is derived from the data rather than guessed — a closed window is
    settled and keeps indefinitely, an open one lasts the day it was taken.
    """
    if conn is None:
        return _fetch_prices(ticker, start, end)
    cached = pit_cache.get_prices(conn, ticker, start, end, today=today)
    if cached is not None:
        return cached
    series = _fetch_prices(ticker, start, end)
    # AN EMPTY SERIES IS NEVER CACHED. `_fetch_prices` returns {} on ANY
    # exception, so an empty result is indistinguishable from a rate-limit or
    # a network blip. Caching it poisons every later run: on 2026-08-28 a
    # rate-limited run stored 145 empty series and the NEXT run returned n=0
    # for every hypothesis — the entire study evaluated to nothing, silently.
    # This is the `_fetch_chain_quotes` defect exactly (see
    # `candidate_marks._fetch_chain`): a fetcher that swallows its own
    # exceptions cannot report failure, so its caller must not read a falsy
    # result as an answer. Refetching a genuinely delisted ticker every run is
    # far cheaper than a study that quietly returns nothing.
    if series:
        pit_cache.put_prices(conn, ticker, start, end, series, fetched_at=today)
    return series


def _universe_key(start: str, end: str) -> str:
    """Cache key for a pinned universe.

    The window is IN the key. Reusing one window's frozen population for a
    different window would be the worst outcome caching could produce —
    silently the wrong sample, with no symptom.
    """
    return f"sweep|{start}|{end}|PHASE2,PHASE3|band"


def _sweep_ncts(start: str, end: str, conn: Optional[Any] = None,
                refresh: bool = False,
                today: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """The study population, pinned. Returns (nct_ids, pinned_at).

    The sweep was the last live input to WHICH ROWS EXIST. CT.gov gains and
    edits trials, and the cap band is applied with TODAY'S market cap, so
    re-running silently re-drew the sample: H3's arms moved 755 -> 736 between
    two runs a day apart with no code change. A study whose population shifts
    under it cannot be compared with itself.

    Pinning is therefore a correctness fix first and a speed fix second —
    `universe.market_caps` makes one uncached, serial yfinance call per
    resolved ticker, which dominated a 30-60 minute run.

    ``pinned_at`` is None when nothing was pinned (no cache, or a fresh
    sweep this run), so the report can say which universe the reader is
    looking at rather than leaving it ambiguous.
    """
    key = _universe_key(start, end)
    if conn is not None and not refresh:
        cached = pit_cache.get_universe(conn, key)
        if cached is not None:
            # An empty list is a real answer — "swept, nothing matched" — and
            # is honoured rather than treated as a miss.
            return list(cached[1]), cached[0]

    ncts = _fresh_sweep(start, end)
    if conn is not None:
        import datetime as _dt
        stamp = today or _dt.date.today().isoformat()
        pit_cache.put_universe(conn, key, stamp, ncts)
        return ncts, stamp
    return ncts, None


def _fresh_sweep(start: str, end: str) -> List[str]:
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
    parser.add_argument("--refresh-universe", action="store_true",
                        help="re-sweep and re-pin the study population; "
                             "changes which rows exist, so two runs either "
                             "side of it are NOT comparable")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.write_prereg:
        print("hash:", prereg.write(args.prereg))
        return 0

    if not prereg.verify(args.prereg):
        print(report.render([], {}, 0, prereg_ok=False))
        return 2

    conn = pit_cache.connect(args.db)
    try:
        ncts, pinned_at = _sweep_ncts(args.start, "2027-12-31", conn=conn,
                                      refresh=args.refresh_universe,
                                      today=args.today)
        universe_n = len(ncts)
        if args.limit:
            ncts = ncts[:args.limit]
        bench = _prices(BENCHMARK, args.start, args.today,
                        conn=conn, today=args.today)
        # EVERY hypothesis is XBI-relative, so an empty benchmark nulls every
        # outcome and the study silently evaluates to nothing. Observed
        # 2026-08-28 under yfinance rate-limiting: three hypotheses printed
        # "n = 0 vs 0 ... UNDERPOWERED", indistinguishable from a real null.
        # Refuse instead — a failed run is not a finding.
        if not bench:
            print(report.render([], {}, 0, prereg_ok=True,
                                run_failed=f"benchmark {BENCHMARK} returned no "
                                           f"prices; every outcome is "
                                           f"{BENCHMARK}-relative"))
            return 3

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
        # (value, cluster, arm) per hypothesis. The cluster is the TICKER,
        # because that is what the outcome belongs to — `outcomes_for` never
        # sees an nct_id, so trials sharing a ticker and a vintage carry the
        # same number and are one observation between them.
        arms: Dict[str, List[S.Observation]] = {k: [] for k, _, _, _ in splits}

        by_ticker: Dict[str, Dict[str, float]] = {}
        counts: Dict[int, int] = {}
        for row in rows:
            if all(fn(row) is None for _, _, _, fn in splits):
                continue
            if row.ticker not in by_ticker:
                by_ticker[row.ticker] = _prices(
                    row.ticker, args.start, args.today,
                    conn=conn, today=args.today)
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
                    arms[key].append((o.relative, row.ticker, bool(side)))

        results = [
            S.compare_clustered(arms[k], key=k,
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
        print(report.render(results, counts, dropped, prereg_ok=True,
                            universe_pinned_at=pinned_at,
                            universe_n=universe_n))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
