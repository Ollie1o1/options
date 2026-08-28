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


class PriceFetchError(RuntimeError):
    """The price SOURCE failed. Not the same as "this ticker has no prices".

    Keeping these apart is the whole point. Three outcomes must stay
    distinguishable, and collapsing them caused three separate bugs in one
    day — empty series cached as answers, a rate-limited benchmark rendering
    as UNDERPOWERED, and single tickers silently dropping out so the sample
    shrank between runs:

        a series          data
        an empty series   we looked; this ticker has no bars (delisted)
        PriceFetchError   we could not look
    """


#: Attempts per ticker before a fetch failure is raised to the caller.
#: Bounded for the same reason `candidate_marks._CHAIN_ATTEMPTS` is: the
#: opposite failure is hammering a source that is already rate-limiting us.
_PRICE_ATTEMPTS = 2

#: Above this share of unreachable tickers the run is refused: the
#: surviving sample is no longer the population that was pinned.
_MAX_FAILED_FRACTION = 0.05


def _history(ticker: str, start: str, end: str) -> Any:
    """Raw bars from yfinance. Raises whatever the library raises."""
    import warnings

    import yfinance as yf
    warnings.filterwarnings("ignore")
    hist = yf.Ticker(ticker).history(start=start, end=end)
    return list(zip(hist.index, hist["Close"]))


def _fetch_prices(ticker: str, start: str, end: str) -> Dict[str, float]:
    """Close series, or PriceFetchError if the source could not be read.

    An EMPTY result is returned normally: a delisted name genuinely has no
    bars, and that is data. Only a raised exception — a transport error, a
    429, a malformed payload — becomes a PriceFetchError.
    """
    try:
        rows = _history(ticker, start, end)
    except Exception as exc:
        raise PriceFetchError(f"{ticker}: {exc}") from exc
    try:
        return {d.date().isoformat(): float(c) for d, c in rows}
    except Exception as exc:
        raise PriceFetchError(f"{ticker}: unreadable payload: {exc}") from exc


def _is_complete(series: Dict[str, float], expect_through: str) -> bool:
    """Does this series reach the date the benchmark reached?

    A truncated response is NON-EMPTY, so it slips past every falsy check and
    gets cached — which is how two runs on one pinned universe returned 1544
    and 1538 rows with identical ticker counts. The benchmark defines the
    trading calendar for the window, so its last bar is the yardstick.

    A genuinely delisted name also fails this, and that is the right outcome:
    its data is real but we cannot tell it apart from a truncation, so it is
    refetched rather than frozen into the cache.
    """
    if not series:
        return False
    return max(series) >= expect_through


def _prices(ticker: str, start: str, end: str, conn: Optional[Any] = None,
            today: Optional[str] = None,
            expect_through: Optional[str] = None) -> Dict[str, float]:
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

    # Retry a source failure, then RAISE. Returning {} here is what let a
    # rate-limited ticker vanish from the study without a trace, shrinking the
    # sample between two runs on the same pinned universe.
    last: Optional[PriceFetchError] = None
    series: Optional[Dict[str, float]] = None
    for _ in range(_PRICE_ATTEMPTS):
        try:
            series = _fetch_prices(ticker, start, end)
            break
        except PriceFetchError as exc:
            last = exc
    if series is None:
        raise last if last else PriceFetchError(f"{ticker}: unknown failure")
    # An empty series is still not cached, even now that it can only mean
    # "delisted". yfinance also returns an empty frame WITHOUT raising when it
    # soft-throttles, so {} remains ambiguous at the library boundary; a
    # rate-limited run once cached 145 of them and the next run returned n=0
    # for every hypothesis. Refetching a genuinely dead ticker each run is far
    # cheaper than a study that quietly evaluates to nothing.
    # Cache only a series we can show is COMPLETE. Empty means "we looked and
    # there is nothing", truncated means "we cannot tell how much we got" —
    # neither may be frozen, because a partial series cached once poisons
    # every later run and silently shrinks the sample.
    if series and (expect_through is None or _is_complete(series, expect_through)):
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
        try:
            bench = _prices(BENCHMARK, args.start, args.today,
                            conn=conn, today=args.today)
        except PriceFetchError as exc:
            bench = {}
            print(report.render([], {}, 0, prereg_ok=True,
                                run_failed=f"benchmark {BENCHMARK} "
                                           f"unreadable: {exc}"))
            return 3
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
        # The benchmark's last bar is the yardstick for every other series:
        # it defines the trading calendar the study actually observed.
        bench_last = max(bench)

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
        failed_fetches: set = set()
        counts: Dict[int, int] = {}
        for row in rows:
            if all(fn(row) is None for _, _, _, fn in splits):
                continue
            if row.ticker not in by_ticker:
                # A source failure is COUNTED, never silently dropped. A
                # ticker that vanishes takes its cluster with it and moves
                # every interval — that is how two runs on one pinned
                # universe returned 1544 and 1545 rows.
                try:
                    by_ticker[row.ticker] = _prices(
                        row.ticker, args.start, args.today,
                        conn=conn, today=args.today,
                        expect_through=bench_last)
                except PriceFetchError:
                    failed_fetches.add(row.ticker)
                    by_ticker[row.ticker] = {}
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
        # A material share of the universe unreachable is a failed run, not a
        # smaller study: the sample is no longer the pinned population.
        reachable = max(len(by_ticker), 1)
        if len(failed_fetches) / reachable > _MAX_FAILED_FRACTION:
            print(report.render(
                [], {}, 0, prereg_ok=True,
                run_failed=f"{len(failed_fetches)} of {len(by_ticker)} tickers "
                           f"could not be priced; the sample is not the "
                           f"pinned universe"))
            return 3
        print(report.render(results, counts, dropped, prereg_ok=True,
                            universe_pinned_at=pinned_at,
                            universe_n=universe_n,
                            failed_fetches=len(failed_fetches)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
