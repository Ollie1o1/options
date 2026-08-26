"""Catalyst calendar CLI.

    python -m src.catalyst                    board, 6-month window
    python -m src.catalyst --window 90d       shorter window
    python -m src.catalyst --phase 3          Ph3 only
    python -m src.catalyst --funded-only      drop names that must raise first
    python -m src.catalyst ANNX               every in-window event for one name
    python -m src.catalyst --mark             resolve elapsed events, no render

The seams (_sweep, _name_index, _market_caps, _amendments, _runway, _implied)
are module-level indirections so tests can replace every network boundary
without touching the modules that own them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.catalyst import board as B
from src.catalyst import (bands, ctgov, design, implied, pdufa, resolve,
                          runway, store, universe)
from src.catalyst.design import Amendments
from src.catalyst.implied import ImpliedMove
from src.catalyst.models import BandCoverage, CatalystEvent, Coverage, Trial
from src.catalyst.runway import Runway

DEEP_TIER_LIMIT = 40


def _sweep(start: str, end: str, phases: Sequence[str]) -> List[Trial]:
    return ctgov.sweep(start, end, phases=phases)


def _name_index() -> Dict[str, str]:
    return resolve.name_index()


def _aliases() -> Dict[str, str]:
    return resolve.load_aliases()


def _market_caps(tickers: Sequence[str]) -> Dict[str, Optional[float]]:
    return universe.market_caps(tickers)


def _amendments(nct_id: str) -> Amendments:
    return design.amendments_for(nct_id)


def _runway(ticker: str, event_date: str) -> Runway:
    return runway.runway_for(ticker, event_date)


def _implied(ticker: str, event_date: str) -> ImpliedMove:
    return implied.implied_move(ticker, event_date)


def _pdufa(start: str, end: str) -> List[Any]:
    return pdufa.pdufa_events(start, end)


def pdufa_rows(today: str, end: str, lookback_days: int = 180
               ) -> List[B.PdufaRow]:
    """Upcoming FDA decision dates, with runway and implied move attached.

    Announcements are searched over the LOOKBACK (a PDUFA date is announced
    months before it lands), then filtered to dates still ahead of us. Rows
    are not cap-banded: an approval decision is material to a company of any
    size, and unlike the trial calendar there is no resolver ceiling to work
    around — EDGAR hands us the ticker directly.
    """
    import datetime as dt

    since = (dt.date.fromisoformat(today)
             - dt.timedelta(days=lookback_days)).isoformat()
    rows: List[B.PdufaRow] = []
    seen = set()
    for event in _pdufa(since, today):
        if not (today <= event.event_date <= end) or event.ticker in seen:
            continue
        seen.add(event.ticker)
        cash, move = Runway(), ImpliedMove()
        try:
            cash = _runway(event.ticker, event.event_date)
        except Exception:
            pass
        try:
            move = _implied(event.ticker, event.event_date)
        except Exception:
            pass
        rows.append(B.PdufaRow(event=event, runway=cash, implied=move))
    return rows


def window(spec: str, today: Optional[str] = None) -> Tuple[str, str]:
    """('2026-08-25', '2027-02-21') from a spec like '6m' or '90d'."""
    start = dt.date.fromisoformat(today) if today else dt.date.today()
    unit, value = spec[-1:].lower(), spec[:-1]
    if not value.isdigit() or unit not in ("m", "d"):
        raise ValueError(f"window must look like '6m' or '90d', got {spec!r}")
    days = int(value) * (30 if unit == "m" else 1)
    return start.isoformat(), (start + dt.timedelta(days=days)).isoformat()


def _deep(event: CatalystEvent,
          coverage: Coverage) -> Tuple[Amendments, Runway, ImpliedMove]:
    """Deep-tier lookups for one event. A failure costs its own column and
    increments the counter; it never propagates."""
    amendments, cash, move = Amendments(), Runway(), ImpliedMove()
    try:
        amendments = _amendments(event.trial.nct_id)
    except Exception:
        coverage.deep_failures += 1
    try:
        cash = _runway(event.ticker, event.event_date)
    except Exception:
        coverage.deep_failures += 1
    try:
        move = _implied(event.ticker, event.event_date)
    except Exception:
        coverage.deep_failures += 1
    return amendments, cash, move


def build_rows(start: str, end: str, phases: Sequence[str] = ("PHASE2", "PHASE3"),
               funded_only: bool = False,
               deep_limit: int = DEEP_TIER_LIMIT,
               today: Optional[str] = None) -> Tuple[List[B.BoardRow], Coverage]:
    """Sweep, resolve, cap-filter, collapse, band, allocate, then deep-fetch.

    ``today`` exists so tests can band deterministically; production passes
    None and gets the real date.
    """
    coverage = Coverage()
    trials = _sweep(start, end, phases)
    coverage.swept = len(trials)

    index, aliases = _name_index(), _aliases()
    resolved: List[Tuple[Trial, str]] = []
    for trial in trials:
        ticker = resolve.resolve(trial.sponsor_name, index, aliases)
        if ticker:
            resolved.append((trial, ticker))
        else:
            coverage.dropped_unresolved += 1
    coverage.resolved = len(resolved)

    caps = _market_caps(sorted({ticker for _, ticker in resolved}))
    events: List[CatalystEvent] = []
    for trial, ticker in resolved:
        mcap = caps.get(ticker)
        if not universe.in_band(mcap):
            coverage.dropped_out_of_band += 1
            continue
        events.append(CatalystEvent(trial=trial, ticker=ticker, mcap=mcap))

    collapsed = B.collapse(events)
    as_of = today or dt.date.today().isoformat()

    # Band first, THEN spend the budget. Taking collapsed[:deep_limit]
    # front-loaded by date: measured 2026-08-26, a 6-month window returned
    # 40 names that all fell inside 2 months and withheld 57 later ones
    # without saying which part of the window had gone missing.
    banded: Dict[str, List[Tuple[CatalystEvent, int]]] = {
        band: [] for band in bands.TRIAL_BANDS}
    for event, others in collapsed:
        banded[bands.band_for(event.event_date, as_of)].append((event, others))

    counts = {band: len(banded[band]) for band in bands.TRIAL_BANDS}
    budget = bands.allocate(counts, deep_limit)
    coverage.bands = [BandCoverage(band=band, found=counts[band],
                                   shown=budget[band])
                      for band in bands.TRIAL_BANDS]

    selected: List[Tuple[CatalystEvent, int]] = []
    for band in bands.TRIAL_BANDS:
        selected.extend(banded[band][:budget[band]])
    selected.sort(key=lambda pair: B.sort_key(pair[0]))

    coverage.shown = len(selected)
    coverage.truncated = max(0, len(collapsed) - len(selected))

    rows: List[B.BoardRow] = []
    for event, others in selected:
        amendments, cash, move = _deep(event, coverage)
        if funded_only and cash.funded_through is not True:
            continue
        rows.append(B.BoardRow(event=event, other_events=others,
                               amendments=amendments, runway=cash, implied=move))
    return rows, coverage


def detail_rows(ticker: str, start: str, end: str,
                phases: Sequence[str] = ("PHASE2", "PHASE3")
                ) -> Tuple[List[B.BoardRow], Coverage]:
    """Every in-window event for ONE ticker, one row each.

    Deliberately skips the market-cap band: the band exists to keep the board
    short, and you asked about this name explicitly."""
    coverage = Coverage()
    trials = _sweep(start, end, phases)
    coverage.swept = len(trials)
    index, aliases = _name_index(), _aliases()
    wanted = ticker.upper()

    events: List[CatalystEvent] = []
    for trial in trials:
        resolved = resolve.resolve(trial.sponsor_name, index, aliases)
        if not resolved:
            coverage.dropped_unresolved += 1
            continue
        coverage.resolved += 1
        if resolved == wanted:
            events.append(CatalystEvent(trial=trial, ticker=resolved))

    caps = _market_caps([wanted]) if events else {}
    rows: List[B.BoardRow] = []
    for event in sorted(events, key=B.sort_key):
        priced = CatalystEvent(trial=event.trial, ticker=event.ticker,
                               mcap=caps.get(wanted))
        amendments, cash, move = _deep(priced, coverage)
        rows.append(B.BoardRow(event=priced, other_events=0,
                               amendments=amendments, runway=cash,
                               implied=move))
    return rows, coverage


def run_detail(args: argparse.Namespace) -> int:
    start, end = window(args.window)
    rows, coverage = detail_rows(args.ticker, start, end)
    print(B.render(rows, coverage))
    return 0


def run_board(args: argparse.Namespace) -> int:
    start, end = window(args.window)
    phases = {"2": ("PHASE2",), "3": ("PHASE3",)}.get(args.phase,
                                                      ("PHASE2", "PHASE3"))
    rows, coverage = build_rows(start, end, phases=phases,
                                funded_only=args.funded_only,
                                deep_limit=args.limit)
    reg = [] if args.no_pdufa else pdufa_rows(start, end)
    conn = store.connect(args.db)
    try:
        today = dt.date.today().isoformat()
        for row in rows:
            store.upsert_event(conn, row.event, today)
            store.add_mark(conn, row.event.event_id, today,
                           row.event.event_date, row.event.trial.status,
                           row.implied.spot)
    finally:
        conn.close()
    print(B.render(rows, coverage, pdufa=reg))
    return 0


def run_mark(args: argparse.Namespace) -> int:
    """Re-observe elapsed events. No rendering — this is the scheduled path."""
    conn = store.connect(args.db)
    try:
        today = dt.date.today().isoformat()
        pending = store.outstanding(conn, today)
        for event_id, _ticker, _date in pending:
            nct = event_id.split(":")[0]
            fresh = design.amendments_for(nct)
            store.add_mark(conn, event_id, today, None, fresh.status_now, None)
        print(f"marked {len(pending)} elapsed events")
    finally:
        conn.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.catalyst",
        description="Dated Ph2/Ph3 catalysts for small/mid-cap biotech. "
                    "Sorted by date, never ranked.")
    parser.add_argument("ticker", nargs="?",
                        help="show every in-window event for one ticker")
    parser.add_argument("--window", default="6m", help="e.g. 6m, 90d")
    parser.add_argument("--phase", choices=["2", "3", "all"], default="all")
    parser.add_argument("--funded-only", action="store_true",
                        help="only names funded through their own catalyst")
    parser.add_argument("--limit", type=int, default=DEEP_TIER_LIMIT,
                        help=f"how many names to deep-fetch "
                             f"(default {DEEP_TIER_LIMIT}); the board states "
                             f"how many it withheld")
    parser.add_argument("--no-pdufa", action="store_true",
                        help="skip the FDA decision-date section")
    parser.add_argument("--db", default=store.DEFAULT_DB)
    parser.add_argument("--mark", action="store_true",
                        help="resolve elapsed events into catalyst_marks")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.mark:
            return run_mark(args)
        return run_detail(args) if args.ticker else run_board(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
