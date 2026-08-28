"""Point-in-time reconstruction: what was knowable on date X.

THIS IS THE REUSABLE ASSET, not the backtest. Three later workstreams need
the same question answered — derived base rates, empirical slippage, and
implied-vs-realised — so the reconstruction lives here and studies consume it.

Two independent lookahead mechanisms, both mandatory:

  * TRIAL STATE — CT.gov versions every record. We take the latest version
    dated <= as_of and never read a later one. Verified 2026-08-25 on
    NCT06510816: v0 stated "2026-10" (month precision), v5 stated
    "2026-10-31" (day). Reading the final state would leak the answer.

  * FINANCIALS — XBRL points carry a `filed` date, and the lag is material,
    not cosmetic: ANNX's period ending 2025-12-31 was filed 2026-08-12. Using
    a figure before it was filed is lookahead however innocuous it looks.

The versioned payload nests under `study` but is otherwise the same
protocolSection shape as the v2 API, so `ctgov.parse_studies` is reused rather
than duplicated.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.request
from typing import Any, Dict, List, Optional

from src.catalyst import ctgov, pit_cache, runway
from src.catalyst.design import OUTCOME_EDIT_FLAG_THRESHOLD, Amendments
from src.catalyst.models import Trial
from src.catalyst.runway import Runway

HISTORY_BASE = "https://clinicaltrials.gov/api/int/studies"
TIMEOUT = 30


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "options-screener/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _fetch_versions(nct_id: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return list(_get_json(f"{HISTORY_BASE}/{nct_id}/history").get("changes") or [])
    except Exception:
        return None


def _fetch_study(nct_id: str, version: int) -> Optional[Dict[str, Any]]:
    try:
        return _get_json(f"{HISTORY_BASE}/{nct_id}/history/{version}")
    except Exception:
        return None


def version_at(versions: List[Dict[str, Any]], as_of: str) -> Optional[int]:
    """Latest version number dated <= as_of, or None if the trial did not yet
    exist. None is never silently replaced by version 0 — a trial registered
    after the vantage date is genuinely absent, not merely early."""
    best: Optional[int] = None
    best_date = ""
    for entry in versions or []:
        date = str(entry.get("date") or "")
        if not date or date > as_of:
            continue
        if date >= best_date:
            best_date, best = date, int(entry.get("version", 0))
    return best


#: The moduleLabel marking a PROTOCOL outcome-measure edit.
#:
#: "Outcome Measures (Results)" is a different label and is deliberately NOT
#: counted: posting results is not amending an endpoint. Validated 2026-08-27
#: against the live `outcomesUpdateCount` on 12 of 12 cached trials, including
#: one with three results-section updates whose live count stayed at its three
#: protocol edits. This is a restriction of the live statistic, not a new one.
_OUTCOME_LABEL = "Outcome Measures"


def amendments_as_of(versions: Optional[List[Dict[str, Any]]],
                     as_of: str) -> Amendments:
    """Amendment history as it stood on ``as_of``.

    `design.amendments_for` counts every change ever recorded, so an endpoint
    amended in 2025 marked a row "amended" at a 2023 vintage — lookahead in
    the one feature H2 is about. The dated version list needed to answer this
    correctly was already in the cache.

    An empty or missing list, and a trial whose first version postdates
    ``as_of``, both yield ``available=False`` rather than a confident zero:
    "we could not look" and "nothing changed" are different answers, and a
    trial registered after the vantage date is genuinely absent.
    """
    seen = [v for v in (versions or [])
            if str(v.get("date") or "") and str(v["date"]) <= as_of]
    if not seen:
        return Amendments()
    seen.sort(key=lambda v: str(v.get("date")))
    outcomes = sum(1 for v in seen
                   if _OUTCOME_LABEL in (v.get("moduleLabels") or []))
    flags: List[str] = []
    if outcomes >= OUTCOME_EDIT_FLAG_THRESHOLD:
        flags.append(f"outcome measures edited {outcomes}x")
    return Amendments(
        versions=len(seen),
        outcomes_updated=outcomes,
        status_now=seen[-1].get("status"),
        flags=tuple(flags),
        available=True,
    )


def _versions(nct_id: str, conn: sqlite3.Connection,
              as_of: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """The version list, refetched when the cached one predates ``as_of``.

    The list GROWS with every amendment, so an entry fetched before ``as_of``
    cannot answer for it. Passing ``as_of`` is what makes the cache renew when
    it must: a 2023 vintage is served from cache forever, while a live board
    asking about today refetches once a day.
    """
    cached = pit_cache.get_versions(conn, nct_id, as_of=as_of)
    if cached is not None:
        return cached
    fetched = _fetch_versions(nct_id)
    if fetched is None:
        return None
    pit_cache.put_versions(conn, nct_id, fetched)
    return fetched


def _study(nct_id: str, version: int,
           conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    cached = pit_cache.get_study(conn, nct_id, version)
    if cached is not None:
        return cached
    fetched = _fetch_study(nct_id, version)
    if fetched is None:
        return None
    pit_cache.put_study(conn, nct_id, version, fetched)
    return fetched


def trial_as_of(nct_id: str, as_of: str,
                conn: sqlite3.Connection) -> Optional[Trial]:
    """The trial as it was recorded on ``as_of``, or None."""
    versions = _versions(nct_id, conn, as_of=as_of)
    if not versions:
        return None
    version = version_at(versions, as_of)
    if version is None:
        return None
    payload = _study(nct_id, version, conn)
    if not payload:
        return None
    study = payload.get("study") or payload
    trials = ctgov.parse_studies({"studies": [study]})
    return trials[0] if trials else None


def _fetch_facts(cik: int) -> Optional[Dict[str, Any]]:
    try:
        from src.insider.edgar import _get
        return dict(_get(runway.FACTS_URL.format(cik=cik), timeout=60).json())
    except Exception:
        return None


def facts_as_of(facts: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    """A copy of ``facts`` containing only points FILED on or before as_of.

    A point with no `filed` date is dropped rather than kept: we cannot show
    that it was knowable, and assuming it was is the whole error this guards.
    The input is never mutated — callers reuse one cached payload across many
    vantage dates.
    """
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    out: Dict[str, Any] = {}
    for concept, node in gaap.items():
        units: Dict[str, Any] = {}
        for unit, items in (node.get("units") or {}).items():
            units[unit] = [p for p in items
                           if p.get("filed") and str(p["filed"]) <= as_of]
        out[concept] = {"units": units}
    return {"facts": {"us-gaap": out}}


def _facts(cik: int, conn: sqlite3.Connection,
           as_of: Optional[str] = None) -> Optional[Dict[str, Any]]:
    cached = pit_cache.get_facts(conn, cik, as_of=as_of)
    if cached is not None:
        return cached
    fetched = _fetch_facts(cik)
    if fetched is None:
        return None
    pit_cache.put_facts(conn, cik, fetched)
    return fetched


def runway_as_of(cik: int, as_of: str, event_date: str,
                 conn: sqlite3.Connection) -> Runway:
    """Cash runway as it was computable on ``as_of``. Never raises."""
    import datetime as dt

    facts = _facts(cik, conn, as_of=as_of)
    if not facts:
        return Runway()
    visible = facts_as_of(facts, as_of)
    cash, cash_at, cash_basis = runway.liquidity(visible)
    flow = runway.concept_points(visible, runway.FLOW_CONCEPT)
    if cash is None or cash_at is None or not flow:
        return Runway(cash=cash)
    burn, burn_basis = runway.quarterly_burn(flow)
    basis = f"{cash_basis} @ {cash_at}; burn {burn_basis}"
    if burn is None:
        return Runway(cash=cash)
    if burn <= 0:
        return Runway(cash=cash, cash_generative=True, burn_basis=basis)
    quarters = cash / burn
    try:
        end = (dt.date.fromisoformat(cash_at)
               + dt.timedelta(days=quarters * runway.DAYS_PER_QUARTER))
        runway_end: Optional[str] = end.isoformat()
    except (ValueError, OverflowError):
        runway_end = None
    return Runway(cash=cash, burn_per_quarter=burn, quarters=quarters,
                  runway_end=runway_end,
                  funded_through=runway.funded_through(runway_end, event_date),
                  burn_basis=basis)


def _index() -> Dict[str, str]:
    from src.catalyst import resolve
    return resolve.name_index()


def _aliases() -> Dict[str, str]:
    from src.catalyst import resolve
    return resolve.load_aliases()


def _caps(tickers: Any) -> Dict[str, Any]:
    from src.catalyst import universe
    return universe.market_caps(sorted(tickers))


def board_as_of(as_of: str, nct_ids: Any, conn: sqlite3.Connection,
                horizon_days: int = 365) -> Any:
    """The board as it would have printed on ``as_of``.

    MARKET CAP IS TODAY'S, NOT THE VINTAGE'S. yfinance exposes no historical
    market cap, and reconstructing shares-outstanding per vintage is a bigger
    project than this study. The band is a universe definition rather than a
    feature under test, so the contamination is bounded — but it is real, and
    the report states it rather than hiding it.
    """
    import datetime as dt

    from src.catalyst import resolve, universe
    from src.catalyst.models import CatalystEvent, Coverage

    coverage = Coverage()
    horizon = (dt.date.fromisoformat(as_of)
               + dt.timedelta(days=horizon_days)).isoformat()

    # `swept` counts the population resolution is ATTEMPTED on, i.e. after the
    # horizon filter — not every trial that merely existed. Counting existence
    # here while computing `resolved` over the horizon-filtered subset would
    # print a percentage whose numerator and denominator come from different
    # populations, which is the defect this codebase keeps rediscovering.
    seen = []
    for nct_id in nct_ids:
        trial = trial_as_of(nct_id, as_of, conn)
        if trial is None:
            continue
        if not (as_of < trial.event_date <= horizon):
            continue
        coverage.swept += 1
        seen.append(trial)

    index, aliases = _index(), _aliases()
    resolved = []
    for trial in seen:
        ticker = resolve.resolve(trial.sponsor_name, index, aliases)
        if ticker:
            resolved.append((trial, ticker))
        else:
            coverage.dropped_unresolved += 1
    coverage.resolved = len(resolved)

    caps = _caps({t for _, t in resolved})
    events = []
    for trial, ticker in resolved:
        mcap = caps.get(ticker)
        if not universe.in_band(mcap):
            coverage.dropped_out_of_band += 1
            continue
        events.append(CatalystEvent(trial=trial, ticker=ticker, mcap=mcap))
    return events, coverage
