"""Cash runway from EDGAR XBRL company-concept data.

The single most decision-relevant fact about a pre-revenue biotech facing a
dated readout: does it reach its own catalyst without raising? A company that
must issue equity first is a different proposition, and the dilution usually
lands before the data does.

Concepts used (both us-gaap, both free):
  CashAndCashEquivalentsAtCarryingValue        instant — latest cash balance
  NetCashProvidedByUsedInOperatingActivities   duration — operating cash flow

THE DURATION IS NOT OPTIONAL. XBRL publishes operating cash flow as
year-to-date periods of varying length — 89, 180, 272 and 364 days all appear
in the same series, often republished under a later fiscal-year label. A value
without its window is meaningless, so ``parse_concept`` carries ``start`` and
``quarterly_burn`` normalises by the real span. See that function for the live
failure that forced this.

Operating cash flow is NEGATIVE for a burner, so burn is its sign flip. A
company with POSITIVE operating cash flow has no runway limit to compute:
quarters stays None and cash_generative is True. Reporting "0 quarters" for a
profitable company would be a number describing something other than its label.

Everything is failure-safe: unknown is None, never zero.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CASH_CONCEPT = "CashAndCashEquivalentsAtCarryingValue"
FLOW_CONCEPT = "NetCashProvidedByUsedInOperatingActivities"
DAYS_PER_QUARTER = 91.3125

# Clinical-stage biotechs hold most of their money in short-term marketable
# securities, NOT in "cash and equivalents". Measured 2026-08-25: CGON reported
# $19.6m of CashAndCashEquivalentsAtCarryingValue and $1,008.6m of
# MarketableSecuritiesCurrent on the same date. Reading cash alone gave 0.6
# quarters of runway and a "RAISE BEFORE" verdict for a company sitting on over
# a billion dollars.
#
# Only the FIRST present tag is added to cash. Filers differ in which they use,
# and summing several risks double-counting the same securities — undercounting
# liquidity is the safe direction for a runway estimate, overcounting is not.
SHORT_TERM_CONCEPTS = (
    "MarketableSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "OtherShortTermInvestments",
)


@dataclass(frozen=True)
class Runway:
    cash: Optional[float] = None
    burn_per_quarter: Optional[float] = None
    quarters: Optional[float] = None
    runway_end: Optional[str] = None
    funded_through: Optional[bool] = None
    cash_generative: bool = False
    burn_basis: Optional[str] = None


def _cik(ticker: str) -> Optional[int]:
    from src.insider.edgar import cik_for
    return cik_for(ticker)


def _facts(cik: int) -> Dict[str, Any]:
    """All XBRL facts for a filer in ONE request, rather than one request per
    concept — we need several concepts and SEC asks clients to be polite."""
    from src.insider.edgar import _get
    return dict(_get(FACTS_URL.format(cik=cik), timeout=60).json())


def concept_points(facts: Dict[str, Any],
                   concept: str) -> List[Tuple[Optional[str], str, float]]:
    """(start, end, value) points for one us-gaap concept inside companyfacts."""
    node = ((facts.get("facts") or {}).get("us-gaap") or {}).get(concept)
    if not node:
        return []
    points: List[Tuple[Optional[str], str, float]] = []
    for unit, items in (node.get("units") or {}).items():
        if unit != "USD":
            continue
        points.extend(parse_concept({"units": {unit: items}}, "USD"))
    return sorted(points, key=lambda p: p[1])


def liquidity(facts: Dict[str, Any]) -> Tuple[Optional[float], Optional[str],
                                              Optional[str]]:
    """(total, as_of, basis) — cash plus short-term investments.

    Matched at the SAME reporting date: adding a cash balance from one quarter
    to securities from another would invent money that never coexisted.
    """
    cash_points = concept_points(facts, CASH_CONCEPT)
    if not cash_points:
        return None, None, None
    _, as_of, cash = cash_points[-1]

    for concept in SHORT_TERM_CONCEPTS:
        for start, end, value in reversed(concept_points(facts, concept)):
            if end == as_of:
                return cash + value, as_of, f"cash + {concept}"
    return cash, as_of, "cash only"


def parse_concept(payload: Dict[str, Any],
                  unit: str) -> List[Tuple[Optional[str], str, float]]:
    """(start, end, value) points for one unit, sorted by end date.

    ``start`` is None for INSTANT concepts such as a cash balance, and a date
    for DURATION concepts such as operating cash flow. Carrying it is not
    bookkeeping: XBRL reports cash flow as year-to-date periods of wildly
    different lengths, and a value is meaningless without the window it covers.

    Duplicate frames are common — the same period is republished under a later
    fiscal-year label — so callers must not treat frame count as observation
    count.
    """
    points: List[Tuple[Optional[str], str, float]] = []
    seen = set()
    for item in (payload.get("units") or {}).get(unit) or []:
        end, val = item.get("end"), item.get("val")
        if not end or val is None:
            continue
        start = item.get("start")
        key = (start, end, val)
        if key in seen:
            continue
        seen.add(key)
        try:
            points.append((str(start) if start else None, str(end), float(val)))
        except (TypeError, ValueError):
            continue
    return sorted(points, key=lambda p: p[1])


def _span_days(start: Optional[str], end: str) -> Optional[int]:
    if not start:
        return None
    try:
        return (_as_date(end) - _as_date(start)).days
    except (ValueError, IndexError):
        return None


def quarterly_burn(
        flow_points: List[Tuple[Optional[str], str, float]]
) -> Tuple[Optional[float], Optional[str]]:
    """Cash burn per quarter, normalised by each frame's ACTUAL duration.

    Returns (burn_per_quarter, basis) where basis names the period used, or
    (None, None) when no usable frame exists. A POSITIVE operating cash flow
    yields a negative burn, which the caller reads as cash-generative.

    Measured 2026-08-25, and the reason this function exists: SRPT's most
    recent frame was a 180-day year-to-date figure of -$5.6m. Dividing it by 4
    implied a $1.4m quarterly burn and 407 quarters of runway, ending in 2128.
    A full-year frame is preferred over a fresher short one because a single
    quarter of working-capital swing dominates a short window.
    """
    annual = [p for p in flow_points if (_span_days(p[0], p[1]) or 0) >= 300]
    chosen = annual[-1] if annual else None
    if chosen is None:
        dated = [p for p in flow_points if _span_days(p[0], p[1])]
        chosen = dated[-1] if dated else None
    if chosen is None:
        return None, None
    start, end, value = chosen
    days = _span_days(start, end)
    if not days:
        return None, None
    return (-value) * (DAYS_PER_QUARTER / days), f"{start}..{end} ({days}d)"


def _as_date(text: str) -> dt.date:
    parts = text.split("-")
    if len(parts) == 2:
        return dt.date(int(parts[0]), int(parts[1]), 1)
    return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))


def funded_through(runway_end: Optional[str],
                   event_date: Optional[str]) -> Optional[bool]:
    """True if the money outlasts the catalyst. None if either is unknown —
    an unknown is not a No."""
    if not runway_end or not event_date:
        return None
    try:
        return _as_date(runway_end) >= _as_date(event_date)
    except (ValueError, IndexError):
        return None


def runway_for(ticker: str, event_date: str) -> Runway:
    """Cash, burn, and whether the company reaches ``event_date`` funded."""
    cik = _cik(ticker)
    if cik is None:
        return Runway()
    try:
        facts = _facts(cik)
        cash, as_of, cash_basis = liquidity(facts)
        flow_points = concept_points(facts, FLOW_CONCEPT)
    except Exception:
        return Runway()
    if cash is None or as_of is None or not flow_points:
        return Runway()

    burn_per_quarter, burn_basis = quarterly_burn(flow_points)
    basis = f"{cash_basis} @ {as_of}; burn {burn_basis}"
    if burn_per_quarter is None:
        return Runway(cash=cash)
    if burn_per_quarter <= 0:
        return Runway(cash=cash, cash_generative=True, burn_basis=basis)

    quarters = cash / burn_per_quarter
    try:
        end = _as_date(as_of) + dt.timedelta(days=quarters * DAYS_PER_QUARTER)
        runway_end: Optional[str] = end.isoformat()
    except (ValueError, OverflowError):
        runway_end = None
    return Runway(cash=cash, burn_per_quarter=burn_per_quarter,
                  quarters=quarters, runway_end=runway_end,
                  funded_through=funded_through(runway_end, event_date),
                  burn_basis=basis)
