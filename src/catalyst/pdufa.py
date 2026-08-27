"""PDUFA dates from 8-K full text — a firmer catalyst than trial completion.

WHY THIS EXISTS. A primary completion date is soft: roughly half are
month-precision estimates, they slip constantly, and topline follows 1-3 months
later. A PDUFA date is a REGULATORY DECISION DATE — firm, day-precision, with a
binary public outcome on the day. It is the better event by every measure that
matters.

Source is EDGAR full-text search, free and keyless. Two properties make it
unusually clean:

  * The ticker is embedded in the hit itself — "Harmony Biosciences Holdings,
    Inc.  (HRMY)  (CIK 0001802665)" — so this SIDESTEPS the sponsor-name
    resolver entirely, and with it the 27% coverage ceiling that caps the trial
    calendar. A company announcing its own PDUFA date is by definition a filer.
  * Volume is small and high-signal: 36 8-Ks mentioned a PDUFA date across
    three months (measured 2026-08-26).

A YEAR-ONLY MENTION IS NOT A DATE. Filings say both "Target PDUFA Date April 1,
2027" and "Target PDUFA Date in 2028" in the same paragraph. The second is
extracted as nothing; turning it into 2028-01-01 would fabricate a precision
the company did not state.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SEARCH = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
TIMEOUT = 30

# How far after the word PDUFA a date may sit and still be its date. Filings
# put it within a few words ("PDUFA goal date of December 15, 2026"); a date a
# paragraph away belongs to something else.
_WINDOW = 80

_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), start=1)}
_DATE_RE = re.compile(
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})", re.I)
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)")
_CIK_RE = re.compile(r"CIK\s+(\d+)")
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    filed: str
    form: str
    doc_url: str


@dataclass(frozen=True)
class PdufaEvent:
    ticker: str
    cik: int
    event_date: str
    filed: str
    doc_url: str


def _user_agent() -> str:
    import os
    contact = (os.environ.get("SEC_EDGAR_CONTACT") or "").strip()
    return f"options-screener/1.0 ({contact})" if contact else "options-screener/1.0"


THROTTLE_S = 0.3
_last_request = 0.0


def _get(url: str) -> str:
    """Throttled fetch. SEC asks clients to stay under 10 req/s and enforces it.

    Measured while building this: an unthrottled loop over 63 filings returned
    ZERO usable documents — every fetch was refused, and because each failure
    is individually failure-safe the whole thing reported "0 PDUFA dates"
    rather than an error. A silent zero from rate limiting looks exactly like
    a genuine absence of data, which is the worst way for this to fail.
    """
    global _last_request
    import time
    wait = THROTTLE_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "ignore")


def parse_hit(hit: Dict[str, Any]) -> Optional[Filing]:
    """A search hit into a Filing, or None if it carries no ticker.

    No ticker means no tradeable instrument, so the row is worthless here —
    dropped rather than guessed at."""
    source = hit.get("_source") or {}
    names = source.get("display_names") or []
    if not names:
        return None
    name = names[0]
    ticker_m, cik_m = _TICKER_RE.search(name), _CIK_RE.search(name)
    if not ticker_m or not cik_m:
        return None
    ident = str(hit.get("_id") or "")
    accession, _, document = ident.partition(":")
    cik = int(cik_m.group(1))
    return Filing(
        ticker=ticker_m.group(1),
        cik=cik,
        filed=str(source.get("file_date") or ""),
        form=str(source.get("form") or ""),
        doc_url=f"{ARCHIVE}/{cik}/{accession.replace('-', '')}/{document}",
    )


def extract_dates(text: str) -> List[str]:
    """ISO dates stated within `_WINDOW` characters after a PDUFA mention.

    Order-preserving and deduplicated — filings routinely restate the same
    date, and counting it twice would imply two events."""
    out: List[str] = []
    for m in re.finditer(r"PDUFA", text, re.I):
        window = text[m.end():m.end() + _WINDOW]
        found = _DATE_RE.search(window)
        if not found:
            continue
        month = _MONTHS[found.group(1).lower()]
        try:
            iso = f"{int(found.group(3)):04d}-{month:02d}-{int(found.group(2)):02d}"
        except (TypeError, ValueError):
            continue
        if iso not in out:
            out.append(iso)
    return out


def _search(start: str, end: str, query: str = '"PDUFA date"',
            forms: str = "8-K") -> List[Dict[str, Any]]:
    params = {"q": query, "forms": forms, "dateRange": "custom",
              "startdt": start, "enddt": end}
    payload = json.loads(_get(f"{SEARCH}?{urllib.parse.urlencode(params)}"))
    return list(((payload.get("hits") or {}).get("hits")) or [])


def _document_text(url: str) -> Optional[str]:
    """Filing text with tags stripped, or None if it will not load."""
    try:
        raw = _get(url)
    except Exception:
        return None
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", raw))


def pdufa_events(start: str, end: str) -> List[PdufaEvent]:
    """PDUFA dates announced in 8-Ks filed between start and end.

    Never raises — this is one input among several and must not take a board
    down. A filing that will not load is skipped, not guessed at.
    """
    try:
        hits = _search(start, end)
    except Exception:
        return []
    events: List[PdufaEvent] = []
    seen = set()
    for hit in hits:
        filing = parse_hit(hit)
        if filing is None:
            continue
        text = _document_text(filing.doc_url)
        if not text:
            continue
        for iso in extract_dates(text):
            key = (filing.ticker, iso)
            if key in seen:
                continue
            seen.add(key)
            events.append(PdufaEvent(ticker=filing.ticker, cik=filing.cik,
                                     event_date=iso, filed=filing.filed,
                                     doc_url=filing.doc_url))
    return sorted(events, key=lambda e: e.event_date)
