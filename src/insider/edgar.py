"""Polite SEC EDGAR I/O. Free, no key; SEC asks for a contact User-Agent and
≤10 req/s — we send the contact and throttle well under the limit.

Endpoints (all free JSON/XML):
  - https://www.sec.gov/files/company_tickers.json        ticker → CIK
  - https://data.sec.gov/submissions/CIK{cik:010d}.json   recent filings
  - https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}   Form 4 XML

Every function is failure-safe (None/[] on any error) — insider data is an
overlay, never a reason the screener fails to start.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

def _user_agent() -> str:
    """SEC asks every client to identify itself with a contact address. That
    contact is per-operator, never baked into the repo: set SEC_EDGAR_CONTACT
    (e.g. "you@example.com") to send it. Unset, we still identify the software
    but SEC may throttle or 403 us — which is fine, every caller below is
    failure-safe and insider data is an overlay, never a startup dependency."""
    contact = (os.environ.get("SEC_EDGAR_CONTACT") or "").strip()
    return f"options-screener/1.0 ({contact})" if contact else "options-screener/1.0"


HEADERS = {"User-Agent": _user_agent()}
THROTTLE_S = 0.3
TICKER_CACHE = os.path.join("data", "edgar_tickers.json")
TICKER_CACHE_DAYS = 30

_last_request = 0.0


def _get(url: str, timeout: int = 20):
    """Throttled GET with the SEC-required headers."""
    global _last_request
    import requests
    wait = THROTTLE_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


#: In-process memo for the ticker map, as (mtime, mapping).
#:
#: The disk cache already has a 30-day TTL; this only removes the REPEATED
#: PARSE. `cik_for` reopens and json.loads a ~10k-entry file on every lookup,
#: which cost 3.5s across 2,234 lookups in the catalyst backtest — 24% of the
#: run — all of it re-reading bytes that had not changed.
#:
#: Keyed by the file's mtime so a refreshed cache is picked up rather than
#: shadowed: the memo is a speed layer, never a second and longer expiry.
_TICKER_MEMO: Optional[Tuple[float, Dict[str, int]]] = None


def reset_ticker_map() -> None:
    """Drop the in-process ticker map. Tests, and any deliberate refresh."""
    global _TICKER_MEMO
    _TICKER_MEMO = None


def _ticker_map() -> Dict[str, int]:
    """ticker → CIK, disk-cached for 30 days and memoized in process."""
    global _TICKER_MEMO
    try:
        mtime = os.path.getmtime(TICKER_CACHE)
        age_days = (time.time() - mtime) / 86400.0
        if age_days <= TICKER_CACHE_DAYS:
            if _TICKER_MEMO is not None and _TICKER_MEMO[0] == mtime:
                return _TICKER_MEMO[1]
            with open(TICKER_CACHE) as f:
                mapping = json.load(f)
            # An empty map is NEVER memoized: `_ticker_map` returns {} when it
            # can neither read nor fetch, and freezing that would poison every
            # later lookup in the process.
            if mapping:
                _TICKER_MEMO = (mtime, mapping)
            return mapping
    except (OSError, ValueError):
        pass
    try:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        mapping = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
        os.makedirs(os.path.dirname(TICKER_CACHE) or ".", exist_ok=True)
        with open(TICKER_CACHE, "w") as f:
            json.dump(mapping, f)
        return mapping
    except Exception:
        return {}


def cik_for(ticker: str) -> Optional[int]:
    return _ticker_map().get((ticker or "").upper())


def recent_form4(cik: int, max_filings: int = 25,
                 since_days: int = 120) -> List[Dict[str, Any]]:
    """Recent Form 4 filings for a CIK: [{accession, document, filed}]."""
    try:
        data = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        recent = (data.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        out = []
        cutoff = time.strftime("%Y-%m-%d",
                               time.localtime(time.time() - since_days * 86400))
        for i, form in enumerate(forms):
            if form != "4":
                continue
            filed = (recent.get("filingDate") or [""] * len(forms))[i]
            if filed and filed < cutoff:
                continue
            out.append({
                "accession": (recent.get("accessionNumber") or [""] * len(forms))[i],
                "document": (recent.get("primaryDocument") or [""] * len(forms))[i],
                "filed": filed,
            })
            if len(out) >= max_filings:
                break
        return out
    except Exception:
        return []


def fetch_form4_xml(cik: int, accession: str, document: str) -> Optional[str]:
    """Fetch one Form 4 primary document as raw XML.

    ``primaryDocument`` often carries an ``xslF345X0N/`` prefix, which serves
    the XSL-rendered HTML; the raw XML is the same filename without the
    prefix."""
    try:
        acc = accession.replace("-", "")
        doc = document.split("/")[-1]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
        return _get(url).text
    except Exception:
        return None
