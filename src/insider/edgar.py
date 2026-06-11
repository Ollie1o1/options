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
from typing import Any, Dict, List, Optional

HEADERS = {"User-Agent": "options-screener/1.0 (oliver.raczka.shue@gmail.com)"}
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


def _ticker_map() -> Dict[str, int]:
    """ticker → CIK, disk-cached for 30 days."""
    try:
        age_days = (time.time() - os.path.getmtime(TICKER_CACHE)) / 86400.0
        if age_days <= TICKER_CACHE_DAYS:
            with open(TICKER_CACHE) as f:
                return json.load(f)
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
