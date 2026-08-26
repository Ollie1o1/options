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


def _versions(nct_id: str, conn: sqlite3.Connection) -> Optional[List[Dict[str, Any]]]:
    cached = pit_cache.get_versions(conn, nct_id)
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
    versions = _versions(nct_id, conn)
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
