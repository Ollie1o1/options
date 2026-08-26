"""ClinicalTrials.gov API v2 client — the documented, free, keyless endpoint.

Query shape verified live 2026-08-25: a 2026-09-01..2027-03-01 window returns
599 industry-sponsored PHASE3 studies and 1,104 PHASE2.

The date carries TWO qualifiers we keep rather than normalise away:
``type`` is CT.gov's ESTIMATED/ACTUAL flag, and the string itself may be
"2027-03" (month) or "2026-10-31" (day). An estimated month is not a tradeable
date. Callers that flatten this distinction are lying to the reader.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from src.catalyst.models import Trial

BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ("NCTId,BriefTitle,Phase,PrimaryCompletionDateStruct,LeadSponsorName,"
          "OverallStatus,EnrollmentCount,DesignAllocation,DesignMasking,"
          "PrimaryOutcomeMeasure,Condition")
PAGE_SIZE = 200
TIMEOUT = 60


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "options-screener/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _precision(date_text: str) -> str:
    return "day" if len(date_text) == 10 else "month"


def _first_outcome(outcomes: Dict[str, Any]) -> Optional[str]:
    items = outcomes.get("primaryOutcomes") or []
    if not items:
        return None
    measure = (items[0] or {}).get("measure")
    return measure or None


def parse_studies(payload: Dict[str, Any]) -> List[Trial]:
    """Trials from one page. A study with no primary completion date is
    SKIPPED, never defaulted — an absent date is not a date."""
    out: List[Trial] = []
    for study in payload.get("studies") or []:
        p = study.get("protocolSection") or {}
        status = p.get("statusModule") or {}
        struct = status.get("primaryCompletionDateStruct") or {}
        date_text = struct.get("date")
        if not date_text:
            continue
        ident = p.get("identificationModule") or {}
        design = p.get("designModule") or {}
        info = design.get("designInfo") or {}
        # Sorted so "lowest" and "highest" are well defined regardless of the
        # order CT.gov happens to serialise them in.
        phases = sorted(design.get("phases") or [])
        enrollment = (design.get("enrollmentInfo") or {}).get("count")
        out.append(Trial(
            nct_id=ident.get("nctId") or "",
            sponsor_name=((p.get("sponsorCollaboratorsModule") or {})
                          .get("leadSponsor") or {}).get("name") or "",
            brief_title=ident.get("briefTitle") or "",
            phase=phases[0] if phases else "",
            phases=tuple(phases),
            event_date=date_text,
            date_precision=_precision(date_text),
            date_type=struct.get("type") or "ESTIMATED",
            status=status.get("overallStatus") or "",
            enrollment=int(enrollment) if enrollment is not None else None,
            allocation=info.get("allocation"),
            masking=(info.get("maskingInfo") or {}).get("masking"),
            primary_outcome=_first_outcome(p.get("outcomesModule") or {}),
            conditions=tuple((p.get("conditionsModule") or {}).get("conditions") or ()),
        ))
    return out


def _url(start: str, end: str, phase: str, token: Optional[str]) -> str:
    advanced = (f"AREA[LeadSponsorClass]INDUSTRY AND AREA[Phase]{phase} AND "
                f"AREA[PrimaryCompletionDate]RANGE[{start},{end}]")
    params = {"filter.advanced": advanced, "fields": FIELDS,
              "pageSize": str(PAGE_SIZE), "countTotal": "true"}
    if token:
        params["pageToken"] = token
    return f"{BASE}?{urllib.parse.urlencode(params)}"


def sweep(start: str, end: str,
          phases: Sequence[str] = ("PHASE2", "PHASE3"),
          max_pages: int = 50) -> List[Trial]:
    """All industry-sponsored trials in ``phases`` with primary completion in
    [start, end]. Deduped by NCT id — a trial registered as PHASE2/PHASE3
    comes back under both queries. Returns [] on any failure; the caller
    reports that as zero coverage rather than an empty board."""
    seen: Dict[str, Trial] = {}
    try:
        for phase in phases:
            token: Optional[str] = None
            for _ in range(max_pages):
                payload = _get_json(_url(start, end, phase, token))
                for trial in parse_studies(payload):
                    seen.setdefault(trial.nct_id, trial)
                token = payload.get("nextPageToken")
                if not token:
                    break
    except Exception:
        return []
    return list(seen.values())
