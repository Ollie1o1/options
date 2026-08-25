"""Trial amendment history from ClinicalTrials.gov.

WARNING — UNDOCUMENTED ENDPOINT. /api/int/ is the internal API behind the
website's "Study Record Versions" page, not the supported v2 contract. It may
change or vanish without notice. Every failure path here therefore returns
Amendments(available=False) and the board prints "amendment history
unavailable". This is the only undocumented dependency in the package, and it
costs exactly one column when it breaks.

What the payload gives us, verified 2026-08-25 on NCT06510816: 11 versions,
each with the moduleLabels that changed, plus a top-level outcomesUpdateCount
of 3. A primary endpoint edited repeatedly mid-trial is a recorded fact worth
seeing. It is NOT a prediction, and nothing here should ever be scored.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://clinicaltrials.gov/api/int/studies"
TIMEOUT = 30
OUTCOME_EDIT_FLAG_THRESHOLD = 2


@dataclass(frozen=True)
class Amendments:
    versions: int = 0
    outcomes_updated: int = 0
    status_now: Optional[str] = None
    flags: Tuple[str, ...] = ()
    available: bool = False


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "options-screener/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def fetch_history(nct_id: str) -> Optional[Dict[str, Any]]:
    """Version list for one study, or None. Never raises."""
    try:
        return _get_json(f"{BASE}/{nct_id}/history")
    except Exception:
        return None


def parse_history(payload: Dict[str, Any]) -> Amendments:
    """Amendments from the version-list payload.

    An empty payload yields available=False rather than a confident zero — 'we
    could not look' and 'nothing changed' are different answers."""
    changes: List[Dict[str, Any]] = list(payload.get("changes") or [])
    if not changes:
        return Amendments()
    outcomes = int(payload.get("outcomesUpdateCount") or 0)
    flags: List[str] = []
    if outcomes >= OUTCOME_EDIT_FLAG_THRESHOLD:
        flags.append(f"outcome measures edited {outcomes}x")
    return Amendments(
        versions=len(changes),
        outcomes_updated=outcomes,
        status_now=changes[-1].get("status"),
        flags=tuple(flags),
        available=True,
    )


def amendments_for(nct_id: str) -> Amendments:
    payload = fetch_history(nct_id)
    return parse_history(payload) if payload else Amendments()
