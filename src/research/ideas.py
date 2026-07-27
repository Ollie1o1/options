"""Idea backlog loader for the research desk Ideas tab.

Local file read, no network — so unlike the other panels this one is loaded
synchronously in ``collect.build`` rather than on a fetcher thread. Fail-safe in
the same way everything else on the desk is: a missing or malformed ideas.json
degrades to an empty panel, never a dead page.
"""
import json
import os

DEFAULT_PATH = "ideas.json"

# Display order. Live work first, graveyard last — the dead entries exist so
# settled questions are not re-litigated, not to be read every time.
_STATUS_ORDER = {"validated": 0, "testing": 1, "proposed": 2,
                 "shipped": 3, "dead": 4}
_CONVICTION_ORDER = {"high": 0, "medium": 1, "low": 2}


def load(path: str = DEFAULT_PATH) -> dict:
    """Return {'updated', 'note', 'ideas': [...], 'counts': {...}} or {} when
    the backlog is absent/unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}

    ideas = [i for i in (raw.get("ideas") or []) if isinstance(i, dict)]
    ideas.sort(key=lambda i: (
        _STATUS_ORDER.get(str(i.get("status", "")).lower(), 9),
        _CONVICTION_ORDER.get(str(i.get("conviction", "")).lower(), 9),
        str(i.get("title", "")),
    ))

    counts = {}
    for i in ideas:
        key = str(i.get("status", "unknown")).lower()
        counts[key] = counts.get(key, 0) + 1

    return {"updated": raw.get("updated"), "note": raw.get("note"),
            "ideas": ideas, "counts": counts}
