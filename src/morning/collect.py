"""Gather the morning-briefing sidecar. Every panel is individually fail-safe:
one dead fetch degrades to a placeholder panel, never a dead page."""
import threading
import time
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _EASTERN = None

SCHEMA_VERSION = 1
PANEL_IDS = ("health", "market", "vol", "macro_events", "signals",
             "portfolio", "gate", "notes")

# Static caveats that must appear on every page: the honesty footer.
_STANDING_NOTES = [
    "All signal panels are display-only; nothing here feeds scoring or execution.",
    "Real money is OFF until the forward-cohort gate (n>=50) fires.",
    "Quotes are 15+ min delayed free data; use for planning, not live execution.",
]


def _now_eastern(now=None):
    if now is not None:
        return now
    if _EASTERN is not None:
        return datetime.now(_EASTERN)
    return datetime.now()


def session_phase(now=None) -> str:
    """'pre-market' | 'open' | 'closed' by US/Eastern wall clock."""
    ts = _now_eastern(now)
    if ts.isoweekday() > 5:
        return "closed"
    hhmm = ts.hour * 100 + ts.minute
    if hhmm < 400:
        return "closed"
    if hhmm < 930:
        return "pre-market"
    return "open" if hhmm < 1600 else "closed"


def _safe(panel_id, fn, panels, failures, default=None):
    try:
        panels[panel_id] = fn()
    except Exception as exc:
        panels[panel_id] = default
        failures.append(f"{panel_id}: {type(exc).__name__}: {exc}")


def _desk_notes(panels, failures):
    notes = list(_STANDING_NOTES)
    for f in failures:
        notes.append(f"Panel unavailable this morning — {f}")
    return notes


def _default_fetchers():
    # Each entry is (panel_id, zero-arg callable); grown panel by panel.
    return []


def build(now=None, slow=True, budget_s=20.0, _fetchers=None) -> dict:
    ts = _now_eastern(now)
    date_s = ts.strftime("%Y-%m-%d")
    meta = {
        "schema": SCHEMA_VERSION,
        "date": date_s,
        "generated_at": ts.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "session": session_phase(ts),
        "sidecar": f"{date_s}.json",
        "title": f"Morning Briefing — {date_s}",
    }
    fetchers = _fetchers if _fetchers is not None else _default_fetchers()
    panels = {pid: None for pid in PANEL_IDS}
    failures = []

    if slow and fetchers:
        threads = []
        for pid, fn in fetchers:
            t = threading.Thread(target=_safe, args=(pid, fn, panels, failures),
                                 daemon=True)
            t.start()
            threads.append(t)
        deadline = time.monotonic() + budget_s
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        for (pid, _), t in zip(fetchers, threads):
            if t.is_alive() and panels.get(pid) is None:
                failures.append(f"{pid}: TimeoutError: budget {budget_s:.0f}s exceeded")
    else:
        for pid, fn in fetchers:
            _safe(pid, fn, panels, failures)

    panels["notes"] = _desk_notes(panels, failures)
    return {"meta": meta, "panels": panels, "failures": failures}
