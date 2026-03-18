#!/usr/bin/env python3
"""
Macro event gatekeeper.

Checks whether the current time falls within a configurable window around
known high-impact macro events (FOMC, CPI, NFP) and returns a quality-score
penalty when active.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default event calendar (2026)
# ---------------------------------------------------------------------------

_DEFAULT_EVENTS: List[Dict] = [
    # FOMC 2026 (8 meetings)
    {"name": "FOMC", "date": "2026-01-29"},
    {"name": "FOMC", "date": "2026-03-19"},
    {"name": "FOMC", "date": "2026-05-07"},
    {"name": "FOMC", "date": "2026-06-17"},
    {"name": "FOMC", "date": "2026-07-29"},
    {"name": "FOMC", "date": "2026-09-16"},
    {"name": "FOMC", "date": "2026-11-04"},
    {"name": "FOMC", "date": "2026-12-16"},
    # CPI releases (monthly)
    {"name": "CPI", "date": "2026-01-15"},
    {"name": "CPI", "date": "2026-02-12"},
    {"name": "CPI", "date": "2026-03-12"},
    {"name": "CPI", "date": "2026-04-10"},
    {"name": "CPI", "date": "2026-05-13"},
    {"name": "CPI", "date": "2026-06-11"},
    {"name": "CPI", "date": "2026-07-14"},
    {"name": "CPI", "date": "2026-08-13"},
    {"name": "CPI", "date": "2026-09-11"},
    {"name": "CPI", "date": "2026-10-13"},
    {"name": "CPI", "date": "2026-11-12"},
    {"name": "CPI", "date": "2026-12-10"},
    # NFP / Jobs Reports (first Friday of each month)
    {"name": "NFP", "date": "2026-01-09"},
    {"name": "NFP", "date": "2026-02-06"},
    {"name": "NFP", "date": "2026-03-06"},
    {"name": "NFP", "date": "2026-04-03"},
    {"name": "NFP", "date": "2026-05-01"},
    {"name": "NFP", "date": "2026-06-05"},
    {"name": "NFP", "date": "2026-07-10"},
    {"name": "NFP", "date": "2026-08-07"},
    {"name": "NFP", "date": "2026-09-04"},
    {"name": "NFP", "date": "2026-10-02"},
    {"name": "NFP", "date": "2026-11-06"},
    {"name": "NFP", "date": "2026-12-04"},
]


_staleness_warned = False


def _load_events_from_config(config: dict) -> list:
    """Return macro events from config override or fall back to _DEFAULT_EVENTS."""
    return config.get("macro_events") or _DEFAULT_EVENTS


def check_macro_event_window(
    config: Optional[Dict] = None,
    window_hours: Optional[int] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check whether now falls within `window_hours` of a macro event.

    Returns (is_active, event_name, event_date_str).
    Uses config keys:
        macro_event_window_hours  (default 48)
        macro_events              (optional list override)
    """
    cfg = config or {}
    if window_hours is None:
        window_hours = int(cfg.get("macro_event_window_hours", 48))

    events = _load_events_from_config(cfg)

    now = datetime.now(tz=timezone.utc)
    window = timedelta(hours=window_hours)

    for event in events:
        try:
            event_dt = datetime.strptime(event["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # Only trigger within the forward-looking window (up to window_hours ahead)
            # and a short look-back (6h) so an event that just started still fires.
            delta = event_dt - now
            if timedelta(hours=-6) <= delta <= window:
                return True, event["name"], event["date"]
        except Exception:
            continue

    global _staleness_warned
    if not _staleness_warned and events:
        dates = []
        for e in events:
            try:
                dates.append(datetime.strptime(e["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc))
            except Exception:
                pass
        if dates:
            most_recent = max(dates)
            if (now - most_recent) > timedelta(days=60):
                logging.getLogger(__name__).warning(
                    "Macro event calendar may be stale — most recent event is %s. "
                    "Update _DEFAULT_EVENTS in macro_analyzer.py for accurate gating.",
                    most_recent.strftime("%Y-%m-%d"),
                )
                _staleness_warned = True

    return False, None, None


def get_macro_penalty(
    config: Optional[Dict] = None,
) -> Tuple[float, bool, Optional[str]]:
    """
    Return (penalty_fraction, is_active, description).

    penalty_fraction is config["macro_penalty"] (default -0.15) when a macro
    event is active, otherwise 0.0.
    """
    cfg = config or {}
    penalty = float(cfg.get("macro_penalty", -0.15))

    is_active, event_name, event_date = check_macro_event_window(cfg)
    if is_active:
        desc = f"{event_name} ({event_date})"
        return penalty, True, desc
    return 0.0, False, None
