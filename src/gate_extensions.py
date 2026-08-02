"""The clock behind the gate's bounded EXTEND.

The v2 redesign (docs/GATE_REDESIGN_SPEC.md, signed 2026-07-31) added the
terminal condition v1 lacked: a gate may EXTEND at most twice, two weeks each,
and must then resolve READY or STOP. That bound was implemented as a single
integer in config.json — `gate.extensions_used` — with a note telling the
operator to "increment when an EXTEND is acted on".

Nobody ever did, and nobody could have done it correctly, because the integer
records only how many extensions were granted and never when one STARTED. Two
consequences followed, both observed on 2026-08-01:

* Checkpoints run daily, so an EXTEND verdict reprinted "extension 1 of 2"
  every single day, indefinitely. The counter never advanced on its own.
* With no window start date, no code could tell whether a two-week extension
  had elapsed — so the terminal condition could not fire, and the unbounded
  EXTEND that v2 was specifically designed to remove was still there, now
  wearing a bound that nothing enforced.

This module supplies the missing half: a dated window per gate, advanced by the
calendar rather than by an operator remembering to edit a config field. An
extension is CONSUMED when its window expires, not when it opens, so the
allowance means what the spec says — two extensions of two weeks is 28 days of
extra evidence-gathering, after which the gate must resolve.

State lives in `status/gate_extensions.json`, deliberately NOT in config.json.
config.json is policy: hand-edited, reviewed, and meaningful to a reader. This
is state: machine-advanced on every checkpoint. Mixing the two is what produced
a counter that read 0 through nine weeks of gating.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

# Two weeks per extension, matching the signed spec's "EXTEND capped at 2x2
# weeks". The cap on how many is GATE_V2_MAX_EXTENSIONS in phase1_checkpoint —
# this module owns the length of a window, not how many are allowed.
EXTENSION_DAYS = 14

DEFAULT_STATE_PATH = os.path.join("status", "gate_extensions.json")

# The gates that carry an extension allowance. Keys of the state file.
LONG_CALL = "long_call"
SHORT_PREMIUM = "short_premium"


@dataclass(frozen=True)
class ExtensionState:
    """A gate's extension standing, resolved to a particular date.

    ``extensions_used`` is what the decision rule reads. ``window_opened`` is
    the date the current extension began, or None when no extension is running.
    ``expired_now`` reports how many windows this resolve step closed, which is
    what makes the advance visible in the checkpoint rather than silent.
    """

    extensions_used: int
    window_opened: Optional[str]
    expired_now: int = 0

    @property
    def is_open(self) -> bool:
        return self.window_opened is not None

    def days_remaining(self, today: str, days: int = EXTENSION_DAYS) -> Optional[int]:
        """Days left in the open window, or None when none is running."""
        if self.window_opened is None:
            return None
        return int((_parse(self.window_opened) + timedelta(days=days)
                    - _parse(today)).days)

    def as_dict(self) -> Dict[str, Any]:
        return {"extensions_used": self.extensions_used,
                "window_opened": self.window_opened}


def _parse(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def resolve(entry: Optional[Dict[str, Any]], today: Optional[str] = None,
            max_extensions: int = 2, days: int = EXTENSION_DAYS) -> ExtensionState:
    """Advance a gate's extension standing to ``today``, before it is decided.

    Any window whose two weeks have elapsed is closed and counted. The loop
    rolls forward rather than closing a single window, because checkpoints are
    not guaranteed to run: this repo's schedulers have been dead since
    2026-06-15, and a gate that under-counts extensions merely because nobody
    ran the job would hand back the unbounded EXTEND by a different route.

    Counting stops at ``max_extensions`` — beyond the allowance the number
    carries no further meaning, and the decision rule resolves STOP regardless.
    """
    today = today or _today()
    entry = entry or {}
    used = int(entry.get("extensions_used") or 0)
    opened = entry.get("window_opened") or None

    expired = 0
    while opened is not None and used < max_extensions:
        closes = _parse(opened) + timedelta(days=days)
        if _parse(today) < closes:
            break
        used += 1
        expired += 1
        # The next window starts where the last one ended, not today: a
        # checkpoint that runs late must not gift back the days it missed.
        opened = closes.isoformat() if used < max_extensions else None

    if used >= max_extensions:
        opened = None

    return ExtensionState(extensions_used=used, window_opened=opened,
                          expired_now=expired)


def apply_verdict(state: ExtensionState, decision: str,
                  today: Optional[str] = None) -> ExtensionState:
    """Open, hold or close the window according to the verdict just reached.

    An EXTEND with no window running opens one dated today. Any other verdict
    closes whatever was open: READY and STOP are terminal, and a gate that has
    resolved is not extending. Holding an open window across a resolved verdict
    is what would let a later flip back to EXTEND silently inherit a stale
    clock.
    """
    today = today or _today()
    if str(decision).upper() == "EXTEND":
        if state.window_opened is None:
            return ExtensionState(state.extensions_used, today, state.expired_now)
        return state
    return ExtensionState(state.extensions_used, None, state.expired_now)


def load(path: str = DEFAULT_STATE_PATH,
         seed: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Read the state file, seeding from config's counters on first run.

    ``seed`` carries the legacy `gate.extensions_used` /
    `gate.short_premium_extensions_used` values so the migration loses nothing:
    whatever the operator had recorded by hand is the starting count.
    """
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if not isinstance(data, dict):
        data = {}

    for key in (LONG_CALL, SHORT_PREMIUM):
        entry = data.get(key)
        if not isinstance(entry, dict):
            data[key] = {"extensions_used": int((seed or {}).get(key, 0) or 0),
                         "window_opened": None}
    return data


def save(data: Dict[str, Any], path: str = DEFAULT_STATE_PATH) -> None:
    """Persist the state file, creating its directory if need be."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(data)
    payload["_note"] = (
        "Machine-advanced gate extension state — do NOT hand-edit to change a "
        "verdict. `window_opened` is the date the current 2-week EXTEND began; "
        "an extension is counted when its window EXPIRES, not when it opens. "
        "Policy (how many extensions are allowed, how long each runs) lives in "
        "config.json and src/gate_extensions.py, not here.")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def describe(state: ExtensionState, today: str, max_extensions: int = 2,
             days: int = EXTENSION_DAYS) -> str:
    """One line for the checkpoint, so the clock is visible in the report."""
    if state.extensions_used >= max_extensions and not state.is_open:
        return (f"Extension allowance exhausted ({state.extensions_used} of "
                f"{max_extensions}) — the gate must resolve.")
    if not state.is_open:
        return (f"No extension running ({state.extensions_used} of "
                f"{max_extensions} used).")
    left = state.days_remaining(today, days)
    closes = (_parse(state.window_opened) + timedelta(days=days)).isoformat()
    return (f"Extension {state.extensions_used + 1} of {max_extensions} running "
            f"since {state.window_opened}, closes {closes} "
            f"({left} day{'' if left == 1 else 's'} left).")
