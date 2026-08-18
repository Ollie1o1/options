"""Whether this scan's board ordering can be trusted, and saying so if not.

`rank_by_verdict` is failure-safe: if quotes are missing or anything raises, a
board still renders rather than a scan dying. It used to buy that safety by
falling back to `sort_values("quality_score")`, which the ranking guard
allowlisted as "a board rendered in a discredited order beats a board that does
not render."

The hole in that reasoning is that `quality_score` is not merely uninformative.
Its TOP quintile is the worst cell in the ledger — 31.6% win rate and -19.9%
return on capital, against +5.2% for the [0.55, 0.65) bucket — and every caller
truncates the result with `.head(N)`. So the fallback surfaced the worst
candidates first, silently, on exactly the runs where the data was already bad.

The fallback now sorts by nothing: scan order carries no claim, which is the
honest state when the ordering key cannot be computed. This module carries the
fact that it happened so the board can say so, because an ordering that changed
meaning without telling anyone is the defect, not the ordering itself.

Scan-scoped, like `iv_crosscheck`: `reset()` at the top of a scan, and
`mark_degraded` announces only on the first failure so six display call sites
falling back in one scan produce one banner rather than six.
"""
from __future__ import annotations

import threading
from typing import List, Optional

_lock = threading.Lock()
_reason: Optional[str] = None
_announced = False


def reset() -> None:
    """Clear the health state. Called at the top of every scan — otherwise a
    degraded scan would keep warning through every later scan in the same
    interactive session."""
    global _reason, _announced
    with _lock:
        _reason = None
        _announced = False


def mark_degraded(reason: str) -> bool:
    """Record that ordering fell back. Returns True only the FIRST time in a
    scan, so the caller can announce once rather than once per board."""
    global _reason, _announced
    with _lock:
        if _reason is None:
            _reason = str(reason or "unknown")
        first, _announced = (not _announced), True
        return first


def is_degraded() -> bool:
    with _lock:
        return _reason is not None


def reason() -> Optional[str]:
    with _lock:
        return _reason


def render() -> List[str]:
    """Banner lines, or empty when ordering is healthy."""
    with _lock:
        why = _reason
    if why is None:
        return []
    return [
        "! RANKING UNAVAILABLE — verdict computation failed, so these rows "
        "are in scan order and are NOT ranked.",
        f"  cause: {why}",
        "  Do not read position as quality. Judge each row on WORTH, Cost% "
        "and Risk, or re-run when quotes recover.",
    ]
