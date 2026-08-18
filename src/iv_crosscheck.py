"""Scan-wide tally of the IV cross-check, reported once instead of per ticker.

`cross_validate_iv` re-solves each contract's IV from its own mid price and
adopts the solved value where Yahoo's reported IV fails verification. Contracts
are flagged, never dropped, so this is a data-quality repair rather than a
filter — but it used to announce itself with one INFO line per ticker. On a
111-ticker scan that is 111 lines written to stderr while a tqdm bar renders on
stdout: two streams into one terminal, and the two shred each other.

Demoting the line to DEBUG would have been the easy fix and the wrong one. The
correction rate varies enormously by name — 4 of 113 contracts on one ticker
against 9 of 17 on another — and "which names does Yahoo price badly" is a
signal worth keeping. A fix that makes a varying number invisible has destroyed
information, not cleaned it up.

So the counts accumulate here and surface once, after scoring, which clears the
screen AND makes the rate comparable across names for the first time. The
per-contract provenance stays at DEBUG, and every corrected pick still carries
its own "IV corrected (yahoo X% -> solved Y%)" tag on the card.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# How many offenders the summary names. Enough to see a pattern, few enough
# that the summary stays one glance rather than a second report.
MAX_WORST = 4

# Below this rate a ticker is not worth naming — a 1% correction rate is the
# guard working normally, not a broken quote feed.
_WORST_MIN_RATE = 0.05

_lock = threading.Lock()
_counts: Dict[str, List[int]] = {}


@dataclass(frozen=True)
class Offender:
    symbol: str
    corrected: int
    total: int

    @property
    def rate(self) -> float:
        return (self.corrected / self.total) if self.total else 0.0


@dataclass(frozen=True)
class Summary:
    corrected: int
    total: int
    tickers: int                       # tickers with at least one correction
    worst: Tuple[Offender, ...] = field(default_factory=tuple)

    @property
    def pct(self) -> float:
        return (100.0 * self.corrected / self.total) if self.total else 0.0


def reset() -> None:
    """Clear the tally. Called at the start of every scan — the interactive
    loop runs many scans in one process, and without this the second scan
    reports the first one's contracts as well as its own."""
    with _lock:
        _counts.clear()


def record(symbol: str, corrected: int, total: int) -> None:
    """Add one ticker's cross-check result. Never raises: a reporting tally
    must not be able to take down a scan."""
    try:
        sym = str(symbol or "?")
        c, t = int(corrected), int(total)
    except (TypeError, ValueError):
        return
    with _lock:
        slot = _counts.setdefault(sym, [0, 0])
        slot[0] += c
        slot[1] += t


def summary() -> Optional[Summary]:
    """The tally, or None when there is nothing worth a line.

    None both when no ticker was recorded and when none needed correcting —
    a clean scan should not spend a line saying so.
    """
    with _lock:
        items = [(s, c, t) for s, (c, t) in _counts.items()]
    corrected = sum(c for _, c, _ in items)
    if corrected <= 0:
        return None
    total = sum(t for _, _, t in items)
    dirty = [Offender(s, c, t) for s, c, t in items if c > 0 and t > 0]
    # Ranked by RATE, not by count: 9 of 17 is a worse feed than 54 of 212,
    # and ranking by count buries the genuinely broken names behind the
    # merely large ones. Symbol breaks ties so the output is deterministic.
    worst = sorted(
        (o for o in dirty if o.rate >= _WORST_MIN_RATE),
        key=lambda o: (-o.rate, o.symbol),
    )[:MAX_WORST]
    return Summary(corrected=corrected, total=total,
                   tickers=len(dirty), worst=tuple(worst))


def render(s: Optional[Summary]) -> List[str]:
    """Plain lines for the scan output. No colour and no theme lookup — the
    caller owns presentation, and this stays importable by tests."""
    if s is None:
        return []
    lines = [
        f"IV cross-check: {s.corrected:,} of {s.total:,} contracts corrected "
        f"({s.pct:.1f}%) across {s.tickers} ticker"
        f"{'s' if s.tickers != 1 else ''} — Yahoo IV failed verification",
    ]
    if s.worst:
        worst = "  ".join(
            f"{o.symbol} {o.corrected}/{o.total} ({o.rate * 100:.0f}%)"
            for o in s.worst
        )
        lines.append(f"  worst: {worst}")
    lines.append("  (per-contract detail at DEBUG log level)")
    return lines
