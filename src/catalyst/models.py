"""Shapes shared across the catalyst package.

Deliberately logic-free and sibling-free: store.py and board.py need these
types without needing the network modules that produce them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

PRIMARY_COMPLETION = "PRIMARY_COMPLETION"


@dataclass(frozen=True)
class Trial:
    """One ClinicalTrials.gov study, as the API gave it.

    ``event_date`` may be day-precision ("2026-10-31") or month-precision
    ("2027-03"); ``date_precision`` says which. ``date_type`` is CT.gov's own
    ESTIMATED/ACTUAL flag. Both are carried rather than normalised away — an
    estimated month is not a tradeable date and the board must be able to say
    so.
    """

    nct_id: str
    sponsor_name: str
    brief_title: str
    phase: str
    """The LOWEST registered phase — what the base-rate prior keys on.

    A trial registered PHASE1/PHASE2 is earlier-stage than a pure Phase 2, so
    taking the lowest never claims more maturity than the trial has. Use
    ``phase_label`` to show a reader what it actually is, and ``phases`` when
    ordering by how far along it is."""

    event_date: str
    date_precision: str
    date_type: str
    status: str
    enrollment: Optional[int] = None
    allocation: Optional[str] = None
    masking: Optional[str] = None
    primary_outcome: Optional[str] = None
    conditions: Tuple[str, ...] = ()
    phases: Tuple[str, ...] = ()
    """Every phase the trial is registered under, ascending.

    Measured 2026-08-25: 40 of 130 industry trials matching a PHASE2 sweep were
    registered across phases — 16 as PHASE1/PHASE2 and 4 as PHASE2/PHASE3.
    Keeping only the first printed a bare "PH1", which reads as a Phase 1 trial
    leaking past a Ph2/Ph3 filter rather than the Ph1/2 trial it is."""

    @property
    def phase_label(self) -> str:
        """Display form: "PH3", or "PH1/2" for a multi-phase registration."""
        ordered = self.phases or ((self.phase,) if self.phase else ())
        if not ordered:
            return ""
        numbers = [p.replace("PHASE", "") for p in ordered]
        return "PH" + "/".join(numbers)

    @property
    def top_phase(self) -> str:
        """The furthest-along registered phase — used for ordering only."""
        ordered = self.phases or ((self.phase,) if self.phase else ())
        return max(ordered) if ordered else ""


@dataclass(frozen=True)
class CatalystEvent:
    """A trial that resolved to a ticker and survived the cap band."""

    trial: Trial
    ticker: str
    mcap: Optional[float] = None
    event_type: str = PRIMARY_COMPLETION

    @property
    def event_id(self) -> str:
        return f"{self.trial.nct_id}:{self.event_type}"

    @property
    def event_date(self) -> str:
        return self.trial.event_date

    @property
    def phase(self) -> str:
        return self.trial.phase


@dataclass
class Coverage:
    """What the run saw and what it threw away.

    Printed on every board. A tool with 27% coverage that says so is honest;
    the same tool silently showing 27% is a lie of omission.
    """

    swept: int = 0
    resolved: int = 0
    dropped_unresolved: int = 0
    dropped_out_of_band: int = 0
    deep_failures: int = 0
    shown: Optional[int] = None
    truncated: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        pct = (100.0 * self.resolved / self.swept) if self.swept else 0.0
        return (f"swept {self.swept} trials, resolved {self.resolved} "
                f"({pct:.1f}%), dropped {self.dropped_unresolved} unresolved / "
                f"{self.dropped_out_of_band} out-of-band, "
                f"{self.deep_failures} deep lookups failed")

    def truncation_note(self) -> Optional[str]:
        """Names the deep-tier cap withheld, or None.

        A board that quietly shows the first N of a longer list is the same
        failure shape as unreported coverage: the reader cannot tell the
        difference between "these are all of them" and "these are the ones we
        got round to". So this is stated, never implied.
        """
        if self.truncated <= 0:
            return None
        total = (self.shown or 0) + self.truncated
        return (f"showing {self.shown or 0} of {total} names — {self.truncated} "
                f"not fetched (raise with --limit)")
