"""Board rendering.

SORTED BY DATE. NOT SCORED. The pre-registered rank race of 2026-08-24 found
that no candidate key ordered a board better than chance, every CI containing
zero. Adding a composite "catalyst score" here would repeat a mistake this repo
has already paid for. Materiality is columns the reader judges.

Two labelling rules that are not cosmetic:
  * the date is PRIMARY COMPLETION, never "readout" — topline typically follows
    1-3 months later, and the source does not contain a readout date;
  * unknown renders as "unknown", never as 0. A zero runway and an unmeasured
    runway are opposite claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import src.formatting as fmt
from src.catalyst import baserates
from src.catalyst.design import Amendments
from src.catalyst.implied import ImpliedMove
from src.catalyst.models import CatalystEvent, Coverage
from src.catalyst.pdufa import PdufaEvent
from src.catalyst.runway import Runway

_PHASE_ORDER = {"PHASE3": 3, "PHASE2": 2}


@dataclass
class BoardRow:
    event: CatalystEvent
    other_events: int = 0
    amendments: Amendments = field(default_factory=Amendments)
    runway: Runway = field(default_factory=Runway)
    implied: ImpliedMove = field(default_factory=ImpliedMove)


@dataclass
class PdufaRow:
    """A regulatory decision date, which is NOT a trial.

    It carries no phase, enrollment or masking because an 8-K does not state
    them. Those fields are absent rather than defaulted — inventing them to
    reuse BoardRow would put fabricated design data on screen.
    """

    event: PdufaEvent
    runway: Runway = field(default_factory=Runway)
    implied: ImpliedMove = field(default_factory=ImpliedMove)


def format_event_date(date: str, precision: str, date_type: str) -> str:
    """Render a date so its precision survives.

    "2026-10-31 (est)" and "~2027-03 (est, month)" are different objects and a
    reader must be able to tell them apart at a glance."""
    if precision == "month":
        return f"~{date} (est, month)" if date_type == "ESTIMATED" else f"~{date} (month)"
    return f"{date} (est)" if date_type == "ESTIMATED" else date


def sort_key(event: CatalystEvent) -> Tuple[str, int, int]:
    """Soonest first; ties to the later phase, then larger enrollment. Both
    tie-breaks are facts about the trial, not judgements about it."""
    return (event.event_date,
            -_PHASE_ORDER.get(event.trial.top_phase or event.phase, 0),
            -(event.trial.enrollment or 0))


def collapse(events: Sequence[CatalystEvent]) -> List[Tuple[CatalystEvent, int]]:
    """One entry per ticker: its soonest event, plus a count of the others."""
    by_ticker: Dict[str, List[CatalystEvent]] = {}
    for event in events:
        by_ticker.setdefault(event.ticker, []).append(event)
    out: List[Tuple[CatalystEvent, int]] = []
    for group in by_ticker.values():
        ordered = sorted(group, key=sort_key)
        out.append((ordered[0], len(ordered) - 1))
    return sorted(out, key=lambda pair: sort_key(pair[0]))


def _mcap(value: Optional[float]) -> str:
    if value is None:
        return "  mcap n/a"
    return f"${value / 1e6:,.0f}M" if value < 1e9 else f"${value / 1e9:,.2f}B"


def _runway_line(runway: Runway) -> str:
    if runway.cash is None:
        return "  runway     unknown (no XBRL cash concept reported)"
    cash = f"${runway.cash / 1e6:,.0f}m"
    if runway.cash_generative:
        return f"  runway     cash {cash}, cash-generative -> no burn limit"
    if runway.quarters is None:
        return f"  runway     cash {cash}, burn unknown"
    burn = f"${(runway.burn_per_quarter or 0) / 1e6:,.0f}m/q"
    verdict = "unknown"
    if runway.funded_through is True:
        verdict = "FUNDED THROUGH"
    elif runway.funded_through is False:
        verdict = "RAISE BEFORE"
    return (f"  runway     cash {cash}, burn {burn} -> {runway.quarters:.1f} q "
            f"-> {verdict}")


def _amend_line(amendments: Amendments) -> str:
    if not amendments.available:
        return "  amend      amendment history unavailable"
    parts = [f"{amendments.versions} versions"]
    parts.extend(amendments.flags)
    return "  amend      " + "; ".join(parts)


def _implied_line(implied: ImpliedMove) -> Optional[str]:
    if implied.move_pct is None:
        return None
    return (f"  implied    {implied.expiry} straddle -> +/- "
            f"{implied.move_pct * 100:.0f}%")


def _design_line(row: BoardRow) -> str:
    t = row.event.trial
    bits = []
    if t.enrollment:
        bits.append(f"n={t.enrollment}")
    if t.allocation:
        bits.append(t.allocation.lower())
    if t.masking and t.masking != "NONE":
        bits.append(f"{t.masking.lower()}-masked")
    elif t.masking == "NONE":
        bits.append("open-label")
    return "  design     " + (", ".join(bits) if bits else "not reported")


def _pdufa_section(rows: Sequence[PdufaRow], width: int) -> List[str]:
    """Regulatory decisions, rendered apart from trial readouts.

    A PDUFA date is FIRM and day-precision, so it prints bare — no "(est)",
    no "~". That visual difference from the trial rows is the point: those are
    estimates that slip, these are decision dates.
    """
    lines: List[str] = [
        fmt.style("REGULATORY DECISIONS  (PDUFA)", "heading"),
        fmt.draw_separator(width),
    ]
    for row in sorted(rows, key=lambda r: r.event.event_date):
        e = row.event
        lines.append(fmt.style(
            f"{e.ticker:<6} {e.event_date}   FDA DECISION DATE"
            f"      announced {e.filed}", "emph"))
        lines.append(_runway_line(row.runway))
        implied_line = _implied_line(row.implied)
        if implied_line:
            lines.append(implied_line)
        lines.append(f"  source     {e.doc_url}")
        lines.append("")
    lines.append("  Firm decision dates, not estimates. NOT ranked, and an "
                 "approval is not a share price.")
    return lines


def render(rows: Sequence[BoardRow], coverage: Coverage,
           width: int = 100,
           pdufa: Sequence[PdufaRow] = ()) -> str:
    """The board. Date-sorted, never scored."""
    lines: List[str] = [
        fmt.style("UPCOMING CLINICAL CATALYSTS", "heading"),
        fmt.draw_separator(width),
    ]
    truncation = coverage.truncation_note()
    if not rows and pdufa:
        lines.append("  no trial readouts in the window after filtering")
        lines.append("")
        lines.extend(_pdufa_section(pdufa, width))
        lines.append(fmt.draw_separator(width))
        lines.append("  " + coverage.summary())
        return "\n".join(lines)
    if not rows:
        lines.append("  no catalysts in the window after filtering")
        lines.append(fmt.draw_separator(width))
        lines.append("  " + coverage.summary())
        if truncation:
            lines.append("  " + truncation)
        return "\n".join(lines)

    for row in rows:
        t = row.event.trial
        header = (f"{row.event.ticker:<6} {_mcap(row.event.mcap):>10}   "
                  f"{format_event_date(t.event_date, t.date_precision, t.date_type):<22} "
                  f"{t.phase_label} PRIMARY COMPLETION")
        if row.other_events == 1:
            header += "   +1 more event"
        elif row.other_events > 1:
            header += f"   +{row.other_events} more events"
        lines.append(fmt.style(header, "emph"))
        lines.append(f"  asset      {fmt.truncate(t.brief_title, width - 13)}")
        lines.append(_design_line(row))
        lines.append(_amend_line(row.amendments))
        lines.append(_runway_line(row.runway))
        implied_line = _implied_line(row.implied)
        if implied_line:
            lines.append(implied_line)
        prior = baserates.describe(t.phase, baserates.area_for(t.conditions))
        if prior:
            lines.append(f"  prior      {prior}")
        lines.append("")

    if pdufa:
        lines.extend(_pdufa_section(pdufa, width))

    lines.append(fmt.draw_separator(width))
    lines.append("  " + coverage.summary())
    if truncation:
        lines.append("  " + fmt.style(truncation, "warn"))
    lines.append("  Sorted by date. NOT ranked - no key on this board has been "
                 "shown to order it.")
    lines.append("  Primary completion is not topline; expect a 1-3 month lag.")
    return "\n".join(lines)
