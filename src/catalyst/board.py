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

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import src.formatting as fmt
import src.ui as ui
from src.catalyst import bands, baserates, relative
from src.catalyst.design import Amendments
from src.catalyst.implied import ImpliedMove
from src.catalyst.models import CatalystEvent, Coverage
from src.catalyst.pdufa import PdufaEvent
from src.catalyst.runway import Runway

_PHASE_ORDER = {"PHASE3": 3, "PHASE2": 2}

#: How many of the soonest names get a full detail block. Bound board-wide
#: rather than to the near band: NEXT_30's population swings with the
#: calendar (17 of 40 names on 2026-08-26), so binding detail to it would
#: make output length unpredictable run to run.
DETAIL_TOP_DEFAULT = 8

#: Compact-row column widths. Fixed so bands align with one another.
_W_TICKER, _W_MCAP, _W_DATE, _W_PHASE, _W_RUNWAY, _W_IMPLIED = 7, 8, 13, 6, 11, 6

LEGEND_ROWS: Tuple[Tuple[str, str], ...] = (
    ("runway", "cash ÷ quarterly burn, measured against the event date"),
    ("implied", "straddle-implied move for the expiry spanning the event"),
    ("amend", "registry protocol revisions since the trial started"),
    ("prior", "historical phase→approval base rate for the therapeutic "
              "area — what happened to other drugs, not a forecast for "
              "this one"),
    ("~2026-09", "month-precision date; banded at mid-month"),
)


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


def _design_body(row: BoardRow) -> str:
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
    return ", ".join(bits) if bits else fmt.style("not reported", "muted")


def _amend_body(row: BoardRow, sup: relative.Superlatives) -> str:
    a = row.amendments
    if not a.available:
        return fmt.style("amendment history unavailable", "muted")
    parts = [f"{a.versions} versions"]
    parts.extend(a.flags)
    body = " · ".join(parts)
    note = relative.note_for(row.event.ticker, "amend", sup)
    return body + (fmt.style(f"   ({note})", "muted") if note else "")


def _runway_body(row: BoardRow, sup: relative.Superlatives) -> str:
    r = row.runway
    if r.cash is None:
        return fmt.style("unknown (no XBRL cash concept reported)", "muted")
    cash = f"${r.cash / 1e6:,.0f}m"
    if r.cash_generative:
        return f"cash {cash} · " + fmt.style("cash-generative → no burn limit",
                                             "good")
    if r.quarters is None:
        return f"cash {cash}, " + fmt.style("burn unknown", "muted")
    burn = f"${(r.burn_per_quarter or 0) / 1e6:,.0f}m/q"
    if r.funded_through is True:
        verdict = fmt.style("FUNDED THROUGH", "good")
    elif r.funded_through is False:
        verdict = fmt.style("RAISE BEFORE", "bad")
    else:
        verdict = fmt.style("unknown", "muted")
    body = f"cash {cash} · {burn} → {r.quarters:.1f} q → {verdict}"
    note = relative.note_for(row.event.ticker, "runway", sup)
    return body + (fmt.style(f"   ({note})", "muted") if note else "")


def _implied_body(row: BoardRow, sup: relative.Superlatives) -> Optional[str]:
    if row.implied.move_pct is None:
        return None
    body = (f"{row.implied.expiry} straddle → "
            + fmt.style(f"±{row.implied.move_pct * 100:.0f}%", "accent"))
    note = relative.note_for(row.event.ticker, "implied", sup)
    return body + (fmt.style(f"   ({note})", "muted") if note else "")


def _runway_cell(runway: Runway) -> Tuple[str, str]:
    """(text, style) for a compact row's runway cell.

    Green/red is reserved for verdict sign — funded-through IS a verdict, so
    it earns colour; the quarter count alone does not.
    """
    if runway.cash is None:
        return "runway n/a", "muted"
    if runway.cash_generative:
        return "cash-gen", "good"
    if runway.quarters is None:
        return "burn n/a", "muted"
    if runway.funded_through is True:
        return f"{runway.quarters:.1f}q FUND", "good"
    if runway.funded_through is False:
        return f"{runway.quarters:.1f}q RAISE", "bad"
    return f"{runway.quarters:.1f}q", "value"


def _implied_cell(implied: ImpliedMove) -> Tuple[str, str]:
    """(text, style). Never sized by magnitude — a wide move has no valence."""
    if implied.move_pct is None:
        return "—", "muted"
    return f"±{implied.move_pct * 100:.0f}%", "accent"


def _date_cell(trial) -> Tuple[str, str]:
    """Precision reads as brightness: a firm day is brighter than a month."""
    text = format_event_date(trial.event_date, trial.date_precision,
                             trial.date_type)
    return text, "muted" if trial.date_precision == "month" else "value"


def _compact_row(row: BoardRow, width: int) -> List[str]:
    """One scannable line plus a dim asset subline."""
    trial = row.event.trial
    date_text, date_style = _date_cell(trial)
    runway_text, runway_style = _runway_cell(row.runway)
    implied_text, implied_style = _implied_cell(row.implied)
    extra = f"+{row.other_events}" if row.other_events else ""
    line = "  " + " ".join([
        fmt.style(ui.pad(row.event.ticker, _W_TICKER), "heading"),
        fmt.style(ui.pad(_mcap(row.event.mcap), _W_MCAP, "right"), "label"),
        fmt.style(ui.pad(date_text, _W_DATE), date_style),
        ui.pad(trial.phase_label, _W_PHASE),
        fmt.style(ui.pad(runway_text, _W_RUNWAY), runway_style),
        fmt.style(ui.pad(implied_text, _W_IMPLIED, "right"), implied_style),
        fmt.style(extra, "muted"),
    ]).rstrip()
    asset = ui.clip(trial.brief_title, width - 6)
    return [line, "    " + fmt.style(asset, "muted")]


def _full_block(row: BoardRow, sup: relative.Superlatives,
                width: int) -> List[str]:
    """The detail block, on the shared 11-char label gutter."""
    trial = row.event.trial
    date_text, date_style = _date_cell(trial)
    header = "  " + " ".join([
        fmt.style(ui.pad(row.event.ticker, _W_TICKER), "heading"),
        fmt.style(ui.pad(_mcap(row.event.mcap), _W_MCAP, "right"), "label"),
        fmt.style(ui.pad(date_text, _W_DATE + 2), date_style),
        fmt.style(f"{trial.phase_label} PRIMARY COMPLETION", "emph"),
    ])
    if row.other_events:
        plural = "" if row.other_events == 1 else "s"
        header += fmt.style(f"   +{row.other_events} more event{plural}", "muted")

    lines = [header,
             ui.kv_line("asset", ui.clip(trial.brief_title, width - 16)),
             ui.kv_line("design", _design_body(row)),
             ui.kv_line("amend", _amend_body(row, sup)),
             ui.kv_line("runway", _runway_body(row, sup))]
    implied_body = _implied_body(row, sup)
    if implied_body:
        lines.append(ui.kv_line("implied", implied_body))
    prior = baserates.describe(trial.phase, baserates.area_for(trial.conditions))
    if prior:
        # The caveat that used to ride on every one of these lines is now a
        # single footnote in the legend.
        lines.append(ui.kv_line("prior",
                                fmt.style(prior.split("(")[0].strip(), "value")))
    return lines


def _pdufa_section(rows: Sequence[PdufaRow], width: int) -> List[str]:
    """Regulatory decisions, rendered apart from trial readouts.

    A PDUFA date is FIRM and day-precision, so it prints bare — no "(est)",
    no "~". That visual difference from the trial rows is the point: those are
    estimates that slip, these are decision dates.
    """
    # No superlatives here: a PdufaRow carries no amendment history, and a
    # regulatory section is almost always below relative.MIN_N anyway, so
    # there is no spread to describe.
    sup = relative.Superlatives()
    lines: List[str] = ["", ui.rule(width, title=bands.BAND_TITLES[bands.FIRM])]
    for row in sorted(rows, key=lambda r: r.event.event_date):
        e = row.event
        lines.append("  " + " ".join([
            fmt.style(ui.pad(e.ticker, _W_TICKER), "heading"),
            fmt.style(ui.pad(e.event_date, _W_DATE + 2), "emph"),
            fmt.style("FDA DECISION DATE", "emph"),
            fmt.style(f"   announced {e.filed}", "muted"),
        ]))
        lines.append(ui.kv_line("runway", _runway_body(row, sup)))
        implied_body = _implied_body(row, sup)
        if implied_body:
            lines.append(ui.kv_line("implied", implied_body))
        lines.append(ui.kv_line("source", fmt.style(e.doc_url, "muted")))
        lines.append("")
    lines.append("  " + fmt.style("Firm decision dates, not estimates. NOT "
                                  "ranked, and an approval is not a share "
                                  "price.", "muted"))
    return lines


def _header_context(coverage: Coverage, window_label: str) -> List[str]:
    """Window, per-band coverage, and the truncation hint — at the TOP.

    This used to be the last four lines of a 312-line board, which meant a
    reader learned that 57 names were withheld only after scrolling past
    everything that was not.
    """
    out: List[str] = []
    if window_label:
        out.append(fmt.style(window_label, "label"))
    parts = [f"{bands.SHORT_TITLES[b.band]} {b.shown}/{b.found}"
             for b in coverage.bands if b.found]
    if parts:
        out.append(fmt.style("  ·  ".join(parts), "label"))
    note = coverage.truncation_note()
    if note:
        out.append(fmt.style(note, "warn"))
    return out


def _footer(coverage: Coverage, width: int, legend: bool) -> List[str]:
    """Legend and the standing caveats — each stated ONCE."""
    lines: List[str] = [""]
    if legend:
        body = [ui.kv_line(label, fmt.style(text, "muted"), indent=0)
                for label, text in LEGEND_ROWS]
        lines.append(ui.card("LEGEND", body, width, boxed=True))
    lines.append(ui.rule(width))
    lines.append("  " + fmt.style(coverage.summary(), "muted"))
    if legend:
        lines.extend([
            "  " + fmt.style("Sorted by date within band. NOT ranked — no key "
                             "on this board has been shown to order it.", "muted"),
            "  " + fmt.style("Primary completion is not topline; expect a 1–3 "
                             "month lag.", "muted"),
            "  " + fmt.style("Amendments, funded-through and phase were tested "
                             "2026-08-26 (n~2,100): NO EVIDENCE any of them "
                             "predicts returns.", "muted"),
            "  " + fmt.style("Superlatives describe this board's spread. They "
                             "do not rank and they do not recommend.", "muted"),
        ])
    return lines


def render(rows: Sequence[BoardRow], coverage: Coverage,
           width: int = 100,
           pdufa: Sequence[PdufaRow] = (),
           today: Optional[str] = None,
           detail_top: int = DETAIL_TOP_DEFAULT,
           legend: bool = True,
           window_label: str = "") -> str:
    """The board: banded by time, never scored.

    Full detail goes to the soonest ``detail_top`` rows BOARD-WIDE rather
    than to whichever band is populated. NEXT_30's size swings with the
    calendar — on 2026-08-26 it held 17 of 40 names — so binding detail to
    it would make output length unpredictable run to run.
    """
    as_of = today or dt.date.today().isoformat()
    ordered = sorted(rows, key=lambda r: sort_key(r.event))
    detailed = {id(r) for r in ordered[:max(0, detail_top)]}
    sup = relative.compute(ordered)

    lines: List[str] = [ui.banner("UPCOMING CLINICAL CATALYSTS",
                                  _header_context(coverage, window_label),
                                  width)]

    if pdufa:
        lines.extend(_pdufa_section(pdufa, width))

    grouped: Dict[str, List[BoardRow]] = {b: [] for b in bands.TRIAL_BANDS}
    for row in ordered:
        grouped[bands.band_for(row.event.event_date, as_of)].append(row)

    for band in bands.TRIAL_BANDS:
        members = grouped[band]
        if not members:          # an empty band prints NO header
            continue
        lines.append("")
        lines.append(ui.rule(width, title=bands.BAND_TITLES[band]))
        for row in members:
            if id(row) in detailed:
                lines.extend(_full_block(row, sup, width))
            else:
                lines.extend(_compact_row(row, width))
            lines.append("")

    if not ordered:
        lines.append("  " + fmt.style(
            "no trial readouts in the window after filtering" if pdufa
            else "no catalysts in the window after filtering", "muted"))

    lines.extend(_footer(coverage, width, legend))
    return "\n".join(lines)
