"""Render a MacroContext into a quant-desk situational-awareness panel.

Pure string building (uses src.formatting where available, plain text
otherwise). Never raises.
"""
from __future__ import annotations

from src.macro_pulse.context import MacroContext

try:
    import src.formatting as fmt  # noqa: F401
    _HAS_FMT = True
except Exception:  # pragma: no cover
    _HAS_FMT = False


def _hr(width: int = 92) -> str:
    return "-" * width


def _pulse_context(ctx: MacroContext) -> str:
    if ctx.pulse_pctile is None:
        return f"history building ({ctx.n_history}/30)"
    z = f", z {ctx.pulse_z:+.1f}" if ctx.pulse_z is not None else ""
    return f"{ctx.pulse_pctile:.0f}th pct{z}"


def render(ctx: MacroContext) -> str:
    lines: list[str] = []
    lines.append("  MACRO PULSE  —  AI market-weather (situational awareness)")
    lines.append(_hr())

    event = (f"  ⚠ {ctx.event_name} {ctx.event_date}"
             if ctx.event_active else "")
    lines.append(
        f"  Pulse {ctx.pulse:+.2f} ({_pulse_context(ctx)})  ·  "
        f"{ctx.lean}  ·  conf {ctx.confidence}%  "
        f"({ctx.n_items} items / {ctx.n_sources} src){event}")
    lines.append("")

    if ctx.headline:
        lines.append("  Read:")
        lines.append(f"    {ctx.headline}")
        lines.append("")

    if ctx.themes:
        lines.append("  Themes (by news volume):")
        for t in ctx.themes:
            pct = (f"{t.pctile:.0f}th pct" if t.pctile is not None
                   else "building")
            tags = ", ".join(f"{lbl}({etf})" for lbl, etf in t.sectors) or "—"
            lines.append(f"    {t.theme:<14s} {t.score:+.2f} ({pct})  "
                         f"→ {tags}")
            if t.read:
                lines.append(f"        {t.read}")
            if t.top_headline:
                lines.append(f"        top: {t.top_headline[:84]}")
        lines.append("")

    if ctx.what_would_flip:
        lines.append(f"  What would flip this: {ctx.what_would_flip}")
        lines.append("")

    if ctx.next_events:
        ev = ", ".join(f"{e['name']} {e['date']}" for e in ctx.next_events)
        lines.append(f"  Risk events ahead: {ev}")

    src_tag = ctx.narrative_source or "deterministic"
    lines.append(_hr())
    lines.append(
        "  Honest read: this is a RISK / TIMING / REGIME overlay for "
        "situational awareness and sizing —")
    lines.append(
        f"  not directional alpha. Scoring weights untouched. "
        f"[narrative: {src_tag}]")
    return "\n".join(lines)
