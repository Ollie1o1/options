"""AI synthesis of a MacroContext into a market-weather narrative.

Reuses ai_scorer's OpenRouter free chain via safe_chat_complete (fallback
models + circuit breaker built in). Any failure -> deterministic template.
Never raises.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.macro_pulse.context import MacroContext

logger = logging.getLogger(__name__)


def _build_prompt(ctx: MacroContext) -> tuple[str, str]:
    system = (
        "You are a macro desk strategist. You are given a QUANTIFIED summary of "
        "current market-moving news (a trust- and recency-weighted pulse, "
        "per-theme scores with historical percentiles, and top headlines). "
        "Write a concise situational-awareness read. Be specific and sober; do "
        "NOT give buy/sell advice or price targets. Return ONLY valid JSON: "
        '{"headline": "2-3 sentences on the dominant market story", '
        '"themes": [{"theme": "name", "read": "one line incl. affected sectors", '
        '"sectors": ["ETF", ...]}], '
        '"what_would_flip": "one line: the event/data that would change this read"}'
    )
    lines = [
        f"Pulse: {ctx.pulse:+.2f} ({ctx.lean})"
        + (f", {ctx.pulse_pctile:.0f}th pct of last {ctx.n_history} readings"
           if ctx.pulse_pctile is not None else ", history building"),
        f"Confidence: {ctx.confidence}% ({ctx.n_items} items, {ctx.n_sources} sources)",
    ]
    if ctx.event_active:
        lines.append(f"Event window: {ctx.event_name} on {ctx.event_date}")
    lines.append("Themes:")
    for t in ctx.themes:
        tag = ", ".join(f"{lbl}({etf})" for lbl, etf in t.sectors)
        pct = f"{t.pctile:.0f}th pct" if t.pctile is not None else "no history"
        lines.append(f"  - {t.theme}: score {t.score:+.2f} ({pct}); "
                     f"sectors {tag}; top: {t.top_headline}")
    return system, "\n".join(lines)


def _apply_ai_json(ctx: MacroContext, data: dict) -> bool:
    headline = str(data.get("headline", "")).strip()
    if not headline:
        return False
    ctx.headline = headline
    ctx.what_would_flip = str(data.get("what_would_flip", "")).strip()
    reads = {str(t.get("theme", "")).lower(): str(t.get("read", "")).strip()
             for t in data.get("themes", []) if isinstance(t, dict)}
    for t in ctx.themes:
        t.read = reads.get(t.theme.lower(), t.read)
    return True


def _deterministic_narrate(ctx: MacroContext) -> MacroContext:
    if ctx.themes:
        lead = ctx.themes[0]
        sect = ", ".join(lbl for lbl, _ in lead.sectors) or "broad market"
        ctx.headline = (
            f"News flow leans {ctx.lean.lower()} (pulse {ctx.pulse:+.2f}). "
            f"Dominant theme is {lead.theme.replace('_', ' ')}, most relevant to "
            f"{sect}.")
    else:
        ctx.headline = (f"No strong themes in current news flow "
                        f"(pulse {ctx.pulse:+.2f}, {ctx.lean.lower()}).")
    for t in ctx.themes:
        tags = ", ".join(f"{lbl} ({etf})" for lbl, etf in t.sectors) or "broad"
        direction = "supportive" if t.score >= 0 else "pressuring"
        hot = (f", {t.pctile:.0f}th-pct hot" if t.pctile is not None
               and t.pctile >= 80 else "")
        t.read = f"{direction} for {tags}{hot}."
    ctx.what_would_flip = ("A high-trust headline reversing the leading theme, "
                           "or an upcoming macro print.")
    ctx.narrative_source = "deterministic"
    return ctx


def narrate(ctx: MacroContext, *, scorer=None) -> MacroContext:
    try:
        if scorer is None:
            from src.ai_scorer import AIScorer
            scorer = AIScorer()
        system, user = _build_prompt(ctx)
        raw: Optional[str] = scorer.safe_chat_complete(
            system=system, user=user, max_tokens=600)
        if raw:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                if _apply_ai_json(ctx, data):
                    ctx.narrative_source = "ai"
                    return ctx
    except Exception as exc:  # any AI/parse failure -> fallback
        logger.debug("macro_pulse synth AI failed: %s", exc)
    return _deterministic_narrate(ctx)
