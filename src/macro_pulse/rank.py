"""End-of-scan AI macro ranking — opt-in, one batched call for all tickers.

Default scans never call this. When the user opts in, a SINGLE AI call ranks the
scanned tickers by macro tailwind/headwind with a one-line reason each. A
deterministic ranking (weighted sum of each ticker's relevant theme scores)
always backs it and is the fallback when the AI is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.macro_pulse.context import MacroContext
from src.macro_pulse.ticker import BROAD_THEMES, themes_for_ticker

try:
    import src.formatting as _fmt
    _C = _fmt.Colors
    _HAS_FMT = True
except Exception:  # pragma: no cover
    _HAS_FMT = False

logger = logging.getLogger(__name__)

# Sector-specific themes count full; broad market themes are diluted so a
# ticker's own sector drives its score more than the market backdrop.
_BROAD_WEIGHT = 0.5


@dataclass
class RankRow:
    symbol: str
    sector: Optional[str]
    score: float
    lean: str
    reason: str
    narrative_source: str = "deterministic"


def _lean(score: float) -> str:
    if score >= 0.08:
        return "TAILWIND"
    if score <= -0.08:
        return "HEADWIND"
    return "NEUTRAL"


def _deterministic_rank(symbols: list[str], ctx: MacroContext,
                        sectors: dict[str, Optional[str]]) -> list[RankRow]:
    theme_score = {t.theme: t.score for t in ctx.themes}
    rows: list[RankRow] = []
    for sym in symbols:
        sector = sectors.get(sym)
        relevant = themes_for_ticker(sym, sector)
        net = 0.0
        contribs: list[tuple[str, float]] = []
        for th in relevant:
            if th not in theme_score:
                continue
            w = _BROAD_WEIGHT if th in BROAD_THEMES else 1.0
            c = w * theme_score[th]
            net += c
            contribs.append((th, c))
        # Explain the NET lean: pick the biggest contributor that agrees with
        # the net sign (a negative net should cite what's dragging, not a small
        # offsetting positive). Fall back to largest-magnitude if none agree.
        if contribs:
            if net >= 0:
                agree = [c for c in contribs if c[1] >= 0]
                verb = "tailwind from"
            else:
                agree = [c for c in contribs if c[1] < 0]
                verb = "headwind from"
            pick = max(agree or contribs, key=lambda kv: abs(kv[1]))
            reason = f"{verb} {pick[0].replace('_', ' ')}"
        else:
            reason = "no active macro theme touches this name"
        rows.append(RankRow(symbol=sym, sector=sector, score=round(net, 3),
                            lean=_lean(net), reason=reason))
    rows.sort(key=lambda r: -r.score)
    return rows


def _build_prompt(symbols: list[str], ctx: MacroContext,
                  sectors: dict[str, Optional[str]]) -> tuple[str, str]:
    system = (
        "You are a macro desk strategist. Rank the given tickers by how the "
        "CURRENT macro/news backdrop helps or hurts each one over the next 1-4 "
        "weeks. Use only the supplied themes; no price targets, no buy/sell "
        "calls. Return ONLY valid JSON: "
        '{"ranking": [{"symbol": "X", "lean": "TAILWIND|HEADWIND|NEUTRAL", '
        '"reason": "one short clause"}, ...]} ordered best macro tailwind first.'
    )
    theme_lines = "\n".join(
        f"  - {t.theme}: {t.score:+.2f} ({t.read})" for t in ctx.themes)
    sym_lines = "\n".join(
        f"  - {s} (sector: {sectors.get(s) or 'unknown'})" for s in symbols)
    user = (f"Market pulse {ctx.pulse:+.2f} ({ctx.lean}).\n"
            f"Active themes:\n{theme_lines}\n\nTickers:\n{sym_lines}")
    return system, user


def _apply_ai(symbols: list[str], data: dict,
              det: list[RankRow]) -> Optional[list[RankRow]]:
    ranking = data.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return None
    det_by = {r.symbol: r for r in det}
    want = {s.upper() for s in symbols}
    out: list[RankRow] = []
    seen: set[str] = set()
    for item in ranking:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).upper()
        if sym not in want or sym in seen:
            continue
        base = det_by.get(sym)
        if base is None:
            continue
        seen.add(sym)
        reason = str(item.get("reason", "")).strip() or base.reason
        lean = str(item.get("lean", "")).strip().upper() or base.lean
        if lean not in ("TAILWIND", "HEADWIND", "NEUTRAL"):
            lean = base.lean
        out.append(RankRow(symbol=sym, sector=base.sector, score=base.score,
                          lean=lean, reason=reason, narrative_source="ai"))
    if len(seen) < len(want):  # AI dropped names — not trustworthy, fall back
        return None
    return out


def rank_tickers(symbols: list[str], ctx: MacroContext, *,
                 sectors: Optional[dict[str, Optional[str]]] = None,
                 scorer=None, use_ai: bool = False) -> list[RankRow]:
    sectors = sectors or {}
    det = _deterministic_rank(symbols, ctx, sectors)
    if scorer is None and not use_ai:
        return det  # no AI requested — pure deterministic, zero tokens
    try:
        if scorer is None:
            from src.ai_scorer import AIScorer
            scorer = AIScorer()
        system, user = _build_prompt(symbols, ctx, sectors)
        raw = scorer.safe_chat_complete(system=system, user=user, max_tokens=700)
        if raw:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                ai = _apply_ai(symbols, json.loads(m.group()), det)
                if ai:
                    return ai
    except Exception as exc:
        logger.debug("macro_pulse rank AI failed: %s", exc)
    return det


# ── Render ───────────────────────────────────────────────────────────────────
def _col_lean(lean: str) -> str:
    if not _HAS_FMT:
        return lean
    color = {"TAILWIND": _C.GREEN, "HEADWIND": _C.RED}.get(lean, _C.YELLOW)
    return _fmt.colorize(f"{lean:<8s}", color)


def render_ranking(rows: list[RankRow]) -> str:
    src = rows[0].narrative_source if rows else "deterministic"
    lines = ["  MACRO RANKING — tickers by current tailwind/headwind"
             f"  [{src}]", "  " + "-" * 90]
    for i, r in enumerate(rows, 1):
        sec = (r.sector or "—")[:18]
        lines.append(f"  {i:>2}. {r.symbol:<6s} {_col_lean(r.lean)} "
                     f"{r.score:+.2f}  {sec:<18s} {r.reason}")
    lines.append("  " + "-" * 90)
    lines.append("  Macro overlay only — situational awareness, not a buy/sell "
                 "signal. Scoring weights untouched.")
    return "\n".join(lines)
