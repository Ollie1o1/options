"""Scan-type-aware macro focus.

Per-ticker scans get a sector-focused read ("how the tape affects AAPL / tech");
discovery / cross-ticker scans get the general market panel. Pure + offline:
reuses the already-synthesized daily MacroContext, so there is no extra AI cost
per ticker. Rendering is colored via src.formatting when available.
"""
from __future__ import annotations

import copy
from typing import Optional

from src.macro_pulse.context import MacroContext

try:
    import src.formatting as _fmt
    _C = _fmt.Colors
    _HAS_FMT = True
except Exception:  # pragma: no cover
    _HAS_FMT = False

# Always-relevant market-wide themes (apply to every ticker).
BROAD_THEMES: list[str] = ["fed_rates", "inflation", "geopolitics", "jobs"]

# yfinance sector string -> sector-specific macro themes.
THEME_BY_SECTOR: dict[str, list[str]] = {
    "Technology":             ["earnings_tech"],
    "Communication Services": ["earnings_tech"],
    "Energy":                 ["energy", "geopolitics"],
    "Financial Services":     ["fed_rates"],
    "Industrials":            ["trade_tariffs"],
    "Basic Materials":        ["trade_tariffs", "energy"],
    "Consumer Cyclical":      ["inflation"],
    "Consumer Defensive":     ["inflation"],
    "Healthcare":             [],
    "Utilities":              ["fed_rates"],
    "Real Estate":            ["fed_rates"],
}

# Ticker-level overrides where the yfinance sector misses the real driver.
TICKER_OVERRIDES: dict[str, list[str]] = {
    "COIN": ["crypto"], "MSTR": ["crypto"], "MARA": ["crypto"],
    "RIOT": ["crypto"], "IBIT": ["crypto"],
}


def themes_for_sector(sector: Optional[str]) -> list[str]:
    """Sector-specific themes unioned with the always-on broad themes."""
    specific = THEME_BY_SECTOR.get(sector or "", None)
    if specific is None:
        return list(BROAD_THEMES)
    out = list(specific)
    for t in BROAD_THEMES:
        if t not in out:
            out.append(t)
    return out


def themes_for_ticker(symbol: Optional[str], sector: Optional[str]) -> list[str]:
    base = themes_for_sector(sector)
    for t in TICKER_OVERRIDES.get((symbol or "").upper(), []):
        if t not in base:
            base.insert(0, t)
    return base


def focus(ctx: MacroContext, sector: Optional[str],
          symbol: Optional[str] = None) -> MacroContext:
    """Return a copy whose themes are restricted to those relevant to the
    ticker's sector (sector-specific first, then broad), preserving order."""
    relevant = themes_for_ticker(symbol, sector)
    rank = {name: i for i, name in enumerate(relevant)}
    kept = [t for t in ctx.themes if t.theme in rank]
    kept.sort(key=lambda t: rank[t.theme])
    out = copy.copy(ctx)
    out.themes = kept
    return out


# ── Rendering ────────────────────────────────────────────────────────────────
def _col(text: str, color: str) -> str:
    if _HAS_FMT:
        return _fmt.colorize(text, color)
    return text


def _sentiment_color(score: float) -> str:
    if not _HAS_FMT:
        return ""
    if score >= 0.08:
        return _C.GREEN
    if score <= -0.08:
        return _C.RED
    return _C.YELLOW


def sentiment_glyph(score: float) -> str:
    if score >= 0.08:
        return "▲"
    if score <= -0.08:
        return "▼"
    return "▬"


def _hr(width: int = 92) -> str:
    return "-" * width


def render_ticker(ctx: MacroContext, symbol: Optional[str],
                  sector: Optional[str]) -> str:
    """Colored macro read. With a symbol -> sector-focused; without -> general
    market-wide read ('discovery' mode)."""
    general = not symbol
    view = ctx if general else focus(ctx, sector, symbol)

    lines: list[str] = []
    if general:
        title = "  MACRO — MARKET-WIDE READ (discovery scan)"
    else:
        sec = sector or "broad market"
        title = f"  MACRO — how the tape affects {symbol} ({sec})"
    lines.append(title)
    lines.append(_hr())

    pulse_col = _sentiment_color(ctx.pulse)
    lines.append(
        "  Market pulse " + _col(f"{ctx.pulse:+.2f} {ctx.lean}", pulse_col)
        + f"  ·  conf {ctx.confidence}%  ({ctx.n_items} items / {ctx.n_sources} src)")
    if ctx.headline:
        lines.append(f"  {ctx.headline}")
    lines.append("")

    if view.themes:
        label = "  Relevant themes:" if not general else "  Themes:"
        lines.append(label)
        for t in view.themes:
            g = sentiment_glyph(t.score)
            tag = ", ".join(etf for _, etf in t.sectors) or "—"
            head = f"{g} {t.theme:<14s} {t.score:+.2f}"
            lines.append("    " + _col(head, _sentiment_color(t.score))
                         + f"  → {tag}")
            if t.read:
                lines.append(f"        {t.read}")
            if t.top_headline:
                lines.append("        " + _col(f"news: {t.top_headline[:80]}",
                                               _C.DIM if _HAS_FMT else ""))
        lines.append("")

    if ctx.what_would_flip:
        lines.append(f"  What would flip this: {ctx.what_would_flip}")
        lines.append("")

    src_tag = ctx.narrative_source or "deterministic"
    lines.append(_hr())
    lines.append(
        "  Honest read: RISK / TIMING / REGIME overlay for awareness & sizing — "
        "not directional alpha.")
    lines.append(f"  Scoring weights untouched. [narrative: {src_tag}]")
    return "\n".join(lines)
