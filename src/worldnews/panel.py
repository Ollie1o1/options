"""Rendering: full CLI panel + the one-line dashboard pulse."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _lean(pulse: float) -> str:
    if pulse >= 0.25:
        return "BULLISH"
    if pulse >= 0.08:
        return "BULLISH-LEAN"
    if pulse <= -0.25:
        return "BEARISH"
    if pulse <= -0.08:
        return "BEARISH-LEAN"
    return "NEUTRAL"


def next_events(limit: int = 3, today: Optional[str] = None) -> List[Dict[str, str]]:
    """Upcoming macro events from the verified calendar (macro_analyzer)."""
    from src.macro_analyzer import _DEFAULT_EVENTS
    today = today or datetime.now().strftime("%Y-%m-%d")
    ahead = sorted((e for e in _DEFAULT_EVENTS if e["date"] >= today),
                   key=lambda e: e["date"])
    return ahead[:limit]


def pulse_line(agg: Dict[str, Any], crowd: Optional[Dict[str, Any]] = None,
               today: Optional[str] = None) -> str:
    """One-liner for the regime dashboard."""
    parts = [f"World pulse: {agg['pulse']:+.2f} {_lean(agg['pulse'])}",
             f"bulls {agg['bull_pct']:.0%} / bears {agg['bear_pct']:.0%}",
             f"conf {agg['confidence']}% ({agg['n_items']} items, "
             f"{agg['n_sources']} src)"]
    if crowd and crowd.get("bull_ratio") is not None:
        parts.append(f"crowd {crowd['bull_ratio']:.0%} bull ({crowd['tagged']} tagged)")
    ev = next_events(1, today=today)
    if ev:
        parts.append(f"next: {ev[0]['name']} {ev[0]['date'][5:]}")
    return " | ".join(parts)


def render_full(agg: Dict[str, Any], crowd: Optional[Dict[str, Any]] = None,
                today: Optional[str] = None) -> str:
    lines = ["World-news market pulse — trust- and recency-weighted", ""]
    lines.append(f"  Pulse: {agg['pulse']:+.2f}  →  {_lean(agg['pulse'])}")
    lines.append(f"  Direction split: {agg['bull_pct']:.0%} bullish / "
                 f"{agg['bear_pct']:.0%} bearish (of directional headlines)")
    lines.append(f"  Confidence: {agg['confidence']}%  "
                 f"({agg['n_items']} headlines from {agg['n_sources']} publishers; "
                 f"more items, more sources, more agreement → higher)")
    if crowd and crowd.get("bull_ratio") is not None:
        lines.append(f"  Crowd (StockTwits SPY/QQQ, weight-LOW): "
                     f"{crowd['bull_ratio']:.0%} bullish of {crowd['tagged']} tagged")
    themes = sorted(agg["themes"].items(), key=lambda kv: -kv[1]["n"])
    if themes:
        lines.append("\n  Themes:")
        for name, th in themes:
            if name == "other":
                continue
            lines.append(f"    {name:<14s} {th['score']:+.2f}  ({th['n']} items)")
    if agg["top"]:
        lines.append("\n  Highest-impact headlines:")
        for t in agg["top"]:
            lines.append(f"    [{t['sentiment']:+.2f}] {t['title'][:90]}  "
                         f"({t['source']})")
    ev = next_events(3, today=today)
    if ev:
        lines.append("\n  Risk events ahead: " + ", ".join(
            f"{e['name']} {e['date']}" for e in ev))
    lines.append("\n  Honest read: public news at retail speed times RISK, not "
                 "direction. Use it to size and to avoid event windows — not "
                 "as a buy signal. Overlay only; scoring weights untouched.")
    return "\n".join(lines)
