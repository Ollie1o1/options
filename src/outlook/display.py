"""Startup display for the sector/asset outlook: a compact box + a plain-English
leaders/laggards narrative, served cache-first so it never slows startup.

The narrative speaks the way the user asked — "semis have led and momentum/trend
favor more," "treasuries are lagging" — grounded honestly in the factors
(momentum/trend = "has been doing well"; the thesis is continuation). Bearish is
framed as relative weakness, never a confident short (see OUTLOOK_FINDINGS.md).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.outlook.backtest import SECTOR_LABEL

CACHE_PATH = "reports/outlook_cache.json"

try:
    from src import formatting as fmt
    _HAS_FMT = True
except Exception:
    fmt = None
    _HAS_FMT = False


def _name(ticker: str) -> str:
    return SECTOR_LABEL.get(ticker, ticker)


def _phrase(drivers: str) -> str:
    """Turn a driver string into plain language."""
    d = drivers.lower()
    mom = "momentum +" in d
    trend = "trend +" in d
    rel = "rel-strength vs mkt +" in d
    if mom and trend:
        return "strong momentum and a solid uptrend"
    if trend and rel:
        return "an uptrend and leading the market"
    if mom:
        return "strong momentum"
    if trend:
        return "a healthy uptrend"
    if rel:
        return "leadership versus the market"
    return "improving relative strength"


def narrative(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[str]:
    """Build 1-3 plain-English lines on the leaders and laggards."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)
    leaders = [r for r in ordered if r["direction"] == "BULLISH"][:3]
    if not leaders:  # no strong favorites (e.g. choppy market) — take the top 2
        leaders = ordered[:2]
    laggards = ordered[-2:]

    lines: List[str] = []
    if leaders:
        if len(leaders) == 1:
            r = leaders[0]
            lines.append(f"Leading: {_name(r['ticker'])} ({r['ticker']}) — "
                         f"{_phrase(r['drivers'])}; has outperformed and the trend "
                         f"favors more.")
        else:
            head = leaders[0]
            others = ", ".join(_name(r["ticker"]) for r in leaders[1:])
            lines.append(f"Leading: {_name(head['ticker'])} ({head['ticker']}) "
                         f"with {others} — {_phrase(head['drivers'])}; these have "
                         f"been doing well and momentum/trend favor continuation.")
    lag_names = ", ".join(f"{_name(r['ticker'])} ({r['ticker']})" for r in laggards)
    lines.append(f"Lagging: {lag_names} — weak momentum/trend; relatively soft, "
                 f"lean underweight (not a high-confidence short).")
    return lines


# ── cache ──────────────────────────────────────────────────────────────────────
def save_outlook_cache(rows: List[Dict[str, Any]], lines: List[str],
                       path: str = CACHE_PATH) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "rows": rows, "narrative": lines}, fh)
    except Exception:
        pass


def load_outlook_cache(path: str = CACHE_PATH) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _cache_age_hours(cached: Dict[str, Any]) -> float:
    try:
        t = datetime.strptime(cached["as_of"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
    except Exception:
        return 1e9


def compute_outlook_cache(cfg: Optional[Dict[str, Any]] = None,
                          path: str = CACHE_PATH) -> None:
    """Compute the live outlook and write the cache (slow; run in background)."""
    from src.outlook.engine import load_outlook_config
    from src.outlook.backtest import live_outlook
    cfg = cfg or load_outlook_config()
    rows = live_outlook(cfg)
    if rows:
        save_outlook_cache(rows, narrative(rows, cfg), path)


# ── rendering ──────────────────────────────────────────────────────────────────
def _c(text: str, color, bold: bool = False) -> str:
    if _HAS_FMT and fmt and color:
        return fmt.colorize(text, color, bold=bold)
    return text


def print_outlook_box(width: int = 90, refresh_if_stale_hours: float = 20.0,
                      cache_path: str = CACHE_PATH) -> None:
    """Print the outlook box from cache (instant). Kick a background refresh if
    the cache is missing or stale, so the next run is current. Never blocks."""
    cached = load_outlook_cache(cache_path)

    if cached is None or _cache_age_hours(cached) > refresh_if_stale_hours:
        import threading
        threading.Thread(target=lambda: compute_outlook_cache(path=cache_path),
                         daemon=True).start()
    if cached is None:
        return  # nothing to show yet; next run will have it

    rows = cached.get("rows") or []
    lines = cached.get("narrative") or []
    if not rows:
        return

    arrow = {"BULLISH": "▲", "BEARISH": "▼"}
    colors = ({"BULLISH": fmt.Colors.GREEN, "BEARISH": fmt.Colors.RED,
               "NEUTRAL": fmt.Colors.YELLOW} if _HAS_FMT and fmt else {})
    label = {"BULLISH": "FAVOR", "BEARISH": "AVOID", "NEUTRAL": "NEUTRAL"}
    ordered = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)
    # show the strongest 4 and weakest 2
    show = ordered[:4] + (["…"] if len(ordered) > 6 else []) + ordered[-2:] \
        if len(ordered) > 6 else ordered

    inner = width - 4
    title = " SECTOR OUTLOOK — next ~1-3 months "
    side = (width - len(title) - 2) // 2
    bar = ("┌" + "─" * side + title + "─" * (width - len(title) - 2 - side) + "┐")
    bot = "└" + "─" * (width - 2) + "┘"
    print()
    print(_c(bar, fmt.Colors.CYAN if _HAS_FMT else "", bold=True))
    for r in show:
        if r == "…":
            print(_c(f"│  {'·· (middle of pack) ··'.ljust(inner)}│", fmt.Colors.DIM if _HAS_FMT else ""))
            continue
        nm = _name(r["ticker"])
        a = arrow.get(r["direction"], "▬")
        line = f"{a} {label[r['direction']]:<7} {r['ticker']:<5} {nm:<16} {r.get('conviction',50):>3}   {r.get('drivers','')}"
        print(_c(f"│  {line[:inner].ljust(inner)}│", colors.get(r["direction"], "")))
    print(_c(bot, fmt.Colors.CYAN if _HAS_FMT else "", bold=True))
    for ln in lines:
        # wrap to width
        words, cur = ln.split(), ""
        for w in words:
            if len(cur) + len(w) + 1 > width - 4:
                print(_c("  " + cur, fmt.Colors.DIM if _HAS_FMT else ""))
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            print(_c("  " + cur, fmt.Colors.DIM if _HAS_FMT else ""))
    print(_c(f"  as of {cached.get('as_of','?')} · long picks ~66-72% right at 2-3mo (see OUTLOOK_FINDINGS.md)",
             fmt.Colors.DIM if _HAS_FMT else ""))
    print()


__all__ = ["narrative", "save_outlook_cache", "load_outlook_cache",
           "compute_outlook_cache", "print_outlook_box"]
