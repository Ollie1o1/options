"""Quant context layer: theme->sector map, historical percentiles, event flag.

Pure except for the history-db helpers (load_history / persist_reading), which
read/write a small local SQLite. Everything degrades to None/[] on cold-start
or error — never raises.
"""
from __future__ import annotations

import json as _json
import os as _os
import sqlite3 as _sqlite
import statistics as _stats
from dataclasses import dataclass, field
from datetime import datetime as _dt, timezone as _tz
from typing import Optional

# theme -> [(human label, ETF proxy), ...]. 'other' is intentionally absent.
THEME_SECTORS: dict[str, list[tuple[str, str]]] = {
    "geopolitics":   [("defense", "ITA"), ("energy", "XLE")],
    "earnings_tech": [("semis", "SOXX"), ("big-tech", "QQQ")],
    "fed_rates":     [("financials", "XLF"), ("rates", "TLT"), ("growth", "QQQ")],
    "inflation":     [("rates", "TLT"), ("energy", "XLE"), ("broad", "SPY")],
    "trade_tariffs": [("industrials", "XLI"), ("semis", "SOXX"), ("china", "FXI")],
    "jobs":          [("rates", "TLT"), ("broad", "SPY")],
    "energy":        [("energy", "XLE"), ("oil", "USO")],
    "crypto":        [("crypto", "IBIT"), ("miners", "MSTR")],
}


def sectors_for(theme: str) -> list[tuple[str, str]]:
    return list(THEME_SECTORS.get(theme, []))


@dataclass
class ThemeRead:
    theme: str
    score: float
    n: int
    pctile: Optional[float]
    z: Optional[float]
    sectors: list[tuple[str, str]]
    top_headline: str
    read: str = ""


@dataclass
class MacroContext:
    pulse: float
    pulse_pctile: Optional[float]
    pulse_z: Optional[float]
    lean: str
    confidence: int
    n_items: int
    n_sources: int
    bull_pct: float
    themes: list[ThemeRead]
    event_active: bool
    event_name: Optional[str]
    event_date: Optional[str]
    next_events: list[dict] = field(default_factory=list)
    n_history: int = 0
    headline: str = ""
    what_would_flip: str = ""
    narrative_source: str = ""


# ── Stats ────────────────────────────────────────────────────────────────────
_MIN_SAMPLES = 5


def percentile(value: float, samples: list[float]) -> Optional[float]:
    if len(samples) < _MIN_SAMPLES:
        return None
    n_le = sum(1 for s in samples if s <= value)
    return round(100.0 * n_le / len(samples), 1)


def zscore(value: float, samples: list[float]) -> Optional[float]:
    if len(samples) < _MIN_SAMPLES:
        return None
    try:
        sd = _stats.pstdev(samples)
    except _stats.StatisticsError:
        return None
    if sd == 0:
        return None
    return round((value - _stats.fmean(samples)) / sd, 2)


# ── History db ───────────────────────────────────────────────────────────────
def default_db_path() -> str:
    return _os.path.join("data", "macro_pulse.db")


def _connect(db_path: str) -> _sqlite.Connection:
    _os.makedirs(_os.path.dirname(_os.path.abspath(db_path)), exist_ok=True)
    conn = _sqlite.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pulse_history "
        "(ts TEXT, pulse REAL, themes_json TEXT)"
    )
    return conn


def load_history(db_path: str, limit: int = 30) -> list[dict]:
    if not _os.path.exists(db_path):
        return []
    try:
        conn = _sqlite.connect(db_path)
        rows = conn.execute(
            "SELECT pulse, themes_json FROM pulse_history "
            "ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    except _sqlite.Error:
        return []
    out = []
    for pulse, themes_json in rows:
        try:
            themes = _json.loads(themes_json) if themes_json else {}
        except (ValueError, TypeError):
            themes = {}
        out.append({"pulse": pulse, "themes": themes})
    return out


def persist_reading(pulse: float, theme_scores: dict[str, float],
                    db_path: str) -> None:
    try:
        conn = _connect(db_path)
        conn.execute(
            "INSERT INTO pulse_history (ts, pulse, themes_json) VALUES (?, ?, ?)",
            (_dt.now(_tz.utc).isoformat(), float(pulse),
             _json.dumps(theme_scores)),
        )
        conn.commit()
        conn.close()
    except _sqlite.Error:
        pass  # best-effort; history is a nicety, not a requirement


# ── Assembly ─────────────────────────────────────────────────────────────────
def enrich(agg: dict, *, db_path: Optional[str] = None,
           history_limit: int = 30, config: Optional[dict] = None) -> MacroContext:
    from src.worldnews import panel as _wn_panel

    db_path = db_path or default_db_path()
    hist = load_history(db_path, limit=history_limit)
    pulse_samples = [r["pulse"] for r in hist]

    pulse = float(agg.get("pulse", 0.0))
    themes_raw = agg.get("themes", {}) or {}
    top = agg.get("top", []) or []

    theme_reads: list[ThemeRead] = []
    for name, th in themes_raw.items():
        if name == "other":
            continue
        score = float(th.get("score", 0.0))
        theme_samples = [r["themes"].get(name) for r in hist
                         if r["themes"].get(name) is not None]
        head = next((t.get("title", "") for t in top
                     if t.get("theme") == name), "")
        theme_reads.append(ThemeRead(
            theme=name, score=score, n=int(th.get("n", 0)),
            pctile=percentile(score, theme_samples),
            z=zscore(score, theme_samples),
            sectors=sectors_for(name), top_headline=head))
    theme_reads.sort(key=lambda t: -t.n)

    try:
        active, ev_name, ev_date = _macro_event(config)
    except Exception:
        active, ev_name, ev_date = False, None, None

    try:
        nxt = _wn_panel.next_events(3)
    except Exception:
        nxt = []

    return MacroContext(
        pulse=pulse,
        pulse_pctile=percentile(pulse, pulse_samples),
        pulse_z=zscore(pulse, pulse_samples),
        lean=_wn_panel._lean(pulse),
        confidence=int(agg.get("confidence", 0)),
        n_items=int(agg.get("n_items", 0)),
        n_sources=int(agg.get("n_sources", 0)),
        bull_pct=float(agg.get("bull_pct", 0.5)),
        themes=theme_reads,
        event_active=active, event_name=ev_name, event_date=ev_date,
        next_events=nxt, n_history=len(hist))


def _macro_event(config: Optional[dict]):
    from src.macro_analyzer import check_macro_event_window
    return check_macro_event_window(config)
