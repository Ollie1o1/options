"""Squeeze display surfaces — banner, calls mini-board, scan summary board.

Display-layer only. All styling goes through fmt.style / src.ui components
(quant-desk UI discipline); nothing here mutates scores or logs trades.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src import formatting as fmt
from src import ui
from src.squeeze.detector import SETUP, WATCH, SqueezeSetup

WIDTH = 100

# yfinance SI is the bi-monthly FINRA number — often weeks stale.
_STALENESS_CAVEAT = "SI is the bi-monthly FINRA print via yfinance — often weeks stale; confirm before sizing"


def banner(setup: SqueezeSetup, ticker: str, width: int = WIDTH) -> Optional[str]:
    """The loud squeeze read for one ticker; None when grade is NONE."""
    if setup.grade not in (SETUP, WATCH):
        return None
    glyph = fmt.GLYPHS.get("squeeze", "^")
    if setup.grade == SETUP:
        head = fmt.style(f"{glyph} SHORT-SQUEEZE SETUP — {ticker}", "good", bold=True)
    else:
        head = fmt.style(f"{glyph} SQUEEZE WATCH — {ticker}", "warn", bold=True)
    body = [fmt.style(f"evidence {setup.points} pts — bullish squeeze read; "
                      "verdict is display-only (scores unchanged)", "muted")]
    body += [f"{fmt.GLYPHS.get('bullet', '*')} {line}" for line in setup.evidence]
    body.append(fmt.style(f"{fmt.GLYPHS.get('warn', '!')} {_STALENESS_CAVEAT}", "muted"))
    return ui.card(head, body, width, boxed=True, accent=(setup.grade == SETUP))


def _fmt_num(value, spec: str, na: str = "—") -> str:
    try:
        f = float(value)
        if math.isnan(f):
            return na
        return format(f, spec)
    except (TypeError, ValueError):
        return na


def call_board(df: pd.DataFrame, ticker: str, top_n: int = 3,
               width: int = WIDTH) -> Optional[str]:
    """Calls-only slice of an enriched chain, ranked by existing quality_score.

    The squeeze thesis is long the underlying, so surface the best calls even
    when the mode's own ranking picked puts (the NBIS 2026-07-16 failure).
    """
    if df is None or len(df) == 0 or "type" not in df.columns:
        return None
    calls = df[df["type"] == "call"]
    if calls.empty:
        return None
    if "quality_score" in calls.columns:
        calls = calls.sort_values("quality_score", ascending=False)
    calls = calls.head(top_n)

    cols = [
        {"h": "Strike", "w": 9, "align": "right"},
        {"h": "Expiry", "w": 10},
        {"h": "DTE", "w": 4, "align": "right"},
        {"h": "Delta", "w": 6, "align": "right"},
        {"h": "Prem", "w": 8, "align": "right"},
        {"h": "Sprd%", "w": 6, "align": "right"},
        {"h": "Net EV", "w": 8, "align": "right"},
        {"h": "Score", "w": 6, "align": "right"},
    ]
    rows = []
    for _, r in calls.iterrows():
        rows.append([
            f"${_fmt_num(r.get('strike'), '.1f')}",
            str(r.get("expiration", "—"))[:10],
            _fmt_num(r.get("dte"), ".0f"),
            _fmt_num(r.get("delta"), "+.2f"),
            f"${_fmt_num(r.get('premium'), '.2f')}",
            _fmt_num(pd.to_numeric(r.get("spread_pct"), errors="coerce"), ".1f"),
            f"${_fmt_num(r.get('ev_per_contract'), '+.0f')}",
            _fmt_num(r.get("quality_score"), ".2f"),
        ])
    title = fmt.style(f"SQUEEZE CALLS — {ticker} (long side of the setup)", "heading")
    return ui.card(title, ui.table(cols, rows).splitlines(), width)


def squeeze_scan_board(per_ticker: list, width: int = WIDTH) -> str:
    """SQUEEZE-mode summary: one row per scanned candidate.

    ``per_ticker``: list of dicts with keys ticker, setup (SqueezeSetup),
    best_call (str or None).
    """
    cols = [
        {"h": "Ticker", "w": 7},
        {"h": "Grade", "w": 6},
        {"h": "Pts", "w": 3, "align": "right"},
        {"h": "SI%", "w": 5, "align": "right"},
        {"h": "Cover", "w": 6, "align": "right"},
        {"h": "Trend", "w": 8},
        {"h": "Best call", "w": 22},
    ]
    rows = []
    ordered = sorted(per_ticker,
                     key=lambda d: (d["setup"].grade != SETUP,
                                    d["setup"].grade != WATCH,
                                    -d["setup"].points))
    for item in ordered:
        s: SqueezeSetup = item["setup"]
        grade_style = {"SETUP": "good", "WATCH": "warn"}.get(s.grade, "muted")
        rows.append([
            item["ticker"],
            fmt.style(s.grade, grade_style, bold=(s.grade == SETUP)),
            str(s.points),
            _fmt_num(s.si_pct, ".1f"),
            _fmt_num(s.days_to_cover, ".1f") + "d",
            s.trend or "—",
            item.get("best_call") or "—",
        ])
    title = fmt.style("SQUEEZE BOARD — high-short-float candidates", "heading")
    body = ui.table(cols, rows).splitlines()
    body.append(fmt.style(f"{fmt.GLYPHS.get('warn', '!')} {_STALENESS_CAVEAT}", "muted"))
    return ui.card(title, body, width)
