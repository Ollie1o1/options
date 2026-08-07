"""Squeeze display surfaces — banner, calls mini-board, scan summary board.

Display-layer only. All styling goes through fmt.style / src.ui components
(quant-desk UI discipline); nothing here mutates scores or logs trades.
"""
from __future__ import annotations

import math
from typing import List, Optional

import pandas as pd

from src import formatting as fmt
from src import ui
from src.squeeze.detector import SETUP, WATCH, SqueezeSetup
from src.squeeze.universe import SqueezeUniverse

WIDTH = 100

# yfinance SI is the bi-monthly FINRA number — often weeks stale.
_STALENESS_CAVEAT = "SI is the bi-monthly FINRA print via yfinance — often weeks stale; confirm before sizing"


def sourcing_lines(uni: SqueezeUniverse) -> List[str]:
    """Where this scan's candidates came from — momentum cohort vs. fill.

    The two screens have different measured base rates (P(+20% in 42d) 50.5%
    vs 39.0%, docs/SQUEEZE_BACKTEST.md), so which one a name arrived on is
    part of reading the board, not sourcing trivia.
    """
    tickers = list(uni.tickers or [])
    momentum = list(uni.momentum or [])
    lines: List[str] = []
    if uni.source == "fallback":
        lines.append(fmt.style(
            f"{fmt.GLYPHS.get('warn', '!')} Finviz screens unavailable — scanning the "
            f"hardcoded fallback list ({len(tickers)} names, refreshed by hand and "
            "likely stale)", "warn"))
        return lines
    head = f"{len(momentum)} of {len(tickers)} cleared the +10% week filter"
    if momentum:
        lines.append(fmt.style(
            f"{head}: {', '.join(momentum[:10])}"
            f"{'...' if len(momentum) > 10 else ''}", "good"))
    else:
        lines.append(fmt.style(
            f"{head} — no momentum cohort today; these are short-float rank only",
            "muted"))
    lines.append(fmt.style(
        "  measured: heavy SI + upward momentum hits +20% in 42d 50.5% of the "
        "time vs 39.0% on SI alone (22.5% base)", "muted"))
    return lines


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


# The move to rank against: median path max of the backtest's best cohort
# (top-5% SI and 5d return >= +10%) was +20.5% over 42 trading days.
SQUEEZE_TARGET_MOVE = 0.20


def convexity_multiple(row, target_move: float = SQUEEZE_TARGET_MOVE) -> Optional[float]:
    """What one contract pays if the underlying makes the measured move.

    Intrinsic at expiry over premium paid. That assumes nothing about vol and
    ignores extrinsic the contract would still hold if the move came early, so
    it is conservative — and conservative in a stated direction, which a
    repriced number with a guessed IV would not be.

    The shape is what earns it: deep ITM scores low (all premium, little
    leverage), a strike the move never reaches scores zero, and the ranking
    peaks just out of the money. Returns None when it cannot be computed, so
    "unknown" never sorts as "pays nothing".
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    spot = _num(get("underlying", get("spot")))
    strike = _num(get("strike"))
    premium = _num(get("premium"))
    if spot is None or strike is None or premium is None:
        return None
    if spot <= 0 or premium <= 0:
        return None
    return max(0.0, spot * (1.0 + target_move) - strike) / premium


def _num(value) -> Optional[float]:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _rank_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Calls from an enriched chain, best convexity first.

    Falls back to quality_score when no row carries a spot, so a chain without
    ``underlying`` still ranks on something rather than on row order.
    """
    calls = df[df["type"] == "call"].copy()
    if calls.empty:
        return calls
    mult = calls.apply(convexity_multiple, axis=1)
    if mult.notna().any():
        calls["_convexity"] = mult
        return calls.sort_values("_convexity", ascending=False, na_position="last")
    calls["_convexity"] = None
    if "quality_score" in calls.columns:
        return calls.sort_values("quality_score", ascending=False)
    return calls


def best_call_label(df: pd.DataFrame) -> Optional[str]:
    """The one contract the scan board names — same ranking as call_board.

    Shared so the summary board's "Best call" column and the per-ticker board's
    top row can never name different contracts.
    """
    if df is None or len(df) == 0 or "type" not in df.columns:
        return None
    ranked = _rank_calls(df)
    if ranked.empty:
        return None
    row = ranked.iloc[0]
    strike = _num(row.get("strike"))
    if strike is None:
        return None
    return f"${strike:g}C {str(row.get('expiration', ''))[:10]}".strip()


def call_board(df: pd.DataFrame, ticker: str, top_n: int = 3,
               width: int = WIDTH) -> Optional[str]:
    """Calls-only slice of an enriched chain, ranked for convexity.

    The squeeze thesis is long the underlying, so surface the best calls even
    when the mode's own ranking picked puts (the NBIS 2026-07-16 failure).

    Ranked by ``convexity_multiple`` rather than quality_score: the scorer
    rewards probability of profit, which on a call ladder means deep ITM, and
    the backtest is explicit that this is a right-tail trade — SETUP names show
    a fatter right tail *and* a worse median. quality_score stays on the board
    as ``Score`` so the scorer's own verdict is still visible.
    """
    if df is None or len(df) == 0 or "type" not in df.columns:
        return None
    calls = _rank_calls(df)
    if calls.empty:
        return None
    calls = calls.head(top_n)

    cols = [
        {"h": "Strike", "w": 9, "align": "right"},
        {"h": "Expiry", "w": 10},
        {"h": "DTE", "w": 4, "align": "right"},
        {"h": "Delta", "w": 6, "align": "right"},
        {"h": "Prem", "w": 8, "align": "right"},
        {"h": "Sprd%", "w": 6, "align": "right"},
        {"h": "+20%", "w": 7, "align": "right"},
        {"h": "Net EV", "w": 8, "align": "right"},
        {"h": "Score", "w": 6, "align": "right"},
    ]
    rows = []
    for _, r in calls.iterrows():
        _mult = r.get("_convexity")
        rows.append([
            f"${_fmt_num(r.get('strike'), '.1f')}",
            str(r.get("expiration", "—"))[:10],
            _fmt_num(r.get("dte"), ".0f"),
            _fmt_num(r.get("delta"), "+.2f"),
            f"${_fmt_num(r.get('premium'), '.2f')}",
            # spread_pct is a fraction pipeline-wide ((ask-bid)/mid); the
            # column header is a percent, as in cli_display.
            _fmt_num(pd.to_numeric(r.get("spread_pct"), errors="coerce") * 100, ".1f"),
            "—" if _mult is None or pd.isna(_mult) else f"{float(_mult):.1f}x",
            f"${_fmt_num(r.get('ev_per_contract'), '+.0f')}",
            _fmt_num(r.get("quality_score"), ".2f"),
        ])
    title = fmt.style(f"SQUEEZE CALLS — {ticker} (long side of the setup)", "heading")
    body = ui.table(cols, rows).splitlines()
    body.append(fmt.style(
        f"{fmt.GLYPHS.get('bullet', '*')} +20% = contract value if the underlying makes the "
        "cohort's median move, at expiry, over premium paid — ranked on this, not on PoP",
        "muted"))
    body.append(fmt.style(
        f"{fmt.GLYPHS.get('warn', '!')} dividing by premium favours cheap, short-dated "
        "contracts; the cohort's 50.5% hit rate is over 42 trading days, so check DTE "
        "before reading a multiple as that trade", "muted"))
    return ui.card(title, body, width)


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
