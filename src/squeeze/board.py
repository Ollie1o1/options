"""Squeeze display surfaces — banner, calls mini-board, scan summary board.

Display-layer only. All styling goes through fmt.style / src.ui components
(quant-desk UI discipline); nothing here mutates scores or logs trades.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

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

# The horizon the multiple is valued at: the close of the measured window,
# 42 trading days out, in the calendar days `dte` carries.
SQUEEZE_WINDOW_DAYS = 59

# The window that move was measured over, in the units `dte` uses.
# `dte` is calendar days (T_years * 365), and 42 trading days is 42 * 7/5 =
# 58.8 of them before holidays — so a "30-day" floor would cover barely half
# the window. Ranking divides by premium, which favours the cheapest contract
# on the board, and the cheapest is always the shortest-dated: without this
# floor the board systematically picks options that expire before the measured
# move has had time to happen.
SQUEEZE_MIN_DTE = 60


def convexity_multiple(row, target_move: float = SQUEEZE_TARGET_MOVE,
                       rfr: float = 0.045) -> Optional[float]:
    """What one contract is worth if the underlying makes the measured move.

    Valued at a fixed horizon — the close of the measured window — rather than
    at expiry, with the contract's own IV held constant. That is what makes
    multiples comparable across DTE: intrinsic-at-expiry is a floor that
    loosens the longer the contract, so a 132-day call that catches the move
    and still holds ~73 days of extrinsic scored below a 13-day one purely for
    costing more. Holding IV constant is the neutral assumption — squeezes
    usually bid vol on the way up and crush it after, and neither is assumed.

    Falls back to intrinsic when there is no usable IV or no time left at the
    horizon, which is the same conservative floor as before. The shape is
    unchanged: deep ITM scores low on leverage and the ranking peaks just out
    of the money. Returns None when it cannot be computed at all, so "unknown"
    never sorts as "pays nothing".
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    spot = _num(get("underlying", get("spot")))
    strike = _num(get("strike"))
    premium = _num(get("premium"))
    if spot is None or strike is None or premium is None:
        return None
    if spot <= 0 or premium <= 0:
        return None

    target = spot * (1.0 + target_move)
    intrinsic = max(0.0, target - strike)

    dte = _num(get("dte"))
    iv = _num(get("impliedVolatility"))
    if iv is None:
        iv = _num(get("iv_solved"))
    if dte is None or iv is None or iv <= 0:
        return intrinsic / premium

    years_left = (dte - SQUEEZE_WINDOW_DAYS) / 365.0
    if years_left <= 0:
        return intrinsic / premium

    try:
        from src.utils import bs_call
        q = _num(get("dividend_yield")) or 0.0
        value = bs_call(target, strike, years_left, rfr, iv, q)
    except Exception:
        return intrinsic / premium
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return intrinsic / premium
    # Never below the floor: a repricing that undercuts intrinsic is a bad
    # number, not a discovery.
    return max(float(value), intrinsic) / premium


def _num(value) -> Optional[float]:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# Wealthsimple: $0 commission, spread is the whole cost. [[project_broker_costs]]
_DEFAULT_COMMISSION = 0.0


def breakeven_vol(row, rfr: float = 0.045,
                  commission: float = _DEFAULT_COMMISSION) -> Optional[float]:
    """Realized vol at which this contract's NET EV is zero.

    The scan's `ev_per_contract` prices fair value on *trailing* realized vol
    (`hv_252d` first), so on a name that just ran +18% on heavy short interest
    it is negative by construction — it measures against precisely what the
    squeeze thesis disputes. This number takes no view on future vol at all: it
    reports the vol the trade needs, and lets the reader supply the view.

    Gross EV is zero at exactly IV, which is the definition of implied vol and
    tells you nothing. So this solves for NET zero — the vol that also repays
    round-trip friction — by inverting Black-Scholes at ``premium + costs``.
    The gap over IV is what crossing the spread costs in vol points.
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    spot = _num(get("underlying", get("spot")))
    strike = _num(get("strike"))
    premium = _num(get("premium"))
    dte = _num(get("dte"))
    # Checked one by one rather than `None in (...)`: a membership test does
    # not narrow Optional, so mypy still sees None on every use below.
    if spot is None or strike is None or premium is None or dte is None:
        return None
    if premium <= 0 or dte <= 0:
        return None

    spread = _num(get("spread_pct")) or 0.0
    # Mirrors trade_analysis.net_ev_per_contract: half the spread per side,
    # both sides, per share; commission is per contract so scale it down.
    cost_per_share = spread * premium + (2.0 * commission / 100.0)
    try:
        from src.data_quality import implied_vol_from_price
        return implied_vol_from_price("call", spot, strike, dte / 365.0, rfr,
                                      premium + cost_per_share,
                                      _num(get("dividend_yield")) or 0.0)
    except Exception:
        return None


def breakeven_vol_premium_ref(
    row, rfr: float = 0.045, commission: float = _DEFAULT_COMMISSION,
) -> Tuple[Optional[float], Optional[float]]:
    """``(breakeven_vol, reference_vol)`` — both solved from the same premium.

    The board prints the breakeven and, beside it, the gap over the contract's
    IV, captioned as what crossing the spread costs in vol points. That caption
    is only true if the two numbers are solved off the same price. It used to
    subtract the *vendor's* reported ``impliedVolatility``, which is quoted
    against a different price than the mid the breakeven is built on, so the
    printed gap was the spread cost plus the vendor's disagreement with the mid.

    Measured over 29,769 archived CBOE call snapshots (2026-07-01 onward): the
    vendor gap has a mean absolute size of 0.92vp against a median true spread
    cost of 1.40vp, and **16.7% of contracts printed a negative number** — that
    is, crossing the spread shown as a discount, which it can never be. The live
    scan reads yfinance rather than CBOE, where agreement with the mid is
    weaker still.

    Solving the reference from ``premium`` makes the difference the cost by
    construction: same model, same inputs, one price apart. Returns
    ``(None, None)`` when the row cannot support either solve, so an unknown
    never prints as a zero cost.
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    be = breakeven_vol(row, rfr=rfr, commission=commission)
    if be is None:
        return None, None
    spot = _num(get("underlying", get("spot")))
    strike = _num(get("strike"))
    premium = _num(get("premium"))
    dte = _num(get("dte"))
    if spot is None or strike is None or premium is None or dte is None:
        return None, None
    try:
        from src.data_quality import implied_vol_from_price
        ref = implied_vol_from_price("call", spot, strike, dte / 365.0, rfr,
                                     premium,
                                     _num(get("dividend_yield")) or 0.0)
    except Exception:
        return None, None
    if ref is None:
        return None, None
    return be, ref


def _apply_dte_floor(calls: pd.DataFrame):
    """Keep only contracts that outlive the measured window.

    Returns ``(frame, floored)``. When nothing clears the floor the original
    frame comes back with ``floored=False`` so the caller can show it under a
    warning — a name whose chain simply lists no long expiries is worth seeing,
    and an empty board is what this whole fix set out to stop printing.
    Rows without a ``dte`` are kept: unknown is not the same as too short.
    """
    if "dte" not in calls.columns:
        return calls, False
    dte = pd.to_numeric(calls["dte"], errors="coerce")
    long_enough = calls[dte.isna() | (dte >= SQUEEZE_MIN_DTE)]
    if long_enough.empty:
        return calls, False
    return long_enough, True


def _rank_calls(df: pd.DataFrame, rfr: float = 0.045) -> pd.DataFrame:
    """Calls from an enriched chain, best convexity first.

    Falls back to quality_score when no row carries a spot, so a chain without
    ``underlying`` still ranks on something rather than on row order.
    """
    calls = df[df["type"] == "call"].copy()
    if calls.empty:
        return calls
    calls, _ = _apply_dte_floor(calls)
    mult = calls.apply(convexity_multiple, axis=1, rfr=rfr)
    if mult.notna().any():
        calls["_convexity"] = mult
        return calls.sort_values("_convexity", ascending=False, na_position="last")
    calls["_convexity"] = None
    if "quality_score" in calls.columns:
        return calls.sort_values("quality_score", ascending=False)
    return calls


def best_call_label(df: pd.DataFrame, rfr: float = 0.045) -> Optional[str]:
    """The one contract the scan board names — same ranking as call_board.

    Shared so the summary board's "Best call" column and the per-ticker board's
    top row can never name different contracts.
    """
    if df is None or len(df) == 0 or "type" not in df.columns:
        return None
    ranked = _rank_calls(df, rfr=rfr)
    if ranked.empty:
        return None
    row = ranked.iloc[0]
    strike = _num(row.get("strike"))
    if strike is None:
        return None
    return f"${strike:g}C {str(row.get('expiration', ''))[:10]}".strip()


def call_board(df: pd.DataFrame, ticker: str, top_n: int = 3,
               width: int = WIDTH, rfr: float = 0.045) -> Optional[str]:
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
    calls = _rank_calls(df, rfr=rfr)
    if calls.empty:
        return None
    _, _floored = _apply_dte_floor(df[df["type"] == "call"])
    calls = calls.head(top_n)

    cols = [
        {"h": "Strike", "w": 9, "align": "right"},
        {"h": "Expiry", "w": 10},
        {"h": "DTE", "w": 4, "align": "right"},
        {"h": "Delta", "w": 6, "align": "right"},
        {"h": "Prem", "w": 8, "align": "right"},
        {"h": "Sprd%", "w": 6, "align": "right"},
        {"h": "+20%", "w": 7, "align": "right"},
        {"h": "IV", "w": 6, "align": "right"},
        {"h": "BE vol", "w": 8, "align": "right"},
        {"h": "Net EV", "w": 8, "align": "right"},
        {"h": "Score", "w": 6, "align": "right"},
    ]
    rows = []
    for _, r in calls.iterrows():
        _mult = r.get("_convexity")
        _iv = pd.to_numeric(r.get("impliedVolatility"), errors="coerce")
        # Reference solved from this contract's own premium, not the vendor's
        # IV — otherwise the "+N" carries the vendor's disagreement with the
        # mid on top of the spread it claims to be measuring.
        _be, _ref = breakeven_vol_premium_ref(r, rfr=rfr)
        if _be is None or _ref is None:
            _be_cell = "—"
        else:
            _be_cell = f"{_be * 100:.0f} +{(_be - _ref) * 100:.0f}"
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
            _fmt_num(_iv * 100 if _iv is not None else None, ".0f"),
            _be_cell,
            f"${_fmt_num(r.get('ev_per_contract'), '+.0f')}",
            _fmt_num(r.get("quality_score"), ".2f"),
        ])
    title = fmt.style(f"SQUEEZE CALLS — {ticker} (long side of the setup)", "heading")
    body = ui.table(cols, rows).splitlines()
    body.append(fmt.style(
        f"{fmt.GLYPHS.get('bullet', '*')} +20% = contract value if the underlying makes the "
        f"cohort's median move, priced at day {SQUEEZE_WINDOW_DAYS} with IV held constant, "
        "over premium paid — ranked on this, not on PoP", "muted"))
    body.append(fmt.style(
        f"{fmt.GLYPHS.get('bullet', '*')} valuing at the window's close, not at expiry, is "
        "what makes these comparable across DTE — a longer contract keeps the extrinsic it "
        "has left. Constant IV assumes neither the squeeze bid nor the crush after",
        "muted"))
    body.append(fmt.style(
        f"{fmt.GLYPHS.get('bullet', '*')} BE vol = the realized vol this contract needs for "
        "net EV zero. The +N is what crossing the spread costs in vol points — both solved "
        "from this contract's own premium, so the gap is the spread and nothing else. Takes "
        "no view on future vol; Net EV does, and its view is trailing hv_252d", "muted"))
    if _floored:
        body.append(fmt.style(
            f"{fmt.GLYPHS.get('bullet', '*')} DTE >= {SQUEEZE_MIN_DTE}d only — the cohort's "
            "50.5% is measured over 42 trading days (~59 calendar), and ranking on premium "
            "would otherwise always pick the shortest-dated contract", "muted"))
    else:
        body.append(fmt.style(
            f"{fmt.GLYPHS.get('warn', '!')} every call in the scan's DTE window is shorter "
            f"than the measured window ({SQUEEZE_MIN_DTE}d) — these multiples are NOT the "
            "cohort's 50.5% trade; the move has less time to happen than the number assumes. "
            "The chain may list longer expiries the scan did not fetch", "warn"))
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
