"""Fast, profit-honest Options Scanner, Live/Archived Chain Matrix, Intel Hub, and Multi-Leg Risk Engine for the TUI Desk.

Design invariants:
  - Instant offline-first defaults: reads local `data/chain_archive.db`, `data/equity_ohlcv.db`,
    `data/occ_oi.db`, and `data/dolt_options.db` in milliseconds with zero network stalls.
  - Explicit opt-in live Yahoo chain fetches (always run on background worker threads).
  - Cost & CRA TFSA honesty: every contract and multi-leg spread reports real bid-ask
    round-trip friction %, walk-away limit price (`build_walk_away_ladder`), max risk,
    IV vs RV20, and explicit TFSA vs TAXABLE_ONLY tagging.
  - Reuses existing canonical engines (`src.structure.chain_data`, `src.alpha.options_for_shares`,
    `src.uoa`, `src.occ_archive`, `src.vol_intel`, `src.radar.calendar_spreads`, `src.utils` BS).
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path
from src.structure.chain_data import (
    ChainContract,
    ChainSlice,
    load_archived_chain_slice,
    load_live_chain_slice,
)
from src.structure.execution_plan import WalkAwayLadder, build_walk_away_ladder
from src.utils import bs_delta, bs_gamma, bs_price, bs_theta, bs_vega

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptionLeg:
    """One contract leg inside a single-leg or multi-leg options structure."""

    opt_type: str               # "call" | "put"
    side: str                   # "BUY" | "SELL"
    strike: float
    expiration: str
    dte: int
    bid: float
    ask: float
    mid: float
    iv: float                   # decimal, e.g. 0.28
    delta: float = 0.0
    theta: float = 0.0
    open_interest: int = 0
    volume: int = 0
    qty: int = 1

    @property
    def signed_qty(self) -> int:
        return self.qty if self.side.upper() == "BUY" else -self.qty


@dataclass(frozen=True)
class OptionScanRow:
    """One screened single-leg or multi-leg options candidate for `[7] OPTIONS`."""

    ticker: str
    strategy: str               # "Bull Put" | "Bear Call" | "Long Call" | "Long Put" | "Iron Condor" | "Calendar" | "Short Put"
    strategy_key: str           # "bull_put" | "bear_call" | "long_call" | "long_put" | "iron_condor" | "calendar" | "short_put"
    expiration: str
    dte: int
    strikes_label: str
    spot: float
    short_delta: Optional[float]
    mid_price: float            # credit (+) for credit structures, debit (+) for debit structures
    is_credit: bool
    walk_away_limit: float
    friction_pct: float         # round-trip spread cost / reward or debit (0.12 = 12%)
    max_risk_usd: float
    max_profit_usd: float
    ror_pct: Optional[float]    # max_profit / max_risk * 100 (or None if unbounded)
    iv_pct: Optional[float]
    iv_vs_rv: Optional[float]   # ATM IV / RV20 ratio
    skew_vp: Optional[float]
    min_oi: int
    total_vol: int
    tfsa_legal: bool
    evidence_badge: str         # "PROMISING" | "WATCHLIST" | "CAUTION" | "EXPERIMENTAL"
    risk_flags: Tuple[str, ...] = ()
    legs: Tuple[OptionLeg, ...] = ()
    source: str = "chain_archive"
    as_of: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChainPairRow:
    """One strike row in the T-shaped Call | Strike | Put Chain Matrix."""

    strike: float
    call: Optional[ChainContract]
    put: Optional[ChainContract]
    gex_m: float = 0.0
    tags: Tuple[str, ...] = ()  # e.g. ("SPOT", "CALL_WALL", "PUT_WALL", "GEX_FLIP", "EM_HI", "EM_LO")


@dataclass
class ChainMatrix:
    """Full interactive options chain view for one ticker & expiration."""

    ticker: str
    spot: float
    expiration: str
    dte: int
    available_expirations: List[Tuple[str, int]]  # [(exp_str, dte), ...]
    rows: List[ChainPairRow]
    source: str                 # "chain_archive" | "live"
    snap_date: str
    atm_iv_pct: Optional[float] = None
    rv20_pct: Optional[float] = None
    iv_vs_rv: Optional[float] = None
    skew_25d_vp: Optional[float] = None
    expected_move_usd: Optional[float] = None
    expected_move_pct: Optional[float] = None
    median_spread_pct: Optional[float] = None
    net_gex_m: Optional[float] = None
    gex_regime: str = "UNKNOWN"
    gamma_flip: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    gex_by_strike: List[Dict[str, float]] = field(default_factory=list)
    smile_points: List[Tuple[float, float]] = field(default_factory=list)   # [(strike, iv_pct)]
    term_points: List[Tuple[int, float]] = field(default_factory=list)      # [(dte, atm_iv_pct)]
    earnings_days: Optional[int] = None
    note: str = ""


@dataclass
class MultiLegAnalysis:
    """Live Greeks, friction, walk-away ladder, payoff curves, and 2D risk grid for 1-4 legs."""

    ticker: str
    spot: float
    structure_name: str
    legs: List[OptionLeg]
    is_credit: bool
    tfsa_legal: bool
    net_mid: float              # positive credit or positive debit per share
    net_cross: float            # natural credit or natural debit per share
    friction_pct: float
    max_risk_usd: float
    max_profit_usd: Optional[float]
    breakevens: List[float]
    delta_dollars: float
    gamma_dollars: float
    theta_per_day_usd: float
    vega_per_1pct_usd: float
    pop_mkt: float
    ladder: WalkAwayLadder
    prices: List[float]
    expiry_pnl: List[float]
    shocked_pnl: List[float]
    days_elapsed: int
    iv_shock_pct: float
    # 2D risk grid: rows = iv_shocks (high->low), cols = price_shocks (low->high)
    grid_price_shocks: List[float]
    grid_iv_shocks: List[float]
    grid_pnl: List[List[float]]


@dataclass
class OptionsIntelBundle:
    """Zero-network Options Intelligence snapshot (Flow + Vol/VRP + Skew/GEX + Calendars)."""

    uoa_rows: List[Dict[str, Any]] = field(default_factory=list)
    occ_rows: List[Dict[str, Any]] = field(default_factory=list)
    vol_rows: List[Dict[str, Any]] = field(default_factory=list)
    skew_rows: List[Dict[str, Any]] = field(default_factory=list)
    calendar_rows: List[Dict[str, Any]] = field(default_factory=list)
    as_of: str = ""


# ── Constants & Strategy Metadata ────────────────────────────────────────────

SCAN_STRATEGIES: Tuple[Tuple[str, str], ...] = (
    ("All Strategies", "all"),
    ("Bull Put Spreads [PROMISING · TAX]", "bull_put"),
    ("Long Calls [TFSA OK]", "long_call"),
    ("Long Puts [TFSA OK]", "long_put"),
    ("Calendars [WATCHLIST · TAX]", "calendar"),
    ("Bear Call Spreads [CAUTION · TAX]", "bear_call"),
    ("Iron Condors [CAUTION · TAX]", "iron_condor"),
    ("Short Puts [TAXABLE]", "short_put"),
)

DEFAULT_LIVE_SCAN_UNIVERSE: Tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "SMH", "XLK", "XLE", "XLF", "XLV", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
)


def _clean_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _load_rv20_map(ohlcv_db: str, symbols: Sequence[str]) -> Dict[str, float]:
    """Return `{TICKER: rv20_decimal}` from local OHLCV DB (fast, zero network)."""
    if not symbols:
        return {}
    try:
        from src.equity_playbooks.indicators import trailing_rv
        from src.radar.scanners import load_local_ohlcv_safe

        ohlcv = load_local_ohlcv_safe(ohlcv_db, symbols=list(symbols), lookback_days=40)
        out: Dict[str, float] = {}
        for sym, (_ds, cs, _vs) in ohlcv.items():
            if len(cs) >= 22:
                rv = float(trailing_rv(cs, 20)[-1])
                if math.isfinite(rv) and rv > 0.01:
                    out[sym.upper()] = rv
        return out
    except Exception:
        return {}


def _load_earnings_map() -> Dict[str, int]:
    try:
        from src.radar.scanners import _load_upcoming_earnings_map

        return _load_upcoming_earnings_map()
    except Exception:
        return {}


def _contract_to_leg(
    c: ChainContract,
    *,
    side: str,
    expiration: str,
    dte: int,
    fallback_iv: float = 0.25,
) -> OptionLeg:
    iv = float(c.iv) if (c.iv is not None and 0.02 <= c.iv <= 4.0) else float(fallback_iv)
    return OptionLeg(
        opt_type=c.opt_type,
        side=side.upper(),
        strike=float(c.strike),
        expiration=expiration,
        dte=int(dte),
        bid=float(c.bid),
        ask=float(c.ask),
        mid=float(c.mid),
        iv=iv,
        delta=float(c.delta or 0.0),
        theta=float(c.theta or 0.0),
        open_interest=int(c.open_interest or 0),
        volume=int(c.volume or 0),
        qty=1,
    )


# ── 1. Multi-Strategy Options Scanner ────────────────────────────────────────


def _build_candidates_for_slice(
    sl: ChainSlice,
    spot: float,
    *,
    rv20: Optional[float],
    earnings_days: Optional[int],
    strategy_filter: str,
    max_friction_pct: float = 0.30,
) -> List[OptionScanRow]:
    """Extract clean, friction-honest option candidates from one ChainSlice."""
    out: List[OptionScanRow] = []
    atm_iv = sl.atm_iv
    iv_pct = round(atm_iv * 100.0, 1) if atm_iv is not None else None
    iv_vs_rv = round(atm_iv / rv20, 2) if (atm_iv is not None and rv20 is not None and rv20 > 0.02) else None
    skew = sl.skew_25d_vp

    base_flags: List[str] = []
    if earnings_days is not None and 0 <= earnings_days <= sl.dte:
        base_flags.append(f"EARN_{earnings_days}d")

    want_all = strategy_filter == "all"

    # A) Bull Put Credit Spread (highest historically validated credit edge in repo)
    if want_all or strategy_filter == "bull_put":
        sp = sl.match_by_delta("put", 0.25, min_bid=0.10)
        if sp is not None:
            width_tgt = max(1.0, spot * 0.05)
            lp = sl.match_contract("put", sp.strike - width_tgt, max_rel_dist=0.06, min_bid=0.02)
            if lp is not None and lp.strike < sp.strike:
                width = sp.strike - lp.strike
                cr_mid = sp.mid - lp.mid
                cr_cross = sp.bid - lp.ask
                if cr_mid > 0.10 and cr_cross > 0 and cr_mid < width:
                    rt_spread = (sp.ask - sp.bid) + (lp.ask - lp.bid)
                    fric = rt_spread / cr_mid
                    if fric <= max_friction_pct:
                        lad = build_walk_away_ladder(
                            structure="Bull Put",
                            is_credit=True,
                            p_mid=cr_mid,
                            p_cross=cr_cross,
                            ev_view_gross_per_share=0.0,
                        )
                        max_risk = max(1.0, (width - cr_mid) * 100.0)
                        max_prof = cr_mid * 100.0
                        flags = list(base_flags) + ["TAXABLE_ONLY"]
                        if fric > 0.20:
                            flags.append("WIDE_SPREAD")
                        if iv_vs_rv is not None and iv_vs_rv < 0.90:
                            flags.append("VOL_CHEAP_SHORT")
                        badge = "PROMISING" if (60 <= sl.dte <= 120 and fric <= 0.25) else "WATCHLIST"
                        bp_legs: Tuple[OptionLeg, ...] = (
                            _contract_to_leg(sp, side="SELL", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                            _contract_to_leg(lp, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                        )
                        out.append(
                            OptionScanRow(
                                ticker=sl.symbol,
                                strategy="Bull Put",
                                strategy_key="bull_put",
                                expiration=sl.expiration,
                                dte=sl.dte,
                                strikes_label=f"{sp.strike:g}/{lp.strike:g}P",
                                spot=round(spot, 2),
                                short_delta=round(abs(sp.delta or 0.25), 2),
                                mid_price=round(cr_mid, 2),
                                is_credit=True,
                                walk_away_limit=lad.p_walk,
                                friction_pct=round(fric, 4),
                                max_risk_usd=round(max_risk, 0),
                                max_profit_usd=round(max_prof, 0),
                                ror_pct=round(max_prof / max_risk * 100.0, 1),
                                iv_pct=iv_pct,
                                iv_vs_rv=iv_vs_rv,
                                skew_vp=skew,
                                min_oi=min(sp.open_interest, lp.open_interest),
                                total_vol=sp.volume + lp.volume,
                                tfsa_legal=False,
                                evidence_badge=badge,
                                risk_flags=tuple(flags),
                                legs=bp_legs,
                                source=sl.source,
                                as_of=sl.snap_date,
                                summary=f"Sell {sp.strike:g}P / Buy {lp.strike:g}P · Cr ${cr_mid:.2f} · Min ${lad.p_walk:.2f}",
                            )
                        )

    # B) Bear Call Credit Spread (explicitly tagged CAUTION due to historical negative expectancy)
    if want_all or strategy_filter == "bear_call":
        sc = sl.match_by_delta("call", 0.25, min_bid=0.10)
        if sc is not None:
            width_tgt = max(1.0, spot * 0.05)
            lc = sl.match_contract("call", sc.strike + width_tgt, max_rel_dist=0.06, min_bid=0.02)
            if lc is not None and lc.strike > sc.strike:
                width = lc.strike - sc.strike
                cr_mid = sc.mid - lc.mid
                cr_cross = sc.bid - lc.ask
                if cr_mid > 0.10 and cr_cross > 0 and cr_mid < width:
                    rt_spread = (sc.ask - sc.bid) + (lc.ask - lc.bid)
                    fric = rt_spread / cr_mid
                    if fric <= max_friction_pct:
                        lad = build_walk_away_ladder(
                            structure="Bear Call",
                            is_credit=True,
                            p_mid=cr_mid,
                            p_cross=cr_cross,
                            ev_view_gross_per_share=0.0,
                        )
                        max_risk = max(1.0, (width - cr_mid) * 100.0)
                        max_prof = cr_mid * 100.0
                        flags = list(base_flags) + ["TAXABLE_ONLY", "HIST_WEAK_EDGE"]
                        bc_legs: Tuple[OptionLeg, ...] = (
                            _contract_to_leg(sc, side="SELL", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                            _contract_to_leg(lc, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                        )
                        out.append(
                            OptionScanRow(
                                ticker=sl.symbol,
                                strategy="Bear Call",
                                strategy_key="bear_call",
                                expiration=sl.expiration,
                                dte=sl.dte,
                                strikes_label=f"{sc.strike:g}/{lc.strike:g}C",
                                spot=round(spot, 2),
                                short_delta=round(abs(sc.delta or 0.25), 2),
                                mid_price=round(cr_mid, 2),
                                is_credit=True,
                                walk_away_limit=lad.p_walk,
                                friction_pct=round(fric, 4),
                                max_risk_usd=round(max_risk, 0),
                                max_profit_usd=round(max_prof, 0),
                                ror_pct=round(max_prof / max_risk * 100.0, 1),
                                iv_pct=iv_pct,
                                iv_vs_rv=iv_vs_rv,
                                skew_vp=skew,
                                min_oi=min(sc.open_interest, lc.open_interest),
                                total_vol=sc.volume + lc.volume,
                                tfsa_legal=False,
                                evidence_badge="CAUTION",
                                risk_flags=tuple(flags),
                                legs=bc_legs,
                                source=sl.source,
                                as_of=sl.snap_date,
                                summary=f"Sell {sc.strike:g}C / Buy {lc.strike:g}C · Cr ${cr_mid:.2f} · Caution",
                            )
                        )

    # C) Long Call (TFSA-legal single-leg debit; prefers liquid ~0.40-0.55Δ)
    if want_all or strategy_filter == "long_call":
        lc = sl.match_by_delta("call", 0.45, min_bid=0.15)
        if lc is not None and lc.mid > 0.15:
            fric = (lc.ask - lc.bid) / lc.mid
            if fric <= max_friction_pct:
                lad = build_walk_away_ladder(
                    structure="Long Call",
                    is_credit=False,
                    p_mid=lc.mid,
                    p_cross=lc.ask,
                    ev_view_gross_per_share=lc.mid * 1.15,
                    exit_half_spread_per_share=lc.half_spread,
                )
                flags = list(base_flags)
                if iv_vs_rv is not None and iv_vs_rv >= 1.25:
                    flags.append("IV_RICH_FOR_LONG")
                if fric > 0.12:
                    flags.append("WIDE_SPREAD")
                badge = "WATCHLIST"
                lc_legs: Tuple[OptionLeg, ...] = (_contract_to_leg(lc, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),)
                out.append(
                    OptionScanRow(
                        ticker=sl.symbol,
                        strategy="Long Call",
                        strategy_key="long_call",
                        expiration=sl.expiration,
                        dte=sl.dte,
                        strikes_label=f"${lc.strike:g}C",
                        spot=round(spot, 2),
                        short_delta=round(abs(lc.delta or 0.45), 2),
                        mid_price=round(lc.mid, 2),
                        is_credit=False,
                        walk_away_limit=lad.p_step25,
                        friction_pct=round(fric, 4),
                        max_risk_usd=round(lc.mid * 100.0, 0),
                        max_profit_usd=round(spot * 0.20 * 100.0, 0),
                        ror_pct=None,
                        iv_pct=round((lc.iv or atm_iv or 0.0) * 100.0, 1) if (lc.iv or atm_iv) else None,
                        iv_vs_rv=iv_vs_rv,
                        skew_vp=skew,
                        min_oi=lc.open_interest,
                        total_vol=lc.volume,
                        tfsa_legal=True,
                        evidence_badge=badge,
                        risk_flags=tuple(flags),
                        legs=lc_legs,
                        source=sl.source,
                        as_of=sl.snap_date,
                        summary=f"Buy {lc.strike:g}C (Δ {abs(lc.delta or 0):.2f}) · Mid ${lc.mid:.2f} · TFSA OK",
                    )
                )

    # D) Long Put (TFSA-legal hedge / directional)
    if want_all or strategy_filter == "long_put":
        lp = sl.match_by_delta("put", 0.40, min_bid=0.15)
        if lp is not None and lp.mid > 0.15:
            fric = (lp.ask - lp.bid) / lp.mid
            if fric <= max_friction_pct:
                lad = build_walk_away_ladder(
                    structure="Long Put",
                    is_credit=False,
                    p_mid=lp.mid,
                    p_cross=lp.ask,
                    ev_view_gross_per_share=lp.mid * 1.15,
                    exit_half_spread_per_share=lp.half_spread,
                )
                flags = list(base_flags)
                if iv_vs_rv is not None and iv_vs_rv >= 1.25:
                    flags.append("IV_RICH_FOR_LONG")
                lp_legs: Tuple[OptionLeg, ...] = (_contract_to_leg(lp, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),)
                out.append(
                    OptionScanRow(
                        ticker=sl.symbol,
                        strategy="Long Put",
                        strategy_key="long_put",
                        expiration=sl.expiration,
                        dte=sl.dte,
                        strikes_label=f"${lp.strike:g}P",
                        spot=round(spot, 2),
                        short_delta=round(abs(lp.delta or 0.40), 2),
                        mid_price=round(lp.mid, 2),
                        is_credit=False,
                        walk_away_limit=lad.p_step25,
                        friction_pct=round(fric, 4),
                        max_risk_usd=round(lp.mid * 100.0, 0),
                        max_profit_usd=round(max(0.0, lp.strike - lp.mid) * 100.0, 0),
                        ror_pct=None,
                        iv_pct=round((lp.iv or atm_iv or 0.0) * 100.0, 1) if (lp.iv or atm_iv) else None,
                        iv_vs_rv=iv_vs_rv,
                        skew_vp=skew,
                        min_oi=lp.open_interest,
                        total_vol=lp.volume,
                        tfsa_legal=True,
                        evidence_badge="WATCHLIST",
                        risk_flags=tuple(flags),
                        legs=lp_legs,
                        source=sl.source,
                        as_of=sl.snap_date,
                        summary=f"Buy {lp.strike:g}P (Δ -{abs(lp.delta or 0):.2f}) · Mid ${lp.mid:.2f} · TFSA OK",
                    )
                )

    # E) Iron Condor (4-leg defined risk — tagged CAUTION per historical book record)
    if strategy_filter == "iron_condor" or (want_all and sl.dte >= 25):
        sp = sl.match_by_delta("put", 0.20, min_bid=0.08)
        sc = sl.match_by_delta("call", 0.20, min_bid=0.08)
        if sp is not None and sc is not None and sp.strike < spot < sc.strike:
            w_tgt = max(1.0, spot * 0.04)
            lp = sl.match_contract("put", sp.strike - w_tgt, max_rel_dist=0.05, min_bid=0.01)
            lc = sl.match_contract("call", sc.strike + w_tgt, max_rel_dist=0.05, min_bid=0.01)
            if lp is not None and lc is not None and lp.strike < sp.strike and lc.strike > sc.strike:
                cr_mid = (sp.mid - lp.mid) + (sc.mid - lc.mid)
                cr_cross = (sp.bid - lp.ask) + (sc.bid - lc.ask)
                w_max = max(sp.strike - lp.strike, lc.strike - sc.strike)
                if cr_mid > 0.15 and cr_cross > 0 and cr_mid < w_max:
                    rt_spr = (sp.ask - sp.bid) + (lp.ask - lp.bid) + (sc.ask - sc.bid) + (lc.ask - lc.bid)
                    fric = rt_spr / cr_mid
                    if fric <= max_friction_pct:
                        lad = build_walk_away_ladder(
                            structure="Iron Condor",
                            is_credit=True,
                            p_mid=cr_mid,
                            p_cross=cr_cross,
                            ev_view_gross_per_share=0.0,
                        )
                        max_risk = max(1.0, (w_max - cr_mid) * 100.0)
                        max_prof = cr_mid * 100.0
                        flags = list(base_flags) + ["TAXABLE_ONLY", "HIST_WEAK_EDGE"]
                        ic_legs: Tuple[OptionLeg, ...] = (
                            _contract_to_leg(lp, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                            _contract_to_leg(sp, side="SELL", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                            _contract_to_leg(sc, side="SELL", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                            _contract_to_leg(lc, side="BUY", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),
                        )
                        out.append(
                            OptionScanRow(
                                ticker=sl.symbol,
                                strategy="Iron Condor",
                                strategy_key="iron_condor",
                                expiration=sl.expiration,
                                dte=sl.dte,
                                strikes_label=f"{lp.strike:g}/{sp.strike:g}P·{sc.strike:g}/{lc.strike:g}C",
                                spot=round(spot, 2),
                                short_delta=round(max(abs(sp.delta or 0.2), abs(sc.delta or 0.2)), 2),
                                mid_price=round(cr_mid, 2),
                                is_credit=True,
                                walk_away_limit=lad.p_walk,
                                friction_pct=round(fric, 4),
                                max_risk_usd=round(max_risk, 0),
                                max_profit_usd=round(max_prof, 0),
                                ror_pct=round(max_prof / max_risk * 100.0, 1),
                                iv_pct=iv_pct,
                                iv_vs_rv=iv_vs_rv,
                                skew_vp=skew,
                                min_oi=min(lp.open_interest, sp.open_interest, sc.open_interest, lc.open_interest),
                                total_vol=lp.volume + sp.volume + sc.volume + lc.volume,
                                tfsa_legal=False,
                                evidence_badge="CAUTION",
                                risk_flags=tuple(flags),
                                legs=ic_legs,
                                source=sl.source,
                                as_of=sl.snap_date,
                                summary=f"IC {lp.strike:g}/{sp.strike:g}-{sc.strike:g}/{lc.strike:g} · Cr ${cr_mid:.2f}",
                            )
                        )

    # F) Short Put (Cash-secured / taxable single-leg)
    if strategy_filter == "short_put":
        sp = sl.match_by_delta("put", 0.25, min_bid=0.15)
        if sp is not None and sp.mid > 0.15:
            fric = (sp.ask - sp.bid) / sp.mid
            if fric <= max_friction_pct:
                lad = build_walk_away_ladder(
                    structure="Short Put",
                    is_credit=True,
                    p_mid=sp.mid,
                    p_cross=sp.bid,
                    ev_view_gross_per_share=0.0,
                )
                max_risk = max(1.0, (sp.strike - sp.mid) * 100.0)
                max_prof = sp.mid * 100.0
                flags = list(base_flags) + ["TAXABLE_ONLY", "CSP_CASH_REQ"]
                sp_legs: Tuple[OptionLeg, ...] = (_contract_to_leg(sp, side="SELL", expiration=sl.expiration, dte=sl.dte, fallback_iv=atm_iv or 0.25),)
                out.append(
                    OptionScanRow(
                        ticker=sl.symbol,
                        strategy="Short Put",
                        strategy_key="short_put",
                        expiration=sl.expiration,
                        dte=sl.dte,
                        strikes_label=f"-${sp.strike:g}P",
                        spot=round(spot, 2),
                        short_delta=round(abs(sp.delta or 0.25), 2),
                        mid_price=round(sp.mid, 2),
                        is_credit=True,
                        walk_away_limit=lad.p_walk,
                        friction_pct=round(fric, 4),
                        max_risk_usd=round(max_risk, 0),
                        max_profit_usd=round(max_prof, 0),
                        ror_pct=round(max_prof / max_risk * 100.0, 1),
                        iv_pct=iv_pct,
                        iv_vs_rv=iv_vs_rv,
                        skew_vp=skew,
                        min_oi=sp.open_interest,
                        total_vol=sp.volume,
                        tfsa_legal=False,
                        evidence_badge="WATCHLIST",
                        risk_flags=tuple(flags),
                        legs=sp_legs,
                        source=sl.source,
                        as_of=sl.snap_date,
                        summary=f"Sell {sp.strike:g}P (CSP) · Cr ${sp.mid:.2f} · ROR {max_prof/max_risk*100:.1f}%",
                    )
                )

    return out


def scan_options_desk(
    *,
    strategy: str = "all",
    symbols: Optional[Sequence[str]] = None,
    chain_db: str = "data/chain_archive.db",
    ohlcv_db: str = "data/equity_ohlcv.db",
    target_dte: int = 45,
    min_dte: int = 14,
    max_dte: int = 125,
    live: bool = False,
    max_friction_pct: float = 0.28,
) -> Tuple[List[OptionScanRow], str]:
    """Scan local `chain_archive.db` (instant) or live Yahoo chains into `OptionScanRow`s.

    Returns `(rows, status_note)`.
    """
    strat = (strategy or "all").lower().strip()
    earnings = _load_earnings_map()
    rows: List[OptionScanRow] = []

    # Calendar spreads handled via the dedicated term-structure scanner
    if strat in ("all", "calendar") and not live:
        try:
            from src.radar.calendar_spreads import scan_etf_calendar_spreads

            cals = scan_etf_calendar_spreads(chain_db=chain_db, max_friction_pct=max_friction_pct)
            for c in cals:
                iv_f = c.front_iv_pct / 100.0
                iv_b = c.back_iv_pct / 100.0
                legs = (
                    OptionLeg(c.opt_type, "SELL", c.strike, c.front_exp, c.front_dte, 0.0, 0.0, 0.0, iv_f),
                    OptionLeg(c.opt_type, "BUY", c.strike, c.back_exp, c.back_dte, c.net_debit_mid, c.net_debit_cross, c.net_debit_mid, iv_b),
                )
                rows.append(
                    OptionScanRow(
                        ticker=c.ticker,
                        strategy="Calendar",
                        strategy_key="calendar",
                        expiration=f"{c.front_exp[5:]}/{c.back_exp[5:]}",
                        dte=c.front_dte,
                        strikes_label=f"${c.strike:g}{c.opt_type[0].upper()} {c.front_dte}d/{c.back_dte}d",
                        spot=c.spot,
                        short_delta=0.50,
                        mid_price=c.net_debit_mid,
                        is_credit=False,
                        walk_away_limit=c.net_debit_mid,
                        friction_pct=round(c.friction_pct, 4),
                        max_risk_usd=round(c.net_debit_mid * 100.0, 0),
                        max_profit_usd=round(c.net_debit_mid * 65.0, 0),
                        ror_pct=65.0,
                        iv_pct=c.front_iv_pct,
                        iv_vs_rv=None,
                        skew_vp=round(c.iv_diff_vp, 1),
                        min_oi=0,
                        total_vol=0,
                        tfsa_legal=False,
                        evidence_badge="WATCHLIST",
                        risk_flags=("TAXABLE_ONLY", c.term_regime),
                        legs=legs,
                        source="chain_archive",
                        as_of="",
                        summary=f"Sell {c.front_dte}d / Buy {c.back_dte}d ${c.strike:g}{c.opt_type[0].upper()} · Front +{c.iv_diff_vp:.1f}vp",
                    )
                )
        except Exception:
            pass
        if strat == "calendar":
            rows.sort(key=lambda r: (r.friction_pct, -(r.skew_vp or 0.0), r.ticker))
            return rows, f"Archive calendars: {len(rows)} found"

    # Determine symbols & latest spots
    if live:
        scan_syms = [s.upper() for s in (symbols or DEFAULT_LIVE_SCAN_UNIVERSE[:10])]
        rv_map = _load_rv20_map(ohlcv_db, scan_syms)
        eff_target = 90 if strat == "bull_put" else target_dte
        for tk in scan_syms:
            sl = load_live_chain_slice(
                tk,
                target_dte=eff_target,
                min_dte=min_dte,
                max_dte=max_dte,
                build_rnd=False,
            )
            if sl is None:
                continue
            # Estimate spot from near-ATM call/put strikes if not carried on slice
            all_k = [c.strike for c in sl.calls + sl.puts if c.delta is not None]
            spot_est = 0.0
            if sl.calls:
                atm_c = min(sl.calls, key=lambda c: abs(abs(c.delta or 0.5) - 0.5))
                spot_est = atm_c.strike
            elif all_k:
                spot_est = float(np.median(all_k))
            if spot_est <= 0:
                continue
            rows.extend(
                _build_candidates_for_slice(
                    sl,
                    spot_est,
                    rv20=rv_map.get(tk),
                    earnings_days=earnings.get(tk),
                    strategy_filter=strat,
                    max_friction_pct=max_friction_pct,
                )
            )
        note = f"Live Yahoo scan: {len(rows)} candidates across {len(scan_syms)} tickers"
    else:
        full = repo_path(chain_db)
        if not os.path.exists(full):
            return rows, "No chain_archive.db found — click Live Scan or run `opt alpha archive-chains`"
        spots: Dict[str, float] = {}
        try:
            con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "chain_snapshots" not in tables:
                    return rows, "chain_archive.db empty"
                if symbols:
                    sym_list = [s.upper() for s in symbols]
                else:
                    sym_list = [
                        str(r[0]).upper()
                        for r in con.execute("SELECT DISTINCT symbol FROM chain_snapshots ORDER BY symbol").fetchall()
                        if r and r[0]
                    ]
                for tk in sym_list:
                    r_sp = con.execute(
                        "SELECT spot FROM chain_snapshots WHERE symbol = ? AND spot > 0 ORDER BY snap_date DESC LIMIT 1",
                        (tk,),
                    ).fetchone()
                    if r_sp and r_sp[0]:
                        spots[tk] = float(r_sp[0])
            finally:
                con.close()
        except Exception as exc:
            return rows, f"Archive read err: {exc}"

        rv_map = _load_rv20_map(ohlcv_db, list(spots.keys()))
        # Scan both a 35d swing slice and a 90d O1 wide-spread slice when strat is all/bull_put
        dte_targets = [(90, 60, 125)] if strat == "bull_put" else (
            [(35, 14, 58), (90, 60, 125)] if strat == "all" else [(target_dte, min_dte, max_dte)]
        )
        seen_keys: set[Tuple[str, str, str, str]] = set()
        for tk, sp in spots.items():
            for tgt, lo_d, hi_d in dte_targets:
                sl = load_archived_chain_slice(
                    tk,
                    target_dte=tgt,
                    min_dte=lo_d,
                    max_dte=hi_d,
                    spot=sp,
                    chain_db=chain_db,
                    build_rnd=False,
                )
                if sl is None:
                    continue
                for cand in _build_candidates_for_slice(
                    sl,
                    sp,
                    rv20=rv_map.get(tk),
                    earnings_days=earnings.get(tk),
                    strategy_filter=strat,
                    max_friction_pct=max_friction_pct,
                ):
                    key = (cand.ticker, cand.strategy_key, cand.expiration, cand.strikes_label)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        rows.append(cand)
        note = f"Archive scan: {len(rows)} candidates across {len(spots)} archived chains"

    # Evidence-first deterministic ordering: PROMISING > WATCHLIST > CAUTION, then low friction, then liquidity
    badge_rank = {"PROMISING": 0, "WATCHLIST": 1, "CAUTION": 2, "EXPERIMENTAL": 3}
    rows.sort(
        key=lambda r: (
            badge_rank.get(r.evidence_badge, 9),
            len([f for f in r.risk_flags if f.startswith("EARN_") or f == "HIST_WEAK_EDGE"]),
            r.friction_pct,
            -r.min_oi,
            r.ticker,
        )
    )
    return rows, note


def filter_option_rows(rows: Sequence[OptionScanRow], query: str) -> List[OptionScanRow]:
    """Composable micro-query filter for the Options Scanner.

    Supported tokens (space-separated, AND-ed):
      - Strategy: `bull_put`, `bear_call`, `long_call`, `long_put`, `condor`, `calendar`, `short_put`, `credit`, `debit`
      - Account/Clean: `tfsa`, `taxable`, `clean`, `promising`, `iv_cheap`, `iv_rich`
      - Numeric bounds: `dte>=30`, `dte<=90`, `fric<15%`, `risk<500`, `delta<0.35`, `oi>=500`
      - Ticker/text substring: `spy`, `nvda`, `2026-12`
    """
    q = (query or "").strip().lower()
    if not q:
        return list(rows)
    tokens = q.split()
    out: List[OptionScanRow] = []
    for r in rows:
        ok = True
        for tok in tokens:
            if tok in ("bull_put", "bear_call", "long_call", "long_put", "iron_condor", "calendar", "short_put"):
                ok = r.strategy_key == tok
            elif tok == "condor":
                ok = r.strategy_key == "iron_condor"
            elif tok == "credit":
                ok = r.is_credit
            elif tok == "debit":
                ok = not r.is_credit
            elif tok == "tfsa":
                ok = r.tfsa_legal
            elif tok == "taxable":
                ok = not r.tfsa_legal
            elif tok == "clean":
                ok = not any(f.startswith("EARN_") or f in ("WIDE_SPREAD", "HIST_WEAK_EDGE") for f in r.risk_flags)
            elif tok == "promising":
                ok = r.evidence_badge == "PROMISING"
            elif tok == "iv_cheap":
                ok = r.iv_vs_rv is not None and r.iv_vs_rv <= 0.95
            elif tok == "iv_rich":
                ok = r.iv_vs_rv is not None and r.iv_vs_rv >= 1.15
            elif tok.startswith("dte>="):
                try:
                    ok = r.dte >= int(tok[5:])
                except ValueError:
                    ok = False
            elif tok.startswith("dte<="):
                try:
                    ok = r.dte <= int(tok[5:])
                except ValueError:
                    ok = False
            elif tok.startswith("fric<"):
                try:
                    lim = float(tok[5:].rstrip("%")) / (100.0 if "%" in tok else 1.0)
                    ok = r.friction_pct <= lim
                except ValueError:
                    ok = False
            elif tok.startswith("risk<"):
                try:
                    ok = r.max_risk_usd <= float(tok[5:].lstrip("$"))
                except ValueError:
                    ok = False
            elif tok.startswith("delta<"):
                try:
                    ok = r.short_delta is not None and r.short_delta <= float(tok[6:])
                except ValueError:
                    ok = False
            elif tok.startswith("oi>="):
                try:
                    ok = r.min_oi >= int(tok[4:])
                except ValueError:
                    ok = False
            else:
                hay = f"{r.ticker} {r.strategy} {r.expiration} {r.strikes_label} {r.summary} {' '.join(r.risk_flags)}".lower()
                ok = tok in hay
            if not ok:
                break
        if ok:
            out.append(r)
    return out


# ── 2. Interactive T-Shaped Chain Matrix Loader ──────────────────────────────


def load_chain_matrix(
    ticker: str,
    *,
    expiration: Optional[str] = None,
    target_dte: int = 35,
    band_pct: float = 0.15,
    chain_db: str = "data/chain_archive.db",
    ohlcv_db: str = "data/equity_ohlcv.db",
    allow_live: bool = False,
) -> Optional[ChainMatrix]:
    """Load a full T-shaped Call/Strike/Put Chain Matrix + GEX + IV Smile + Term Structure."""
    sym = str(ticker or "SPY").upper().strip()
    rv_map = _load_rv20_map(ohlcv_db, [sym])
    rv20 = rv_map.get(sym)
    earnings = _load_earnings_map().get(sym)

    avail_exps: List[Tuple[str, int]] = []
    term_points: List[Tuple[int, float]] = []
    gex_rows_raw: List[Dict[str, Any]] = []
    spot: Optional[float] = None
    sl: Optional[ChainSlice] = None

    if not allow_live:
        full = repo_path(chain_db)
        if os.path.exists(full):
            try:
                con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
                try:
                    r_dt = con.execute(
                        "SELECT MAX(snap_date) FROM chain_snapshots WHERE symbol = ?",
                        (sym,),
                    ).fetchone()
                    if r_dt and r_dt[0]:
                        snap_date = str(r_dt[0])
                        snap_dt = dt.date.fromisoformat(snap_date[:10])
                        r_sp = con.execute(
                            "SELECT spot FROM chain_snapshots WHERE symbol = ? AND snap_date = ? AND spot > 0 LIMIT 1",
                            (sym, snap_date),
                        ).fetchone()
                        if r_sp and r_sp[0]:
                            spot = float(r_sp[0])
                        exp_rows = con.execute(
                            "SELECT DISTINCT expiration FROM chain_snapshots WHERE symbol = ? AND snap_date = ? ORDER BY expiration",
                            (sym, snap_date),
                        ).fetchall()
                        for (e_str,) in exp_rows:
                            if not e_str:
                                continue
                            try:
                                dte_i = (dt.date.fromisoformat(str(e_str)[:10]) - snap_dt).days
                            except ValueError:
                                continue
                            if dte_i > 0:
                                avail_exps.append((str(e_str)[:10], dte_i))

                        # Pull all rows on snap_date for GEX & Term Structure
                        all_snap = con.execute(
                            "SELECT strike, type, expiration, bid, ask, iv, delta, gamma, theta, open_interest, volume "
                            "FROM chain_snapshots WHERE symbol = ? AND snap_date = ?",
                            (sym, snap_date),
                        ).fetchall()
                        iv_by_exp: Dict[str, List[Tuple[float, float]]] = {}
                        for k, tp, exp, b, a, iv_v, dl, gm, th, oi, vol in all_snap:
                            kf = _clean_float(k)
                            ivf = _clean_float(iv_v)
                            if ivf is not None and ivf > 5.0:
                                ivf = ivf / 100.0
                            gex_rows_raw.append(
                                {
                                    "strike": kf,
                                    "type": tp,
                                    "gamma": _clean_float(gm),
                                    "open_interest": _clean_float(oi),
                                }
                            )
                            if spot and kf and ivf and 0.02 <= ivf <= 4.0:
                                iv_by_exp.setdefault(str(exp)[:10], []).append((abs(kf - spot), ivf))
                        for exp_s, dte_i in avail_exps:
                            pts = iv_by_exp.get(exp_s) or []
                            if pts:
                                pts.sort(key=lambda x: x[0])
                                med_iv = float(np.median([v for _, v in pts[:4]])) * 100.0
                                term_points.append((dte_i, round(med_iv, 1)))
                finally:
                    con.close()
            except Exception:
                pass

        if avail_exps:
            chosen_exp = expiration if (expiration and any(e == expiration for e, _ in avail_exps)) else None
            if chosen_exp is None:
                chosen_exp = min(avail_exps, key=lambda x: abs(x[1] - target_dte))[0]
            chosen_dte = dict(avail_exps)[chosen_exp]
            sl = load_archived_chain_slice(
                sym,
                target_dte=chosen_dte,
                min_dte=max(1, chosen_dte - 1),
                max_dte=chosen_dte + 1,
                spot=spot,
                chain_db=chain_db,
                max_snap_age_days=45,
                build_rnd=False,
            )

    if sl is None and allow_live:
        try:
            import yfinance as yf

            tkr = yf.Ticker(sym)
            exps_raw = list(tkr.options or ())
            today = dt.date.today()
            for e in exps_raw:
                try:
                    d = (dt.date.fromisoformat(e[:10]) - today).days
                    if d > 0:
                        avail_exps.append((e[:10], d))
                except ValueError:
                    continue
            hist = tkr.history(period="5d")
            if hist is not None and not hist.empty:
                spot = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass
        if avail_exps:
            chosen_exp = expiration if (expiration and any(e == expiration for e, _ in avail_exps)) else None
            if chosen_exp is None:
                chosen_exp = min(avail_exps, key=lambda x: abs(x[1] - target_dte))[0]
            chosen_dte = dict(avail_exps)[chosen_exp]
            sl = load_live_chain_slice(
                sym,
                target_dte=chosen_dte,
                min_dte=max(1, chosen_dte - 1),
                max_dte=chosen_dte + 1,
                spot=spot,
                build_rnd=False,
            )
            if sl is not None and spot and spot > 0:
                T = max(sl.dte, 1) / 365.0
                for c in sl.calls + sl.puts:
                    if c.iv and c.open_interest:
                        gm = float(bs_gamma(spot, c.strike, T, 0.045, c.iv))
                        gex_rows_raw.append(
                            {
                                "strike": c.strike,
                                "type": c.opt_type,
                                "gamma": gm,
                                "open_interest": float(c.open_interest),
                            }
                        )
                if sl.atm_iv is not None:
                    term_points = [(sl.dte, round(sl.atm_iv * 100.0, 1))]

    if sl is None:
        return None

    if spot is None or spot <= 0:
        ks = [c.strike for c in sl.calls + sl.puts]
        spot = float(np.median(ks)) if ks else 100.0

    # Dealer GEX & OI walls
    from src.alpha.options_for_shares import compute_dealer_gex

    gex_info = compute_dealer_gex(gex_rows_raw, spot) if gex_rows_raw else {}
    gex_by_k = {float(d["strike"]): float(d["gex_m"]) for d in (gex_info.get("by_strike") or [])}
    call_wall = gex_info.get("call_wall")
    put_wall = gex_info.get("put_wall")
    gamma_flip = gex_info.get("gamma_flip")

    # Expected 1-sigma move for this expiry
    em_usd = em_pct = None
    if sl.atm_iv is not None and sl.dte > 0:
        em_frac = sl.atm_iv * math.sqrt(sl.dte / 365.0)
        em_usd = round(spot * em_frac, 2)
        em_pct = round(em_frac * 100.0, 1)

    # Pair Calls & Puts by Strike
    calls_by_k = {c.strike: c for c in sl.calls}
    puts_by_k = {p.strike: p for p in sl.puts}
    all_strikes = sorted(set(calls_by_k.keys()) | set(puts_by_k.keys()))
    if band_pct > 0 and spot > 0:
        filtered_k = [k for k in all_strikes if abs(k / spot - 1.0) <= band_pct]
        if len(filtered_k) >= 5:
            all_strikes = filtered_k

    # Closest strike to spot / expected move bounds / gamma flip
    spot_k = min(all_strikes, key=lambda k: abs(k - spot)) if all_strikes else None
    flip_k = min(all_strikes, key=lambda k: abs(k - float(gamma_flip))) if (all_strikes and gamma_flip is not None) else None
    em_hi_k = min(all_strikes, key=lambda k: abs(k - (spot + em_usd))) if (all_strikes and em_usd) else None
    em_lo_k = min(all_strikes, key=lambda k: abs(k - (spot - em_usd))) if (all_strikes and em_usd) else None

    pair_rows: List[ChainPairRow] = []
    smile_pts: List[Tuple[float, float]] = []
    for k in all_strikes:
        call_c: Optional[ChainContract] = calls_by_k.get(k)
        put_c: Optional[ChainContract] = puts_by_k.get(k)
        tags: List[str] = []
        if k == spot_k:
            tags.append("SPOT")
        if call_wall is not None and abs(k - float(call_wall)) < 1e-6:
            tags.append("C-WALL")
        if put_wall is not None and abs(k - float(put_wall)) < 1e-6:
            tags.append("P-WALL")
        if flip_k is not None and k == flip_k:
            tags.append("FLIP")
        if em_hi_k is not None and k == em_hi_k:
            tags.append("+1σ")
        if em_lo_k is not None and k == em_lo_k:
            tags.append("-1σ")
        pair_rows.append(
            ChainPairRow(
                strike=k,
                call=call_c,
                put=put_c,
                gex_m=round(gex_by_k.get(k, 0.0), 2),
                tags=tuple(tags),
            )
        )
        # OTM smile IV
        otm_c = put_c if k < spot else call_c
        if otm_c is not None and otm_c.iv is not None and 0.02 <= otm_c.iv <= 3.5:
            smile_pts.append((k, round(otm_c.iv * 100.0, 1)))

    atm_iv_pct = round(sl.atm_iv * 100.0, 1) if sl.atm_iv is not None else None
    rv20_pct = round(rv20 * 100.0, 1) if rv20 is not None else None
    iv_vs_rv = round(sl.atm_iv / rv20, 2) if (sl.atm_iv is not None and rv20 is not None and rv20 > 0.02) else None

    return ChainMatrix(
        ticker=sym,
        spot=round(spot, 2),
        expiration=sl.expiration,
        dte=sl.dte,
        available_expirations=avail_exps or [(sl.expiration, sl.dte)],
        rows=pair_rows,
        source=sl.source,
        snap_date=sl.snap_date,
        atm_iv_pct=atm_iv_pct,
        rv20_pct=rv20_pct,
        iv_vs_rv=iv_vs_rv,
        skew_25d_vp=sl.skew_25d_vp,
        expected_move_usd=em_usd,
        expected_move_pct=em_pct,
        median_spread_pct=round(sl.median_half_spread_pct * 200.0, 1) if sl.median_half_spread_pct is not None else None,
        net_gex_m=gex_info.get("net_gex_m"),
        gex_regime=str(gex_info.get("gex_regime") or "UNKNOWN"),
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        gex_by_strike=list(gex_info.get("by_strike") or []),
        smile_points=smile_pts,
        term_points=sorted(term_points, key=lambda x: x[0]),
        earnings_days=earnings,
    )


# ── 3. Multi-Leg Pricer, Greeks & 2D Risk Surface ────────────────────────────


def classify_legs(legs: Sequence[OptionLeg]) -> Tuple[str, bool]:
    """Return `(structure_name, tfsa_legal)` from 1-4 staged option legs."""
    n = len(legs)
    if n == 0:
        return "Empty", True
    if n == 1:
        lg = legs[0]
        if lg.side.upper() == "BUY":
            return (f"Long {'Call' if lg.opt_type == 'call' else 'Put'}", True)
        return (f"Short {'Call' if lg.opt_type == 'call' else 'Put'}", False)
    if n == 2:
        a, b = legs[0], legs[1]
        if a.expiration != b.expiration and a.opt_type == b.opt_type:
            return ("Calendar / Diagonal", False)
        if a.opt_type == b.opt_type and a.side != b.side:
            short_l = a if a.side == "SELL" else b
            long_l = b if a.side == "SELL" else a
            if a.opt_type == "put":
                return ("Bull Put" if short_l.strike > long_l.strike else "Bear Put Debit", False)
            return ("Bear Call" if short_l.strike < long_l.strike else "Bull Call Debit", False)
        if a.side == "BUY" and b.side == "BUY" and a.opt_type != b.opt_type:
            return ("Long Straddle" if abs(a.strike - b.strike) < 1e-6 else "Long Strangle", True)
        return ("2-Leg Custom", False)
    if n == 4:
        return ("Iron Condor / 4-Leg", False)
    return (f"{n}-Leg Custom", False)


def analyze_custom_legs(
    ticker: str,
    spot: float,
    legs: Sequence[OptionLeg],
    *,
    days_elapsed: int = 0,
    iv_shock_pct: float = 0.0,
    r: float = 0.045,
) -> MultiLegAnalysis:
    """Compute exact Greeks, friction, walk-away ladder, payoff curves, and 2D risk grid for 1-4 legs."""
    leg_list = list(legs)
    struct_name, tfsa_legal = classify_legs(leg_list)

    # Net mid & net cross from trader's perspective:
    # signed_qty > 0 => BUY (pays ask at cross, mid at mid); < 0 => SELL (gets bid at cross, mid at mid)
    net_debit_mid = sum(lg.signed_qty * lg.mid for lg in leg_list)
    net_debit_cross = sum(
        (lg.qty * lg.ask if lg.signed_qty > 0 else -lg.qty * lg.bid)
        for lg in leg_list
    )
    is_credit = net_debit_mid < 0
    net_mid = round(abs(net_debit_mid), 2)
    net_cross = round(max(0.01, -net_debit_cross) if is_credit else max(0.01, net_debit_cross), 2)
    rt_spread = sum(lg.qty * max(0.0, lg.ask - lg.bid) for lg in leg_list)
    fric = rt_spread / max(net_mid, 0.05)

    # Greeks at spot
    d_tot = g_tot = th_tot = v_tot = 0.0
    min_dte = max(1, min((lg.dte for lg in leg_list), default=30))
    for lg in leg_list:
        T = max(lg.dte, 1) / 365.0
        sig = max(0.05, lg.iv or 0.25)
        sq = lg.signed_qty * 100.0
        d_tot += sq * float(bs_delta(lg.opt_type, spot, lg.strike, T, r, sig))
        g_tot += sq * float(bs_gamma(spot, lg.strike, T, r, sig))
        th_tot += sq * float(bs_theta(lg.opt_type, spot, lg.strike, T, r, sig))
        v_tot += sq * float(bs_vega(spot, lg.strike, T, r, sig))

    # Price grid (±22% around spot) for expiry and shocked T+days curve
    grid = np.linspace(spot * 0.78, spot * 1.22, 45)
    exp_pnl = np.zeros_like(grid)
    shk_pnl = np.zeros_like(grid)
    iv_mult = max(0.25, 1.0 + iv_shock_pct / 100.0)
    eff_elapsed = max(0, min(days_elapsed, max(0, min_dte - 1)))

    for lg in leg_list:
        sq = lg.signed_qty * 100.0
        sig = max(0.05, lg.iv or 0.25)
        # Value at earliest leg's expiry
        rem_exp = max(0, lg.dte - min_dte) / 365.0
        if rem_exp <= 1e-5:
            val_exp = np.maximum(grid - lg.strike, 0.0) if lg.opt_type == "call" else np.maximum(lg.strike - grid, 0.0)
        else:
            val_exp = np.asarray(bs_price(lg.opt_type, grid, lg.strike, rem_exp, r, sig), dtype=float)
        exp_pnl += sq * (val_exp - lg.mid)

        # Value at T + days_elapsed with IV shock
        rem_t = max(1, lg.dte - eff_elapsed) / 365.0
        val_shk = np.asarray(bs_price(lg.opt_type, grid, lg.strike, rem_t, r, sig * iv_mult), dtype=float)
        shk_pnl += sq * (val_shk - lg.mid)

    # Breakevens where expiry P&L crosses zero
    breakevens: List[float] = []
    for i in range(1, len(grid)):
        y0, y1 = float(exp_pnl[i - 1]), float(exp_pnl[i])
        if (y0 < 0 <= y1) or (y0 > 0 >= y1):
            frac = abs(y0) / max(abs(y1 - y0), 1e-9)
            breakevens.append(round(float(grid[i - 1] + frac * (grid[i] - grid[i - 1])), 2))

    # Max risk & max profit over wide domain
    wide = np.linspace(max(0.01, spot * 0.25), spot * 2.25, 160)
    wide_pnl = np.zeros_like(wide)
    for lg in leg_list:
        sq = lg.signed_qty * 100.0
        rem_exp = max(0, lg.dte - min_dte) / 365.0
        if rem_exp <= 1e-5:
            val = np.maximum(wide - lg.strike, 0.0) if lg.opt_type == "call" else np.maximum(lg.strike - wide, 0.0)
        else:
            val = np.asarray(bs_price(lg.opt_type, wide, lg.strike, rem_exp, r, max(0.05, lg.iv or 0.25)), dtype=float)
        wide_pnl += sq * (val - lg.mid)

    min_pnl = float(np.min(wide_pnl))
    max_pnl = float(np.max(wide_pnl))
    max_risk_usd = round(max(1.0, -min_pnl), 0)
    has_unbounded_upside = sum(lg.signed_qty for lg in leg_list if lg.opt_type == "call") > 0
    max_profit_usd = None if has_unbounded_upside else round(max(0.0, max_pnl), 0)

    # Lognormal market-implied PoP at expiry
    avg_iv = float(np.mean([max(0.05, lg.iv or 0.25) for lg in leg_list])) if leg_list else 0.25
    T_exp = min_dte / 365.0
    sd = max(avg_iv * math.sqrt(T_exp), 0.02)
    mu = math.log(spot) + (r - 0.5 * avg_iv * avg_iv) * T_exp
    pdf = np.exp(-0.5 * ((np.log(grid) - mu) / sd) ** 2) / (grid * sd * math.sqrt(2.0 * math.pi))
    weights = pdf / max(float(np.sum(pdf)), 1e-12)
    pop_mkt = float(np.sum(weights[exp_pnl > 0]))

    # 2D Price x IV Shock P&L grid (9 price cols x 5 IV rows, fast vectorized)
    p_shocks = [-0.15, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.15]
    iv_shocks = [0.30, 0.15, 0.0, -0.15, -0.30]  # top=IV up, bottom=IV crush
    S_vec = spot * (1.0 + np.array(p_shocks, dtype=float))
    grid_2d: List[List[float]] = []
    for iv_s in iv_shocks:
        row_pnl = np.zeros_like(S_vec)
        mult = max(0.25, 1.0 + iv_s)
        for lg in leg_list:
            sq = lg.signed_qty * 100.0
            rem_t = max(1, lg.dte - eff_elapsed) / 365.0
            sig = max(0.05, (lg.iv or 0.25) * mult)
            val = np.asarray(bs_price(lg.opt_type, S_vec, lg.strike, rem_t, r, sig), dtype=float)
            row_pnl += sq * (val - lg.mid)
        grid_2d.append([round(float(x), 0) for x in row_pnl])

    ladder = build_walk_away_ladder(
        structure=struct_name,
        is_credit=is_credit,
        p_mid=net_mid,
        p_cross=net_cross,
        ev_view_gross_per_share=net_mid * (1.0 if is_credit else 1.10),
        exit_half_spread_per_share=rt_spread * 0.5,
    )

    return MultiLegAnalysis(
        ticker=ticker.upper(),
        spot=round(spot, 2),
        structure_name=struct_name,
        legs=leg_list,
        is_credit=is_credit,
        tfsa_legal=tfsa_legal,
        net_mid=net_mid,
        net_cross=net_cross,
        friction_pct=round(fric, 4),
        max_risk_usd=max_risk_usd,
        max_profit_usd=max_profit_usd,
        breakevens=breakevens,
        delta_dollars=round(d_tot * spot / 100.0, 0),
        gamma_dollars=round(g_tot * spot * spot * 0.0001, 1),
        theta_per_day_usd=round(th_tot, 2),
        vega_per_1pct_usd=round(v_tot, 2),
        pop_mkt=round(pop_mkt, 3),
        ladder=ladder,
        prices=[round(float(x), 2) for x in grid],
        expiry_pnl=[round(float(x), 1) for x in exp_pnl],
        shocked_pnl=[round(float(x), 1) for x in shk_pnl],
        days_elapsed=eff_elapsed,
        iv_shock_pct=round(iv_shock_pct, 1),
        grid_price_shocks=p_shocks,
        grid_iv_shocks=iv_shocks,
        grid_pnl=grid_2d,
    )


# ── 4. Calibrated PoP Lookup (Shipped Model Only — Never Fabricated) ─────────

_CAL_POP_CACHE: Dict[str, Tuple[Optional[Any], Optional[Any]]] = {}


def describe_cal_pop_for_row(row: OptionScanRow) -> Optional[str]:
    """Return canonical `CalPoP` description if a shipped reliability-checked model exists, else None."""
    try:
        import json
        import pandas as pd
        from src.pop_calibration import DEFAULT_MODEL_PATH, describe_row, load_model

        full = repo_path(DEFAULT_MODEL_PATH)
        if not os.path.exists(full):
            return None
        ck = f"{full}:{os.path.getmtime(full):.0f}"
        if ck not in _CAL_POP_CACHE:
            model = load_model(full)
            rel_df = None
            if model is not None:
                with open(full, encoding="utf-8") as fh:
                    raw_rel = (json.load(fh) or {}).get("reliability") or []
                if raw_rel:
                    rel_df = pd.DataFrame(raw_rel)
            _CAL_POP_CACHE.clear()
            _CAL_POP_CACHE[ck] = (model, rel_df)
        model, rel_df = _CAL_POP_CACHE[ck]
        if model is None or rel_df is None or len(rel_df) == 0:
            return None
        width = 0.0
        if len(row.legs) >= 2:
            strikes = sorted(lg.strike for lg in row.legs)
            width = max(0.0, strikes[-1] - strikes[0])
        payload = {
            "strategy_name": row.strategy,
            "delta": row.short_delta if row.short_delta is not None else 0.25,
            "dte": row.dte,
            "entry_iv": (row.iv_pct / 100.0) if row.iv_pct is not None else 0.25,
            "iv_rank_score": 0.5,
            "credit_to_width": (row.mid_price / width) if (row.is_credit and width > 0) else 0.0,
        }
        return describe_row(payload, model, rel_df)
    except Exception:
        return None


# ── 5. Options Intelligence & Flow Hub (Zero Network) ────────────────────────


def load_options_intel(
    *,
    chain_db: str = "data/chain_archive.db",
    ohlcv_db: str = "data/equity_ohlcv.db",
    occ_db: str = "data/occ_oi.db",
) -> OptionsIntelBundle:
    """Aggregate local UOA flow, OCC surges, Vol-Intel VRP, Skew/GEX, and Calendars."""
    bundle = OptionsIntelBundle()

    # 1. Day-over-day UOA from chain_archive.db
    full_ch = repo_path(chain_db)
    if os.path.exists(full_ch):
        try:
            from src.uoa import uoa_report

            rep = uoa_report(db_path=full_ch)
            bundle.as_of = str(rep.get("date") or "")
            bundle.uoa_rows = list((rep.get("symbols") or [])[:25])
        except Exception:
            pass

    # 2. OCC market-wide OI & Volume surges
    try:
        from src.occ_archive import scan_occ_uoa

        bundle.occ_rows = [r.to_dict() for r in scan_occ_uoa(db_path=occ_db, limit=25)]
    except Exception:
        pass

    # 3. Vol-Intel VRP (IV vs RV20 & IV moves)
    if os.path.exists(full_ch):
        try:
            from src.vol_intel.engine import build_rows

            movers, vrp_rows = build_rows(chain_db=full_ch, ohlcv_db=repo_path(ohlcv_db))
            mv_by_sym = {m["symbol"]: m for m in movers}
            merged_vol: List[Dict[str, Any]] = []
            for v in vrp_rows:
                sym = v["symbol"]
                m = mv_by_sym.get(sym, {})
                merged_vol.append(
                    {
                        "symbol": sym,
                        "iv_pct": round(float(v["iv"]) * 100.0, 1) if v.get("iv") is not None else None,
                        "rv_pct": round(float(v["rv"]) * 100.0, 1) if v.get("rv") is not None else None,
                        "vrp_vp": round(float(v["vrp"]) * 100.0, 1) if v.get("vrp") is not None else None,
                        "iv_rv_ratio": round(float(v["ratio"]), 2) if v.get("ratio") is not None else None,
                        "iv_chg_vp": round(float(m["d_iv"]) * 100.0, 1) if m.get("d_iv") is not None else None,
                        "rv_pctile": round(float(v["rv_pctile"]) * 100.0, 0) if v.get("rv_pctile") is not None else None,
                    }
                )
            merged_vol.sort(key=lambda r: -(abs(r.get("vrp_vp") or 0.0)))
            bundle.vol_rows = merged_vol[:30]
        except Exception:
            pass

    # 4. Option-implied Skew + Dealer GEX across archived universe
    if os.path.exists(full_ch):
        try:
            from src.alpha.options_for_shares import scan_option_signals_for_shares

            bundle.skew_rows = [s.to_dict() for s in scan_option_signals_for_shares(chain_db=chain_db)[:30]]
        except Exception:
            pass

    # 5. Taxable ETF Calendars
    if os.path.exists(full_ch):
        try:
            from src.radar.calendar_spreads import scan_etf_calendar_spreads

            bundle.calendar_rows = [c.to_dict() for c in scan_etf_calendar_spreads(chain_db=chain_db)[:15]]
        except Exception:
            pass

    return bundle
