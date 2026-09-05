#!/usr/bin/env python3
"""
Paper Trading Manager for Options Screener.
Handles logging forward tests and updating open positions using SQLite.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from .data_fetching import get_risk_free_rate as _get_rfr
    _HAS_RFR = True
except ImportError:
    _HAS_RFR = False

# Use optimized yfinance from data_fetching (with curl_cffi session + caching)
def _get_yf_and_session():
    """Get lazily-initialized yfinance and curl_cffi session from data_fetching."""
    from . import data_fetching
    data_fetching._init_yfinance()
    data_fetching._init_yf_session()
    return data_fetching.yf, data_fetching._yf_session

from .utils import is_short_position as _is_short_position
from .utils import bs_delta as _bs_delta
from .capital_risk import capital_at_risk, within_budget
from .book_sizing import (SizingDecision, book_equity, load_sizing_config,
                          open_risk, size)
from .earnings_gate import (PROJECTED_THROUGH, THROUGH, UNKNOWN,
                            cached_earnings_dates, load_earnings_gate_config,
                            project_next_earnings, refuses, verdict_for_trade)

# ── Chain-quote memo ─────────────────────────────────────────────────────────
# _fetch_chain_quotes already serves every leg on a (ticker, expiration) from
# one request, but marking a book calls it from several places — position
# marks, shadow marks, the risk gate — so the same pair is refetched once per
# call site. Measured 2026-08-07: 113 calls against 38 distinct pairs, 17.0s of
# a 22.8s squeeze scan.
#
# The TTL is short on purpose. These are live bid/ask that mark open positions,
# and a stale mark is a worse failure than a slow one
# (docs/MARK_TRUSTWORTHINESS_SPEC.md). 60s dedupes within one run without
# carrying quotes across runs — the same reasoning as portfolio_risk._SPOT_CACHE.
import time as _pm_time
_CHAIN_QUOTE_CACHE: dict = {}   # {(ticker, expiration): (quotes, timestamp)}
_CHAIN_QUOTE_TTL = 60


def reset_chain_quote_cache() -> None:
    """Drop memoized chain quotes. Tests + forced refresh."""
    _CHAIN_QUOTE_CACHE.clear()


# ── Mark provenance ──────────────────────────────────────────────────────────
# Every mark carries where it came from, because "what is this contract worth"
# and "is that number trustworthy enough to write an exit into the ledger
# forever" are two different questions. Preference order is
# MID -> LAST -> CLOSE -> MODEL; only the first three are market observations.
# A MODEL mark is a fabricated price (Black-Scholes at the row's entry IV) and
# must never fire a price-based exit — see docs/MARK_TRUSTWORTHINESS_SPEC.md.
MARK_MID = "mid"
MARK_LAST = "last"
MARK_CLOSE = "close"
MARK_MODEL = "model"
MARKET_MARK_SOURCES: Tuple[str, ...] = (MARK_MID, MARK_LAST, MARK_CLOSE)

# Sigma used by the model fallback when the row has no usable stored entry IV
# (pre-v16 rows never backfilled). Name-specific IV strictly dominates it.
DEFAULT_MODEL_SIGMA = 0.30
# An entry_iv above this is a data error, not a vol — fall back rather than
# price a contract at 500%+ vol.
_MAX_SANE_SIGMA = 5.0

# Suffix stamped on an exit_reason whose exit price came from a model mark, so
# the ledger carries the provenance of the fill it recorded.
MODEL_MARK_SUFFIX = " (model mark)"

# Refuse an auto-logged CREDIT trade when round-trip friction exceeds this
# fraction of the credit received. Above 1.0 the trade cannot profit at any win
# rate; 0.50 still demands roughly 2:1 accuracy just to clear the spread.
# Measured on the 2026-07-31 cohort: 31 of 188 short-premium trades were above
# 1.0, and excluding those above 0.50 moved the family's median return on risk
# from +12.6% to +28.1%. Set to null to disable.
DEFAULT_MAX_FRICTION_TO_CREDIT = 0.50

# Legs per credit structure, for costing the round trip.
_CREDIT_LEG_COUNTS = {"Bull Put": 2, "Bear Call": 2, "Short Put": 1,
                      "Iron Condor": 4, "Credit Spread": 2}


def _mid_from_quote(bid: Any, ask: Any) -> Optional[float]:
    """Bid/ask midpoint, or None when the book is unusable.

    A mid is taken only from a two-sided, uncrossed book: both sides present,
    both > 0, and ask >= bid (the crossed-quote guard from the 2026-07-13
    audit fixes). Anything else falls through to the next mark source.
    """
    try:
        b = float(bid)
        a = float(ask)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(b) and np.isfinite(a)):
        return None
    if b <= 0 or a <= 0 or a < b:
        return None
    mid = (a + b) / 2.0
    return mid if mid > 0 else None


def _model_sigma(entry_iv: Any) -> float:
    """Sigma for the model fallback: the row's stored entry IV when usable.

    Entry IV is itself an approximation — vol moves after entry — but it is
    name-specific and strictly dominates a global 30 constant. Marking an
    80-vol name at 0.30 is what made model marks dangerous in the first place.
    """
    try:
        iv = float(entry_iv)
    except (TypeError, ValueError):
        return DEFAULT_MODEL_SIGMA
    if not np.isfinite(iv) or iv <= 0 or iv > _MAX_SANE_SIGMA:
        return DEFAULT_MODEL_SIGMA
    return iv


def _normalize_exit_rules(config: dict) -> dict:
    """Pull context-aware exit rules from config with legacy fallback.

    New schema (config.json → exit_rules):
      time_exit_dte, min_days_held
      short_premium: take_profit_{ge_21_dte,7_to_21_dte,lt_7_dte},
                     stop_loss_premium_multiple, stop_loss_on_strike_breach,
                     strike_breach_buffer, stop_loss_delta_multiple
      spread: take_profit, stop_loss
      long_option: take_profit, take_profit_delta, stop_loss

    Legacy keys (take_profit, stop_loss) are used as fallbacks only.
    """
    raw = (config or {}).get("exit_rules", {}) or {}
    legacy_tp = float(raw.get("take_profit", 0.50))
    legacy_sl = float(raw.get("stop_loss", -0.25))

    short_r = raw.get("short_premium", {}) or {}
    spread_r = raw.get("spread", {}) or {}
    long_r = raw.get("long_option", {}) or {}

    return {
        "time_exit_dte": int(raw.get("time_exit_dte", 21)),
        "min_days_held": int(raw.get("min_days_held", 3)),
        "short": {
            "tp_ge_21":       float(short_r.get("take_profit_ge_21_dte", legacy_tp)),
            "tp_7_21":        float(short_r.get("take_profit_7_to_21_dte", legacy_tp * 0.70)),
            "tp_lt_7":        float(short_r.get("take_profit_lt_7_dte", legacy_tp * 0.50)),
            "sl_prem_mult":   float(short_r.get("stop_loss_premium_multiple", 2.0)),
            "sl_strike":      bool(short_r.get("stop_loss_on_strike_breach", True)),
            "sl_strike_buf":  float(short_r.get("strike_breach_buffer", 0.0)),
            "sl_delta_mult":  float(short_r.get("stop_loss_delta_multiple", 2.5)),
            "legacy_sl":      legacy_sl,
        },
        "spread": {
            "tp": float(spread_r.get("take_profit", 0.50)),
            "sl": float(spread_r.get("stop_loss", -1.0)),
        },
        "long": {
            "tp":       float(long_r.get("take_profit", 1.00)),
            "tp_delta": float(long_r.get("take_profit_delta", 0.80)),
            "sl":       float(long_r.get("stop_loss", -0.50)),
        },
    }


def _tp_for_dte(rules_short: dict, dte: int) -> float:
    if dte >= 21:
        return rules_short["tp_ge_21"]
    if dte >= 7:
        return rules_short["tp_7_21"]
    return rules_short["tp_lt_7"]


def _evaluate_short_single_leg_exit(
    rules: dict,
    option_type: str,
    strike: float,
    spot: Optional[float],
    entry_price: float,
    current_price: float,
    entry_delta: Optional[float],
    entry_iv: Optional[float],
    dte: int,
    days_held: int,
    rfr: float,
) -> Tuple[bool, Optional[str], float]:
    """Evaluate context-aware exits for a short single-leg option.

    Returns (should_close, reason_or_None, pnl_raw_mark_to_market).
    Trigger priority (first fires wins):
      1. Take profit (DTE-tiered)
      2. Time exit (≤ time_exit_dte, min_days_held satisfied)
      3. Stop loss — strike breach, premium multiple, delta multiple
    """
    short = rules["short"]
    pnl_raw = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0

    tp_target = _tp_for_dte(short, dte)
    if pnl_raw >= tp_target:
        return True, f"Take Profit ({tp_target*100:.0f}% @ {dte}d)", pnl_raw

    if 0 < dte <= rules["time_exit_dte"] and days_held >= rules["min_days_held"]:
        return True, f"Time Exit ({dte}d to expiry)", pnl_raw

    # Strike-breach stop (defensive — short strike tested)
    if short["sl_strike"] and spot is not None and strike > 0:
        buf = short["sl_strike_buf"]
        ot = (option_type or "").lower()
        if ot == "call" and spot >= strike * (1.0 + buf):
            return True, "Stop Loss (strike breached)", pnl_raw
        if ot == "put" and spot <= strike * (1.0 - buf):
            return True, "Stop Loss (strike breached)", pnl_raw

    # Premium-multiple stop (e.g. premium ≥ 2× entry ⇒ pnl_raw ≤ -1.0)
    sl_prem = -(short["sl_prem_mult"] - 1.0)
    if pnl_raw <= sl_prem:
        return True, f"Stop Loss ({short['sl_prem_mult']:.1f}× premium)", pnl_raw

    # Delta-multiple early warning (requires entry_delta + entry_iv + spot)
    if (
        entry_delta is not None and entry_iv is not None and spot is not None
        and abs(entry_delta) > 1e-4 and entry_iv > 0 and dte > 0
    ):
        try:
            T = max(dte / 365.0, 1 / 365.0)
            cur_delta = _bs_delta((option_type or "call").lower(), float(spot), float(strike), T, rfr, float(entry_iv))
            if abs(cur_delta) >= short["sl_delta_mult"] * abs(entry_delta):
                return True, f"Stop Loss (Δ {abs(cur_delta):.2f} ≥ {short['sl_delta_mult']:.1f}× entry)", pnl_raw
        except Exception:
            pass

    return False, None, pnl_raw


def _classify_structure(row) -> str:
    """Determine row structure from new schema columns; falls back to strategy_name."""
    try:
        sn = str(row["strategy_name"] or "").lower() if "strategy_name" in row.keys() else ""
    except Exception:
        sn = ""
    try:
        sp = row["short_put_strike"] if "short_put_strike" in row.keys() else None
        sc = row["short_call_strike"] if "short_call_strike" in row.keys() else None
    except Exception:
        sp, sc = None, None
    if (sp not in (None, "", 0) and sc not in (None, "", 0)) or "iron condor" in sn:
        return "iron_condor"
    try:
        ls = row["long_strike"] if "long_strike" in row.keys() else None
    except Exception:
        ls = None
    if ls not in (None, "", 0) or any(k in sn for k in ("bull put", "bear call")):
        return "spread"
    if sn.startswith("spread:"):
        return "spread"
    return "single"


def _intrinsic_value(option_type: str, spot: float, strike: float) -> float:
    """Intrinsic value of a single option at expiry (never negative)."""
    if (option_type or "").lower() == "call":
        return max(0.0, float(spot) - float(strike))
    return max(0.0, float(strike) - float(spot))


def _legs_for_row(row) -> List[Tuple[float, str, int]]:
    """Decompose a trade row into (strike, option_type, qty) legs.

    qty is -1 for short legs, +1 for long legs. Iron condors stored without
    call (or put) strikes — a known shape in legacy rows — degrade gracefully
    to whichever legs are present instead of dropping the entire row (which
    previously left them un-markable and OPEN forever).
    """
    structure = _classify_structure(row)
    if structure == "iron_condor":
        legs: List[Tuple[float, str, int]] = []
        for col, opt_t, qty in (
            ("short_put_strike",  "put",  -1),
            ("long_put_strike",   "put",  +1),
            ("short_call_strike", "call", -1),
            ("long_call_strike",  "call", +1),
        ):
            try:
                v = row[col] if col in row.keys() else None
            except (KeyError, IndexError):
                v = None
            if v is None or v in ("", 0):
                continue
            try:
                legs.append((float(v), opt_t, qty))
            except (TypeError, ValueError):
                continue
        return legs
    if structure == "spread":
        opt_type = str(row["type"] or "").lower()
        if opt_type not in ("put", "call"):
            sn = str(row["strategy_name"] or "").lower()
            opt_type = "put" if "bull put" in sn else "call"
        ls = row["long_strike"] if "long_strike" in row.keys() else None
        try:
            long_strike = float(ls) if ls is not None and ls not in ("", 0) else None
        except (TypeError, ValueError):
            long_strike = None
        if long_strike is None:
            # Legacy SPREAD:long:width:max_loss fallback
            try:
                long_strike = float(str(row["strategy_name"] or "").split(":")[1])
            except (ValueError, IndexError):
                return []
        return [
            (float(row["strike"]), opt_type, -1),
            (long_strike,          opt_type, +1),
        ]
    return [(
        float(row["strike"]),
        str(row["type"] or "").lower(),
        -1 if _is_short_position(row["strategy_name"] or "") else +1,
    )]


def _leg_exit_columns(row, leg_quotes: Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]],
                      structure: str) -> Dict[str, Optional[float]]:
    """Per-leg exit bid/ask, keyed by the schema v23 column name for this
    leg's role. A role whose strike is not in `leg_quotes` (missing chain
    data for that leg) contributes (None, None) for its two columns —
    never a fabricated value."""
    def _q(strike_key: str, opt_type: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        try:
            raw = row[strike_key] if strike_key in row.keys() else None
            strike = float(raw) if raw is not None and raw not in ("", 0) else None
        except (TypeError, ValueError, KeyError):
            strike = None
        if strike is None or opt_type is None:
            return None, None
        return leg_quotes.get((strike, opt_type), (None, None))

    if structure == "iron_condor":
        sp_b, sp_a = _q("short_put_strike", "put")
        lp_b, lp_a = _q("long_put_strike", "put")
        sc_b, sc_a = _q("short_call_strike", "call")
        lc_b, lc_a = _q("long_call_strike", "call")
        return {
            "short_put_bid_exit": sp_b, "short_put_ask_exit": sp_a,
            "long_put_bid_exit": lp_b, "long_put_ask_exit": lp_a,
            "short_call_bid_exit": sc_b, "short_call_ask_exit": sc_a,
            "long_call_bid_exit": lc_b, "long_call_ask_exit": lc_a,
        }

    opt_t = str(row["type"] or "").lower()
    try:
        short_strike = float(row["strike"])
    except (TypeError, ValueError, KeyError):
        short_strike = None
    s_b, s_a = (leg_quotes.get((short_strike, opt_t), (None, None))
               if short_strike is not None else (None, None))
    l_b, l_a = _q("long_strike", opt_t)
    return {
        "short_bid_exit": s_b, "short_ask_exit": s_a,
        "long_bid_exit": l_b, "long_ask_exit": l_a,
    }


def _legs_intrinsic_close_value(legs: List[Tuple[float, str, int]], spot: float) -> float:
    """Debit required to flatten a multi-leg structure at expiry.

    cost-to-close = sum(-qty * intrinsic). For a credit structure (shorts have
    qty=-1, longs qty=+1) this is the debit paid to buy it back — 0 when every
    leg expires out-of-the-money, so the seller keeps the full credit.
    """
    return sum(-qty * _intrinsic_value(opt_t, spot, k) for k, opt_t, qty in legs)


def _evaluate_multileg_exit(
    rules: dict,
    entry_credit: float,
    current_credit_to_close: float,
    dte: int,
    days_held: int,
) -> Tuple[bool, Optional[str], float]:
    """TP / SL / time-exit evaluation for credit spreads & iron condors.

    pnl_raw = (entry_credit - current_credit) / entry_credit. Positive when
    the structure has decayed (premium seller's profit). spread.tp / spread.sl
    in config are interpreted as fractions of credit collected.
    """
    if entry_credit <= 0:
        return False, None, 0.0
    pnl_raw = (entry_credit - current_credit_to_close) / entry_credit
    tp = rules["spread"]["tp"]
    sl = rules["spread"]["sl"]
    if pnl_raw >= tp:
        return True, f"Take Profit ({tp*100:.0f}% of credit)", pnl_raw
    if pnl_raw <= sl:
        return True, f"Stop Loss ({abs(sl)*100:.0f}% of credit)", pnl_raw
    if 0 < dte <= rules["time_exit_dte"] and days_held >= rules["min_days_held"]:
        return True, f"Time Exit ({dte}d to expiry)", pnl_raw
    return False, None, pnl_raw


def _evaluate_long_single_leg_exit(
    rules: dict,
    option_type: str,
    strike: float,
    spot: Optional[float],
    entry_price: float,
    current_price: float,
    entry_iv: Optional[float],
    dte: int,
    days_held: int,
    rfr: float,
) -> Tuple[bool, Optional[str], float]:
    """Exits for long single-leg: TP on profit or deep-ITM delta; SL on loss; time exit."""
    lng = rules["long"]
    pnl_raw = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    if pnl_raw >= lng["tp"]:
        return True, f"Take Profit ({lng['tp']*100:.0f}%)", pnl_raw

    if 0 < dte <= rules["time_exit_dte"] and days_held >= rules["min_days_held"]:
        return True, f"Time Exit ({dte}d to expiry)", pnl_raw

    # Deep-ITM TP via delta
    if entry_iv is not None and spot is not None and entry_iv > 0 and dte > 0:
        try:
            T = max(dte / 365.0, 1 / 365.0)
            cur_delta = _bs_delta((option_type or "call").lower(), float(spot), float(strike), T, rfr, float(entry_iv))
            if abs(cur_delta) >= lng["tp_delta"]:
                return True, f"Take Profit (Δ {abs(cur_delta):.2f} deep ITM)", pnl_raw
        except Exception:
            pass

    if pnl_raw <= lng["sl"]:
        return True, f"Stop Loss ({lng['sl']*100:.0f}%)", pnl_raw

    return False, None, pnl_raw


# Realistic execution cost constants (deprecated fallbacks — use config.json paper_trading section)
# Sourced from src.execution_costs so there is ONE number to change when the
# broker changes; see the note there for why the fallback is not 0.0.
from src.cost_calibration import OUT_OF_RANGE_REASON as _OUT_OF_RANGE_REASON
from src.cost_calibration import entry_dte as _entry_dte
from src.cost_calibration import in_calibration as _in_calibration
from src.execution_costs import FALLBACK_COMMISSION_PER_CONTRACT
from src.paths import repo_path

COMMISSION_PER_CONTRACT = FALLBACK_COMMISSION_PER_CONTRACT  # $ per contract per leg
SLIPPAGE_PER_SHARE = 0.05        # $ per share (1 typical options tick, ~half spread)
# Round-trip friction per share = entry slippage + exit slippage + 2 commissions
_FRICTION_PER_SHARE = (2 * SLIPPAGE_PER_SHARE) + (2 * COMMISSION_PER_CONTRACT / 100.0)

# Days an auto-logged contract blocks a second auto-log of the same
# (ticker, strategy, strike, expiration). Config key auto_log.dedup_window_days;
# 0 or null disables the guard. Only the automated feeders are gated by it — a
# deliberate manual entry is always the operator's call.
DEFAULT_DEDUP_WINDOW_DAYS = 3

# Leg columns beyond the anchor `strike` that define a structure. A single-leg
# row has NULL in all of them and so still matches only other single legs; a
# spread differs on long_strike, a condor on either wing.
_DEDUP_LEG_COLUMNS = (
    "long_strike",
    "short_call_strike", "long_call_strike",
    "short_put_strike", "long_put_strike",
)


def _leg_strike(value: Any) -> Optional[float]:
    """A leg strike as a float, or None for an absent/unusable/zero leg.

    Zero collapses to None on purpose: the auto-log payloads default a missing
    wing to 0 (``row.get("long_strike", 0)``) while ``log_trade`` writes NULL
    for one that was never set, and the two must not read as different legs."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f == 0.0:
        return None
    return f

_SCHEMA_VERSION = 24
_MIGRATIONS = {
    1: [],
    2: ["ALTER TABLE trades ADD COLUMN pnl_usd REAL"],
    3: [
        "ALTER TABLE trades ADD COLUMN pop_score REAL",
        "ALTER TABLE trades ADD COLUMN ev_score REAL",
        "ALTER TABLE trades ADD COLUMN rr_score REAL",
        "ALTER TABLE trades ADD COLUMN liquidity_score REAL",
        "ALTER TABLE trades ADD COLUMN momentum_score REAL",
        "ALTER TABLE trades ADD COLUMN iv_rank_score REAL",
        "ALTER TABLE trades ADD COLUMN theta_score REAL",
    ],
    4: [
        # Track AI score at entry to measure AI IC vs technical IC separately
        "ALTER TABLE trades ADD COLUMN ai_score REAL",
        "ALTER TABLE trades ADD COLUMN ai_confidence REAL",
    ],
    5: [
        # Store entry Greeks and IV for accurate stress testing and P&L attribution
        "ALTER TABLE trades ADD COLUMN entry_iv REAL",
        "ALTER TABLE trades ADD COLUMN entry_delta REAL",
        "ALTER TABLE trades ADD COLUMN entry_gamma REAL",
        "ALTER TABLE trades ADD COLUMN entry_vega REAL",
        "ALTER TABLE trades ADD COLUMN entry_theta REAL",
        "ALTER TABLE trades ADD COLUMN dividend_yield REAL",
    ],
    6: [
        # Expanded component scores for per-component IC validation
        "ALTER TABLE trades ADD COLUMN iv_edge_score REAL",
        "ALTER TABLE trades ADD COLUMN vrp_score REAL",
        "ALTER TABLE trades ADD COLUMN iv_mispricing_score REAL",
        "ALTER TABLE trades ADD COLUMN skew_align_score REAL",
        "ALTER TABLE trades ADD COLUMN vega_risk_score REAL",
        "ALTER TABLE trades ADD COLUMN term_structure_score REAL",
    ],
    7: [
        # Remaining 14 composite_weights components — full IC coverage of all 27 weights
        "ALTER TABLE trades ADD COLUMN catalyst_score REAL",
        "ALTER TABLE trades ADD COLUMN em_realism_score REAL",
        "ALTER TABLE trades ADD COLUMN gamma_theta_score REAL",
        "ALTER TABLE trades ADD COLUMN gex_score REAL",
        "ALTER TABLE trades ADD COLUMN gamma_magnitude_score REAL",
        "ALTER TABLE trades ADD COLUMN gamma_pin_score REAL",
        "ALTER TABLE trades ADD COLUMN iv_velocity_score REAL",
        "ALTER TABLE trades ADD COLUMN max_pain_score REAL",
        "ALTER TABLE trades ADD COLUMN oi_change_score REAL",
        "ALTER TABLE trades ADD COLUMN option_rvol_score REAL",
        "ALTER TABLE trades ADD COLUMN pcr_score REAL",
        "ALTER TABLE trades ADD COLUMN sentiment_score_norm REAL",
        "ALTER TABLE trades ADD COLUMN spread_score REAL",
        "ALTER TABLE trades ADD COLUMN trader_pref_score REAL",
    ],
    8: [
        # Tag each trade with the weight profile that produced it, so scans under
        # different weight configurations can be compared head-to-head later.
        "ALTER TABLE trades ADD COLUMN weight_profile TEXT",
        "CREATE INDEX IF NOT EXISTS idx_dedup_profile ON trades(ticker, strike, expiration, type, weight_profile, date)",
    ],
    9: [
        # Record which exit rule fired (Take Profit / Stop Loss / Strike Breach / Delta Touch / Time Exit)
        "ALTER TABLE trades ADD COLUMN exit_reason TEXT",
    ],
    10: [
        # Multi-leg structural columns. For single-leg trades these stay NULL.
        # For credit spreads: long_strike, spread_width, net_credit, max_profit_usd, max_loss_usd.
        # For iron condors: also short_call_strike, long_call_strike, short_put_strike, long_put_strike,
        # plus net_delta. The 'strike' column continues to hold the short-put strike for spreads
        # (and the short-put strike for iron condors) so existing dedup keys keep working.
        "ALTER TABLE trades ADD COLUMN long_strike REAL",
        "ALTER TABLE trades ADD COLUMN spread_width REAL",
        "ALTER TABLE trades ADD COLUMN net_credit REAL",
        "ALTER TABLE trades ADD COLUMN max_profit_usd REAL",
        "ALTER TABLE trades ADD COLUMN max_loss_usd REAL",
        "ALTER TABLE trades ADD COLUMN short_call_strike REAL",
        "ALTER TABLE trades ADD COLUMN long_call_strike REAL",
        "ALTER TABLE trades ADD COLUMN short_put_strike REAL",
        "ALTER TABLE trades ADD COLUMN long_put_strike REAL",
        "ALTER TABLE trades ADD COLUMN net_delta REAL",
    ],
    11: [
        # Fractional position size; default 1.0 keeps equity/options path inert
        "ALTER TABLE trades ADD COLUMN quantity REAL DEFAULT 1.0",
    ],
    12: [
        # paper_only=1 marks trades excluded from the Long-Call validation cohort
        # (e.g. Bear Call, Long Put, Iron Condor). paper_only=0 (default) means the
        # trade is eligible for real-money edge validation.
        "ALTER TABLE trades ADD COLUMN paper_only INTEGER DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_paper_only ON trades(paper_only)",
    ],
    13: [
        # era partitions the book around the 2026-06-16 real-marks/screener overhaul
        # (net-of-cost EV ranking + quant read + portfolio guard). Every row that
        # existed before the overhaul backfills to 'pre_data' via the column default;
        # log_trade tags new trades 'finalized'. Lets P&L be read in the two eras the
        # user asked for: before vs after the data work.
        "ALTER TABLE trades ADD COLUMN era TEXT DEFAULT 'pre_data'",
        "CREATE INDEX IF NOT EXISTS idx_era ON trades(era)",
    ],
    14: [
        # lottery_edge marks whether a Lottery-sleeve ticket cleared the evidence
        # bar (cheap IV + reachable strike + catalyst/momentum) at entry. Lets the
        # sleeve report edge-flagged vs unflagged hit-rate — the core validation of
        # whether selection beats a blind far-OTM basket. NULL on every non-lottery row.
        "ALTER TABLE trades ADD COLUMN lottery_edge INTEGER",
    ],
    15: [
        # High-water mark: the highest premium observed while the position was
        # tracked (sampled at each update_positions/check_pnl run — daily-ish
        # granularity, NOT a tick high). Answers "how high did it actually go
        # after I exited at +100%?" so missed 3x/5x multipliers become data.
        "ALTER TABLE trades ADD COLUMN max_price_seen REAL",
        "ALTER TABLE trades ADD COLUMN max_price_date TEXT",
    ],
    16: [
        # Dollars the position ties up until it closes, resolved per structure by
        # src/capital_risk.py. Stored rather than re-derived because every call
        # site was rolling its own `max_loss_usd or entry_price * 100`, which
        # costs a cash-secured put at the credit received instead of the
        # collateral — understating a 77.5-strike short put by ~50x. Strategy
        # comparisons and the auto-log budget gate both read this column.
        # NULL means risk could not be bounded from the stored fields.
        "ALTER TABLE trades ADD COLUMN capital_at_risk REAL",
    ],
    17: [
        # duplicate_of holds the entry_id this row double-counts, when a row has
        # been RULED a double-log rather than merely flagged as a candidate by
        # reports/duplicate_trades_audit.md. NULL on every genuine row.
        #
        # A column rather than a deletion, because the audit's own rule is that
        # the ledger records what happened and rewriting it silently is worse
        # than the double-count it fixes. Marking is reversible and auditable;
        # DELETE is neither. Cohort and track-record queries exclude
        # `duplicate_of IS NOT NULL`, so a marked row stops inflating the
        # evidence without vanishing from the record.
        "ALTER TABLE trades ADD COLUMN duplicate_of INTEGER",
        "CREATE INDEX IF NOT EXISTS idx_duplicate_of ON trades(duplicate_of)",
    ],
    18: [
        # What the fill actually was, under src/execution_truth.py's three
        # policies. The scan path prices every leg at the bid/ask MID
        # (options_screener.py:2160 sets `premium = mid`) and charges slippage
        # only on exit, so entry friction has always been modelled as zero.
        # Measured against archived CBOE quotes for 30 logged Bull Puts,
        # crossing costs $0.35/share — 27% of the credit — which moves the
        # breakeven win rate on that structure from 58% to over 70%. The book
        # wins 70.4%. An edge and a loss are on opposite sides of that number.
        #
        # `entry_price` KEEPS ITS v17 MEANING and is never rewritten by this
        # migration or the restate script: every existing reader stays correct.
        # Analysis that wants the honest number reads `entry_price_fill`.
        # NULL everywhere until scripts/restate_execution.py or a new log_*
        # call populates it — absent, not assumed.
        "ALTER TABLE trades ADD COLUMN entry_price_mid REAL",
        "ALTER TABLE trades ADD COLUMN entry_price_fill REAL",
        "ALTER TABLE trades ADD COLUMN entry_price_cross REAL",
        # 'mid' | 'limit' | 'cross' — which policy entry_price_fill reflects.
        "ALTER TABLE trades ADD COLUMN fill_policy TEXT",
        # 'live_quote' when the legs carried a real two-sided quote at entry,
        # 'modeled' when the half-spread table supplied it, 'unknown' when
        # neither could. Never pool the three into one headline number.
        "ALTER TABLE trades ADD COLUMN fill_source TEXT",
        "CREATE INDEX IF NOT EXISTS idx_fill_source ON trades(fill_source)",
    ],
    19: [
        # Shadow-tracking: what a CLOSED trade went on to do.
        #
        # The stop fires on 40 of 82 single-leg long trades, realising -60.3%
        # from an average peak of +16.6%. Whether the stop helps or hurts is
        # unanswerable from the ledger as it stood: max_price_seen stops
        # updating the moment the position closes, so there is no post-exit
        # path to compare the realised outcome against.
        #
        # shadow_until is the date to keep marking to (the original expiry).
        # The post_exit_* columns are written by shadow_mark and are READ-ONLY
        # to every existing consumer — status, exit_price, exit_reason, pnl_usd
        # and max_price_seen are never touched, so the realised record stands
        # exactly as it was. The high answers "could it have recovered"; the
        # last answers "where did it actually end up". Judging a stop needs both.
        "ALTER TABLE trades ADD COLUMN shadow_until TEXT",
        "ALTER TABLE trades ADD COLUMN post_exit_max_price REAL",
        "ALTER TABLE trades ADD COLUMN post_exit_max_date TEXT",
        "ALTER TABLE trades ADD COLUMN post_exit_last_price REAL",
        "ALTER TABLE trades ADD COLUMN post_exit_last_date TEXT",
        "CREATE INDEX IF NOT EXISTS idx_shadow_until ON trades(shadow_until)",
    ],
    20: [
        # Which post-composite adjustments fired at entry.
        #
        # quality_score is a 27-component weighted average and then ~20 hand-set
        # additions and multipliers. Measured 2026-08-07: those adjustments can
        # subtract 1.28 and add 0.47, against a composite whose whole documented
        # range spans 0.54 and whose observed spread on a clean chain was 0.29.
        # One `decay_warning` at -0.20 outweighs any single component; two
        # penalties outweigh all 27 together. Not one of the constants has ever
        # been measured.
        #
        # It could not be measured, either: the ledger stored every component
        # score and no record of which flags fired, so `flag -> outcome` had no
        # data behind it. This column is that data. It is written at entry and
        # never updated, holds a comma-separated list of flag names (empty when
        # none fired), and is READ-ONLY to every existing consumer — no score,
        # cohort or verdict reads it.
        #
        # Rows logged before this migration carry NULL, which is not "no flags
        # fired" but "not recorded". Any analysis must exclude NULL rather than
        # treat it as empty, or it will read the entire pre-2026-08-07 book as
        # having had a clean bill of health.
        "ALTER TABLE trades ADD COLUMN score_adjustments TEXT",
    ],
    21: [
        # The EV numbers the board actually decides on, kept instead of discarded.
        #
        # `decide_verdict` reads net EV against the error bar this contract's own
        # vega implies, and that comparison drives the TAKE / MARGINAL / SKIP
        # call, the pick_ranking EV gate, and the WORTH grade on every card. None
        # of the four inputs survived the scan: the ledger stored `ev_score`, a
        # within-scan rank, and nothing else. So on 2026-08-09, asked whether a
        # contract graded STRONG went on to beat one graded THIN, the book had no
        # answer — 851 closed trades and not one raw EV among them.
        #
        # A rank is not a level. `ev_score` is `rank_norm` over one scan's
        # candidates, so it cannot be compared across scans and carries no
        # information about how large an edge was, only where it sat that day.
        # These four are levels, in dollars per contract, and they make
        # `net_ev / noise` reconstructable after the fact.
        #
        # Written at entry, never updated, and READ-ONLY to every existing
        # consumer — no score, cohort, gate or verdict reads them back. Rows
        # logged before this migration carry NULL, which means "not recorded",
        # not "zero"; analysis must exclude NULL rather than read the pre-2026
        # -08-10 book as a book of zero-edge trades.
        "ALTER TABLE trades ADD COLUMN entry_ev_net REAL",
        "ALTER TABLE trades ADD COLUMN entry_ev_gross REAL",
        "ALTER TABLE trades ADD COLUMN entry_ev_cost REAL",
        "ALTER TABLE trades ADD COLUMN entry_ev_noise REAL",
    ],
    22: [
        # The budget in force when the trade was logged.
        #
        # Until 2026-08-14 one global number (auto_log.max_capital_at_risk =
        # 4000) governed every log site. It is now the SCHEDULER's budget only;
        # interactive scans choose their own per scan, defaulting to no limit.
        # Recording it is what keeps the book readable: the analysis that
        # matters is "inside the budget +$3,283 (n=247) vs above it -$19,741
        # (n=160)", and once the budget varies per scan you cannot recover it
        # from capital_at_risk alone.
        #
        # NULL means NO LIMIT WAS IN FORCE — not "unknown". The backfill below
        # makes that truthful: the cap shipped 2026-07-29, so rows from that
        # date really had a $4,000 budget, and earlier rows really had none.
        # That is the unbounded-feeder era whose $27k and $83k positions are
        # correctly marked unbudgeted.
        "ALTER TABLE trades ADD COLUMN budget_at_entry REAL",
        "UPDATE trades SET budget_at_entry = 4000.0 "
        "WHERE date >= '2026-07-29' AND budget_at_entry IS NULL",
    ],
    23: [
        # Per-leg bid/ask at entry and exit, for multi-leg structures only.
        # Unblocks repricing the 46% of the closed book that is multi-leg
        # (docs/SINGLE_LEG_REPRICE_20260902.md refused it: entry_price on a
        # spread is a net credit across legs, not any single leg's mid).
        # NULL on every legacy row and on every single-leg row — never zero.
        "ALTER TABLE trades ADD COLUMN short_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN short_ask_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_ask_exit REAL",
        "ALTER TABLE trades ADD COLUMN short_put_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_put_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_put_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_put_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_call_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_call_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_call_bid_entry REAL",
        "ALTER TABLE trades ADD COLUMN long_call_ask_entry REAL",
        "ALTER TABLE trades ADD COLUMN short_put_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN short_put_ask_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_put_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_put_ask_exit REAL",
        "ALTER TABLE trades ADD COLUMN short_call_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN short_call_ask_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_call_bid_exit REAL",
        "ALTER TABLE trades ADD COLUMN long_call_ask_exit REAL",
    ],
    24: [
        # The earnings-gate verdict (src/earnings_gate.py) computed at log_trade
        # time, kept instead of discarded. Reachable values: clear_of_earnings,
        # earnings_unknown, projected_through_earnings (report mode only —
        # refuse mode refuses the trade before the insert, same as an announced
        # through_earnings always does, so that value never lands here). NULL
        # means the check never ran at all — the gate was disabled, or
        # allow_through_earnings bypassed it, or the trade predates this column
        # — never read NULL as "clear". This is instrumentation only: nothing
        # reads this column yet. It exists so a future test of "does clear beat
        # unknown" going forward has data to run on.
        "ALTER TABLE trades ADD COLUMN earnings_state TEXT",
    ],
}


def cost_disclosure(slippage: float, commission: float, fx_rate: float) -> str:
    """One line naming every cost the ledger charged, and where they come from.

    Printed whenever positions auto-close — it is the only place someone who is
    not reading config.json finds out what was deducted. Costs that are zero are
    left out rather than printed as $0.00, so the line always describes what is
    actually happening on this broker.
    """
    parts = []
    if slippage:
        parts.append(f"${slippage:.2f}/share spread x2")
    if commission:
        parts.append(f"${commission:.2f}/contract commission x2")
    if fx_rate:
        parts.append(f"{fx_rate:.1%} CAD/USD conversion x2")
    if not parts:
        return "    [costs: no trading costs configured — see docs/BROKER_COSTS.md]"
    return f"    [costs: {' + '.join(parts)} — see docs/BROKER_COSTS.md]"


_CREDIT_STRUCTURES = {"Bull Put", "Bear Call", "Iron Condor"}
_SHORT_SINGLE_LEGS = {"Short Put", "Short Call"}


def _get_multiplier(ticker: str) -> float:
    """Return the contract multiplier: 1.0 for crypto, 100.0 for stocks."""
    if (ticker or "").upper() in ("BTC", "ETH"):
        return 1.0
    return 100.0


def _earnings_dates_in_window(trade_dict: Dict[str, Any], cfg: Dict[str, Any],
                              time_exit_dte: Any) -> List[str]:
    """The event dates a refusal is about, for the message that names them.

    A refusal the operator cannot check is a refusal they cannot trust, so the
    print says WHICH date rather than only that there was one.
    """
    from .earnings_gate import cached_earnings_dates, horizon_end
    entry = str(trade_dict.get("date") or
                datetime.now().strftime("%Y-%m-%d"))[:10]
    end = horizon_end(trade_dict.get("expiration"), time_exit_dte,
                      str(cfg.get("horizon")))
    if not end:
        return []
    return [d for d in cached_earnings_dates(
        str(trade_dict.get("ticker") or ""), str(cfg.get("cache_path")))
        if entry < str(d)[:10] <= end]


def _lots(quantity: Any) -> float:
    """Contracts held, defaulting to one.

    A row whose quantity is NULL, zero, negative or unparseable is one the
    ledger never sized — every row written before 2026-08-19 is in that state.
    Falling back to 1.0 keeps those rows reading exactly as they always have;
    falling back to 0.0 would erase their P&L.
    """
    try:
        qty = float(quantity)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(qty) or qty <= 0:
        return 1.0
    return qty


def _row_lots(row) -> float:
    """`_lots` for a sqlite3.Row / mapping, tolerating a missing column."""
    try:
        return _lots(row["quantity"])
    except (KeyError, IndexError, TypeError):
        return 1.0


def _sanitize_close_values(
    strategy_name: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    max_loss_floor: float | None = None,
    multiplier: float = 100.0,
    quantity: Any = 1.0,
) -> tuple[float, float, float]:
    """Clamp close-time values to physically possible bounds and derive pnl_usd.

    Caller observed a QQQ Bear Call closed with pnl_pct=+3.58 and exit_price=-1.22 —
    both impossible for a credit spread. Without clamping, anomalies poison the IC
    sample (one outlier flipped the sign of skew_align IC from -0.16 to ~0).

    Bounds by structure:
      - credit spreads (Bull Put / Bear Call / Iron Condor):
          max gain = +1.0 (full credit kept).
          max loss is `-(spread_width / entry_credit - 1)`, which can far exceed -1.0
          (e.g. $0.50 credit on a $5 wide spread → -9.0 / -900%). Caller passes this
          value as ``max_loss_floor``; if not supplied, a permissive -100.0 floor is
          used so a real max-loss close is recorded faithfully rather than truncated.
      - short single legs (Short Put / Short Call): max gain = full credit (+1.0);
        loss is unbounded since premium can multiply against you.
      - long premium (Long Call / Long Put): max loss = full premium (-1.0);
        gain is unbounded.

    exit_price is clamped to >= 0 (negative option prices are impossible).
    pnl_usd is computed deterministically from the sanitized pnl_pct so it can never
    be NULL after a close (caller bug: 115 historical closes were NULL because
    pnl_usd wasn't being written by the auto-exit UPDATE statements).

    ``quantity`` scales the DOLLARS and nothing else — a return is a return at
    any size, so pnl_pct is untouched (it feeds the IC sample). This existed
    only implicitly until 2026-08-19: every ledger row carried the migration
    default of 1.0, so a per-contract figure and a whole-position figure were
    the same number. Position sizing writes 2 and 3, and an unscaled column
    would book a two-lot winner at half its value — into `book_equity`, which
    is what sizes the NEXT position. A missing or absurd value falls back to
    one contract rather than zeroing the trade.
    """
    safe_exit = max(float(exit_price), 0.0) if exit_price is not None else 0.0
    raw_pct = float(pnl_pct) if pnl_pct is not None else 0.0
    if not np.isfinite(raw_pct):
        raw_pct = 0.0

    if strategy_name in _CREDIT_STRUCTURES:
        # Use caller-supplied max_loss_floor when available so true max-loss closes
        # (e.g. -3× credit on a wide spread) are recorded as-is. Fall back to a
        # permissive -100.0 floor — better to preserve magnitude than truncate at -1.0.
        floor = float(max_loss_floor) if (max_loss_floor is not None and np.isfinite(max_loss_floor)) else -100.0
        clamped_pct = max(floor, min(1.0, raw_pct))
    elif strategy_name in _SHORT_SINGLE_LEGS:
        clamped_pct = min(1.0, raw_pct)  # gain capped, loss unbounded
    else:
        clamped_pct = max(-1.0, raw_pct)  # loss capped, gain unbounded (long premium)

    pnl_usd = round(float(entry_price) * clamped_pct * multiplier * _lots(quantity), 2)
    return safe_exit, clamped_pct, pnl_usd


class PaperManager:
    """Manages paper trades stored in a SQLite database."""
    
    def __init__(self, db_path: str = "paper_trades.db", config_path: str = "config.json"):
        # Anchored on assignment so every later use — connect, migrations, the
        # exit enforcer — targets one ledger. A relative path here meant a run
        # from another directory would CREATE a second, empty paper_trades.db
        # and log real trades into it, while check_pnl from the repo root kept
        # showing the old book. Absolute paths (every test fixture) pass through.
        self.db_path = repo_path(db_path)
        # Held open only for `:memory:`, where closing the connection destroys
        # the database. See `_get_connection`.
        self._memory_conn: Optional[sqlite3.Connection] = None
        self.config_path = config_path
        # Load friction costs from config (fall back to module-level constants)
        try:
            with open(repo_path(config_path), 'r') as f:
                _cfg = json.load(f)
            _pt = _cfg.get("paper_trading", {})
            self._commission_per_contract = float(_pt.get("commission_per_contract", COMMISSION_PER_CONTRACT))
            self._slippage_per_share = float(_pt.get("slippage_per_share", SLIPPAGE_PER_SHARE))
            self._fx_conversion_rate = float(_pt.get("fx_conversion_rate", 0.0) or 0.0)
            _cap = (_cfg.get("auto_log") or {}).get("max_capital_at_risk")
            self._max_capital_at_risk = (float(_cap) if _cap is not None
                                         and _cap not in ("", 0) else None)
            _win = (_cfg.get("auto_log") or {}).get("dedup_window_days",
                                                    DEFAULT_DEDUP_WINDOW_DAYS)
            self._dedup_window_days = int(_win) if _win not in (None, "", 0, False) else 0
            _fric = (_cfg.get("auto_log") or {}).get(
                "max_friction_to_credit", DEFAULT_MAX_FRICTION_TO_CREDIT)
            self._max_friction_to_credit = (
                float(_fric) if _fric not in (None, "", 0, False) else None)
            self._sizing_cfg = load_sizing_config(_cfg)
            self._earnings_cfg = load_earnings_gate_config(_cfg)
            self._time_exit_dte = (_cfg.get("exit_rules") or {}).get(
                "time_exit_dte", 21)
        except Exception:
            self._commission_per_contract = COMMISSION_PER_CONTRACT
            self._slippage_per_share = SLIPPAGE_PER_SHARE
            self._fx_conversion_rate = 0.0
            self._max_capital_at_risk = None
            self._dedup_window_days = DEFAULT_DEDUP_WINDOW_DAYS
            self._max_friction_to_credit = DEFAULT_MAX_FRICTION_TO_CREDIT
            # An unreadable config is not permission to size positions off
            # numbers nobody chose: fall back to one contract, today's behaviour.
            self._sizing_cfg = load_sizing_config(None)
            self._earnings_cfg = load_earnings_gate_config(None)
            self._time_exit_dte = 21
        self._friction_per_share = (2 * self._slippage_per_share) + (2 * self._commission_per_contract / 100.0)
        # Count of trades refused for exceeding the budget this session. Callers
        # print it so a feeder that has gone quiet is visibly gated, not broken.
        self.unaffordable_rejected = 0
        # Same idea for near-duplicate auto-log entries: reported separately so a
        # window that logged nothing shows *which* gate held it back.
        self.duplicate_rejected = 0
        # And for credit trades whose bid-ask cost swallows the credit. Counted
        # separately so a quiet feeder names the gate that held it back.
        self.untradeable_rejected = 0
        # And for positions the account cannot size to a whole contract, or
        # that have no room left under the concurrent cap. A book that has gone
        # quiet must be able to say WHICH rule silenced it.
        self.unsized_rejected = 0
        # Short-premium entries refused for holding through a known earnings
        # date, and — counted separately and just as deliberately — entries the
        # gate could not judge because the calendar has no coverage for that
        # symbol. 72% of the book is in the second state, so a gate reporting
        # only its refusals would look far more active than it is.
        self.through_earnings_rejected = 0
        self.earnings_unknown = 0
        # Entries whose PROJECTED next report lands inside the window. Counted
        # even when it does not refuse, because the report-only default exists
        # precisely to measure how often this fires before it acts.
        self.projected_earnings_flagged = 0
        # The last decision `log_trade` reached, for callers that want to show
        # the size they got rather than re-deriving it. None until one is made.
        self.last_sizing_decision: Optional[SizingDecision] = None
        self._init_db()

    # Exit reasons worth a counterfactual. Stop-outs realise -60.3% from an
    # average peak of +16.6% and time exits realise +8.0% against a +60.2%
    # peak, so both are open questions. A take-profit is not: every trade that
    # peaked at +100% was exited at +100%, leaving nothing to learn.
    SHADOW_EXIT_PREFIXES = ("stop loss", "time exit")

    def open_shadow_window(self, entry_id: int, exit_reason: Optional[str]) -> bool:
        """Start tracking what this trade does after being exited.

        Runs to the original expiry, which is the only date by which the
        question "should I have held?" is fully settled."""
        if not exit_reason:
            return False
        low = str(exit_reason).strip().lower()
        if not any(low.startswith(p) for p in self.SHADOW_EXIT_PREFIXES):
            return False
        try:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "UPDATE trades SET shadow_until = expiration "
                    "WHERE entry_id = ? AND expiration IS NOT NULL", (entry_id,))
                return cur.rowcount > 0
        except Exception:
            return False

    def update_shadow_marks(self, today: Optional[str] = None) -> int:
        """Quote every trade with an open shadow window and record the mark.

        Runs after the live book is updated, on the same chain requests shape:
        one option-chain call per (ticker, expiration), shared by every shadowed
        row on that pair. Failure-safe throughout — a shadow mark is research
        data and must never be able to disturb the live book. Returns the
        number of rows marked."""
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        rows = self.shadowed_positions(today=today)
        if not rows:
            return 0

        marked = 0
        chains: Dict[Tuple[str, str], dict] = {}
        for r in rows:
            ticker, expiration = r.get("ticker"), r.get("expiration")
            strike, opt_type = r.get("strike"), str(r.get("type") or "").lower()
            if not ticker or not expiration or strike is None or not opt_type:
                continue
            key = (ticker, expiration)
            if key not in chains:
                try:
                    chains[key] = self._fetch_chain_quotes(ticker, expiration)
                except Exception:
                    chains[key] = {}
            bid, ask = chains[key].get((round(float(strike), 4), opt_type), (None, None))
            mid = _mid_from_quote(bid, ask)
            if mid and self.shadow_mark(r["entry_id"], mid, today):
                marked += 1
        return marked

    def shadow_mark(self, entry_id: int, price: float, today: str) -> bool:
        """Record where a CLOSED trade's premium went after it was exited.

        Writes only the post_exit_* columns. The realised record — status,
        exit_price, exit_reason, pnl_usd, max_price_seen — is never touched, so
        this adds a counterfactual beside the outcome rather than rewriting it.

        Ignored unless the trade has an open shadow window: `shadow_until` set
        and not yet passed. Returns whether a mark was written."""
        if price is None or price <= 0 or not today:
            return False
        try:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "UPDATE trades SET "
                    "post_exit_max_date = CASE WHEN ? > COALESCE(post_exit_max_price, 0) "
                    "THEN ? ELSE post_exit_max_date END, "
                    "post_exit_max_price = MAX(COALESCE(post_exit_max_price, 0), ?), "
                    "post_exit_last_price = ?, post_exit_last_date = ? "
                    "WHERE entry_id = ? AND shadow_until IS NOT NULL "
                    "AND shadow_until >= ?",
                    (price, today, price, price, today, entry_id, today))
                return cur.rowcount > 0
        except Exception:
            return False

    def shadowed_positions(self, today: Optional[str] = None) -> list:
        """Closed trades whose shadow window is still open, for the updater."""
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM trades WHERE status='CLOSED' "
                    "AND shadow_until IS NOT NULL AND shadow_until >= ? "
                    "AND duplicate_of IS NULL", (today,))]
        except Exception:
            return []

    def _friction_to_credit_ratio(self, trade_dict: dict) -> Optional[float]:
        """Round-trip friction as a fraction of the credit received, or None.

        None whenever the question does not apply — a debit structure, or a row
        with no recorded credit. Returning None rather than 0.0 matters: a
        missing credit is a row the guard should not judge, never a free trade.
        """
        strategy = trade_dict.get("strategy_name") or ""
        n_legs = _CREDIT_LEG_COUNTS.get(strategy)
        if n_legs is None:
            return None  # not a credit structure
        credit = trade_dict.get("net_credit")
        if credit in (None, "", 0):
            credit = trade_dict.get("entry_price")
        try:
            credit = float(credit or 0)
        except (TypeError, ValueError):
            return None
        if credit <= 0:
            return None
        # Friction per share, both ways, every leg — the same shape
        # src.execution_costs prices, using this manager's configured costs.
        #
        # `slippage_per_share` is a flat $0.05 assumption. Measured against
        # archived CBOE quotes for 30 logged Bull Puts the real ENTRY cost
        # alone is $0.35/share, so the flat number understates a two-leg credit
        # spread by roughly 3.5x — on the very guard that decides which trades
        # are too small to survive their own market. When the payload carries
        # real leg quotes, measure instead of assume; when it does not, the
        # flat estimate stands and nothing changes for that caller.
        measured = self._measured_slippage_per_share(trade_dict)
        slip = measured if measured is not None else self._slippage_per_share * n_legs
        friction = ((2 * slip)
                    + (2 * self._commission_per_contract * n_legs / 100.0)
                    + self._fx_per_share(credit))
        return friction / credit

    @staticmethod
    def _measured_slippage_per_share(trade_dict: dict) -> Optional[float]:
        """One-way cost of crossing every leg, from real quotes, or None.

        None whenever any leg lacks a usable two-sided market: a structure
        priced from one real quote and one guess is not measured, and falling
        back to the flat estimate is honest where half-counting is not."""
        legs = trade_dict.get("legs")
        if not legs:
            return None
        from . import execution_truth as _et
        crossed = _et.structure_fill(legs, "cross")
        mid = _et.structure_fill(legs, "mid")
        if crossed is None or mid is None:
            return None
        return mid.price - crossed.price

    def _fx_per_share(self, premium: float) -> float:
        """Currency conversion cost per share on a round trip of this premium.

        A CAD account converts to USD to open and back to CAD to close, paying
        the spread each way. Unlike commission this scales with the money moved,
        so it is the dominant remaining cost on large positions and is invisible
        on a per-contract fee schedule. Zero with a USD account.
        """
        if not self._fx_conversion_rate:
            return 0.0
        return abs(float(premium or 0)) * self._fx_conversion_rate * 2

    @contextmanager
    def _get_connection(self):
        """Yield a sqlite3 connection; commits on success, rolls back on error.

        On disk: a fresh connection per operation, WAL, always closed. That is
        the right shape for a ledger several processes touch — the scheduler,
        the launcher and a scan can all be live at once.

        In memory: ONE connection for the manager's lifetime, never closed. An
        in-memory database lives and dies with its connection, so the on-disk
        pattern made `:memory:` a null database — `_init_db()` built the schema
        on a connection that was immediately discarded and the next call saw
        zero tables. Five test modules pass the literal and only pass because
        they never query it; anything that did would raise "no such table".

        Not `file::memory:?cache=shared`, which would make every `:memory:`
        manager in the process share ONE database. That is the isolation bug
        this keyword is chosen to avoid, and the one that existed until
        `repo_path` stopped resolving it to a real file (PR #30).
        """
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                # check_same_thread=False because the on-disk path connects per
                # call and is therefore thread-safe by construction; a single
                # shared handle must not become the reason a threaded caller
                # breaks only under `:memory:`.
                self._memory_conn = sqlite3.connect(
                    ":memory:", timeout=30.0, check_same_thread=False)
            conn = self._memory_conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return                      # deliberately NOT closed

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Creates the trades table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            type TEXT,
            entry_price REAL,
            quality_score REAL,
            strategy_name TEXT,
            status TEXT,
            exit_price REAL,
            exit_date TEXT,
            pnl_pct REAL,
            pnl_usd REAL
        )
        """
        with self._get_connection() as conn:
            conn.execute(query)
        self._migrate_db()

    def _migrate_db(self):
        """Apply incremental schema migrations up to _SCHEMA_VERSION."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA user_version")
            current_version = cur.fetchone()[0]
            for ver in range(current_version + 1, _SCHEMA_VERSION + 1):
                for sql in _MIGRATIONS.get(ver, []):
                    try:
                        cur.execute(sql)
                    except sqlite3.OperationalError:
                        pass  # column may already exist
                cur.execute(f"PRAGMA user_version = {int(ver)}")

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration for exit rules."""
        _default = {
            "exit_rules": {
                "take_profit": 0.50,
                "stop_loss": -0.25
            }
        }
        try:
            with open(repo_path(self.config_path), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug("Config file not found at %s — using defaults", self.config_path)
            return _default
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in config %s: %s — using defaults", self.config_path, exc)
            return _default

    def _is_recent_duplicate(self, trade_dict: Dict[str, Any]) -> bool:
        """True when an automated feeder is about to re-log a contract it
        already logged inside ``auto_log.dedup_window_days``.

        The per-day dedup in the ``*_if_new`` helpers only sees *today*, so a
        catch-up replay of a missed window — the normal way this system fills
        gaps — logs the same contract again the next morning at a slightly
        different price and slips straight past it. Two rows for one decision
        double-count that decision in every cohort statistic and every dollar
        total (see ``scripts/audit_duplicate_trades.py`` for what the ledger
        already accumulated this way).

        Match key is ``(ticker, strategy_name, expiration, every leg strike)`` —
        price is deliberately NOT in it, because a re-log at a drifted quote is
        exactly the case the per-day dedup misses. The window is measured in
        absolute days from the new row's own entry date, so a backdated catch-up
        is caught the same as a forward one.

        Every leg has to be in the key because ``strike`` alone is only the
        *anchor* leg: ``log_spread`` puts the short strike there and
        ``log_iron_condor`` the short put. Two condors on the same ticker and
        expiration can share a short put and differ entirely on the call wing,
        and two Bear Calls can share a short strike at different widths — those
        are different structures with different risk, and matching on the anchor
        alone would silently refuse the second one. ``IS`` rather than ``=`` so
        the NULL leg columns of a single-leg row match each other.

        Fails open: a DB error here must not stop a trade being logged, so it
        returns False and lets the insert proceed.
        """
        window = int(getattr(self, "_dedup_window_days", 0) or 0)
        if window <= 0:
            return False
        try:
            strike = float(trade_dict["strike"])
        except (KeyError, TypeError, ValueError):
            return False
        ticker = str(trade_dict.get("ticker") or "").upper()
        strategy = str(trade_dict.get("strategy_name") or "")
        expiration = str(trade_dict.get("expiration") or "")
        effective_date = trade_dict.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # NULLIF on the column mirrors _leg_strike on the parameter: a stored 0
        # and a stored NULL are the same "no such leg", and must compare equal.
        leg_clauses = " ".join(
            f"AND ROUND(NULLIF({col}, 0), 4) IS ROUND(?, 4)" for col in _DEDUP_LEG_COLUMNS
        )
        leg_values = [_leg_strike(trade_dict.get(col)) for col in _DEDUP_LEG_COLUMNS]

        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    f"""
                    SELECT entry_id, date, entry_price FROM trades
                    WHERE ticker = ?
                      AND COALESCE(strategy_name, '') = ?
                      AND ROUND(strike, 4) = ROUND(?, 4)
                      AND COALESCE(expiration, '') = ?
                      {leg_clauses}
                      AND date(date) IS NOT NULL
                      AND ABS(julianday(date(date)) - julianday(date(?))) <= ?
                    ORDER BY date DESC, entry_id DESC
                    LIMIT 1
                    """,
                    (ticker, strategy, strike, expiration, *leg_values,
                     effective_date, window),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.debug("Dedup check failed (%s) — allowing the insert", exc)
            return False

        if row is None:
            return False

        self.duplicate_rejected += 1
        prior_id, prior_date, prior_price = row
        try:
            prior_price_s = f"${float(prior_price):.2f}"
        except (TypeError, ValueError):
            prior_price_s = "?"
        _new_price = trade_dict.get("entry_price")
        try:
            # `or 0` would print $0.00 for a missing price, which reads as a
            # real number. An absent price is unknown, and says so.
            new_price_s = ("?" if _new_price is None
                           else f"${float(_new_price):.2f}")
        except (TypeError, ValueError):
            new_price_s = "?"
        msg = (
            f"AUTO-LOG DUPLICATE REFUSED: {strategy} on {ticker} ${strike:g} "
            f"exp {expiration} at {new_price_s} matches entry_id {prior_id} "
            f"logged {str(prior_date)[:10]} at {prior_price_s} "
            f"(within the {window}-day auto_log.dedup_window_days window)"
        )
        logger.warning(msg)
        print(f"  ! {msg}")
        return True

    # Ledgers that size themselves before they get here. The crypto book trades
    # fractional coins and caps each position at $1,000 of unit risk
    # (src/core/sizing.py::capped_quantity); book sizing is calibrated on the
    # equity book's own equity and must not overwrite that.
    _SELF_SIZED_TICKERS = ("BTC", "ETH")

    def _resolve_quantity(
        self, trade_dict: Dict[str, Any], budget: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """How many contracts this position gets, and what it ties up at that size.

        Returns ``(None, None)`` when sizing refuses the trade, matching the
        three refusals above it. Otherwise ``(quantity, capital_at_risk)``, both
        of which are what gets STORED — a `capital_at_risk` computed at one
        contract on a two-contract position would be a number describing
        something other than its label, which is the defect class this whole
        change exists to remove.
        """
        self.last_sizing_decision = None

        def _risk_at(qty: float) -> Optional[float]:
            # The multi-leg fields are passed deliberately: without
            # spread_width/net_credit a credit structure with no stored max loss
            # prices as None (unsizable) rather than as width - credit.
            return capital_at_risk(
                strategy_name=trade_dict["strategy_name"],
                entry_price=trade_dict.get("entry_price"),
                strike=trade_dict.get("strike"),
                max_loss_usd=trade_dict.get("max_loss_usd"),
                spread_width=trade_dict.get("spread_width"),
                net_credit=trade_dict.get("net_credit"),
                quantity=qty,
                ticker=trade_dict.get("ticker"),
            )

        try:
            _caller = trade_dict.get("quantity")
            caller_qty = (float(_caller)
                          if _caller is not None and np.isfinite(float(_caller))
                          and float(_caller) > 0 else None)
        except (TypeError, ValueError):
            caller_qty = None
        ticker = str(trade_dict.get("ticker") or "").upper()

        # Three ways past the sizer, all of them deliberate: a self-sizing
        # ledger, a caller that already chose a size, and the manual override.
        if (ticker in self._SELF_SIZED_TICKERS
                or (caller_qty is not None and caller_qty != 1.0)
                or trade_dict.get("allow_unsized")):
            qty = caller_qty if caller_qty is not None else 1.0
            return qty, _risk_at(qty)

        per_contract = _risk_at(1.0)
        with self._get_connection() as conn:
            equity = book_equity(conn, self._sizing_cfg)
            exposure = open_risk(conn, self._sizing_cfg)
        decision = size(per_contract, equity, exposure, self._sizing_cfg)
        self.last_sizing_decision = decision

        if decision.contracts < 1:
            self.unsized_rejected += 1
            shown = (f"${decision.risk_per_contract:,.0f}"
                     if decision.risk_per_contract is not None else "unbounded")
            print(
                f"Skipped {trade_dict['strategy_name']} on "
                f"{trade_dict.get('ticker')}: {shown} of risk per contract "
                f"({decision.reason}) against ${equity:,.0f} of book equity, "
                f"${equity * float(self._sizing_cfg['max_risk_pct']):,.0f} per "
                f"trade and ${exposure:,.0f} already at risk"
            )
            return None, None

        contracts = decision.contracts
        # The budget gate ran BEFORE this and cleared ONE contract. Multiplying
        # up must not carry the position past the cap it just cleared, or the
        # row would store a capital_at_risk above its own budget_at_entry.
        # Clamping can never refuse a trade the budget already admitted.
        if (contracts > 1 and per_contract and budget is not None
                and not trade_dict.get("allow_unaffordable")
                and not within_budget(per_contract * contracts, budget)):
            contracts = max(1, int(float(budget) // per_contract))

        return float(contracts), _risk_at(float(contracts))

    def log_trade(self, trade_dict: Dict[str, Any]):
        """
        Logs a new paper trade to the SQLite database.
        Required keys: ticker, expiration, strike, type, entry_price, quality_score, strategy_name
        Optional keys (entry context):
            ai_score, ai_confidence, entry_iv, entry_delta, entry_gamma, entry_vega, entry_theta, dividend_yield
        Optional keys (per-component scores — full 27-weight coverage for IC calibration):
            pop_score, ev_score, rr_score, liquidity_score, momentum_score, iv_rank_score, theta_score,
            iv_edge_score, vrp_score, iv_mispricing_score, skew_align_score, vega_risk_score, term_structure_score,
            catalyst_score, em_realism_score, gamma_theta_score, gex_score, gamma_magnitude_score,
            gamma_pin_score, iv_velocity_score, max_pain_score, oi_change_score, option_rvol_score,
            pcr_score, sentiment_score_norm, spread_score, trader_pref_score

        Returns True if the row was inserted, False if it was refused for
        exceeding ``auto_log.max_capital_at_risk``. Pass
        ``allow_unaffordable=True`` to log a deliberate over-budget entry.

        Also refused when POSITION SIZING cannot fit a whole contract inside
        ``position_sizing.max_risk_pct`` of book equity, or inside what is left
        under ``max_open_risk_pct`` — a position too big to size is one the
        account cannot afford. ``allow_unsized=True`` bypasses that for one
        deliberate manual entry and logs at quantity 1. The stored ``quantity``
        and ``capital_at_risk`` describe the position actually taken.

        Set ``auto_log=True`` for entries written by an automated feeder: that
        arms the near-duplicate guard below. It is off by default so a manual,
        deliberate entry is never second-guessed.
        """
        if not trade_dict.get("strategy_name"):
            raise ValueError("strategy_name is required; must include 'short'/'long' to set P&L direction")
        if float(trade_dict.get("entry_price", 0)) <= 0:
            raise ValueError(f"Cannot log trade: entry_price must be > 0, got {trade_dict.get('entry_price')}")

        # Near-duplicate gate — automated feeders only.
        if trade_dict.get("auto_log") and self._is_recent_duplicate(trade_dict):
            return False

        # Budget gate. The feeder had no size limit, so 160 of 407 closed cohort
        # trades tied up more than the whole account and carried every dollar of
        # the book's loss. Refuse them at the door rather than measuring a
        # population the account could never trade.
        risk = capital_at_risk(
            strategy_name=trade_dict["strategy_name"],
            entry_price=trade_dict.get("entry_price"),
            strike=trade_dict.get("strike"),
            max_loss_usd=trade_dict.get("max_loss_usd"),
            quantity=trade_dict.get("quantity", 1.0),
            ticker=trade_dict.get("ticker"),
        )
        # Tradeability: a credit trade whose round-trip friction eats the credit
        # cannot profit at any win rate. Measured 2026-07-31 over the logged
        # cohort, 31 of 188 short-premium trades were in that state — micro
        # spreads with $57 of median capital at risk against ~$65 of spread. The
        # affordability gate refuses positions too LARGE for the account; this
        # one refuses positions too SMALL to survive their own market.
        # The friction gate below was calibrated on DTE 10-67 and says nothing
        # about tenors it never saw. Refusing FIRST, with the real reason,
        # keeps "this candidate is too expensive" separate from "this threshold
        # was never measured here" — past 250 DTE the gate refuses a quarter to
        # a half of all candidates, and reporting that as a spread verdict
        # would be inventing a judgement.
        _dte = _entry_dte(trade_dict)
        if not trade_dict.get("allow_untradeable") and not _in_calibration(_dte):
            self.untradeable_rejected += 1
            print(
                f"Skipped {trade_dict['strategy_name']} on {trade_dict.get('ticker')}: "
                f"{_dte} DTE — {_OUT_OF_RANGE_REASON}"
            )
            return False

        _fric_ratio = self._friction_to_credit_ratio(trade_dict)
        if (not trade_dict.get("allow_untradeable")
                and self._max_friction_to_credit is not None
                and _fric_ratio is not None
                and _fric_ratio > self._max_friction_to_credit):
            self.untradeable_rejected += 1
            print(
                f"Skipped {trade_dict['strategy_name']} on {trade_dict.get('ticker')}: "
                f"round-trip friction is {_fric_ratio:.0%} of the credit "
                f"(limit {self._max_friction_to_credit:.0%}) — the spread eats the trade"
            )
            return False

        # The budget that governs THIS trade. Key presence is the signal:
        # present-and-None means the operator explicitly chose no limit;
        # absent means no prompt was ever reached (cron, --auto, a pipe) and
        # the config value applies. A run that never saw the prompt must not
        # be treated as having chosen "no limit".
        _budget = (trade_dict["budget_at_entry"]
                   if "budget_at_entry" in trade_dict
                   else self._max_capital_at_risk)
        if not trade_dict.get("allow_unaffordable") and not within_budget(
            risk, _budget
        ):
            self.unaffordable_rejected += 1
            shown = f"${risk:,.0f}" if risk is not None else "unbounded"
            print(
                f"Skipped {trade_dict['strategy_name']} on {trade_dict.get('ticker')}: "
                f"capital at risk {shown} exceeds the "
                f"${_budget:,.0f} budget"
            )
            return False

        # Earnings. A short-premium structure held across a dated, public,
        # binary event is short a gap nobody priced. Refused here rather than
        # ranked down, because a score penalty is a preference and this is a
        # risk the account should not take at any rank. See src/earnings_gate.py
        # for the WMT trade that prompted it, and for why UNKNOWN is not CLEAR.
        if not trade_dict.get("allow_through_earnings"):
            _earn = verdict_for_trade(
                strategy_name=trade_dict["strategy_name"],
                symbol=trade_dict.get("ticker"),
                entry_date=str(trade_dict.get("date") or
                               datetime.now().strftime("%Y-%m-%d"))[:10],
                expiration=trade_dict.get("expiration"),
                time_exit_dte=self._time_exit_dte,
                cfg=self._earnings_cfg,
            )
            # Kept for analysis even when it doesn't refuse — see migration 23
            # in _MIGRATIONS. None here (also reachable when the config leaves
            # the gate disabled, or allow_through_earnings bypassed this whole
            # branch) means the check never ran — never read it as "clear".
            trade_dict["earnings_state"] = _earn
            if _earn == PROJECTED_THROUGH:
                # An estimate, so it is COUNTED whether or not it refuses —
                # the point of the report-only default is to see how often it
                # fires on the live board before it starts acting.
                self.projected_earnings_flagged += 1
                _proj = project_next_earnings(cached_earnings_dates(
                    str(trade_dict.get("ticker") or ""),
                    str(self._earnings_cfg.get("cache_path"))))
                print(
                    f"  ! {trade_dict['strategy_name']} on "
                    f"{trade_dict.get('ticker')}: no announced date, but its "
                    f"cadence projects a report around {_proj} — inside this "
                    f"trade's window"
                )
            if refuses(_earn, self._earnings_cfg):
                self.through_earnings_rejected += 1
                _when = _earnings_dates_in_window(
                    trade_dict, self._earnings_cfg, self._time_exit_dte)
                print(
                    f"Skipped {trade_dict['strategy_name']} on "
                    f"{trade_dict.get('ticker')}: holds through earnings on "
                    f"{', '.join(_when) if _when else 'a known date'} — "
                    "selling premium across a dated binary event"
                )
                return False
            if _earn == UNKNOWN:
                self.earnings_unknown += 1
                logger.info(
                    "earnings gate: no calendar coverage for %s at %s, logging "
                    "unchecked — run `python -m src.dolt_earnings --dates %s` "
                    "to give the gate something to work with",
                    trade_dict.get("ticker"), trade_dict.get("expiration"),
                    trade_dict.get("ticker"))

        # Sizing — the LAST gate, deliberately. A trade that fails budget or
        # tradeability is refused for that reason rather than for its size, so
        # refusal reasons stay diagnostic.
        #
        # Risk per contract comes from capital_at_risk with the multi-leg
        # fields, never from entry premium: a Bull Put risks `width - credit`
        # and receives its credit, so premium-based sizing (what
        # src/execution/sizing.py does, for long calls) would price the trade at
        # a quarter of its loss and buy several times too many contracts.
        _qty, _sized_risk = self._resolve_quantity(trade_dict, _budget)
        if _qty is None:
            return False
        if _sized_risk is not None:
            risk = _sized_risk

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = """
        INSERT INTO trades (
            date, ticker, expiration, strike, type,
            entry_price, quality_score, strategy_name,
            status, exit_price, exit_date, pnl_pct,
            ai_score, ai_confidence,
            entry_iv, entry_delta, entry_gamma, entry_vega, entry_theta, dividend_yield,
            pop_score, ev_score, rr_score, liquidity_score, momentum_score, iv_rank_score, theta_score,
            iv_edge_score, vrp_score, iv_mispricing_score, skew_align_score, vega_risk_score, term_structure_score,
            catalyst_score, em_realism_score, gamma_theta_score, gex_score, gamma_magnitude_score,
            gamma_pin_score, iv_velocity_score, max_pain_score, oi_change_score, option_rvol_score,
            pcr_score, sentiment_score_norm, spread_score, trader_pref_score,
            score_adjustments,
            entry_ev_net, entry_ev_gross, entry_ev_cost, entry_ev_noise,
            weight_profile,
            long_strike, spread_width, net_credit, max_profit_usd, max_loss_usd,
            short_call_strike, long_call_strike, short_put_strike, long_put_strike, net_delta,
            paper_only, era, lottery_edge, capital_at_risk, budget_at_entry,
            quantity,
            short_bid_entry, short_ask_entry, long_bid_entry, long_ask_entry,
            short_bid_exit, short_ask_exit, long_bid_exit, long_ask_exit,
            short_put_bid_entry, short_put_ask_entry, long_put_bid_entry, long_put_ask_entry,
            short_call_bid_entry, short_call_ask_entry, long_call_bid_entry, long_call_ask_entry,
            short_put_bid_exit, short_put_ask_exit, long_put_bid_exit, long_put_ask_exit,
            short_call_bid_exit, short_call_ask_exit, long_call_bid_exit, long_call_ask_exit,
            earnings_state
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?
        )
        """

        def _float_or_none(key):
            v = trade_dict.get(key)
            if v is None:
                return None
            try:
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except (ValueError, TypeError):
                return None

        params = (
            trade_dict.get("date", now),
            trade_dict["ticker"].upper(),
            trade_dict["expiration"],
            float(trade_dict["strike"]),
            trade_dict["type"].lower(),
            float(trade_dict["entry_price"]),
            float(trade_dict["quality_score"]),
            trade_dict["strategy_name"],
            "OPEN",
            None,   # exit_price
            None,   # exit_date — NULL until close (was "" historically; broke IS NULL filters)
            None,   # pnl_pct
            _float_or_none("ai_score"),        # None if not ranked yet
            _float_or_none("ai_confidence"),   # None if not ranked yet
            _float_or_none("entry_iv"),
            _float_or_none("entry_delta"),
            _float_or_none("entry_gamma"),
            _float_or_none("entry_vega"),
            _float_or_none("entry_theta"),
            _float_or_none("dividend_yield"),
            _float_or_none("pop_score"),
            _float_or_none("ev_score"),
            _float_or_none("rr_score"),
            _float_or_none("liquidity_score"),
            _float_or_none("momentum_score"),
            _float_or_none("iv_rank_score"),
            _float_or_none("theta_score"),
            _float_or_none("iv_edge_score"),
            _float_or_none("vrp_score"),
            _float_or_none("iv_mispricing_score"),
            _float_or_none("skew_align_score"),
            _float_or_none("vega_risk_score"),
            _float_or_none("term_structure_score"),
            _float_or_none("catalyst_score"),
            _float_or_none("em_realism_score"),
            _float_or_none("gamma_theta_score"),
            _float_or_none("gex_score"),
            _float_or_none("gamma_magnitude_score"),
            _float_or_none("gamma_pin_score"),
            _float_or_none("iv_velocity_score"),
            _float_or_none("max_pain_score"),
            _float_or_none("oi_change_score"),
            _float_or_none("option_rvol_score"),
            _float_or_none("pcr_score"),
            _float_or_none("sentiment_score_norm"),
            _float_or_none("spread_score"),
            _float_or_none("trader_pref_score"),
            (trade_dict.get("score_adjustments") or None),
            # The EV level the verdict was taken on, in dollars per contract.
            # `ev_score` beside it is a within-scan rank and cannot answer
            # "how big was the edge" — only these can.
            _float_or_none("ev_per_contract"),
            _float_or_none("ev_gross_per_contract"),
            _float_or_none("ev_cost_per_contract"),
            _float_or_none("ev_noise"),
            trade_dict.get("weight_profile"),
            _float_or_none("long_strike"),
            _float_or_none("spread_width"),
            _float_or_none("net_credit"),
            _float_or_none("max_profit_usd"),
            _float_or_none("max_loss_usd"),
            _float_or_none("short_call_strike"),
            _float_or_none("long_call_strike"),
            _float_or_none("short_put_strike"),
            _float_or_none("long_put_strike"),
            _float_or_none("net_delta"),
            int(trade_dict["paper_only"]) if trade_dict.get("paper_only") is not None else 0,
            # New trades belong to the post-overhaul 'finalized' era unless told otherwise.
            trade_dict.get("era", "finalized"),
            (int(bool(trade_dict["lottery_edge"])) if trade_dict.get("lottery_edge") is not None else None),
            risk,
            _budget,   # budget_at_entry — NULL means no limit was in force
            # Contracts. Until 2026-08-19 this column was never written at all:
            # every row inherited the migration's DEFAULT 1.0, so bet size was
            # the option premium and the book's headline P&L was a sizing
            # artifact. See src/book_sizing.py.
            _qty,
            # Per-leg bid/ask, entry and exit (schema v23). _exit columns are
            # always None here — a trade is always OPEN when first logged;
            # the exit-enforcement loop fills those in via UPDATE on close.
            _float_or_none("short_bid_entry"), _float_or_none("short_ask_entry"),
            _float_or_none("long_bid_entry"), _float_or_none("long_ask_entry"),
            _float_or_none("short_bid_exit"), _float_or_none("short_ask_exit"),
            _float_or_none("long_bid_exit"), _float_or_none("long_ask_exit"),
            _float_or_none("short_put_bid_entry"), _float_or_none("short_put_ask_entry"),
            _float_or_none("long_put_bid_entry"), _float_or_none("long_put_ask_entry"),
            _float_or_none("short_call_bid_entry"), _float_or_none("short_call_ask_entry"),
            _float_or_none("long_call_bid_entry"), _float_or_none("long_call_ask_entry"),
            _float_or_none("short_put_bid_exit"), _float_or_none("short_put_ask_exit"),
            _float_or_none("long_put_bid_exit"), _float_or_none("long_put_ask_exit"),
            _float_or_none("short_call_bid_exit"), _float_or_none("short_call_ask_exit"),
            _float_or_none("long_call_bid_exit"), _float_or_none("long_call_ask_exit"),
            trade_dict.get("earnings_state"),
        )

        with self._get_connection() as conn:
            conn.execute(query, params)

        # Record the size back so a caller holding this dict can report what it
        # actually bought. `log_spread`/`log_iron_condor` hand us a COPY, so
        # they see it only via `last_sizing_decision`.
        trade_dict["quantity"] = _qty
        _lots = f" x{_qty:g}" if _qty != 1.0 else ""
        print(f"Logged {trade_dict['strategy_name']} on {trade_dict['ticker']}{_lots} at ${float(trade_dict['entry_price']):.2f}")
        return True

    def log_trade_if_new(self, trade_dict: Dict[str, Any], auto_log: bool = False) -> bool:
        """Insert a paper trade unless an identical row already exists.

        Dedup key: ``(trade date, ticker, strike, expiration, type, strategy_name,
        weight_profile)``. ``strategy_name`` participates so a Long Call and a
        Short Call at the same strike on the same day don't collide. The trade
        date is whatever ``log_trade`` would store — either the caller's explicit
        ``trade_dict["date"]`` or today's timestamp. ``weight_profile`` may be
        ``None`` for untagged trades — NULL-equal rows still dedup against each
        other because ``IS`` is used instead of ``=``.

        That key only spans one calendar day. ``auto_log=True`` additionally
        arms the multi-day near-duplicate guard in ``log_trade``, which is what
        a catch-up replay needs; it defaults off because this helper is also
        used by the interactive log-trades menu.

        Returns ``True`` if inserted, ``False`` if skipped as duplicate.
        """
        if auto_log:
            trade_dict = dict(trade_dict, auto_log=True)
        ticker = trade_dict["ticker"].upper()
        typ = trade_dict["type"].lower()
        strike = float(trade_dict["strike"])
        expiration = trade_dict["expiration"]
        strategy = trade_dict.get("strategy_name") or ""
        profile = trade_dict.get("weight_profile")
        effective_date = trade_dict.get("date") or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM trades
                WHERE ticker = ?
                  AND strike = ?
                  AND expiration = ?
                  AND type = ?
                  AND COALESCE(strategy_name, '') = ?
                  AND weight_profile IS ?
                  AND date(date) = date(?)
                LIMIT 1
                """,
                (ticker, strike, expiration, typ, strategy, profile, effective_date),
            ).fetchone()
        if row is not None:
            return False
        # False also when the budget gate refuses it — either way nothing was written.
        return self.log_trade(trade_dict)

    def log_spread(self, spread_dict: dict) -> bool:
        """Log a multi-leg credit spread as a single paper trade.

        Routes through ``log_trade`` so component scores, Greeks, and weight_profile
        all persist. Required keys: ``ticker, expiration, short_strike, long_strike,
        type, net_credit``. Optional: every component-score key accepted by
        ``log_trade`` plus ``max_profit``, ``max_loss``, ``quality_score``,
        ``weight_profile``, entry Greeks.

        ``type`` becomes ``strategy_name`` ("Bull Put" / "Bear Call"). The DB
        ``type`` column gets the underlying option type ("put" / "call") inferred
        from the strategy name.
        """
        spread_type = str(spread_dict.get("type", "Spread"))
        opt_type = "put" if "put" in spread_type.lower() else "call"
        short_strike = float(spread_dict.get("short_strike") or 0)
        long_strike = float(spread_dict.get("long_strike") or 0)
        net_credit = float(spread_dict.get("net_credit") or 0)
        max_profit = spread_dict.get("max_profit")
        max_loss = spread_dict.get("max_loss")

        if net_credit <= 0:
            raise ValueError(f"log_spread: net_credit must be > 0, got {net_credit}")

        trade_dict = dict(spread_dict)  # copy so we don't mutate caller's dict
        trade_dict["strike"] = short_strike
        trade_dict["type"] = opt_type
        trade_dict["entry_price"] = net_credit
        trade_dict["strategy_name"] = spread_type
        trade_dict["long_strike"] = long_strike
        trade_dict["spread_width"] = abs(short_strike - long_strike)
        trade_dict["net_credit"] = net_credit
        # Per-leg entry quotes, when the caller supplied them (schema v23).
        # Absent keys must stay absent here too — trade_dict.get() at
        # log_trade's INSERT layer is what turns "key missing" into NULL
        # rather than 0, and inserting an explicit None here would do the
        # same thing, so this is deliberately a straight pass-through, not
        # a coalesce.
        for _key in ("short_bid", "short_ask", "long_bid", "long_ask"):
            if _key in spread_dict:
                trade_dict[f"{_key}_entry"] = spread_dict[_key]
        if max_profit is not None:
            trade_dict["max_profit_usd"] = float(max_profit)
        if max_loss is not None:
            trade_dict["max_loss_usd"] = float(max_loss)
        trade_dict.setdefault("quality_score", 0.5)
        # log_trade requires ticker — make sure case-normalized
        trade_dict["ticker"] = str(spread_dict.get("ticker", "")).upper()

        return self.log_trade(trade_dict)

    def log_iron_condor(self, condor_dict: dict) -> bool:
        """Log an iron condor (4-leg) as a single paper trade.

        Required keys: ``ticker, expiration, short_put_strike, long_put_strike,
        short_call_strike, long_call_strike, total_credit``. The DB ``strike``
        column holds the short-put strike (canonical anchor for dedup); 4-leg
        details persist in named columns.
        """
        sp_strike = float(condor_dict.get("short_put_strike") or 0)
        lp_strike = float(condor_dict.get("long_put_strike") or 0)
        sc_strike = float(condor_dict.get("short_call_strike") or 0)
        lc_strike = float(condor_dict.get("long_call_strike") or 0)
        total_credit = float(condor_dict.get("total_credit") or condor_dict.get("net_credit") or 0)
        max_risk = condor_dict.get("max_risk") or condor_dict.get("max_loss")

        if total_credit <= 0:
            raise ValueError(f"log_iron_condor: total_credit must be > 0, got {total_credit}")

        put_width = sp_strike - lp_strike
        call_width = lc_strike - sc_strike
        spread_width = max(put_width, call_width)

        trade_dict = dict(condor_dict)
        trade_dict["strike"] = sp_strike  # canonical anchor (short put)
        trade_dict["type"] = "put"        # short-put-anchored (matches dedup convention)
        trade_dict["entry_price"] = total_credit
        trade_dict["strategy_name"] = "Iron Condor"
        trade_dict["long_strike"] = lp_strike  # paired long for the anchor leg
        trade_dict["short_put_strike"] = sp_strike
        trade_dict["long_put_strike"] = lp_strike
        trade_dict["short_call_strike"] = sc_strike
        trade_dict["long_call_strike"] = lc_strike
        trade_dict["spread_width"] = spread_width
        trade_dict["net_credit"] = total_credit
        # Per-leg entry quotes, when the caller supplied them (schema v23).
        for _key in ("short_put_bid", "short_put_ask", "long_put_bid",
                    "long_put_ask", "short_call_bid", "short_call_ask",
                    "long_call_bid", "long_call_ask"):
            if _key in condor_dict:
                trade_dict[f"{_key}_entry"] = condor_dict[_key]
        if max_risk is not None:
            trade_dict["max_loss_usd"] = float(max_risk)
        if condor_dict.get("max_profit") is not None:
            trade_dict["max_profit_usd"] = float(condor_dict["max_profit"])
        if condor_dict.get("net_delta") is not None:
            trade_dict["net_delta"] = float(condor_dict["net_delta"])
        trade_dict.setdefault("quality_score", 0.5)
        trade_dict["ticker"] = str(condor_dict.get("ticker", "")).upper()

        return self.log_trade(trade_dict)

    def log_spread_if_new(self, spread_dict: dict, auto_log: bool = False) -> bool:
        """Insert a credit spread unless an identical OPEN row already exists for
        the same (date, ticker, expiration, short_strike, long_strike, strategy, profile).
        Returns True if inserted, False if duplicate.

        ``auto_log=True`` also arms the multi-day near-duplicate guard (see
        ``log_trade_if_new``); default off so manual entries are never refused.
        """
        if auto_log:
            spread_dict = dict(spread_dict, auto_log=True)
        ticker = str(spread_dict.get("ticker", "")).upper()
        strategy = str(spread_dict.get("type", "Spread"))
        short_strike = float(spread_dict.get("short_strike") or 0)
        long_strike = float(spread_dict.get("long_strike") or 0)
        expiration = spread_dict.get("expiration", "")
        profile = spread_dict.get("weight_profile")
        effective_date = spread_dict.get("date") or datetime.now().strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM trades
                WHERE ticker = ?
                  AND strike = ?
                  AND long_strike = ?
                  AND expiration = ?
                  AND strategy_name = ?
                  AND weight_profile IS ?
                  AND date(date) = date(?)
                LIMIT 1
                """,
                (ticker, short_strike, long_strike, expiration, strategy, profile, effective_date),
            ).fetchone()
        if row is not None:
            return False
        # False also when the budget gate refuses it — nothing was written.
        return self.log_spread(spread_dict)

    def log_iron_condor_if_new(self, condor_dict: dict, auto_log: bool = False) -> bool:
        """Same dedup pattern as log_spread_if_new but for 4-leg iron condors,
        including the ``auto_log`` opt-in to the multi-day guard."""
        if auto_log:
            condor_dict = dict(condor_dict, auto_log=True)
        ticker = str(condor_dict.get("ticker", "")).upper()
        sp_strike = float(condor_dict.get("short_put_strike") or 0)
        lp_strike = float(condor_dict.get("long_put_strike") or 0)
        sc_strike = float(condor_dict.get("short_call_strike") or 0)
        lc_strike = float(condor_dict.get("long_call_strike") or 0)
        expiration = condor_dict.get("expiration", "")
        profile = condor_dict.get("weight_profile")
        effective_date = condor_dict.get("date") or datetime.now().strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM trades
                WHERE ticker = ?
                  AND strategy_name = 'Iron Condor'
                  AND short_put_strike = ?
                  AND long_put_strike = ?
                  AND short_call_strike = ?
                  AND long_call_strike = ?
                  AND expiration = ?
                  AND weight_profile IS ?
                  AND date(date) = date(?)
                LIMIT 1
                """,
                (ticker, sp_strike, lp_strike, sc_strike, lc_strike, expiration, profile, effective_date),
            ).fetchone()
        if row is not None:
            return False
        # False also when the budget gate refuses it — nothing was written.
        return self.log_iron_condor(condor_dict)

    def _get_option_symbol(self, ticker: str, expiration: str, strike: float, option_type: str) -> str:
        """Generates a yfinance-compatible option symbol."""
        try:
            exp_date = pd.to_datetime(expiration).strftime('%y%m%d')
            otype = 'C' if option_type.lower() == 'call' else 'P'
            strike_price = f"{int(strike * 1000):08d}"
            return f"{ticker}{exp_date}{otype}{strike_price}"
        except Exception:
            return ""

    def _get_spread_slippage(self, ticker: str, expiration: str, strike: float, option_type: str, entry_price: float) -> float:
        """Return per-share slippage as 30% of the bid-ask spread width, capped at $0.50 and floored at self._slippage_per_share."""
        try:
            symbol = self._get_option_symbol(ticker, expiration, strike, option_type)
            if not symbol:
                return self._slippage_per_share
            yf, session = _get_yf_and_session()
            tkr = yf.Ticker(symbol, session=session)
            bid = getattr(tkr.fast_info, "bid", None)
            ask = getattr(tkr.fast_info, "ask", None)
            if bid is None or ask is None:
                # fallback: use 10% of entry price as spread estimate
                spread = entry_price * 0.10
            else:
                spread = max(0.0, float(ask) - float(bid))
            slippage = spread * 0.30
            return max(self._slippage_per_share, min(slippage, 0.50))
        except Exception:
            return self._slippage_per_share

    # ── Marking open legs ────────────────────────────────────────────────────
    # Split into three seams (chain quotes / traded prices / model) so each
    # rung of the fallback ladder can be exercised on its own in tests without
    # a network hop, and so the chain is fetched ONCE per (ticker, expiration)
    # rather than once per leg.

    def _fetch_chain_quotes(
        self, ticker: str, expiration: str
    ) -> Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]]:
        """Live bid/ask per (strike, option_type) for one (ticker, expiration).

        One option-chain request serves every leg of every row on that pair,
        mirroring how _get_spread_slippage reads the book. Returns {} on any
        failure; the caller then falls through to the traded-price rungs.
        """
        quotes: Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]] = {}
        memo_key = (ticker, str(expiration)[:10])
        cached = _CHAIN_QUOTE_CACHE.get(memo_key)
        if cached is not None:
            memo_quotes, ts = cached
            if _pm_time.monotonic() - ts < _CHAIN_QUOTE_TTL:
                return dict(memo_quotes)
        try:
            yf, session = _get_yf_and_session()
            tkr = yf.Ticker(ticker, session=session)
            chain = tkr.option_chain(str(expiration)[:10])
        except Exception as exc:
            logger.debug("Chain quote fetch failed for %s %s: %s", ticker, expiration, exc)
            return quotes
        for tbl, opt_t in ((getattr(chain, "calls", None), "call"),
                           (getattr(chain, "puts", None), "put")):
            try:
                if tbl is None or getattr(tbl, "empty", True):
                    continue
                strikes = tbl["strike"].tolist()
                bids = tbl["bid"].tolist() if "bid" in tbl else [None] * len(strikes)
                asks = tbl["ask"].tolist() if "ask" in tbl else [None] * len(strikes)
            except Exception as exc:
                logger.debug("Chain quote parse failed for %s %s: %s", ticker, expiration, exc)
                continue
            for k, b, a in zip(strikes, bids, asks):
                try:
                    quotes[(round(float(k), 4), opt_t)] = (b, a)
                except (TypeError, ValueError):
                    continue
        # An empty result is not cached: a transient outage must not pin this
        # pair to "no quotes" and silently degrade every mark for the TTL.
        if quotes:
            _CHAIN_QUOTE_CACHE[memo_key] = (dict(quotes), _pm_time.monotonic())
        return quotes

    def _fetch_traded_mark(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """Last trade, then daily close, for one contract symbol.

        Returns (price, MARK_LAST | MARK_CLOSE) or (None, None). Both rungs are
        real prints, so both may fire an exit — they are merely stale, not
        fabricated.
        """
        import warnings

        try:
            yf, session = _get_yf_and_session()
            tkr = yf.Ticker(symbol, session=session)
            price = None
            try:
                price = getattr(tkr.fast_info, "last_price", None)
            except Exception:
                price = None
            if price is not None and not np.isnan(price) and price > 0:
                return float(price), MARK_LAST
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = tkr.history(period="1d")
            if hist is not None and not hist.empty:
                close = float(hist["Close"].iloc[-1])
                if not np.isnan(close) and close > 0:
                    return close, MARK_CLOSE
        except Exception as exc:
            logger.debug("Traded mark fetch failed for %s: %s", symbol, exc)
        return None, None

    def _model_mark(
        self,
        option_type: str,
        spot: Optional[float],
        strike: float,
        expiration: str,
        sigma: float,
    ) -> Optional[float]:
        """Black-Scholes/American model price at `sigma` — the last resort."""
        if not spot or float(spot) <= 0:
            return None
        try:
            from .utils import american_price
            T = max((datetime.strptime(str(expiration)[:10], "%Y-%m-%d") - datetime.now()).days / 365, 1 / 365)
            rfr = _get_rfr() if _HAS_RFR else 0.045
            price = american_price(option_type, float(spot), float(strike), T, rfr, float(sigma))
        except Exception as exc:
            logger.debug("Model mark failed for %s %s %s: %s", option_type, strike, expiration, exc)
            return None
        if price is None or np.isnan(price) or price <= 0:
            return None
        logger.debug(
            "Model mark used for %s %s %s at sigma=%.4f", option_type, strike, expiration, sigma
        )
        return float(price)

    def _mark_option_leg(
        self,
        key: Tuple[str, str, float, str],
        symbol: str,
        quotes: Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]],
        spot: Optional[float],
        sigma: float,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Mark one leg: live mid -> last trade -> daily close -> model.

        Returns (price, source) or (None, None) if even the model has nothing.
        """
        ticker, expiration, strike, option_type = key
        bid, ask = quotes.get((round(float(strike), 4), str(option_type).lower()), (None, None))
        mid = _mid_from_quote(bid, ask)
        if mid is not None:
            return mid, MARK_MID
        price, source = self._fetch_traded_mark(symbol)
        if price is not None:
            return price, source
        modelled = self._model_mark(option_type, spot, strike, expiration, sigma)
        if modelled is not None:
            return modelled, MARK_MODEL
        return None, None

    # Trade-count thresholds at which a calibration notice should fire (once each)
    _CALIBRATION_THRESHOLDS: Tuple[int, ...] = (25, 50, 100, 200, 400, 800)

    def _calibration_marker_path(self) -> str:
        """Path to the marker file recording the highest threshold already announced."""
        return f"{self.db_path}.calibration_marker.json"

    def _maybe_emit_calibration_threshold_notice(self) -> None:
        """
        After a close-trade run, if the closed-trade count crosses one of the
        CALIBRATION_THRESHOLDS for the first time, print a one-line notice
        pointing the user at `python -m src.backtester --calibrate`.
        Persists state in a marker file so it never re-fires for the same threshold.
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE status='CLOSED' "
                    "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL"
                ).fetchone()
            closed_count = int(row[0]) if row and row[0] is not None else 0
        except Exception as exc:
            logger.debug("Calibration notice: closed-count query failed: %s", exc)
            return

        # Highest threshold the new closed_count has reached
        crossed = max((t for t in self._CALIBRATION_THRESHOLDS if closed_count >= t), default=0)
        if crossed == 0:
            return

        marker_path = self._calibration_marker_path()
        last_fired = 0
        try:
            if os.path.exists(marker_path):
                with open(marker_path, "r") as f:
                    last_fired = int(json.load(f).get("highest_threshold_fired", 0))
        except Exception as exc:
            logger.debug("Calibration notice: marker read failed: %s", exc)
            last_fired = 0

        if crossed <= last_fired:
            return

        # Cross — emit notice and persist new high-water mark
        try:
            with open(marker_path, "w") as f:
                json.dump(
                    {
                        "highest_threshold_fired": crossed,
                        "closed_count_at_fire": closed_count,
                        "fired_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                )
        except Exception as exc:
            logger.debug("Calibration notice: marker write failed: %s", exc)

        bar = "─" * 70
        print()
        print(f"  {bar}")
        print(f"  📊 CALIBRATION MILESTONE — {closed_count} closed paper trades logged")
        print(f"     Reached the {crossed}-trade threshold. Component IC is now")
        print(f"     statistically meaningful enough to recalibrate composite_weights.")
        print(f"     Review:  python -m src.backtester --calibrate")
        print(f"     Apply:   python -m src.backtester --calibrate --apply")
        print(f"  {bar}")
        print()

    def update_positions(self):
        """Updates all OPEN positions using SQLite and checks context-aware exit rules.

        Uses strategy-aware rules from config.exit_rules:
          - short single-leg: DTE-tiered TP, strike-breach + premium-multiple + delta-multiple stops
          - spreads: 50% TP / 1× credit SL
          - long single-leg: 100% TP or deep-ITM delta / -50% SL
        """
        config = self._load_config()
        rules = _normalize_exit_rules(config)
        time_exit_dte = rules["time_exit_dte"]
        spread_tp = rules["spread"]["tp"]
        spread_sl = rules["spread"]["sl"]
        try:
            rfr = _get_rfr() if _HAS_RFR else 0.045
        except Exception:
            rfr = 0.045

        # Fetch open trades
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades WHERE status='OPEN'")
            open_trades = cursor.fetchall()

        if not open_trades:
            return

        today = date.today()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        closed_this_run = []

        import warnings

        # Batch-fetch spot prices for all unique underlying tickers
        unique_tickers = list({row["ticker"] for row in open_trades})
        spot_cache: Dict[str, float] = {}

        def _fetch_spot(t: str) -> Tuple[str, Optional[float]]:
            try:
                yf, session = _get_yf_and_session()
                tkr = yf.Ticker(t, session=session)
                s = getattr(tkr.fast_info, "last_price", None)
                if s and float(s) > 0:
                    return t, float(s)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = tkr.history(period="5d")
                if not hist.empty:
                    val = float(hist["Close"].iloc[-1])
                    if val > 0:
                        return t, val
            except Exception as exc:
                logger.debug("Spot fetch failed for %s: %s", t, exc)
            return t, None

        max_workers = min(len(unique_tickers), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            try:
                for ticker, spot in ex.map(_fetch_spot, unique_tickers, timeout=30):
                    if spot is not None:
                        spot_cache[ticker] = spot
            except TimeoutError:
                logger.warning("Spot price fetch timed out after 30s — proceeding with partial data")

        # Leg decomposition lives at module scope (_legs_for_row) so it is unit
        # testable and shared with the expiry-settlement path below.

        # Compose unique fetch tasks across every leg of every open row.
        # Tasks are keyed by (ticker, expiration, strike, opt_type) so multi-leg
        # rows can pull each leg's mark independently.
        LegKey = Tuple[str, str, float, str]
        _option_fetch_tasks: List[Tuple[LegKey, str]] = []
        _row_legs: Dict[int, List[Tuple[float, str, int]]] = {}
        _leg_sigma: Dict[LegKey, float] = {}
        _seen_legs: set = set()
        _chain_pairs: List[Tuple[str, str]] = []
        for row in open_trades:
            if row["ticker"] not in spot_cache:
                continue
            legs = _legs_for_row(row)
            _row_legs[row["entry_id"]] = legs
            try:
                row_iv = row["entry_iv"] if "entry_iv" in row.keys() else None
            except Exception:
                row_iv = None
            for strike_v, opt_t, _qty in legs:
                key: LegKey = (row["ticker"], row["expiration"], float(strike_v), opt_t)
                if key in _seen_legs:
                    continue
                _seen_legs.add(key)
                # The model fallback prices this leg at the row's own entry IV
                # (schema v16); a shared leg keeps the first row's IV, which is
                # still name-specific and beats the global constant.
                _leg_sigma[key] = _model_sigma(row_iv)
                pair = (row["ticker"], row["expiration"])
                if pair not in _chain_pairs:
                    _chain_pairs.append(pair)
                symbol = self._get_option_symbol(row["ticker"], row["expiration"], strike_v, opt_t)
                if symbol:
                    _option_fetch_tasks.append((key, symbol))

        # One option-chain call per (ticker, expiration) supplies the live
        # bid/ask for every leg on that pair — no per-leg quote request.
        _chain_quotes: Dict[Tuple[str, str], Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]]] = {}
        if _chain_pairs:
            _chain_workers = min(len(_chain_pairs), 8)
            with ThreadPoolExecutor(max_workers=_chain_workers) as ex:
                try:
                    for pair, quotes in zip(
                        _chain_pairs,
                        ex.map(lambda p: self._fetch_chain_quotes(p[0], p[1]), _chain_pairs, timeout=30),
                    ):
                        _chain_quotes[pair] = quotes or {}
                except TimeoutError:
                    logger.warning("Option chain fetch timed out — falling back to traded marks")

        # value = (price, source); source ∈ mid | last | close | model
        option_price_cache: Dict[LegKey, Tuple[float, str]] = {}

        def _fetch_option_price(task_tuple):
            key, symbol = task_tuple
            ticker, expiration, strike, option_type = key
            try:
                price, source = self._mark_option_leg(
                    key,
                    symbol,
                    _chain_quotes.get((ticker, expiration), {}),
                    spot_cache.get(ticker),
                    _leg_sigma.get(key, DEFAULT_MODEL_SIGMA),
                )
                if price is not None and source is not None:
                    return key, (float(price), source)
            except Exception as exc:
                logger.debug("Option price fetch failed for %s: %s", symbol, exc)
            return key, None

        if _option_fetch_tasks:
            _opt_workers = min(len(_option_fetch_tasks), 8)
            with ThreadPoolExecutor(max_workers=_opt_workers) as ex:
                try:
                    for k, marked in ex.map(_fetch_option_price, _option_fetch_tasks, timeout=30):
                        if marked is not None:
                            option_price_cache[k] = marked
                except TimeoutError:
                    logger.warning("Option price fetch timed out — proceeding with partial data")

        for row in open_trades:
            entry_id    = row["entry_id"]
            ticker      = row["ticker"]
            expiration  = row["expiration"]
            strike      = row["strike"]
            option_type = row["type"]
            entry_price = row["entry_price"]

            # Time-based exit: close if DTE ≤ time_exit_dte (avoids gamma risk near expiry)
            try:
                exp_date = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except Exception:
                dte = 999

            # Days held — don't time-exit a trade logged today or yesterday
            try:
                trade_date = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d").date()
                days_held = (today - trade_date).days
            except Exception:
                days_held = 999

            if ticker not in spot_cache:
                continue

            structure = _classify_structure(row)

            # ── Deterministic expiry settlement ──────────────────────────────
            # An expired option is worth exactly its intrinsic value. Live option
            # quotes vanish at/after expiry, so the mark-to-market path below can
            # never fetch a price for these and they hang OPEN forever (observed:
            # expired iron condors stuck for weeks). Settle straight off spot.
            if dte <= 0:
                spot = spot_cache[ticker]
                reason: Optional[str] = "Expired (settled at intrinsic)"
                if structure in ("spread", "iron_condor"):
                    legs = _legs_for_row(row)
                    if not legs:
                        continue
                    try:
                        nc = row["net_credit"] if "net_credit" in row.keys() else None
                        entry_credit = (float(nc) if nc is not None and nc not in ("", 0)
                                        else float(entry_price or 0))
                    except (TypeError, ValueError):
                        entry_credit = float(entry_price or 0)
                    if entry_credit <= 0:
                        continue
                    close_cost = _legs_intrinsic_close_value(legs, spot)
                    pnl_raw = (entry_credit - close_cost) / entry_credit
                    safe_exit, clamped_pct, pnl_usd = _sanitize_close_values(
                        row["strategy_name"] or "", entry_credit, close_cost, pnl_raw,
                        multiplier=_get_multiplier(ticker),
                        quantity=_row_lots(row),
                    )
                else:
                    is_short = _is_short_position(row["strategy_name"] or "")
                    intrinsic = _intrinsic_value(option_type, spot, float(strike))
                    if entry_price and float(entry_price) > 0:
                        gain = (float(entry_price) - intrinsic) if is_short else (intrinsic - float(entry_price))
                        pnl_raw = gain / float(entry_price)
                    else:
                        pnl_raw = 0.0
                    safe_exit, clamped_pct, pnl_usd = _sanitize_close_values(
                        row["strategy_name"] or "", entry_price, intrinsic, pnl_raw,
                        multiplier=_get_multiplier(ticker),
                        quantity=_row_lots(row),
                    )
                closed_this_run.append(
                    f"{ticker} {row['strategy_name']} → {reason} (settle: {pnl_raw:+.1%})"
                )
                with self._get_connection() as conn:
                    conn.execute(
                        "UPDATE trades SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=?, pnl_usd=?, exit_reason=? WHERE entry_id=?",
                        (safe_exit, now, clamped_pct, pnl_usd, reason, entry_id),
                    )
                continue

            # Multi-leg structure path: spreads and iron condors mark-to-market via per-leg prices.
            if structure in ("spread", "iron_condor"):
                legs = _row_legs.get(entry_id, [])
                if not legs:
                    continue
                leg_marks: List[Tuple[int, float]] = []
                leg_quotes: Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]] = {}
                model_legs: List[str] = []
                missing = False
                for strike_v, opt_t, qty in legs:
                    leg_key: LegKey = (ticker, expiration, float(strike_v), opt_t)
                    marked = option_price_cache.get(leg_key)
                    if marked is None:
                        missing = True
                        break
                    lp, lp_source = marked
                    if lp_source == MARK_MODEL:
                        model_legs.append(f"{opt_t} ${float(strike_v):g}")
                    leg_marks.append((qty, lp))
                    # Raw bid/ask, independent of the mid/model collapse
                    # above — already fetched into _chain_quotes by the
                    # batched chain call earlier in this method; no new
                    # network call. A leg _chain_quotes has no entry for
                    # (model-marked, or the chain call failed for this
                    # pair) leaves this a (None, None) pair, which is
                    # exactly "not recorded" for this leg.
                    leg_quotes[(float(strike_v), opt_t)] = (
                        _chain_quotes.get((ticker, expiration), {})
                        .get((float(strike_v), opt_t), (None, None)))
                if missing:
                    continue

                # entry_credit — prefer stored net_credit / total_credit columns, fall back to entry_price
                try:
                    nc = row["net_credit"] if "net_credit" in row.keys() else None
                except Exception:
                    nc = None
                try:
                    entry_credit = (float(nc) if nc is not None and nc not in ("", 0)
                                    else float(entry_price or 0))
                except (TypeError, ValueError):
                    entry_credit = float(entry_price or 0)

                # cost-to-close = sum(-qty × leg_price). For a short credit structure
                # (qty=-1 on shorts, +1 on longs) this is the debit needed to flatten.
                current_credit_to_close = sum(-qty * lp for qty, lp in leg_marks)

                should_close, reason, pnl_raw = _evaluate_multileg_exit(
                    rules, entry_credit, current_credit_to_close, dte, days_held,
                )
                if model_legs:
                    # At least one leg has no market price, so the structure's
                    # cost-to-close is partly fabricated and may not write an
                    # exit — a structure is only as trustworthy as its worst leg.
                    # Only the pure-DTE rule — whose trigger never reads the
                    # mark — is still allowed, and it stamps its provenance.
                    if 0 < dte <= rules["time_exit_dte"] and days_held >= rules["min_days_held"]:
                        should_close = True
                        reason = f"Time Exit ({dte}d to expiry){MODEL_MARK_SUFFIX}"
                    else:
                        logger.warning(
                            "Exit checks skipped for trade #%s (%s %s): no market mark for %s "
                            "— model price used for display only; row stays OPEN",
                            entry_id, ticker, row["strategy_name"], ", ".join(model_legs),
                        )
                        should_close = False
                if should_close:
                    # Friction: 2 commissions × number of legs (round trip), 2 slippage × legs
                    n_legs = len(legs)
                    friction = (2 * self._slippage_per_share * n_legs) + (2 * self._commission_per_contract * n_legs / 100.0)
                    # Currency conversion is charged on the credit moved, not per
                    # leg — a CAD account converts both ways regardless of how
                    # many legs the structure has.
                    friction += self._fx_per_share(entry_credit)
                    friction_fraction = friction / entry_credit if entry_credit > 0 else 0.0
                    # Compute the structural max-loss floor (loss in pct-of-credit units).
                    # For a credit spread: floor = -(width / credit - 1).
                    # For an iron condor: floor = -(max_wing_width / credit - 1).
                    # If width can't be derived (legacy row missing strikes), fall back to None
                    # so _sanitize_close_values applies its permissive default.
                    spread_width_val: float | None = None
                    try:
                        if structure == "iron_condor":
                            sp_v = float(row["short_put_strike"]); lp_sv = float(row["long_put_strike"])
                            sc_v = float(row["short_call_strike"]); lc_v = float(row["long_call_strike"])
                            spread_width_val = max(abs(sp_v - lp_sv), abs(lc_v - sc_v))
                        else:
                            ls_raw = row["long_strike"] if "long_strike" in row.keys() else None
                            ls_f = (float(ls_raw) if ls_raw is not None
                                    and ls_raw not in ("", 0) else None)
                            if ls_f is not None:
                                spread_width_val = abs(float(strike) - ls_f)
                    except (TypeError, ValueError, KeyError):
                        spread_width_val = None
                    if spread_width_val and entry_credit > 0 and spread_width_val > entry_credit:
                        max_loss_floor = -((spread_width_val / entry_credit) - 1.0)
                    else:
                        max_loss_floor = None  # let sanitizer use permissive default
                    # Don't pre-clamp pnl_realistic — let _sanitize_close_values apply the
                    # structural floor below so true max-loss closes are preserved.
                    pnl_realistic = pnl_raw - friction_fraction
                    if structure == "iron_condor":
                        try:
                            sp = float(row["short_put_strike"]); lp_s = float(row["long_put_strike"])
                            sc = float(row["short_call_strike"]); lc = float(row["long_call_strike"])
                            label = f"IC {lp_s:.0f}/{sp:.0f}—{sc:.0f}/{lc:.0f}"
                        except Exception:
                            label = "IC"
                    else:
                        try:
                            ls_v = float(row["long_strike"]) if row["long_strike"] not in (None, "", 0) else None
                        except Exception:
                            ls_v = None
                        if ls_v is None:
                            label = f"SPREAD ${strike:.0f}"
                        else:
                            label = f"SPREAD ${strike:.0f}/{ls_v:.0f}"
                    closed_this_run.append(
                        f"{ticker} {label} → {reason} "
                        f"(mkt: {pnl_raw:+.1%}, after costs: {pnl_realistic:+.1%})"
                    )
                    safe_exit, clamped_pct, pnl_usd = _sanitize_close_values(
                        row["strategy_name"] or "", entry_credit,
                        current_credit_to_close, pnl_realistic,
                        max_loss_floor=max_loss_floor,
                        multiplier=_get_multiplier(ticker),
                        quantity=_row_lots(row),
                    )
                    exit_cols = _leg_exit_columns(row, leg_quotes, structure)
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE trades SET status='CLOSED', exit_price=?, "
                            "exit_date=?, pnl_pct=?, pnl_usd=?, exit_reason=?, "
                            + ", ".join(f"{k}=?" for k in exit_cols) +
                            " WHERE entry_id=?",
                            (safe_exit, now, clamped_pct, pnl_usd, reason,
                             *exit_cols.values(), entry_id),
                        )
                    # Keep marking a stopped-out or time-exited trade to its
                    # original expiry, so "should I have held?" becomes data
                    # instead of a question the ledger cannot answer.
                    self.open_shadow_window(entry_id, reason)
                continue

            # Single-leg path
            single_key: LegKey = (ticker, expiration, float(strike), str(option_type or "").lower())
            marked = option_price_cache.get(single_key)
            current_price, mark_source = marked if marked is not None else (None, None)

            if current_price is not None:
                # High-water mark: sample the premium every run so "how high
                # did it go while I held it" is recorded data, not memory.
                # One statement; RHS reads the OLD row, so date and level stay
                # consistent. Never blocks exit handling. A model mark is not
                # an observation, so it never sets the high-water mark either —
                # that column is ledger data, not display.
                if mark_source in MARKET_MARK_SOURCES:
                    try:
                        with self._get_connection() as conn:
                            conn.execute(
                                "UPDATE trades SET "
                                "max_price_date = CASE WHEN ? > COALESCE(max_price_seen, entry_price) "
                                "THEN ? ELSE max_price_date END, "
                                "max_price_seen = MAX(COALESCE(max_price_seen, entry_price), ?) "
                                "WHERE entry_id=?",
                                (current_price, now, current_price, entry_id),
                            )
                    except Exception:
                        pass
                is_short = _is_short_position(row["strategy_name"] or "")
                spot = spot_cache.get(ticker)
                try:
                    entry_delta = row["entry_delta"] if "entry_delta" in row.keys() else None
                except Exception:
                    entry_delta = None
                try:
                    entry_iv = row["entry_iv"] if "entry_iv" in row.keys() else None
                except Exception:
                    entry_iv = None

                if is_short:
                    should_close, reason, pnl_raw = _evaluate_short_single_leg_exit(
                        rules, option_type, float(strike), spot,
                        entry_price, current_price, entry_delta, entry_iv,
                        dte, days_held, rfr,
                    )
                else:
                    should_close, reason, pnl_raw = _evaluate_long_single_leg_exit(
                        rules, option_type, float(strike), spot,
                        entry_price, current_price, entry_iv,
                        dte, days_held, rfr,
                    )

                if mark_source == MARK_MODEL:
                    # The mark is a model price, not a market observation. A
                    # fabricated number must not write a permanent exit into
                    # the ledger, so every price-based rule (TP, stop, delta,
                    # strike breach) is skipped for this row this run — it
                    # stays OPEN and is re-evaluated when a quote exists, with
                    # expiry settlement as the terminal backstop. The pure-DTE
                    # rule still fires: its trigger never reads the mark.
                    if 0 < dte <= rules["time_exit_dte"] and days_held >= rules["min_days_held"]:
                        should_close = True
                        reason = f"Time Exit ({dte}d to expiry){MODEL_MARK_SUFFIX}"
                    else:
                        logger.warning(
                            "Exit checks skipped for trade #%s (%s %s $%s): no market mark "
                            "— model price used for display only; row stays OPEN",
                            entry_id, ticker, str(option_type).upper(), f"{float(strike):g}",
                        )
                        should_close = False

                if should_close:
                    # Realistic P&L: proportional slippage (30% of bid-ask) + commissions
                    _slip = self._get_spread_slippage(ticker, expiration, strike, option_type, entry_price)
                    _friction = (2 * _slip) + (2 * self._commission_per_contract / 100.0)
                    _friction += self._fx_per_share(entry_price)
                    friction_fraction = _friction / entry_price if entry_price > 0 else 0.0
                    # No floor for short legs — loss can exceed entry premium (e.g. short call bought back at 2x)
                    pnl_realistic = pnl_raw - friction_fraction

                    closed_this_run.append(
                        f"{ticker} {option_type.upper()} ${strike:.0f} → {reason} "
                        f"(mkt: {pnl_raw:+.1%}, after costs: {pnl_realistic:+.1%})"
                    )
                    safe_exit, clamped_pct, pnl_usd = _sanitize_close_values(
                        row["strategy_name"] or "", entry_price,
                        current_price, pnl_realistic,
                        multiplier=_get_multiplier(ticker),
                        quantity=_row_lots(row),
                    )
                    update_query = """
                    UPDATE trades
                    SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=?, pnl_usd=?, exit_reason=?
                    WHERE entry_id=?
                    """
                    with self._get_connection() as conn:
                        conn.execute(update_query, (safe_exit, now, clamped_pct, pnl_usd, reason, entry_id))
                    # Single-leg longs are where the stop question actually
                    # bites: it fires on 40 of 82 of them, realising -60.3%
                    # from an average peak of +16.6%.
                    self.open_shadow_window(entry_id, reason)

        # Counterfactual pass: keep marking stopped-out and time-exited trades
        # to their original expiry. Research data only — wrapped so it can
        # never disturb the live book it runs behind.
        try:
            self.update_shadow_marks()
        except Exception:
            pass

        if closed_this_run:
            print(f"  Auto-closed {len(closed_this_run)} position(s):")
            for msg in closed_this_run:
                print(f"    \u2713 {msg}")
            print(cost_disclosure(self._slippage_per_share,
                                  self._commission_per_contract,
                                  self._fx_conversion_rate))
            self._maybe_emit_calibration_threshold_notice()

    def get_correlated_open_positions(
        self,
        ticker: str,
        lookback_days: int = 60,
        correlation_threshold: float = 0.80,
    ) -> List[Dict]:
        """Return open positions whose underlying is highly correlated with `ticker`.

        Fetches `lookback_days` of daily closes via yfinance for `ticker` and each
        distinct ticker in OPEN trades, then computes Pearson correlation of daily
        returns.  Returns a list of dicts with keys "ticker" and "correlation" for
        any pair where abs(correlation) > correlation_threshold.
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM trades WHERE status='OPEN'"
                ).fetchall()
        except Exception:
            return []

        open_tickers = [r[0] for r in rows if r[0] and r[0].upper() != ticker.upper()]
        if not open_tickers:
            return []

        period = f"{lookback_days}d"

        all_tickers = [ticker] + open_tickers

        def _fetch_returns(sym: str):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                yf, session = _get_yf_and_session()
                hist = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True, session=session)
            if hist.empty:
                return sym, None
            close_col = hist["Close"]
            if isinstance(close_col, pd.DataFrame):
                # MultiIndex result — take first column
                close_col = close_col.iloc[:, 0]
            close_col = close_col.dropna()
            if not isinstance(close_col, pd.Series) or len(close_col) < 5:
                return sym, None
            return sym, close_col.pct_change().dropna()

        # Parallel download — capped at 8 workers to avoid rate-limiting
        hist_map: dict = {}
        try:
            with ThreadPoolExecutor(max_workers=min(len(all_tickers), 8)) as exe:
                futures = {exe.submit(_fetch_returns, sym): sym for sym in all_tickers}
                for fut in as_completed(futures):
                    sym, returns = fut.result()
                    if returns is not None:
                        hist_map[sym] = returns
        except Exception:
            return []

        ref_returns = hist_map.get(ticker)
        if ref_returns is None:
            return []

        correlated = []
        for ot in open_tickers:
            other_returns = hist_map.get(ot)
            if other_returns is None:
                continue
            combined = pd.concat([ref_returns, other_returns], axis=1, join="inner").dropna()
            if len(combined) < 10:
                continue
            corr = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
            if abs(corr) > correlation_threshold:
                correlated.append({"ticker": ot, "correlation": corr})

        return correlated

    def get_position_size_with_correlation(
        self,
        ticker: str,
        base_blended_fraction: float,
    ) -> Tuple[float, str]:
        """Return an (adjusted_fraction, reason_string) pair.

        Loads correlation_threshold from config (default 0.80).  If any open
        position is highly correlated with `ticker`, halves the fraction
        (reduction factor from config, default 0.50).
        """
        try:
            config = self._load_config()
            threshold = config.get("correlation_threshold", 0.80)
            reduction = config.get("correlation_size_reduction", 0.50)
            if not config.get("correlation_aware_sizing", True):
                return base_blended_fraction, ""
        except Exception:
            threshold, reduction = 0.80, 0.50

        correlated = self.get_correlated_open_positions(ticker, correlation_threshold=threshold)
        if not correlated:
            return base_blended_fraction, ""

        top = max(correlated, key=lambda x: abs(x["correlation"]))
        reason = (
            f"Correlation-adjusted: {top['ticker']} r={top['correlation']:.2f} "
            f"→ size reduced by {(1-reduction)*100:.0f}%"
        )
        return base_blended_fraction * reduction, reason

    def get_strategy_breakdown(self) -> List[Dict]:
        """Return win/loss/avg P&L grouped by strategy_name.

        ``return_on_risk`` is total P&L over total capital at risk. Summed
        dollars alone rank strategies partly by position size — risk per trade
        spans two orders of magnitude — so a line can lead on dollars while
        losing money per dollar committed. It is None until the cohort has
        capital_at_risk populated (see scripts/backfill_capital_at_risk.py).
        """
        query = """
            SELECT strategy_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
                   AVG(pnl_pct) as avg_pnl,
                   SUM(pnl_pct) as total_pnl,
                   SUM(CASE WHEN capital_at_risk > 0 THEN capital_at_risk ELSE 0 END) as risk,
                   SUM(CASE WHEN capital_at_risk > 0 AND pnl_usd IS NOT NULL
                            THEN pnl_usd ELSE 0 END) as pnl_on_risk
            FROM trades
            WHERE status = 'CLOSED' AND pnl_pct IS NOT NULL
            GROUP BY strategy_name
            ORDER BY total DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        out = []
        for r in rows:
            risk = r[6] or 0.0
            out.append({
                "strategy": r[0] or "Unknown", "total": r[1], "wins": r[2],
                "losses": r[3], "win_rate": r[2] / r[1] if r[1] else 0,
                "avg_pnl": r[4], "total_pnl": r[5],
                "capital_at_risk": risk,
                "return_on_risk": (r[7] / risk) if risk > 0 else None,
            })
        return out

    def get_all_trades(self) -> pd.DataFrame:
        """Returns all trades as a pandas DataFrame."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM trades", conn)

    def get_performance_summary(self) -> pd.DataFrame:
        """Returns a summary of trading performance with Sharpe, Sortino, and win rate."""
        with self._get_connection() as conn:
            total_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] or 0
            closed_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0
            win_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct > 0").fetchone()[0] or 0
            avg_pnl = conn.execute("SELECT AVG(pnl_pct) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0.0
            sum_pnl = conn.execute("SELECT SUM(pnl_pct) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0.0

        win_rate = win_count / closed_count if closed_count > 0 else 0.0

        # Sharpe and Sortino from closed-trade returns (per-trade, not annualized)
        sharpe_str = "n/a"
        sortino_str = "n/a"
        if closed_count >= 5:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT pnl_pct FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL"
                    ).fetchall()
                returns = np.array([r[0] for r in rows], dtype=float)
                mean_r = np.mean(returns)
                std_r = np.std(returns, ddof=1)
                if std_r > 0:
                    sharpe_str = f"{mean_r / std_r:.3f}"
                downside = returns[returns < 0]
                if len(downside) > 1:
                    sortino_std = np.std(downside, ddof=1)
                    if sortino_std > 0:
                        sortino_str = f"{mean_r / sortino_std:.3f}"
            except Exception as exc:
                logger.debug("Sharpe/Sortino calculation failed: %s", exc)

        summary = {
            "Total Trades": [total_count],
            "Closed Trades": [closed_count],
            "Win Rate": [f"{win_rate:.1%}"],
            "Total PnL %": [f"{sum_pnl:.1%}"],
            "Avg Return": [f"{avg_pnl:.1%}"],
            "Per-Trade Sharpe": [sharpe_str],
            "Per-Trade Sortino": [sortino_str],
        }
        return pd.DataFrame(summary)

    def compute_ic(self) -> dict:
        """Compute Information Coefficient between quality_score and realized pnl_pct.

        IC (Pearson correlation between predicted score and actual P&L) is the key
        metric for validating whether the model has real edge.

        Interpretation:
          IC > 0.10, p < 0.05  →  solid edge, model is predictive
          IC > 0.05, p < 0.20  →  some edge, keep trading to confirm
          IC > 0, not sig      →  weak positive, need more trades
          IC ≤ 0               →  no edge detected

        Requires at least 10 closed trades for a meaningful result.
        """
        df = self.get_all_trades()
        closed = df[
            (df["status"] == "CLOSED")
            & df["pnl_pct"].notna()
            & df["quality_score"].notna()
        ].copy()

        result: dict = {"n": len(closed)}

        if len(closed) < 10:
            result["message"] = (
                f"Need at least 10 closed trades for IC (have {len(closed)}). "
                "Keep paper trading and check back."
            )
            return result

        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError:
            result["message"] = "scipy not installed — pip install scipy"
            return result

        q_scores = closed["quality_score"].values.astype(float)
        pnl = closed["pnl_pct"].values.astype(float)

        ic_p, pval_p = pearsonr(q_scores, pnl)
        ic_s, pval_s = spearmanr(q_scores, pnl)

        if ic_p > 0.10 and pval_p < 0.05:
            interp = "SOLID EDGE — model is predictive of returns"
        elif ic_p > 0.05 and pval_p < 0.20:
            interp = "SOME EDGE — statistically weak, keep trading to confirm"
        elif ic_p > 0:
            interp = "WEAK POSITIVE — not yet statistically significant"
        else:
            interp = "NO EDGE DETECTED — model is not predictive of returns"

        result.update({
            "ic_technical_pearson": round(float(ic_p), 4),
            "p_technical": round(float(pval_p), 4),
            "ic_technical_spearman": round(float(ic_s), 4),
            "interpretation": interp,
        })

        # AI IC if ai_score was recorded at entry
        if "ai_score" in closed.columns:
            ai_valid = closed[closed["ai_score"].notna()].copy()
            if len(ai_valid) >= 10:
                ai_q = ai_valid["ai_score"].values.astype(float) / 100.0
                ai_pnl = ai_valid["pnl_pct"].values.astype(float)
                ai_ic, ai_pval = pearsonr(ai_q, ai_pnl)
                result["ic_ai_pearson"] = round(float(ai_ic), 4)
                result["p_ai"] = round(float(ai_pval), 4)
                result["ai_adds_value"] = bool(ai_ic > ic_p)
                result["ai_ic_note"] = (
                    "AI score outperforms technical" if ai_ic > ic_p
                    else "Technical score outperforms AI"
                )

        return result

if __name__ == "__main__":
    # Test script with temporary database
    test_db = "test_paper_trades.db"
    manager = PaperManager(db_path=test_db)
    
    test_trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-19",
        "strike": 150.0,
        "type": "call",
        "entry_price": 50.0,
        "quality_score": 0.85,
        "strategy_name": "Test Strategy"
    }
    
    manager.log_trade(test_trade)
    manager.update_positions()
    print("\nPerformance Summary:")
    print(manager.get_performance_summary())
    
    # Cleanup test database
    if os.path.exists(test_db):
        os.remove(test_db)
