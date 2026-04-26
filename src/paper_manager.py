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
COMMISSION_PER_CONTRACT = 0.65   # $ per contract per leg (retail ~$0.65, e.g. Tastytrade/TDA)
SLIPPAGE_PER_SHARE = 0.05        # $ per share (1 typical options tick, ~half spread)
# Round-trip friction per share = entry slippage + exit slippage + 2 commissions
_FRICTION_PER_SHARE = (2 * SLIPPAGE_PER_SHARE) + (2 * COMMISSION_PER_CONTRACT / 100.0)

_SCHEMA_VERSION = 10
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
}


class PaperManager:
    """Manages paper trades stored in a SQLite database."""
    
    def __init__(self, db_path: str = "paper_trades.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        # Load friction costs from config (fall back to module-level constants)
        try:
            with open(config_path, 'r') as f:
                _cfg = json.load(f)
            _pt = _cfg.get("paper_trading", {})
            self._commission_per_contract = float(_pt.get("commission_per_contract", COMMISSION_PER_CONTRACT))
            self._slippage_per_share = float(_pt.get("slippage_per_share", SLIPPAGE_PER_SHARE))
        except Exception:
            self._commission_per_contract = COMMISSION_PER_CONTRACT
            self._slippage_per_share = SLIPPAGE_PER_SHARE
        self._friction_per_share = (2 * self._slippage_per_share) + (2 * self._commission_per_contract / 100.0)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yield a sqlite3 connection with WAL mode; commits on success, rollbacks on error, always closes."""
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
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug("Config file not found at %s — using defaults", self.config_path)
            return _default
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in config %s: %s — using defaults", self.config_path, exc)
            return _default

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
        """
        if not trade_dict.get("strategy_name"):
            raise ValueError("strategy_name is required; must include 'short'/'long' to set P&L direction")
        if float(trade_dict.get("entry_price", 0)) <= 0:
            raise ValueError(f"Cannot log trade: entry_price must be > 0, got {trade_dict.get('entry_price')}")

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
            weight_profile,
            long_strike, spread_width, net_credit, max_profit_usd, max_loss_usd,
            short_call_strike, long_call_strike, short_put_strike, long_put_strike, net_delta
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
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
            "",     # exit_date
            None,   # pnl_pct
            trade_dict.get("ai_score"),        # None if not ranked yet
            trade_dict.get("ai_confidence"),   # None if not ranked yet
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
        )

        with self._get_connection() as conn:
            conn.execute(query, params)

        print(f"Logged {trade_dict['strategy_name']} on {trade_dict['ticker']} at ${float(trade_dict['entry_price']):.2f}")

    def log_trade_if_new(self, trade_dict: Dict[str, Any]) -> bool:
        """Insert a paper trade unless an identical row already exists.

        Dedup key: ``(trade date, ticker, strike, expiration, type, weight_profile)``.
        The trade date is whatever ``log_trade`` would store — either the caller's
        explicit ``trade_dict["date"]`` or today's timestamp — so a re-logged row
        matches its prior copy even when re-runs happen on a later calendar day.
        ``weight_profile`` may be ``None`` for untagged trades — NULL-equal rows still
        dedup against each other because ``IS`` is used instead of ``=``.

        Returns ``True`` if inserted, ``False`` if skipped as duplicate.
        """
        ticker = trade_dict["ticker"].upper()
        typ = trade_dict["type"].lower()
        strike = float(trade_dict["strike"])
        expiration = trade_dict["expiration"]
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
                  AND weight_profile IS ?
                  AND date(date) = date(?)
                LIMIT 1
                """,
                (ticker, strike, expiration, typ, profile, effective_date),
            ).fetchone()
        if row is not None:
            return False
        self.log_trade(trade_dict)
        return True

    def log_spread(self, spread_dict: dict) -> None:
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
        if max_profit is not None:
            trade_dict["max_profit_usd"] = float(max_profit)
        if max_loss is not None:
            trade_dict["max_loss_usd"] = float(max_loss)
        trade_dict.setdefault("quality_score", 0.5)
        # log_trade requires ticker — make sure case-normalized
        trade_dict["ticker"] = str(spread_dict.get("ticker", "")).upper()

        self.log_trade(trade_dict)

    def log_iron_condor(self, condor_dict: dict) -> None:
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
        if max_risk is not None:
            trade_dict["max_loss_usd"] = float(max_risk)
        if condor_dict.get("max_profit") is not None:
            trade_dict["max_profit_usd"] = float(condor_dict["max_profit"])
        if condor_dict.get("net_delta") is not None:
            trade_dict["net_delta"] = float(condor_dict["net_delta"])
        trade_dict.setdefault("quality_score", 0.5)
        trade_dict["ticker"] = str(condor_dict.get("ticker", "")).upper()

        self.log_trade(trade_dict)

    def log_spread_if_new(self, spread_dict: dict) -> bool:
        """Insert a credit spread unless an identical OPEN row already exists for
        the same (date, ticker, expiration, short_strike, long_strike, strategy, profile).
        Returns True if inserted, False if duplicate.
        """
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
        self.log_spread(spread_dict)
        return True

    def log_iron_condor_if_new(self, condor_dict: dict) -> bool:
        """Same dedup pattern as log_spread_if_new but for 4-leg iron condors."""
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
        self.log_iron_condor(condor_dict)
        return True

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

        # Build leg list per row: single = 1 leg, spread = 2, iron condor = 4.
        # We fetch each unique (ticker, exp, strike, type) once and reuse.
        def _legs_for_row(row) -> List[Tuple[float, str, int]]:
            structure = _classify_structure(row)
            if structure == "iron_condor":
                try:
                    return [
                        (float(row["short_put_strike"]),  "put",  -1),
                        (float(row["long_put_strike"]),   "put",  +1),
                        (float(row["short_call_strike"]), "call", -1),
                        (float(row["long_call_strike"]),  "call", +1),
                    ]
                except (TypeError, ValueError):
                    return []
            if structure == "spread":
                opt_type = str(row["type"] or "").lower()
                if opt_type not in ("put", "call"):
                    sn = str(row["strategy_name"] or "").lower()
                    opt_type = "put" if "bull put" in sn else "call"
                ls = row["long_strike"] if "long_strike" in row.keys() else None
                try:
                    long_strike = float(ls) if ls not in (None, "", 0) else None
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
            return [(float(row["strike"]), str(row["type"] or "").lower(), -1 if _is_short_position(row["strategy_name"] or "") else +1)]

        # Compose unique fetch tasks across every leg of every open row.
        # Tasks are keyed by (ticker, expiration, strike, opt_type) so multi-leg
        # rows can pull each leg's mark independently.
        LegKey = Tuple[str, str, float, str]
        _option_fetch_tasks: List[Tuple[LegKey, str]] = []
        _row_legs: Dict[int, List[Tuple[float, str, int]]] = {}
        _seen_legs: set = set()
        for row in open_trades:
            if row["ticker"] not in spot_cache:
                continue
            legs = _legs_for_row(row)
            _row_legs[row["entry_id"]] = legs
            for strike_v, opt_t, _qty in legs:
                key: LegKey = (row["ticker"], row["expiration"], float(strike_v), opt_t)
                if key in _seen_legs:
                    continue
                _seen_legs.add(key)
                symbol = self._get_option_symbol(row["ticker"], row["expiration"], strike_v, opt_t)
                if symbol:
                    _option_fetch_tasks.append((key, symbol))

        option_price_cache: Dict[LegKey, float] = {}

        def _fetch_option_price(task_tuple):
            key, symbol = task_tuple
            ticker, expiration, strike, option_type = key
            try:
                yf, session = _get_yf_and_session()
                tkr = yf.Ticker(symbol, session=session)
                price = None
                try:
                    price = getattr(tkr.fast_info, "last_price", None)
                except Exception:
                    pass
                if price is None or np.isnan(price) or price <= 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        hist = tkr.history(period="1d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                if price is None or np.isnan(price) or price <= 0:
                    try:
                        from .utils import american_price
                        S = spot_cache.get(ticker)
                        if S:
                            T = max((datetime.strptime(expiration[:10], "%Y-%m-%d") - datetime.now()).days / 365, 1/365)
                            _rfr = _get_rfr() if _HAS_RFR else 0.045
                            price = american_price(option_type, float(S), float(strike), T, _rfr, 0.30)
                    except Exception:
                        pass
                if price is not None and not np.isnan(price) and price > 0:
                    return key, float(price)
            except Exception as exc:
                logger.debug("Option price fetch failed for %s: %s", symbol, exc)
            return key, None

        if _option_fetch_tasks:
            _opt_workers = min(len(_option_fetch_tasks), 8)
            with ThreadPoolExecutor(max_workers=_opt_workers) as ex:
                try:
                    for k, price in ex.map(_fetch_option_price, _option_fetch_tasks, timeout=30):
                        if price is not None:
                            option_price_cache[k] = price
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

            # Multi-leg structure path: spreads and iron condors mark-to-market via per-leg prices.
            structure = _classify_structure(row)
            if structure in ("spread", "iron_condor"):
                legs = _row_legs.get(entry_id, [])
                if not legs:
                    continue
                leg_marks: List[Tuple[int, float]] = []
                missing = False
                for strike_v, opt_t, qty in legs:
                    leg_key: LegKey = (ticker, expiration, float(strike_v), opt_t)
                    lp = option_price_cache.get(leg_key)
                    if lp is None:
                        missing = True
                        break
                    leg_marks.append((qty, lp))
                if missing:
                    continue

                # entry_credit — prefer stored net_credit / total_credit columns, fall back to entry_price
                try:
                    nc = row["net_credit"] if "net_credit" in row.keys() else None
                except Exception:
                    nc = None
                try:
                    entry_credit = float(nc) if nc not in (None, "", 0) else float(entry_price or 0)
                except (TypeError, ValueError):
                    entry_credit = float(entry_price or 0)

                # cost-to-close = sum(-qty × leg_price). For a short credit structure
                # (qty=-1 on shorts, +1 on longs) this is the debit needed to flatten.
                current_credit_to_close = sum(-qty * lp for qty, lp in leg_marks)

                should_close, reason, pnl_raw = _evaluate_multileg_exit(
                    rules, entry_credit, current_credit_to_close, dte, days_held,
                )
                if should_close:
                    # Friction: 2 commissions × number of legs (round trip), 2 slippage × legs
                    n_legs = len(legs)
                    friction = (2 * self._slippage_per_share * n_legs) + (2 * self._commission_per_contract * n_legs / 100.0)
                    friction_fraction = friction / entry_credit if entry_credit > 0 else 0.0
                    pnl_realistic = max(pnl_raw - friction_fraction, -1.0)
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
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE trades SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=?, exit_reason=? WHERE entry_id=?",
                            (current_credit_to_close, now, pnl_realistic, reason, entry_id),
                        )
                continue

            # Single-leg path
            single_key: LegKey = (ticker, expiration, float(strike), str(option_type or "").lower())
            current_price = option_price_cache.get(single_key)

            if current_price is not None:
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

                if should_close:
                    # Realistic P&L: proportional slippage (30% of bid-ask) + commissions
                    _slip = self._get_spread_slippage(ticker, expiration, strike, option_type, entry_price)
                    _friction = (2 * _slip) + (2 * self._commission_per_contract / 100.0)
                    friction_fraction = _friction / entry_price if entry_price > 0 else 0.0
                    # No floor for short legs — loss can exceed entry premium (e.g. short call bought back at 2x)
                    pnl_realistic = pnl_raw - friction_fraction

                    closed_this_run.append(
                        f"{ticker} {option_type.upper()} ${strike:.0f} → {reason} "
                        f"(mkt: {pnl_raw:+.1%}, after costs: {pnl_realistic:+.1%})"
                    )
                    update_query = """
                    UPDATE trades
                    SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=?, exit_reason=?
                    WHERE entry_id=?
                    """
                    with self._get_connection() as conn:
                        conn.execute(update_query, (current_price, now, pnl_realistic, reason, entry_id))

        if closed_this_run:
            print(f"  Auto-closed {len(closed_this_run)} position(s):")
            for msg in closed_this_run:
                print(f"    \u2713 {msg}")
            print(f"    [costs: ${self._slippage_per_share:.2f}/share slippage ×2 + ${self._commission_per_contract:.2f}/contract commissions ×2]")
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
        """Return win/loss/avg P&L grouped by strategy_name."""
        query = """
            SELECT strategy_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
                   AVG(pnl_pct) as avg_pnl,
                   SUM(pnl_pct) as total_pnl
            FROM trades
            WHERE status = 'CLOSED' AND pnl_pct IS NOT NULL
            GROUP BY strategy_name
            ORDER BY total DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return [
            {"strategy": r[0] or "Unknown", "total": r[1], "wins": r[2],
             "losses": r[3], "win_rate": r[2] / r[1] if r[1] else 0,
             "avg_pnl": r[4], "total_pnl": r[5]}
            for r in rows
        ]

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
