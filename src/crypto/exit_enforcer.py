"""Crypto paper-trade exit enforcer.

Closes any open trade in `paper_trades_crypto.db` that has hit its
take-profit, stop-loss, or time-exit threshold per `config.json` exit_rules
— but priced via Deribit instead of yfinance.

Designed to run hourly (crypto is 24/7) via:
  scripts/enforce_exits_crypto.sh

Handles all crypto strategies:
  • Long Call / Long Put     — TP at +100% (default), SL at -50%
  • Bull Put / Bear Call     — TP at +50% credit captured, SL at structural max
  • Iron Condor              — same rules, 4-leg P&L
  • Calendar Call / Put      — single-strike time spread, debit position
                                (TP at +50% of debit, SL at -50%)

Uses paper_manager._sanitize_close_values() for write-side bound checking.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.paper_manager import (
    PaperManager,
    _sanitize_close_values,
    _CREDIT_STRUCTURES,
)

from . import data_fetching as _df


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CRYPTO_DB_PATH = os.path.join(_PROJECT_ROOT, "paper_trades_crypto.db")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")

# Deribit instrument-name month abbreviations.
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _load_config() -> Dict[str, Any]:
    try:
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _exit_rules() -> Dict[str, Any]:
    cfg = _load_config()
    rules = cfg.get("exit_rules") or {}
    # Crypto-specific overrides could go in config["crypto"]["exit_rules"];
    # for v1 we share the equity rules.
    return {
        "time_exit_dte":      int(rules.get("time_exit_dte", 1)),
        "take_profit_long":   float(rules.get("long_option", {}).get("take_profit", 1.0)),
        "stop_loss_long":     float(rules.get("long_option", {}).get("stop_loss", -0.50)),
        "take_profit_credit": float(rules.get("spread", {}).get("take_profit", 0.50)),
        "stop_loss_credit":   float(rules.get("spread", {}).get("stop_loss", -1.00)),
    }


def _deribit_instrument(currency: str, expiration: str, strike: float, opt_type: str) -> str:
    """Build a Deribit instrument name from components."""
    try:
        d = _dt.datetime.strptime(str(expiration), "%Y-%m-%d")
    except ValueError:
        d = _dt.datetime.strptime(str(expiration), "%Y-%m-%d %H:%M:%S")
    day = d.day
    mon = _MONTHS[d.month - 1]
    yr = d.year % 100
    leg = "C" if str(opt_type).lower() == "call" else "P"
    return f"{currency.upper()}-{day}{mon}{yr:02d}-{int(strike)}-{leg}"


def _chain_lookup(chain: pd.DataFrame, instrument: str) -> Optional[pd.Series]:
    """Find a row in a Deribit chain by instrument_name."""
    if chain is None or chain.empty:
        return None
    matches = chain[chain["instrument_name"] == instrument]
    if matches.empty:
        return None
    return matches.iloc[0]


def _dte_now(expiration: str) -> float:
    """Days-to-expiry from now (UTC)."""
    try:
        d = _dt.datetime.strptime(str(expiration), "%Y-%m-%d")
    except ValueError:
        try:
            d = _dt.datetime.strptime(str(expiration), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return 0.0
    d = d.replace(tzinfo=_dt.timezone.utc)
    now = _dt.datetime.now(_dt.timezone.utc)
    return max(0.0, (d - now).total_seconds() / 86400.0)


def _classify_row(row: sqlite3.Row) -> str:
    """Determine structure: 'iron_condor' | 'spread' | 'calendar' | 'single'."""
    strategy = str(row["strategy_name"] or "").lower()
    if "iron condor" in strategy:
        return "iron_condor"
    if "calendar" in strategy:
        return "calendar"
    if any(s in strategy for s in ("bull put", "bear call")):
        return "spread"
    return "single"


def _parse_back_expiration(strategy_name: str) -> Optional[str]:
    """Extract the 'back YYYY-MM-DD' suffix used to encode calendar back legs."""
    if not strategy_name or "[back " not in strategy_name:
        return None
    try:
        return strategy_name.split("[back ", 1)[1].rstrip("]").strip()
    except (IndexError, AttributeError):
        return None


def _close_row(conn: sqlite3.Connection, entry_id: int, exit_price: float,
                pnl_pct: float, pnl_usd: float, reason: str,
                strategy_name: str, entry_price: float,
                max_loss_floor: Optional[float] = None) -> None:
    """Sanitize and write the close UPDATE."""
    safe_exit, clamped_pct, sanitized_usd = _sanitize_close_values(
        strategy_name or "", entry_price, exit_price, pnl_pct,
        max_loss_floor=max_loss_floor,
    )
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """UPDATE trades SET status='CLOSED', exit_price=?, exit_date=?,
           pnl_pct=?, pnl_usd=?, exit_reason=? WHERE entry_id=?""",
        (safe_exit, now, clamped_pct, sanitized_usd, reason, entry_id),
    )


def enforce_exits(db_path: str = _CRYPTO_DB_PATH, dry_run: bool = False) -> List[str]:
    """Walk the crypto paper ledger and close anything past TP/SL/time exit.

    Returns a list of human-readable strings, one per closed trade.
    """
    if not os.path.exists(db_path):
        return []
    rules = _exit_rules()

    closed_messages: List[str] = []

    # Pre-fetch BTC and ETH chains once; reuse for every leg lookup.
    chain_cache: Dict[str, pd.DataFrame] = {}

    pm = PaperManager(db_path=db_path)  # ensures schema is up-to-date

    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        opens = conn.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_id"
        ).fetchall()
        if not opens:
            return []

        for row in opens:
            entry_id = int(row["entry_id"])
            ticker = str(row["ticker"] or "").upper()
            if ticker not in ("BTC", "ETH"):
                continue
            structure = _classify_row(row)
            strategy = str(row["strategy_name"] or "")
            expiration = str(row["expiration"])

            # Lazy-load chain for this currency.
            if ticker not in chain_cache:
                chain_cache[ticker] = _df.get_options_chain(ticker)
            chain = chain_cache[ticker]

            try:
                if structure == "single":
                    msg = _evaluate_single(conn, row, chain, rules, dry_run)
                elif structure == "spread":
                    msg = _evaluate_spread(conn, row, chain, rules, dry_run)
                elif structure == "iron_condor":
                    msg = _evaluate_iron_condor(conn, row, chain, rules, dry_run)
                elif structure == "calendar":
                    msg = _evaluate_calendar(conn, row, chain, rules, dry_run)
                else:
                    msg = None
            except Exception as e:
                msg = f"  ! id={entry_id} {ticker} eval failed: {type(e).__name__}: {e}"

            if msg:
                closed_messages.append(msg)

    return closed_messages


# ── Per-structure evaluators ─────────────────────────────────────────────

def _evaluate_single(conn: sqlite3.Connection, row: sqlite3.Row,
                      chain: pd.DataFrame, rules: Dict[str, Any],
                      dry_run: bool) -> Optional[str]:
    """Long-premium single-leg (Long Call / Long Put)."""
    ticker = str(row["ticker"]).upper()
    strike = float(row["strike"])
    opt_type = str(row["type"]).lower()
    expiration = str(row["expiration"])
    entry_price = float(row["entry_price"] or 0)
    if entry_price <= 0:
        return None

    instrument = _deribit_instrument(ticker, expiration, strike, opt_type)
    leg = _chain_lookup(chain, instrument)
    dte = _dte_now(expiration)

    # Time-exit takes priority if very close to expiry.
    if dte <= rules["time_exit_dte"]:
        # Use intrinsic value as exit (best approximation if mark unavailable).
        if leg is not None:
            exit_price = float(leg.get("mark_price") or leg.get("mid_price") or 0)
        else:
            spot = _df.get_index_price(ticker) or 0
            exit_price = max(0.0, (spot - strike) if opt_type == "call" else (strike - spot))
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_usd = (exit_price - entry_price)
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), exit_price, pnl_pct, pnl_usd,
                       "Time Exit", str(row["strategy_name"]), entry_price)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Time Exit (mkt: {pnl_pct:+.1%})"

    if leg is None:
        return None  # contract not in current chain — Deribit may have delisted

    current_price = float(leg.get("mark_price") or leg.get("mid_price") or 0)
    if current_price <= 0:
        return None
    pnl_pct = (current_price - entry_price) / entry_price

    if pnl_pct >= rules["take_profit_long"]:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), current_price, pnl_pct,
                       (current_price - entry_price), "Take Profit (+100%)",
                       str(row["strategy_name"]), entry_price)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Take Profit (mkt: {pnl_pct:+.1%})"
    if pnl_pct <= rules["stop_loss_long"]:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), current_price, pnl_pct,
                       (current_price - entry_price), "Stop Loss (-50%)",
                       str(row["strategy_name"]), entry_price)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Stop Loss (mkt: {pnl_pct:+.1%})"
    return None


def _evaluate_spread(conn: sqlite3.Connection, row: sqlite3.Row,
                      chain: pd.DataFrame, rules: Dict[str, Any],
                      dry_run: bool) -> Optional[str]:
    """2-leg credit spread (Bull Put / Bear Call)."""
    ticker = str(row["ticker"]).upper()
    strategy = str(row["strategy_name"] or "")
    expiration = str(row["expiration"])
    short_strike = float(row["strike"])
    try:
        long_strike = float(row["long_strike"]) if row["long_strike"] is not None else 0.0
    except (TypeError, ValueError):
        long_strike = 0.0
    if long_strike <= 0:
        return None
    leg_type = "put" if "bull put" in strategy.lower() else "call"
    entry_credit = float(row["entry_price"] or row["net_credit"] or 0)
    width = abs(short_strike - long_strike)
    if entry_credit <= 0 or width <= 0:
        return None

    short_inst = _deribit_instrument(ticker, expiration, short_strike, leg_type)
    long_inst  = _deribit_instrument(ticker, expiration, long_strike, leg_type)
    short_leg = _chain_lookup(chain, short_inst)
    long_leg  = _chain_lookup(chain, long_inst)
    dte = _dte_now(expiration)

    # Determine cost-to-close = short_value - long_value (we owe short, recover long)
    if short_leg is not None and long_leg is not None:
        sv = float(short_leg.get("mark_price") or short_leg.get("mid_price") or 0)
        lv = float(long_leg.get("mark_price") or long_leg.get("mid_price") or 0)
        cost_to_close = sv - lv
    else:
        cost_to_close = float("nan")

    if dte <= rules["time_exit_dte"]:
        # Settle at intrinsic
        spot = _df.get_index_price(ticker) or 0
        if leg_type == "put":
            sv_intrinsic = max(0.0, short_strike - spot)
            lv_intrinsic = max(0.0, long_strike - spot)
        else:
            sv_intrinsic = max(0.0, spot - short_strike)
            lv_intrinsic = max(0.0, spot - long_strike)
        cost_to_close = sv_intrinsic - lv_intrinsic
        pnl_pct = (entry_credit - cost_to_close) / entry_credit
        pnl_usd = entry_credit - cost_to_close
        if not dry_run:
            max_loss_floor = -((width / entry_credit) - 1.0)
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct, pnl_usd,
                       "Time Exit", strategy, entry_credit, max_loss_floor=max_loss_floor)
        return f"  ✓ {ticker} {strategy} ${short_strike:,.0f}/{long_strike:,.0f} → Time Exit (mkt: {pnl_pct:+.1%})"

    if not (cost_to_close == cost_to_close):  # NaN check
        return None

    pnl_pct = (entry_credit - cost_to_close) / entry_credit

    if pnl_pct >= rules["take_profit_credit"]:
        if not dry_run:
            max_loss_floor = -((width / entry_credit) - 1.0)
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct,
                       entry_credit - cost_to_close, "Take Profit (+50%)",
                       strategy, entry_credit, max_loss_floor=max_loss_floor)
        return f"  ✓ {ticker} {strategy} ${short_strike:,.0f}/{long_strike:,.0f} → Take Profit (mkt: {pnl_pct:+.1%})"

    structural_floor = -((width / entry_credit) - 1.0)
    if pnl_pct <= structural_floor:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct,
                       entry_credit - cost_to_close, "Stop Loss (max)",
                       strategy, entry_credit, max_loss_floor=structural_floor)
        return f"  ✓ {ticker} {strategy} ${short_strike:,.0f}/{long_strike:,.0f} → Stop Loss (mkt: {pnl_pct:+.1%})"
    return None


def _evaluate_iron_condor(conn: sqlite3.Connection, row: sqlite3.Row,
                           chain: pd.DataFrame, rules: Dict[str, Any],
                           dry_run: bool) -> Optional[str]:
    """4-leg iron condor."""
    ticker = str(row["ticker"]).upper()
    expiration = str(row["expiration"])
    try:
        sp = float(row["short_put_strike"]); lp = float(row["long_put_strike"])
        sc = float(row["short_call_strike"]); lc = float(row["long_call_strike"])
    except (TypeError, ValueError):
        return None
    if 0 in (sp, lp, sc, lc):
        return None
    entry_credit = float(row["entry_price"] or 0)
    if entry_credit <= 0:
        return None
    put_width = sp - lp
    call_width = lc - sc
    width = max(put_width, call_width)
    if width <= 0:
        return None

    legs = []
    for strike, opt_type, qty in (
        (sp, "put", -1), (lp, "put", +1),
        (sc, "call", -1), (lc, "call", +1),
    ):
        inst = _deribit_instrument(ticker, expiration, strike, opt_type)
        leg = _chain_lookup(chain, inst)
        if leg is None:
            return None  # missing leg — wait for next run
        legs.append((qty, float(leg.get("mark_price") or leg.get("mid_price") or 0)))
    cost_to_close = sum(-qty * price for qty, price in legs)

    dte = _dte_now(expiration)
    if dte <= rules["time_exit_dte"]:
        spot = _df.get_index_price(ticker) or 0
        # intrinsic settlement
        ic_legs = [
            (-1, max(0.0, sp - spot)), (+1, max(0.0, lp - spot)),
            (-1, max(0.0, spot - sc)), (+1, max(0.0, spot - lc)),
        ]
        cost_to_close = sum(-qty * price for qty, price in ic_legs)
        pnl_pct = (entry_credit - cost_to_close) / entry_credit
        if not dry_run:
            max_loss_floor = -((width / entry_credit) - 1.0)
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct,
                       entry_credit - cost_to_close, "Time Exit",
                       "Iron Condor", entry_credit, max_loss_floor=max_loss_floor)
        return f"  ✓ {ticker} IC {lp:,.0f}/{sp:,.0f}—{sc:,.0f}/{lc:,.0f} → Time Exit (mkt: {pnl_pct:+.1%})"

    pnl_pct = (entry_credit - cost_to_close) / entry_credit
    if pnl_pct >= rules["take_profit_credit"]:
        if not dry_run:
            max_loss_floor = -((width / entry_credit) - 1.0)
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct,
                       entry_credit - cost_to_close, "Take Profit (+50%)",
                       "Iron Condor", entry_credit, max_loss_floor=max_loss_floor)
        return f"  ✓ {ticker} IC → Take Profit (mkt: {pnl_pct:+.1%})"

    structural_floor = -((width / entry_credit) - 1.0)
    if pnl_pct <= structural_floor:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), cost_to_close, pnl_pct,
                       entry_credit - cost_to_close, "Stop Loss (max)",
                       "Iron Condor", entry_credit, max_loss_floor=structural_floor)
        return f"  ✓ {ticker} IC → Stop Loss (mkt: {pnl_pct:+.1%})"
    return None


def _evaluate_calendar(conn: sqlite3.Connection, row: sqlite3.Row,
                        chain: pd.DataFrame, rules: Dict[str, Any],
                        dry_run: bool) -> Optional[str]:
    """Calendar (single-strike time spread) — long back, short front."""
    ticker = str(row["ticker"]).upper()
    front_exp = str(row["expiration"])
    strategy = str(row["strategy_name"] or "")
    back_exp = _parse_back_expiration(strategy)
    if not back_exp:
        return None
    strike = float(row["strike"])
    opt_type = str(row["type"]).lower()
    entry_debit = float(row["entry_price"] or 0)
    if entry_debit <= 0:
        return None

    front_inst = _deribit_instrument(ticker, front_exp, strike, opt_type)
    back_inst  = _deribit_instrument(ticker, back_exp,  strike, opt_type)
    front_leg = _chain_lookup(chain, front_inst)
    back_leg  = _chain_lookup(chain, back_inst)

    front_dte = _dte_now(front_exp)
    if front_dte <= rules["time_exit_dte"]:
        # Front expires roughly worthless if OTM; settle at intrinsic vs back leg residual.
        spot = _df.get_index_price(ticker) or 0
        if opt_type == "call":
            front_settle = max(0.0, spot - strike)
        else:
            front_settle = max(0.0, strike - spot)
        back_value = float(back_leg.get("mark_price") or back_leg.get("mid_price") or 0) if back_leg is not None else 0.0
        # Net value to close = back - front
        current_value = back_value - front_settle
        pnl_pct = (current_value - entry_debit) / entry_debit
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), current_value, pnl_pct,
                       current_value - entry_debit, "Time Exit (front expiry)",
                       "Calendar Call" if opt_type == "call" else "Calendar Put",
                       entry_debit)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Time Exit (mkt: {pnl_pct:+.1%})"

    if front_leg is None or back_leg is None:
        return None
    fv = float(front_leg.get("mark_price") or front_leg.get("mid_price") or 0)
    bv = float(back_leg.get("mark_price")  or back_leg.get("mid_price")  or 0)
    if fv <= 0 or bv <= 0:
        return None
    current_value = bv - fv
    pnl_pct = (current_value - entry_debit) / entry_debit

    # Treat calendar like long premium: TP at +50% (calendars rarely double),
    # SL at -50% — overrides the more aggressive long_option +100%.
    cal_tp = 0.50
    cal_sl = -0.50
    if pnl_pct >= cal_tp:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), current_value, pnl_pct,
                       current_value - entry_debit, "Take Profit (+50%)",
                       "Calendar Call" if opt_type == "call" else "Calendar Put",
                       entry_debit)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Take Profit (mkt: {pnl_pct:+.1%})"
    if pnl_pct <= cal_sl:
        if not dry_run:
            _close_row(conn, int(row["entry_id"]), current_value, pnl_pct,
                       current_value - entry_debit, "Stop Loss (-50%)",
                       "Calendar Call" if opt_type == "call" else "Calendar Put",
                       entry_debit)
        return f"  ✓ {ticker} {strategy_short(row)} ${strike:,.0f} → Stop Loss (mkt: {pnl_pct:+.1%})"
    return None


def strategy_short(row: sqlite3.Row) -> str:
    """Compact name for display ('Long Call' / 'Calendar Put' / etc.)."""
    s = str(row["strategy_name"] or "")
    # strip the [back …] annotation we add for calendars
    return s.split(" [back", 1)[0].strip() or "?"


# ── CLI entry ────────────────────────────────────────────────────────────

def main() -> None:
    """Run the enforcer end-to-end and print a summary."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] crypto_enforce_exits starting")
    if not os.path.exists(_CRYPTO_DB_PATH):
        print(f"  No crypto ledger at {_CRYPTO_DB_PATH} — nothing to do.")
        print(f"[{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M:%S UTC}] done")
        return
    closed = enforce_exits(_CRYPTO_DB_PATH)
    if not closed:
        print("  No exits triggered.")
    else:
        print(f"  Auto-closed {len(closed)} position(s):")
        for line in closed:
            print(line)
    print(f"[{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M:%S UTC}] crypto_enforce_exits done")


if __name__ == "__main__":
    main()
