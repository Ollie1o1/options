#!/usr/bin/env python3
"""
Portfolio viewer — reads open and closed paper trades from paper_trades.db
and displays a clean P/L summary with live price fetching.
"""

import logging
import os
import sqlite3
import shutil
import sys
from contextlib import closing
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add project root and src to sys.path for direct execution
_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_src_path = _file_path.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    try:
        from . import formatting as fmt
    except (ImportError, ValueError):
        import formatting as fmt
    HAS_FMT = fmt.supports_color()
except Exception:
    HAS_FMT = False
    fmt = None

try:
    try:
        from .utils import is_short_position as _is_short
        from .utils import bs_delta, bs_gamma, bs_vega, bs_theta, american_price
    except (ImportError, ValueError):
        from utils import is_short_position as _is_short
        from utils import bs_delta, bs_gamma, bs_vega, bs_theta, american_price
    HAS_BS = True
except Exception:
    HAS_BS = False
    def _is_short(strategy_name: str) -> bool:  # type: ignore[misc]
        s = (strategy_name or "").lower()
        return any(k in s for k in ("short", "credit", "covered", "cash-secured", "cash secured", "naked", "iron condor", "sell"))

try:
    try:
        from .stress_test import print_stress_test, _classify_structure
        from .backtester import print_paper_trade_ic
    except (ImportError, ValueError):
        from stress_test import print_stress_test, _classify_structure
        from backtester import print_paper_trade_ic
    HAS_STRESS = True
except Exception:
    HAS_STRESS = False
    def _classify_structure(trade) -> str:  # type: ignore[misc]
        sn = str(trade.get("strategy_name", "") or "").lower()
        if (trade.get("short_put_strike") and trade.get("short_call_strike")) or "iron condor" in sn:
            return "iron_condor"
        if trade.get("long_strike") or any(k in sn for k in ("bull put", "bear call")):
            return "spread"
        return "single"

try:
    try:
        from .data_fetching import get_risk_free_rate as _get_rfr
    except (ImportError, ValueError):
        from data_fetching import get_risk_free_rate as _get_rfr
    HAS_RFR = True
except Exception:
    HAS_RFR = False

DB_PATH = "paper_trades.db"


def _width() -> int:
    try:
        return max(80, min(shutil.get_terminal_size(fallback=(100, 24)).columns, 120))
    except Exception:
        return 100


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _dte(expiration: str) -> int:
    try:
        exp = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
        return (exp - date.today()).days
    except Exception:
        return 0


def _legs_for_row(r) -> list:
    """Return list of (opt_type, strike, qty_sign) for the given DB row.

    qty_sign: +1 = long leg, -1 = short leg. Used to compute net position
    value as sum(qty * leg_price) so credit spreads / iron condors mark to
    market correctly. Returns [] for unrecognized rows.
    """
    structure = _classify_structure(r)
    if structure == "iron_condor":
        return [
            ("put",  float(r["short_put_strike"]),  -1),
            ("put",  float(r["long_put_strike"]),   +1),
            ("call", float(r["short_call_strike"]), -1),
            ("call", float(r["long_call_strike"]),  +1),
        ]
    if structure == "spread":
        opt_type = str(r.get("type", "")).lower() or ("put" if "bull put" in str(r.get("strategy_name", "")).lower() else "call")
        long_strike = r.get("long_strike")
        if long_strike in (None, "", 0):
            # Legacy SPREAD:long:width:max_loss fallback
            try:
                long_strike = float(str(r.get("strategy_name", "")).split(":")[1])
            except (ValueError, IndexError):
                return []
        return [
            (opt_type, float(r["strike"]),  -1),
            (opt_type, float(long_strike), +1),
        ]
    # Single leg
    sign = -1 if _is(_get_strategy_name(r)) else 1
    return [(str(r["type"]).lower(), float(r["strike"]), sign)]


def _is(strategy_name: str) -> bool:
    return _is_short(strategy_name)


def _get_strategy_name(r) -> str:
    try:
        return r["strategy_name"] or ""
    except Exception:
        return ""


def _fetch_live_price(ticker: str, expiration: str, strike: float, opt_type: str, _retries: int = 2) -> Optional[float]:
    """Fetch live option price via OCC symbol with retry. Returns None on failure."""
    if not HAS_YF:
        return None
    try:
        from pandas import to_datetime
        import numpy as np
        exp = to_datetime(expiration)
        date_str = exp.strftime("%y%m%d")
        otype = "C" if opt_type.lower() == "call" else "P"
        strike_str = f"{int(float(strike) * 1000):08d}"
        occ = f"{ticker.upper()}{date_str}{otype}{strike_str}"

        tkr = yf.Ticker(occ)
        price = None

        try:
            price = getattr(tkr.fast_info, "last_price", None)
            if price and float(price) > 0:
                price = float(price)
        except Exception:
            pass

        if price is None or np.isnan(price) or price <= 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = tkr.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        # Black-Scholes Fallback (mirrors PaperManager)
        if (price is None or np.isnan(price) or price <= 0) and HAS_BS:
            try:
                tkr_spot = yf.Ticker(ticker)
                S = getattr(tkr_spot.fast_info, "last_price", None)
                if not S:
                    hist_s = tkr_spot.history(period="5d")
                    if not hist_s.empty:
                        S = float(hist_s["Close"].iloc[-1])
                
                if S and S > 0:
                    exp_dt = datetime.strptime(expiration[:10], "%Y-%m-%d")
                    T = max((exp_dt - datetime.now()).days / 365.25, 1/365.25)
                    _rfr = _get_rfr() if HAS_RFR else 0.045
                    # Use a standard 30% volatility for fallback if we don't have stored IV
                    price = american_price(opt_type.lower(), float(S), float(strike), T, _rfr, 0.30)
            except Exception:
                pass

        if price is not None and not np.isnan(price) and price > 0:
            return float(price)

    except Exception:
        if _retries > 0:
            import time
            time.sleep(0.5)
            return _fetch_live_price(ticker, expiration, strike, opt_type, _retries - 1)
    return None


def _has_entry_greeks(r) -> bool:
    """Check if a trade row has stored entry Greeks."""
    try:
        keys = r.keys() if hasattr(r, 'keys') else []
        return "entry_delta" in keys and r["entry_delta"] is not None
    except Exception:
        return False


def _print_pnl_attribution(closed_trades: list, stock_prices: dict, width: int):
    """
    Display P&L attribution breakdown: delta, gamma, theta, vega contributions.
    Uses stored entry Greeks from paper trades DB.
    """
    total_delta_pnl = 0.0
    total_gamma_pnl = 0.0
    total_theta_pnl = 0.0
    total_vega_pnl = 0.0
    total_actual_pnl = 0.0
    counted = 0

    for r in closed_trades:
        try:
            entry_delta = float(r["entry_delta"])
            entry_theta = float(r["entry_theta"])
            entry_gamma = float(r["entry_gamma"])
            entry_vega = float(r["entry_vega"])
            entry_price = float(r["entry_price"])
            is_short = _is_short(r.get("strategy_name", ""))
            pnl_ratio = float(r["pnl_pct"]) if r["pnl_pct"] is not None else 0.0

            # We don't have S_entry/S_exit stored, so use pnl_ratio * entry_price * 100 as actual P&L
            actual_pnl = pnl_ratio * entry_price * 100

            # Estimate days held
            try:
                trade_date = datetime.strptime(str(r["date"])[:10], "%Y-%m-%d").date()
                exit_dt = datetime.strptime(str(r["exit_date"])[:10], "%Y-%m-%d").date()
                days_held = max((exit_dt - trade_date).days, 1)
            except Exception:
                days_held = 14  # fallback

            # Theta P&L (daily theta * days held * 100 shares)
            # For short positions, theta P&L has opposite sign
            sign_mult = -1.0 if is_short else 1.0
            theta_pnl = sign_mult * entry_theta * days_held * 100

            # For delta/gamma/vega, we'd need spot price change.
            # Approximate: actual - theta = delta+gamma+vega+residual
            # Attribute remaining proportionally to Greeks magnitude
            remaining = actual_pnl - theta_pnl

            # Without spot price data, attribute remaining based on Greek magnitudes
            # Scale gamma to comparable units: gamma * S * typical_5pct_move
            # (gamma is per-share per $1 move, so raw value is ~0.001-0.05)
            S_approx = float(stock_prices.get(r.get("ticker", ""), 100.0))
            abs_d = abs(entry_delta)
            abs_g = 0.5 * abs(entry_gamma) * (S_approx * 0.05) ** 2 * 100  # quadratic: 0.5 * gamma * (ΔS)^2 * 100 shares
            abs_v = abs(entry_vega)
            total_mag = abs_d + abs_g + abs_v
            if total_mag > 0:
                delta_pnl = remaining * abs_d / total_mag
                gamma_pnl = remaining * abs_g / total_mag
                vega_pnl = remaining * abs_v / total_mag
            else:
                delta_pnl = remaining
                gamma_pnl = 0.0
                vega_pnl = 0.0

            total_delta_pnl += delta_pnl
            total_gamma_pnl += gamma_pnl
            total_theta_pnl += theta_pnl
            total_vega_pnl += vega_pnl
            total_actual_pnl += actual_pnl
            counted += 1
        except Exception:
            continue

    if counted == 0:
        return

    print()
    sep = "  " + "\u2500" * (width - 2)
    header = f"  P&L ATTRIBUTION  ({counted} closed trades with entry Greeks)"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
        print(fmt.colorize(sep, fmt.Colors.DIM))
    else:
        print(header)
        print(sep)

    total_abs = abs(total_delta_pnl) + abs(total_gamma_pnl) + abs(total_theta_pnl) + abs(total_vega_pnl)
    if total_abs == 0:
        total_abs = 1.0

    def _attr_line(name, val):
        pct = val / total_abs * 100 if total_abs > 0 else 0
        sign = "+" if val >= 0 else "-"
        color = ""
        if HAS_FMT and fmt:
            color = fmt.Colors.GREEN if val >= 0 else fmt.Colors.RED
            return fmt.colorize(f"    {name:<8} {sign}${abs(val):>8.0f}  ({pct:>+5.0f}%)", color)
        return f"    {name:<8} {sign}${abs(val):>8.0f}  ({pct:>+5.0f}%)"

    print(_attr_line("Delta:", total_delta_pnl))
    print(_attr_line("Gamma:", total_gamma_pnl))
    print(_attr_line("Theta:", total_theta_pnl))
    print(_attr_line("Vega:", total_vega_pnl))

    residual = total_actual_pnl - (total_delta_pnl + total_gamma_pnl + total_theta_pnl + total_vega_pnl)
    if abs(residual) > 1.0:
        print(_attr_line("Other:", residual))

    if HAS_FMT and fmt:
        print(fmt.colorize(sep, fmt.Colors.DIM))
    else:
        print(sep)


def _print_portfolio_greeks(open_trades: list, width: int):
    """
    Compute and display aggregate portfolio Greeks for open positions.

    Multi-leg structures (vertical credit spreads, iron condors) reprice each
    leg independently and net the Greeks with proper signs, so a short put
    spread doesn't show up as if it were a naked short.
    """
    if not open_trades or not HAS_BS or not HAS_YF:
        return

    unique_tickers = list({r["ticker"] for r in open_trades})
    stock_prices: dict = {}
    for ticker in unique_tickers:
        try:
            tkr = yf.Ticker(ticker)
            p = getattr(tkr.fast_info, "last_price", None)
            if p and float(p) > 0:
                stock_prices[ticker] = float(p)
        except Exception:
            pass

    rfr = _get_rfr() if HAS_RFR else 0.045
    now_dt = datetime.now()
    net_delta = 0.0
    net_gamma_dollar = 0.0
    net_vega = 0.0
    net_theta = 0.0
    counted = 0

    for r in open_trades:
        ticker = r["ticker"]
        S = stock_prices.get(ticker)
        if S is None:
            continue
        try:
            exp_dt = datetime.strptime(r["expiration"][:10], "%Y-%m-%d")
            T = max((exp_dt - now_dt).total_seconds() / (365.25 * 24 * 3600), 1.0 / (365 * 24))
        except Exception:
            continue

        sigma = 0.25
        try:
            stored_iv = r.get("entry_iv") if isinstance(r, dict) else (r["entry_iv"] if "entry_iv" in r.keys() else None)
            if stored_iv is not None:
                sv = float(stored_iv)
                if 0.01 < sv < 5.0:
                    sigma = sv
        except Exception:
            pass

        legs = _legs_for_row(r)
        if not legs:
            continue

        try:
            row_delta = 0.0
            row_gamma = 0.0
            row_vega  = 0.0
            row_theta = 0.0
            for leg_type, leg_strike, leg_qty in legs:
                d = float(bs_delta(leg_type, S, leg_strike, T, rfr, sigma))
                g = float(bs_gamma(S, leg_strike, T, rfr, sigma))
                v = float(bs_vega(S, leg_strike, T, rfr, sigma))
                t = float(bs_theta(leg_type, S, leg_strike, T, rfr, sigma))
                row_delta += leg_qty * d
                row_gamma += leg_qty * g
                row_vega  += leg_qty * v
                row_theta += leg_qty * t
            net_delta        += row_delta
            net_gamma_dollar += 0.5 * row_gamma * (S * 0.01) ** 2 * 100
            net_vega         += row_vega * 100
            net_theta        += row_theta * 100
            counted += 1
        except Exception as _greeks_exc:
            logging.getLogger(__name__).debug("Greeks computation failed for position: %s", _greeks_exc)

    if counted == 0:
        return

    # ── Display ────────────────────────────────────────────────────────────────
    print()
    sep = "  " + "\u2500" * (width - 2)
    header = "  PORTFOLIO GREEKS  (uses entry IV when stored  |  long=+  short=−)"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
        print(fmt.colorize(sep, fmt.Colors.DIM))
    else:
        print(header)
        print("  " + "-" * (width - 2))

    # Delta
    delta_color = (fmt.Colors.GREEN if net_delta > 0.10 else
                   fmt.Colors.RED if net_delta < -0.10 else fmt.Colors.YELLOW) if HAS_FMT and fmt else ""
    delta_label = f"  Net Δ: {net_delta:+.2f}"
    if HAS_FMT and fmt:
        print(fmt.colorize(delta_label, delta_color))
    else:
        print(delta_label)

    # Gamma ($ per 1% stock move)
    gd = net_gamma_dollar
    gcolor = (fmt.Colors.GREEN if gd > 0 else fmt.Colors.RED) if HAS_FMT and fmt else ""
    gamma_label = f"  Net Γ ($/1% move): {gd:+.2f}"
    if HAS_FMT and fmt:
        print(fmt.colorize(gamma_label, gcolor))
    else:
        print(gamma_label)

    # Vega ($ per 1% IV rise)
    vc = (fmt.Colors.GREEN if net_vega > 0 else fmt.Colors.RED) if HAS_FMT and fmt else ""
    vega_label = f"  Net Vega ($/1% IV): {net_vega:+.2f}"
    if HAS_FMT and fmt:
        print(fmt.colorize(vega_label, vc))
    else:
        print(vega_label)

    # Theta ($/day)
    tc = (fmt.Colors.GREEN if net_theta > 0 else fmt.Colors.RED) if HAS_FMT and fmt else ""
    theta_label = f"  Net Θ ($/day): {net_theta:+.2f}"
    if HAS_FMT and fmt:
        print(fmt.colorize(theta_label, tc))
    else:
        print(theta_label)

    # Directional bias warnings
    warnings_list = []
    if abs(net_delta) > 0.5:
        direction = "BULLISH" if net_delta > 0 else "BEARISH"
        warnings_list.append(f"Strong {direction} bias (net delta: {net_delta:+.2f})")
    if net_theta < -5.0:
        warnings_list.append(f"High time decay exposure (net theta: ${net_theta:.2f}/day)")
    if abs(net_vega) > 1.0:
        direction = "long" if net_vega > 0 else "short"
        warnings_list.append(f"Significant {direction} vol exposure (net vega: {net_vega:+.2f})")
    for w in warnings_list:
        warn_line = f"  \u26a0 {w}"
        if HAS_FMT and fmt:
            print(fmt.colorize(warn_line, fmt.Colors.YELLOW, bold=True))
        else:
            print(warn_line)

    note = f"  [{counted}/{len(open_trades)} positions, entry IV when available]"
    if HAS_FMT and fmt:
        print(fmt.colorize(note, fmt.Colors.DIM))
    else:
        print(note)


def view_portfolio():
    """Display paper portfolio from paper_trades.db."""
    width = _width()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not os.path.exists(DB_PATH):
        print("\n  No paper_trades.db found. Log some trades first.\n")
        return

    # Enforce exit rules (TP / SL / strike breach / time exit) BEFORE displaying.
    # Without this, the viewer just shows stale OPEN positions that should have
    # been auto-closed — which is how a -16k drawdown accumulated silently.
    try:
        try:
            from .paper_manager import PaperManager
        except (ImportError, ValueError):
            from paper_manager import PaperManager
        print("  Enforcing exit rules...", end="", flush=True)
        PaperManager(db_path=DB_PATH, config_path="config.json").update_positions()
        print("\r" + " " * 30 + "\r", end="")
    except Exception as _e:
        print(f"\r  (exit enforcement skipped: {_e})")

    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trades ORDER BY date DESC").fetchall()
            all_rows = [dict(r) for r in rows]
    except Exception as e:
        print(f"\n  Error reading database: {e}\n")
        return

    # ── Header ─────────────────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(fmt.draw_box(f"PAPER PORTFOLIO  \u2014  {now_str}", width, double=True))
    else:
        print("=" * width)
        print(f"  PAPER PORTFOLIO  \u2014  {now_str}")
        print("=" * width)

    if not all_rows:
        print("\n  No trades logged yet.\n")
        return

    open_trades  = [r for r in all_rows if r["status"] == "OPEN"]
    closed_trades = [r for r in all_rows if r["status"] == "CLOSED"]

    # ── Open Positions ─────────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(fmt.format_header("  OPEN POSITIONS", ""))
    else:
        print("  OPEN POSITIONS")
    print()

    if not open_trades:
        print(_c("  No open positions.", fmt.Colors.DIM if HAS_FMT and fmt else ""))
    else:
        # Header row
        today_dt = date.today()
        hdr = (
            f"  {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Expiry':<12}"
            f" {'DTE':>4} {'Opened':<11} {'Held':>5} {'Entry $':>8} {'Live $':>8}"
            f" {'P/L $':>10} {'P/L %':>7}"
        )
        sep = "  " + "\u2500" * (width - 2)

        if HAS_FMT and fmt:
            print(fmt.colorize(hdr, fmt.Colors.BOLD))
            print(fmt.colorize(sep, fmt.Colors.DIM))
        else:
            print(hdr)
            print("  " + "-" * (width - 2))

        total_pnl_usd   = 0.0
        total_cost_usd  = 0.0
        fetched_count   = 0

        # Parallel-fetch all live option prices up front. For multi-leg
        # structures (spreads / iron condors) we need ALL legs' marks so the
        # P&L row shows the true net cost-to-close rather than just one leg.
        from concurrent.futures import ThreadPoolExecutor
        _live_tasks_set: set = set()
        for r in open_trades:
            for opt_type, strike, _qty in _legs_for_row(r):
                _live_tasks_set.add((r["ticker"], r["expiration"][:10], strike, opt_type))
        _live_tasks = list(_live_tasks_set)
        _live_prices: dict = {}
        def _fetch_one(args):
            return args, _fetch_live_price(*args)
        if _live_tasks:
            _workers = min(len(_live_tasks), 8)
            with ThreadPoolExecutor(max_workers=_workers) as _ex:
                for key, price in _ex.map(_fetch_one, _live_tasks):
                    _live_prices[key] = price

        for r in open_trades:
            ticker      = r["ticker"]
            structure   = _classify_structure(r)
            strike      = float(r["strike"])
            expiry      = r["expiration"][:10]
            opened      = r["date"][:10]
            entry_price = float(r["entry_price"])
            dte         = _dte(expiry)
            short       = _is_short(r["strategy_name"] or "")
            try:
                entry_dt  = datetime.strptime(r["date"][:10], "%Y-%m-%d").date()
                days_held = (today_dt - entry_dt).days
            except Exception:
                days_held = 0
            held_str = f"{days_held}d"

            # Display label and strike for the type column
            if structure == "iron_condor":
                opt_type_disp = "IC"
                # Show short put / short call
                sp = float(r.get("short_put_strike") or 0)
                sc = float(r.get("short_call_strike") or 0)
                strike_disp = f"{sp:.0f}/{sc:.0f}"
            elif structure == "spread":
                sn_low = (r.get("strategy_name") or "").lower()
                opt_type_disp = "BPS" if "bull put" in sn_low else ("BCS" if "bear call" in sn_low else "SPR")
                long_k = r.get("long_strike")
                try:
                    long_k = float(long_k) if long_k not in (None, "", 0) else None
                except (TypeError, ValueError):
                    long_k = None
                strike_disp = f"{strike:.0f}/{long_k:.0f}" if long_k else f"{strike:.0f}"
            else:
                opt_type_disp = str(r["type"]).upper()
                strike_disp = f"{strike:.2f}"

            # Type column color
            if HAS_FMT and fmt:
                if structure == "single":
                    tc = fmt.Colors.BRIGHT_GREEN if opt_type_disp == "CALL" else fmt.Colors.BRIGHT_RED
                else:
                    tc = fmt.Colors.BRIGHT_CYAN
                type_str = fmt.colorize(f"{opt_type_disp:<5}", tc)
            else:
                type_str = f"{opt_type_disp:<5}"

            # DTE column
            if HAS_FMT and fmt:
                dc = fmt.Colors.BRIGHT_RED if dte < 7 else (fmt.Colors.YELLOW if dte < 14 else fmt.Colors.WHITE)
                dte_str = fmt.colorize(f"{max(dte,0):>4}", dc)
            else:
                dte_str = f"{max(dte,0):>4}"

            # Mark-to-market: for multi-leg structures, sum each leg's mark
            # weighted by qty sign (-1 short, +1 long); position value =
            # sum(qty * leg_now). Single-leg falls through the same path.
            legs = _legs_for_row(r)
            leg_marks = []
            for leg_type, leg_strike, leg_qty in legs:
                lp = _live_prices.get((ticker, expiry, leg_strike, leg_type))
                leg_marks.append((leg_qty, lp))
            all_legs_priced = all(lp is not None and lp > 0 for _, lp in leg_marks)

            if structure == "single":
                live_price = leg_marks[0][1] if leg_marks else None
                live_str = f"${live_price:.2f}" if (live_price is not None and live_price > 0) else None
                if live_price is not None and live_price > 0:
                    pnl_per = (entry_price - live_price) if short else (live_price - entry_price)
                    pnl_usd_row = pnl_per * 100
                    pnl_pct_row = pnl_per / entry_price * 100 if entry_price > 0 else 0.0
                    cost_basis = entry_price * 100
                else:
                    pnl_usd_row = None
                    pnl_pct_row = None
                    cost_basis = 0.0
            else:
                # Spread / iron condor — entry_price is the net credit collected
                if all_legs_priced:
                    # current_credit_to_close = short_now - long_now
                    current_credit = sum(-qty * lp for qty, lp in leg_marks)
                    pnl_per = entry_price - current_credit  # decay = profit for credit seller
                    pnl_usd_row = pnl_per * 100
                    pnl_pct_row = pnl_per / entry_price * 100 if entry_price > 0 else 0.0
                    live_str = f"${current_credit:.2f}"
                    # Cost basis ≈ max_loss (true defined risk) for concentration math
                    ml_col = r.get("max_loss_usd")
                    try:
                        cost_basis = abs(float(ml_col)) if ml_col not in (None, "", 0) else entry_price * 100
                    except (TypeError, ValueError):
                        cost_basis = entry_price * 100
                else:
                    pnl_usd_row = None
                    pnl_pct_row = None
                    live_str = None
                    cost_basis = 0.0

            if pnl_usd_row is not None:
                total_pnl_usd  += pnl_usd_row
                total_cost_usd += cost_basis
                fetched_count  += 1
                sign = "+" if pnl_usd_row >= 0 else "-"
                raw_usd = f"{sign}${abs(pnl_usd_row):.2f}"
                raw_pct = f"{sign}{abs(pnl_pct_row):.1f}%"
                if HAS_FMT and fmt:
                    pc = fmt.Colors.GREEN if pnl_usd_row >= 0 else fmt.Colors.RED
                    usd_str = fmt.colorize(f"{raw_usd:>10}", pc)
                    pct_str = fmt.colorize(f"{raw_pct:>7}", pc)
                else:
                    usd_str = f"{raw_usd:>10}"
                    pct_str = f"{raw_pct:>7}"
                live_render = live_str if live_str is not None else "—"
            else:
                live_render = _c(f"{'—':>8}", fmt.Colors.DIM if HAS_FMT and fmt else "")
                usd_str  = _c(f"{'—':>10}", fmt.Colors.DIM if HAS_FMT and fmt else "")
                pct_str  = _c(f"{'—':>7}", fmt.Colors.DIM if HAS_FMT and fmt else "")

            print(
                f"  {ticker:<7} {type_str} {strike_disp:>8} {expiry:<12}"
                f" {dte_str} {opened:<11} {held_str:>5} ${entry_price:>6.2f} {live_render:>8}"
                f" {usd_str} {pct_str}"
            )

            # Backwards-compat placeholder for the delta-drift sub-line below;
            # multi-leg structures skip that block (see if-guard).
            live_price = leg_marks[0][1] if (structure == "single" and leg_marks) else None

            # Delta drift sub-line
            if live_price is not None and live_price > 0 and HAS_BS:
                try:
                    exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
                    now_dt = datetime.now()
                    T_now = max((exp_dt - now_dt).total_seconds() / (365.25 * 24 * 3600), 1.0 / (365 * 24))  # floor at 1 hour
                    rfr = _get_rfr() if HAS_RFR else 0.045
                    sigma = 0.25
                    try:
                        stored_iv = r["entry_iv"] if "entry_iv" in r.keys() else None
                        if stored_iv is not None:
                            sv = float(stored_iv)
                            if 0.01 < sv < 5.0:
                                sigma = sv
                    except Exception:
                        pass
                    live_underlying = None
                    try:
                        tkr_obj = yf.Ticker(ticker)
                        live_underlying = getattr(tkr_obj.fast_info, "last_price", None) or getattr(tkr_obj.fast_info, "regularMarketPrice", None)
                    except Exception:
                        pass
                    if live_underlying and live_underlying > 0:
                        current_delta = float(bs_delta(r["type"].lower(), live_underlying, strike, T_now, rfr, sigma))
                        entry_delta_val = None
                        try:
                            ed = r["entry_delta"] if "entry_delta" in r.keys() else None
                            if ed is not None:
                                entry_delta_val = float(ed)
                        except Exception:
                            pass
                        if entry_delta_val is not None:
                            drift_note = "gamma exposure increasing" if abs(current_delta) > abs(entry_delta_val) else "gamma exposure decreasing"
                            drift_line = f"    delta: {entry_delta_val:+.2f} \u2192 {current_delta:+.2f}  ({drift_note})"
                            if HAS_FMT and fmt:
                                drift_color = fmt.Colors.YELLOW if abs(current_delta - entry_delta_val) > 0.15 else fmt.Colors.DIM
                                print(fmt.colorize(drift_line, drift_color))
                            else:
                                print(drift_line)
                except Exception:
                    pass

            # 50% max-profit milestone alert (only when live price is available)
            if live_price is not None and live_price > 0:
                tp_threshold = 0.50  # config.get("exit_rules", {}).get("take_profit", 0.50) if config available
                if short:
                    # Short: profit = entry - live (premium decay)
                    profit_pct = (entry_price - live_price) / entry_price if entry_price > 0 else 0.0
                else:
                    # Long: profit = live - entry
                    profit_pct = (live_price - entry_price) / entry_price if entry_price > 0 else 0.0

                if profit_pct >= tp_threshold:
                    milestone_line = f"    ✓ {profit_pct:.0%} profit — consider closing ({tp_threshold:.0%} target reached)"
                    if HAS_FMT and fmt:
                        print(fmt.colorize(milestone_line, fmt.Colors.GREEN, bold=True))
                    else:
                        print(milestone_line)

        # Open totals
        if HAS_FMT and fmt:
            print(fmt.colorize(sep, fmt.Colors.DIM))
        else:
            print("  " + "-" * (width - 2))

        if total_cost_usd > 0:
            total_pct = total_pnl_usd / total_cost_usd * 100
        else:
            total_pct = 0.0
        sign = "+" if total_pnl_usd >= 0 else ""
        raw_total = f"{sign}${abs(total_pnl_usd):.2f}  ({sign}{abs(total_pct):.1f}%)"
        if total_pnl_usd < 0:
            raw_total = f"-${abs(total_pnl_usd):.2f}  (-{abs(total_pct):.1f}%)"
        fetch_note = f"[{fetched_count}/{len(open_trades)} live prices]"
        summary = f"  Unrealized P/L: {raw_total}   {fetch_note}"
        if HAS_FMT and fmt:
            pc = fmt.Colors.GREEN if total_pnl_usd >= 0 else fmt.Colors.RED
            print(fmt.colorize(summary, pc, bold=True))
        else:
            print(summary)

        # Concentration warning — flag if any ticker > 40% of total invested capital
        if total_cost_usd > 0:
            ticker_exp: dict = {}
            for r in open_trades:
                t = r["ticker"]
                ticker_exp[t] = ticker_exp.get(t, 0.0) + float(r["entry_price"]) * 100
            hot = {t: v / total_cost_usd for t, v in ticker_exp.items() if v / total_cost_usd > 0.40}
            if hot:
                conc_msg = "  ⚠  Concentration risk: " + ", ".join(
                    f"{t} {pct:.0%} of book" for t, pct in sorted(hot.items(), key=lambda x: -x[1])
                )
                if HAS_FMT and fmt:
                    print(fmt.colorize(conc_msg, fmt.Colors.YELLOW, bold=True))
                else:
                    print(conc_msg)

        # Portfolio max loss aggregation (defined-risk structures have stored max_loss)
        total_max_loss = 0.0
        has_undefined_risk = False
        for r in open_trades:
            structure = _classify_structure(r)
            sn = str(r.get("strategy_name", ""))
            if structure in ("spread", "iron_condor"):
                ml_col = r.get("max_loss_usd")
                ml_val = None
                try:
                    if ml_col not in (None, "", 0):
                        ml_val = abs(float(ml_col))
                except (TypeError, ValueError):
                    ml_val = None
                if ml_val is None and sn.startswith("SPREAD:"):
                    # Legacy fallback parsing
                    try:
                        parts = sn.split(":")
                        ml_val = abs(float(parts[3])) * 100 if len(parts) >= 4 else None
                    except (ValueError, IndexError):
                        ml_val = None
                if ml_val is None:
                    has_undefined_risk = True
                else:
                    total_max_loss += ml_val
            else:
                # Single-leg: max loss = entry_price * 100 (for longs) or unlimited (shorts)
                ep = abs(float(r.get("entry_price", 0)))
                if _is_short(sn):
                    has_undefined_risk = True
                else:
                    total_max_loss += ep * 100

        if total_max_loss > 0 or has_undefined_risk:
            if total_max_loss > 0:
                risk_str = f"  Portfolio Max Loss: ${total_max_loss:,.0f}"
                if has_undefined_risk:
                    risk_str += "  (+ undefined risk from naked short positions)"
            else:
                risk_str = "  Portfolio Max Loss: N/A  (undefined risk from naked short positions)"
            if HAS_FMT and fmt:
                print(fmt.colorize(risk_str, fmt.Colors.RED))
            else:
                print(risk_str)

        _print_portfolio_greeks(open_trades, width)

        # Stress test — only meaningful with 3+ positions
        if HAS_STRESS and len(open_trades) >= 3:
            try:
                print_stress_test(open_trades, width=width)
            except Exception:
                pass

    # ── Closed Positions ───────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(fmt.format_header("  CLOSED POSITIONS", ""))
    else:
        print("  CLOSED POSITIONS")
    print()

    if not closed_trades:
        print(_c("  No closed trades yet.", fmt.Colors.DIM if HAS_FMT and fmt else ""))
    else:
        hdr = (
            f"  {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Expiry':<12}"
            f" {'Opened':<11} {'Closed':<11} {'Entry $':>8} {'Exit $':>8}"
            f" {'P/L $':>10} {'P/L %':>7} {'Result'}"
        )
        sep = "  " + "\u2500" * (width - 2)

        if HAS_FMT and fmt:
            print(fmt.colorize(hdr, fmt.Colors.BOLD))
            print(fmt.colorize(sep, fmt.Colors.DIM))
        else:
            print(hdr)
            print("  " + "-" * (width - 2))

        closed_pnl_usd = 0.0
        wins = 0

        for r in closed_trades:
            ticker      = r["ticker"]
            opt_type    = r["type"].upper()
            strike      = float(r["strike"])
            expiry      = r["expiration"][:10]
            opened      = r["date"][:10]
            closed_date = (r["exit_date"] or "")[:10] or "—"
            entry_price = float(r["entry_price"])
            exit_price  = float(r["exit_price"]) if r["exit_price"] else 0.0
            pnl_ratio   = float(r["pnl_pct"]) if r["pnl_pct"] is not None else 0.0
            # Compute dollar P/L from actual prices when exit_price is stored,
            # so short positions that lost more than entry premium display correctly.
            # DB pnl_pct is the friction-aware, strategy-aware source of truth
            # (computed in paper_manager._evaluate_*_exit as pnl_raw - friction).
            # Use it to determine win/loss so the row count matches BY STRATEGY
            # and the IC analytics. Fall back to mark-to-market recomputation
            # only when DB lacks pnl_pct (historical rows).
            if r["pnl_pct"] is not None:
                pnl_pct = pnl_ratio * 100
                pnl_usd = pnl_ratio * entry_price * 100
            elif exit_price > 0 and entry_price > 0:
                is_short_closed = _is_short(str(r.get("strategy_name", "")))
                if is_short_closed:
                    pnl_usd = (entry_price - exit_price) * 100
                else:
                    pnl_usd = (exit_price - entry_price) * 100
                pnl_pct = pnl_usd / (entry_price * 100) * 100
            else:
                pnl_usd = 0.0
                pnl_pct = 0.0
            won = pnl_usd > 0
            if won:
                wins += 1
            closed_pnl_usd += pnl_usd

            sign = "+" if pnl_usd >= 0 else ""
            raw_usd = f"{sign}${abs(pnl_usd):.2f}"
            raw_pct = f"{sign}{abs(pnl_pct):.1f}%"
            if pnl_usd < 0:
                raw_usd = f"-${abs(pnl_usd):.2f}"
                raw_pct = f"-{abs(pnl_pct):.1f}%"
            result  = "WIN " if won else "LOSS"

            if HAS_FMT and fmt:
                tc = fmt.Colors.BRIGHT_GREEN if opt_type == "CALL" else fmt.Colors.BRIGHT_RED
                type_str   = fmt.colorize(f"{opt_type:<5}", tc)
                pc         = fmt.Colors.GREEN if won else fmt.Colors.RED
                usd_str    = fmt.colorize(f"{raw_usd:>10}", pc)
                pct_str    = fmt.colorize(f"{raw_pct:>7}", pc)
                result_str = fmt.colorize(result, pc, bold=True)
            else:
                type_str   = f"{opt_type:<5}"
                usd_str    = f"{raw_usd:>10}"
                pct_str    = f"{raw_pct:>7}"
                result_str = result

            print(
                f"  {ticker:<7} {type_str} {strike:>8.2f} {expiry:<12}"
                f" {opened:<11} {closed_date:<11} ${entry_price:>6.2f} ${exit_price:>6.2f}"
                f" {usd_str} {pct_str}  {result_str}"
            )

        if HAS_FMT and fmt:
            print(fmt.colorize(sep, fmt.Colors.DIM))
        else:
            print("  " + "-" * (width - 2))

        n = len(closed_trades)
        win_rate_pct = wins / n * 100 if n > 0 else 0.0
        sign = "+" if closed_pnl_usd >= 0 else "-"
        closed_summary = (
            f"  Realized P/L: {sign}${abs(closed_pnl_usd):.2f}"
            f"   Win Rate: {win_rate_pct:.0f}% ({wins}/{n} trades)"
        )
        if HAS_FMT and fmt:
            pc = fmt.Colors.GREEN if closed_pnl_usd >= 0 else fmt.Colors.RED
            print(fmt.colorize(closed_summary, pc, bold=True))
        else:
            print(closed_summary)

        # ── Enhanced Performance Analytics ─────────────────────────────────
        if n >= 2:
            returns = [float(r["pnl_pct"]) for r in closed_trades if r["pnl_pct"] is not None]
            winning_r = [x for x in returns if x > 0]
            losing_r  = [x for x in returns if x <= 0]
            avg_win  = sum(winning_r) / len(winning_r) if winning_r else 0.0
            avg_loss = sum(losing_r)  / len(losing_r)  if losing_r  else 0.0
            wr = len(winning_r) / len(returns)
            pf = sum(winning_r) / abs(sum(losing_r)) if losing_r and abs(sum(losing_r)) > 1e-12 else (float("inf") if winning_r else 0.0)
            expectancy = wr * avg_win + (1 - wr) * avg_loss

            # Max drawdown on chronologically-ordered USD equity curve.
            # Previous impl summed per-trade pnl_pct (dimensionless) and
            # formatted as a percentage, which produced nonsensical
            # "-1270%" readings. Dollar drawdown is the honest view for a
            # paper portfolio without a fixed starting equity.
            chrono = sorted(
                [r for r in closed_trades if r["pnl_pct"] is not None],
                key=lambda r: (r.get("exit_date") or r.get("date") or "")
            )
            cum_usd, peak_usd, max_dd_usd = 0.0, 0.0, 0.0
            for r in chrono:
                ep = float(r["entry_price"]) if r["entry_price"] else 0.0
                pnl_u = float(r["pnl_pct"]) * ep * 100 if ep > 0 else 0.0
                cum_usd += pnl_u
                peak_usd = max(peak_usd, cum_usd)
                max_dd_usd = max(max_dd_usd, peak_usd - cum_usd)

            pf_str = f"{pf:.2f}x" if pf != float("inf") else "∞"
            analytics = (
                f"  Profit Factor: {pf_str}   Expectancy: {expectancy:+.1%}/trade"
                f"   Max Drawdown: -${max_dd_usd:,.0f}"
                f"   Avg Win: {avg_win:+.1%}  Avg Loss: {avg_loss:+.1%}"
            )
            if HAS_FMT and fmt:
                pf_color = fmt.Colors.GREEN if pf > 1.5 else (fmt.Colors.YELLOW if pf > 1.0 else fmt.Colors.RED)
                print(fmt.colorize(analytics, pf_color))
            else:
                print(analytics)

            # Per-strategy breakdown (only if > 1 strategy present)
            from collections import defaultdict
            strat_map: dict = defaultdict(list)
            for r in closed_trades:
                if r["pnl_pct"] is not None:
                    strat = (r["strategy_name"] or "OTHER").split(":")[0].strip()[:22]
                    strat_map[strat].append(float(r["pnl_pct"]))
            if len(strat_map) > 1:
                print()
                strat_hdr = "  BY STRATEGY"
                if HAS_FMT and fmt:
                    print(fmt.colorize(strat_hdr, fmt.Colors.BRIGHT_CYAN))
                else:
                    print(strat_hdr)
                for strat, rets in sorted(strat_map.items(), key=lambda x: -sum(x[1])):
                    sw = len([x for x in rets if x > 0])
                    sn = len(rets)
                    avg = sum(rets) / sn
                    spf_val = sum(x for x in rets if x > 0)
                    spl_val = abs(sum(x for x in rets if x <= 0))
                    spf = f"{spf_val/spl_val:.2f}x" if spl_val > 0 else "∞"
                    line = f"    {strat:<24} {sw}/{sn} wins  avg {avg:+.1%}  PF {spf}"
                    if HAS_FMT and fmt:
                        lc = fmt.Colors.GREEN if avg > 0 else fmt.Colors.RED
                        print(fmt.colorize(line, lc))
                    else:
                        print(line)

        # ── Strategy Breakdown (from DB query) ────────────────────────
        try:
            from .paper_manager import PaperManager
            pm = PaperManager(db_path=DB_PATH)
            breakdown = pm.get_strategy_breakdown()
            if breakdown:
                print()
                strat_db_hdr = "  STRATEGY BREAKDOWN"
                if HAS_FMT and fmt:
                    print(fmt.colorize(strat_db_hdr, fmt.Colors.BRIGHT_CYAN, bold=True))
                    print(fmt.colorize(sep, fmt.Colors.DIM))
                else:
                    print(strat_db_hdr)
                    print(sep)
                col_hdr = f"    {'Strategy':<24} {'Trades':>6}  {'Win%':>5}  {'Avg P&L':>9}  {'Total P&L':>10}"
                if HAS_FMT and fmt:
                    print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
                else:
                    print(col_hdr)
                for row in breakdown:
                    wr = row["win_rate"] * 100
                    avg = row["avg_pnl"] * 100
                    tot = row["total_pnl"] * 100
                    strat_name = (row["strategy"] or "Unknown").split(":")[0].strip()[:24]
                    line = f"    {strat_name:<24} {row['total']:>6}  {wr:>4.0f}%  {avg:>+8.1f}%  {tot:>+9.1f}%"
                    if HAS_FMT and fmt:
                        lc = fmt.Colors.GREEN if row["avg_pnl"] > 0 else fmt.Colors.RED
                        print(fmt.colorize(line, lc))
                    else:
                        print(line)
        except Exception:
            pass

        # ── P&L Attribution (delta/gamma/theta/vega) ─────────────────────
        # Only for closed trades that have stored entry Greeks
        _attr_trades = [
            r for r in closed_trades
            if r["pnl_pct"] is not None
            and _has_entry_greeks(r)
        ]
        if _attr_trades and HAS_BS:
            _attr_tickers = list({r["ticker"] for r in _attr_trades})
            _attr_prices: dict = {}
            for _t in _attr_tickers:
                try:
                    _p = getattr(yf.Ticker(_t).fast_info, "last_price", None)
                    if _p and float(_p) > 0:
                        _attr_prices[_t] = float(_p)
                except Exception:
                    pass
            _print_pnl_attribution(_attr_trades, _attr_prices, width)

        # Paper trade IC analysis
        if HAS_STRESS and len(closed_trades) >= 5:
            try:
                print_paper_trade_ic(DB_PATH, width=width)
            except Exception:
                pass

    # ── Roll Alerts ────────────────────────────────────────────────────────────
    roll_candidates = []
    today = date.today()
    for r in open_trades:
        try:
            exp_date = datetime.strptime(r["expiration"][:10], "%Y-%m-%d").date()
            dte = (exp_date - today).days
            pnl_pct_val = float(r["pnl_pct"]) if r["pnl_pct"] is not None else 0.0
            if 0 < dte <= 21 and pnl_pct_val > 0.25:
                roll_candidates.append((r["ticker"], r["type"], float(r["strike"]), exp_date, dte, pnl_pct_val))
        except Exception:
            continue

    if roll_candidates:
        print()
        roll_header = "  ROLL ALERTS \u2014 Consider rolling these positions:"
        if HAS_FMT and fmt:
            print(fmt.colorize(roll_header, fmt.Colors.YELLOW, bold=True))
        else:
            print(roll_header)
        for ticker, opt_type, strike, exp, dte, pnl in roll_candidates:
            line = f"  \u2192 {ticker} {str(opt_type).upper()} ${strike} exp {exp} | DTE: {dte}d | P/L: {pnl:.0%} \u2014 consider rolling"
            if HAS_FMT and fmt:
                print(fmt.colorize(line, fmt.Colors.YELLOW))
            else:
                print(line)

    # ── Footer ─────────────────────────────────────────────────────────────────
    print()
    note = "  Live prices may be unavailable outside market hours or for expired contracts."
    if HAS_FMT and fmt:
        print(fmt.draw_separator(width))
        print(fmt.colorize(note, fmt.Colors.DIM))
    else:
        print("-" * width)
        print(note)
    print()


if __name__ == "__main__":
    view_portfolio()
