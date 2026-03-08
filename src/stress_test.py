#!/usr/bin/env python3
"""
Portfolio Greeks-based P&L Stress Test.

Applies first + second order Greeks approximation (delta-gamma approximation) to
estimate P&L across stock move and IV shock scenarios.

Note: This uses an approximation, not full Black-Scholes repricing.
      Results are directionally indicative but will diverge from full repricing
      for large moves (>20%) where gamma convexity terms are significant.
      Does not model term structure changes or smile dynamics.
"""

import math
from typing import Optional, List, Dict, Any

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from . import formatting as fmt
    from .formatting import Colors, BoxChars, supports_color, colorize
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None

try:
    from .utils import bs_delta, bs_gamma, bs_vega
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


# Stock move and IV shock scenario axes
STOCK_MOVES = [-0.20, -0.10, -0.05, 0.00, +0.05, +0.10, +0.20]
IV_SHOCKS = [0.0, 0.10, 0.20]


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _sep(width: int = 90) -> str:
    line = "  " + "\u2500" * (width - 2)
    if HAS_FMT and fmt:
        return fmt.colorize(line, fmt.Colors.DIM)
    return line


def _is_short_position(strategy_name: str) -> bool:
    """Return True if the strategy is a short/credit position."""
    s = (strategy_name or "").lower()
    return any(k in s for k in ("short", "credit", "covered", "cash secured", "naked"))


def _fetch_stock_prices(tickers: List[str]) -> Dict[str, float]:
    """Batch-fetch current stock prices via yfinance fast_info."""
    prices: Dict[str, float] = {}
    if not HAS_YF:
        return prices
    import warnings
    for ticker in set(tickers):
        try:
            tkr = yf.Ticker(ticker)
            p = getattr(tkr.fast_info, "last_price", None)
            if p and float(p) > 0:
                prices[ticker] = float(p)
        except Exception:
            pass
    return prices


def compute_position_greeks(open_trades: list, stock_prices: Optional[Dict[str, float]] = None) -> list:
    """
    Compute per-position Greeks for each open trade.

    For each trade:
    - S = stock_prices[ticker] (fetched if not provided)
    - K = strike, T = DTE/365, sigma = 0.25 (fallback IV)
    - Compute delta, gamma, vega using Black-Scholes functions from utils.
    - sign = -1 if short position, +1 if long.

    Returns list of dicts: {ticker, type, strike, expiry, S, delta, gamma, vega, entry_price, sign}
    """
    if not HAS_UTILS:
        return []

    from datetime import datetime, date as date_type

    if stock_prices is None:
        tickers = [r["ticker"] for r in open_trades]
        stock_prices = _fetch_stock_prices(tickers)

    rfr = 0.045
    now_dt = datetime.now()
    result = []

    for trade in open_trades:
        try:
            ticker = trade["ticker"]
            S = stock_prices.get(ticker)
            if S is None or S <= 0:
                continue
            K = float(trade["strike"])
            opt_type = str(trade.get("type", "call")).lower()
            entry_price = float(trade.get("entry_price", 0.0))
            strategy_name = str(trade.get("strategy_name", ""))
            expiry_str = str(trade.get("expiration", ""))

            try:
                exp_dt = datetime.strptime(expiry_str[:10], "%Y-%m-%d")
                dte_days = max((exp_dt - now_dt).days, 1)
            except Exception:
                dte_days = 30  # fallback

            T = max(dte_days / 365.0, 1.0 / 365)
            sigma = 0.25  # fallback IV

            sign = -1.0 if _is_short_position(strategy_name) else 1.0

            delta = float(bs_delta(opt_type, S, K, T, rfr, sigma))
            gamma = float(bs_gamma(S, K, T, rfr, sigma))
            vega = float(bs_vega(S, K, T, rfr, sigma))  # per 1% IV move per share

            result.append({
                "ticker": ticker,
                "type": opt_type,
                "strike": K,
                "expiry": expiry_str[:10],
                "dte": dte_days,
                "S": S,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "entry_price": entry_price,
                "sign": sign,
            })
        except Exception:
            continue

    return result


def run_stress_test(
    open_trades: list,
    stock_prices: Optional[Dict[str, float]] = None,
) -> Optional["pd.DataFrame"]:
    """
    Apply a grid of stock-move x IV-shock scenarios to the open position book.

    Stock move scenarios: [-20%, -10%, -5%, 0%, +5%, +10%, +20%]
    IV shock scenarios:   [0%, +10%, +20% in absolute IV points]

    For each position and scenario, P&L approximation (first + second order):
        dS = S * dS_pct
        delta_pnl = sign * delta * dS * 100  (per contract, 100 shares)
        gamma_pnl = sign * 0.5 * gamma * dS^2 * 100
        vega_pnl  = sign * vega * dIV_pct_absolute * 100 * 100
                    (vega is per 1% IV per share, so * 100 shares * dIV in percent points)

    Note: vega convention — bs_vega returns value per 1% IV change per share.
    For dIV = 0.10 (10% absolute IV rise = 10 percentage points):
        vega_dollar = sign * vega_per_share * 10_pp * 100_shares

    Aggregates across all positions.
    Returns DataFrame with columns: [stock_move, iv_shock, total_pnl_usd, pnl_pct_of_book, n_positions]
    """
    if not HAS_PD:
        return None

    position_greeks = compute_position_greeks(open_trades, stock_prices)
    if not position_greeks:
        return None

    # Total book value (entry cost basis)
    book_value = sum(abs(p["entry_price"]) * 100 for p in position_greeks)
    if book_value <= 0:
        book_value = 1.0  # avoid div/zero

    rows = []
    for dS_pct in STOCK_MOVES:
        for dIV in IV_SHOCKS:
            total_pnl = 0.0
            counted = 0
            for pos in position_greeks:
                try:
                    S = pos["S"]
                    delta = pos["delta"]
                    gamma = pos["gamma"]
                    vega = pos["vega"]
                    sign = pos["sign"]

                    dS = S * dS_pct

                    # Delta + Gamma P&L (per share × 100 shares per contract)
                    pnl_delta = sign * delta * dS * 100
                    pnl_gamma = sign * 0.5 * gamma * (dS ** 2) * 100

                    # Vega P&L: vega is per 1% IV move per share
                    # dIV is absolute fraction (0.10 = 10 percentage points = 10 one-percent moves)
                    dIV_pp = dIV * 100  # convert fraction to number of 1% moves
                    pnl_vega = sign * vega * dIV_pp * 100  # * 100 shares per contract

                    total_pnl += pnl_delta + pnl_gamma + pnl_vega
                    counted += 1
                except Exception:
                    continue

            rows.append({
                "stock_move": dS_pct,
                "iv_shock": dIV,
                "total_pnl_usd": total_pnl,
                "pnl_pct_of_book": total_pnl / book_value if book_value > 0 else 0.0,
                "n_positions": counted,
            })

    return pd.DataFrame(rows)


def print_stress_test(
    open_trades: list,
    stock_prices: Optional[Dict[str, float]] = None,
    width: int = 90,
) -> None:
    """
    Print a scenario matrix of portfolio P&L under stock-move x IV-shock scenarios.

    Approximation note: Uses delta-gamma first+second order Greeks approximation.
    Results are directionally indicative; large moves will deviate from full repricing.
    """
    n_pos = len(open_trades)
    if n_pos == 0:
        return

    df = run_stress_test(open_trades, stock_prices)
    if df is None or df.empty:
        print("\n  Stress test unavailable (missing dependencies or positions).")
        return

    print()
    header = f"  PORTFOLIO STRESS TEST  \u2014  {n_pos} open position(s)  [delta-gamma approx — not full reprice]"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    # Build pivot: rows = IV shocks, cols = stock moves
    iv_shocks = sorted(df["iv_shock"].unique())
    stock_moves = sorted(df["stock_move"].unique())

    # Book value for coloring (pct_of_book threshold)
    # Color: RED if loss >10% of book, YELLOW if 0-10% loss, GREEN if profit
    def _cell_color(pnl_pct: float) -> str:
        if not HAS_FMT or not fmt:
            return ""
        if pnl_pct < -0.10:
            return fmt.Colors.RED
        elif pnl_pct < 0.0:
            return fmt.Colors.YELLOW
        else:
            return fmt.Colors.GREEN

    # Column header
    move_labels = [f"{int(m*100):+d}%" for m in stock_moves]
    col_header = f"  {'IV Shock':<12}  " + "  ".join(f"{lbl:>8}" for lbl in move_labels)
    if HAS_FMT and fmt:
        print(fmt.colorize(col_header, fmt.Colors.BOLD))
    else:
        print(col_header)

    min_pnl = float("inf")
    min_scenario = None
    max_pnl = float("-inf")

    for iv in iv_shocks:
        iv_label = f"IV +{int(iv*100)}%" if iv > 0 else "IV flat "
        row_parts = [f"  {iv_label:<12}"]
        for sm in stock_moves:
            subset = df[(df["stock_move"] == sm) & (df["iv_shock"] == iv)]
            if subset.empty:
                row_parts.append(f"  {'—':>8}")
                continue
            pnl_usd = float(subset["total_pnl_usd"].iloc[0])
            pnl_pct = float(subset["pnl_pct_of_book"].iloc[0])

            if pnl_usd < min_pnl:
                min_pnl = pnl_usd
                min_scenario = (sm, iv)
            if pnl_usd > max_pnl:
                max_pnl = pnl_usd

            # Format cell
            if abs(pnl_usd) >= 1000:
                cell = f"{pnl_usd:>+8,.0f}"
            else:
                cell = f"{pnl_usd:>+8.0f}"

            color = _cell_color(pnl_pct)
            if HAS_FMT and fmt and color:
                row_parts.append("  " + fmt.colorize(f"{cell:>8}", color))
            else:
                row_parts.append(f"  {cell:>8}")

        print("".join(row_parts))

    print(_sep(width))

    # Max loss scenario
    if min_scenario is not None:
        sm_pct = int(min_scenario[0] * 100)
        iv_pct = int(min_scenario[1] * 100)
        # pnl_pct_of_book for min scenario
        min_subset = df[(df["stock_move"] == min_scenario[0]) & (df["iv_shock"] == min_scenario[1])]
        if not min_subset.empty:
            min_book_pct = float(min_subset["pnl_pct_of_book"].iloc[0]) * 100
            max_loss_line = (
                f"  Max loss scenario: {sm_pct:+d}% stock / +{iv_pct}% IV"
                f" = ${min_pnl:,.0f} ({min_book_pct:+.1f}% of book)"
            )
            if HAS_FMT and fmt:
                print(fmt.colorize(max_loss_line, fmt.Colors.RED, bold=True))
            else:
                print(max_loss_line)

    # Break-even stock move (at flat IV)
    flat_iv_df = df[df["iv_shock"] == 0.0].sort_values("stock_move")
    if not flat_iv_df.empty:
        # Find move where pnl changes sign
        pnls = list(zip(flat_iv_df["stock_move"], flat_iv_df["total_pnl_usd"]))
        breakeven_move = None
        for i in range(len(pnls) - 1):
            sm1, p1 = pnls[i]
            sm2, p2 = pnls[i + 1]
            if (p1 <= 0 <= p2) or (p2 <= 0 <= p1):
                if abs(p2 - p1) > 0:
                    frac = -p1 / (p2 - p1)
                    breakeven_move = sm1 + frac * (sm2 - sm1)
                break
        if breakeven_move is not None:
            be_line = f"  Break-even requires: stock move of {breakeven_move*100:+.1f}% or better (flat IV)"
            if HAS_FMT and fmt:
                print(fmt.colorize(be_line, fmt.Colors.YELLOW))
            else:
                print(be_line)

    note = "  [Approximation: delta + 0.5*gamma*dS^2 + vega*dIV. Not a substitute for full repricing.]"
    if HAS_FMT and fmt:
        print(fmt.colorize(note, fmt.Colors.DIM))
    else:
        print(note)
    print()


__all__ = [
    "compute_position_greeks",
    "run_stress_test",
    "print_stress_test",
    "STOCK_MOVES",
    "IV_SHOCKS",
]
