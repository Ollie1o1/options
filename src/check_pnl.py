#!/usr/bin/env python3
"""
Portfolio viewer — reads open and closed paper trades from paper_trades.db
and displays a clean P/L summary with live price fetching.
"""

import os
import sqlite3
import shutil
import pandas as pd
from datetime import datetime, date
from typing import Optional

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from . import formatting as fmt
    HAS_FMT = fmt.supports_color()
except ImportError:
    HAS_FMT = False
    fmt = None

try:
    from .utils import bs_delta, bs_gamma, bs_vega
    HAS_BS = True
except ImportError:
    HAS_BS = False

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


def _fetch_live_price(ticker: str, expiration: str, strike: float, opt_type: str) -> Optional[float]:
    """Fetch live option price via OCC symbol. Returns None on failure."""
    if not HAS_YF:
        return None
    try:
        from pandas import to_datetime
        exp = to_datetime(expiration)
        date_str = exp.strftime("%y%m%d")
        otype = "C" if opt_type.lower() == "call" else "P"
        strike_str = f"{int(float(strike) * 1000):08d}"
        occ = f"{ticker.upper()}{date_str}{otype}{strike_str}"

        tkr = yf.Ticker(occ)

        try:
            price = getattr(tkr.fast_info, "last_price", None)
            if price and float(price) > 0:
                return float(price)
        except Exception:
            pass

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = tkr.history(period="1d")
        if not hist.empty:
            p = float(hist["Close"].iloc[-1])
            if p > 0:
                return p
    except Exception:
        pass
    return None


def _is_short(strategy_name: str) -> bool:
    s = (strategy_name or "").lower()
    return any(k in s for k in ("short", "credit", "covered", "cash secured", "naked"))


def _print_portfolio_greeks(open_trades: list, width: int):
    """
    Compute and display aggregate portfolio Greeks for open positions.
    Uses current stock prices from yfinance and a 25% IV estimate as fallback.
    Signs: long calls = +Δ, short calls = -Δ, long puts = -Δ, short puts = +Δ.
    """
    if not open_trades or not HAS_BS or not HAS_YF:
        return

    # Batch-fetch unique stock prices (fast, 1 call per ticker)
    unique_tickers = list({r["ticker"] for r in open_trades})
    stock_prices: dict = {}
    import warnings
    for ticker in unique_tickers:
        try:
            tkr = yf.Ticker(ticker)
            p = getattr(tkr.fast_info, "last_price", None)
            if p and float(p) > 0:
                stock_prices[ticker] = float(p)
        except Exception:
            pass

    rfr = 0.045  # ~current risk-free rate
    now_dt = datetime.now()
    net_delta = 0.0
    net_gamma_dollar = 0.0   # $ P&L per 1% stock move across the book
    net_vega = 0.0           # $ P&L per 1% IV rise across the book
    counted = 0

    for r in open_trades:
        ticker = r["ticker"]
        S = stock_prices.get(ticker)
        if S is None:
            continue
        K = float(r["strike"])
        opt_type = r["type"].lower()
        try:
            exp_dt = datetime.strptime(r["expiration"][:10], "%Y-%m-%d")
            T = max((exp_dt - now_dt).total_seconds() / (365.25 * 24 * 3600), 1.0 / 365)
        except Exception:
            continue

        sigma = 0.25  # fallback IV (25%)
        is_short = _is_short(r["strategy_name"] or "")
        sign = -1.0 if is_short else 1.0

        try:
            delta = float(bs_delta(opt_type, S, K, T, rfr, sigma))
            gamma = float(bs_gamma(S, K, T, rfr, sigma))
            vega  = float(bs_vega(S, K, T, rfr, sigma))   # already per 1% IV per share
            net_delta       += sign * delta
            # dollar gamma: $ gain per 1% stock move = 0.5 × gamma × (S×0.01)² × 100 shares
            net_gamma_dollar += sign * 0.5 * gamma * (S * 0.01) ** 2 * 100
            net_vega        += sign * vega * 100   # per contract (×100 shares)
            counted += 1
        except Exception:
            pass

    if counted == 0:
        return

    # ── Display ────────────────────────────────────────────────────────────────
    print()
    sep = "  " + "\u2500" * (width - 2)
    header = "  PORTFOLIO GREEKS  (IV est. 25%  |  long=+  short=−)"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
        print(fmt.colorize(sep, fmt.Colors.DIM))
    else:
        print(header)
        print("  " + "-" * (width - 2))

    # Delta
    delta_color = (fmt.Colors.GREEN if net_delta > 0.10 else
                   fmt.Colors.RED if net_delta < -0.10 else fmt.Colors.YELLOW) if HAS_FMT and fmt else ""
    delta_label = f"  Net Δ (directional): {net_delta:+.2f} contracts"
    delta_hint  = ("  → book is net LONG (profits if market rises)" if net_delta > 0.10 else
                   "  → book is net SHORT (profits if market falls)" if net_delta < -0.10 else
                   "  → roughly delta-neutral")
    if HAS_FMT and fmt:
        print(fmt.colorize(delta_label, delta_color))
        print(fmt.colorize(delta_hint, fmt.Colors.DIM))
    else:
        print(delta_label)
        print(delta_hint)

    # Gamma ($ per 1% stock move)
    gd = net_gamma_dollar
    gcolor = (fmt.Colors.GREEN if gd > 0 else fmt.Colors.RED) if HAS_FMT and fmt else ""
    gamma_label = f"  Net Γ ($ per 1% stock move): {gd:+.2f}"
    gamma_hint  = ("  → long gamma: profits accelerate with big moves" if gd > 0 else
                   "  → short gamma: losses accelerate with big moves — watch carefully")
    if HAS_FMT and fmt:
        print(fmt.colorize(gamma_label, gcolor))
        print(fmt.colorize(gamma_hint, fmt.Colors.DIM))
    else:
        print(gamma_label)
        print(gamma_hint)

    # Vega ($ per 1% IV rise)
    vc = (fmt.Colors.GREEN if net_vega > 0 else fmt.Colors.RED) if HAS_FMT and fmt else ""
    vega_label = f"  Net Vega ($ per 1% IV rise): {net_vega:+.2f}"
    vega_hint  = ("  → long vega: profits if IV expands" if net_vega > 0 else
                  "  → short vega: profits if IV contracts (premium selling)")
    if HAS_FMT and fmt:
        print(fmt.colorize(vega_label, vc))
        print(fmt.colorize(vega_hint, fmt.Colors.DIM))
    else:
        print(vega_label)
        print(vega_hint)

    note = f"  Computed from {counted}/{len(open_trades)} positions with live stock prices."
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

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        all_rows = conn.execute("SELECT * FROM trades ORDER BY date DESC").fetchall()
        conn.close()
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
        hdr = (
            f"  {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Expiry':<12}"
            f" {'DTE':>4} {'Opened':<11} {'Entry $':>8} {'Live $':>8}"
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

        for r in open_trades:
            ticker      = r["ticker"]
            opt_type    = r["type"].upper()
            strike      = float(r["strike"])
            expiry      = r["expiration"][:10]
            opened      = r["date"][:10]
            entry_price = float(r["entry_price"])
            dte         = _dte(expiry)
            short       = _is_short(r["strategy_name"] or "")

            live_price = _fetch_live_price(ticker, expiry, strike, r["type"])

            # Type column
            if HAS_FMT and fmt:
                tc = fmt.Colors.BRIGHT_GREEN if opt_type == "CALL" else fmt.Colors.BRIGHT_RED
                type_str = fmt.colorize(f"{opt_type:<5}", tc)
            else:
                type_str = f"{opt_type:<5}"

            # DTE column
            if HAS_FMT and fmt:
                dc = fmt.Colors.BRIGHT_RED if dte < 7 else (fmt.Colors.YELLOW if dte < 14 else fmt.Colors.WHITE)
                dte_str = fmt.colorize(f"{max(dte,0):>4}", dc)
            else:
                dte_str = f"{max(dte,0):>4}"

            if live_price is not None and live_price > 0:
                pnl_per = (entry_price - live_price) if short else (live_price - entry_price)
                pnl_usd = pnl_per * 100
                pnl_pct = pnl_per / entry_price * 100 if entry_price > 0 else 0.0
                total_pnl_usd  += pnl_usd
                total_cost_usd += entry_price * 100
                fetched_count  += 1

                sign = "+" if pnl_usd >= 0 else ""
                raw_usd = f"{sign}${abs(pnl_usd):.2f}"
                raw_pct = f"{sign}{abs(pnl_pct):.1f}%"
                if pnl_usd < 0:
                    raw_usd = f"-${abs(pnl_usd):.2f}"
                    raw_pct = f"-{abs(pnl_pct):.1f}%"

                live_str = f"${live_price:.2f}"

                if HAS_FMT and fmt:
                    pc = fmt.Colors.GREEN if pnl_usd >= 0 else fmt.Colors.RED
                    usd_str = fmt.colorize(f"{raw_usd:>10}", pc)
                    pct_str = fmt.colorize(f"{raw_pct:>7}", pc)
                else:
                    usd_str = f"{raw_usd:>10}"
                    pct_str = f"{raw_pct:>7}"
            else:
                live_str = _c(f"{'—':>8}", fmt.Colors.DIM if HAS_FMT and fmt else "")
                usd_str  = _c(f"{'—':>10}", fmt.Colors.DIM if HAS_FMT and fmt else "")
                pct_str  = _c(f"{'—':>7}", fmt.Colors.DIM if HAS_FMT and fmt else "")

            print(
                f"  {ticker:<7} {type_str} {strike:>8.2f} {expiry:<12}"
                f" {dte_str} {opened:<11} ${entry_price:>6.2f} {live_str:>8}"
                f" {usd_str} {pct_str}"
            )

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

        _print_portfolio_greeks(open_trades, width)

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
            pnl_usd     = pnl_ratio * entry_price * 100
            pnl_pct     = pnl_ratio * 100
            won         = pnl_usd > 0
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
        win_rate = wins / n * 100 if n > 0 else 0.0
        sign = "+" if closed_pnl_usd >= 0 else "-"
        closed_summary = (
            f"  Realized P/L: {sign}${abs(closed_pnl_usd):.2f}"
            f"   Win Rate: {win_rate:.0f}% ({wins}/{n} trades)"
        )
        if HAS_FMT and fmt:
            pc = fmt.Colors.GREEN if closed_pnl_usd >= 0 else fmt.Colors.RED
            print(fmt.colorize(closed_summary, pc, bold=True))
        else:
            print(closed_summary)

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
