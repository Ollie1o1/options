#!/usr/bin/env python3
"""
Portfolio viewer — reads open and closed paper trades from paper_trades.db
and displays a clean P/L summary with live price fetching.
"""

import os
import sqlite3
import shutil
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
