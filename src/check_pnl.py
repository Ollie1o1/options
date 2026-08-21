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
from src.paths import repo_path

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
        from . import ui
    except (ImportError, ValueError):
        import formatting as fmt  # type: ignore[no-redef]
        import ui  # type: ignore[no-redef]
    HAS_FMT = fmt.supports_color()
except Exception:
    HAS_FMT = False
    # Sentinels, not modules: every use is guarded by HAS_FMT / _HAS_UI_CP.
    fmt = None  # type: ignore[assignment]
    ui = None  # type: ignore[assignment]

# `ui` is always bound above (to None on failure), so test the value, not the name.
_HAS_UI_CP = ui is not None


class _NoProgress:
    """Fallback bar for the no-`ui` path, so callers never branch on None."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def _progress(total: int, desc: str):
    """A progress bar, degrading to a no-op when the UI layer is unavailable.

    `ui` is None whenever its import failed, so calling ui.progress_bar directly
    would turn a cosmetic missing-dependency case into an AttributeError in the
    middle of pricing the book.
    """
    if not _HAS_UI_CP:
        return _NoProgress()
    return ui.progress_bar(total, desc)

try:
    try:
        from .utils import is_short_position as _is_short
        from .utils import bs_delta, bs_gamma, bs_vega, bs_theta, american_price
    except (ImportError, ValueError):
        from utils import is_short_position as _is_short  # type: ignore[no-redef]
        from utils import (  # type: ignore[no-redef]
            bs_delta, bs_gamma, bs_vega, bs_theta, american_price)
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
        from stress_test import (  # type: ignore[no-redef]
            print_stress_test, _classify_structure)
        from backtester import print_paper_trade_ic  # type: ignore[no-redef]
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
        from data_fetching import get_risk_free_rate as _get_rfr  # type: ignore[no-redef]
    HAS_RFR = True
except Exception:
    HAS_RFR = False

DB_PATH = repo_path("paper_trades.db")


def _num_or_none(value: Any) -> Optional[float]:
    """A ledger column as a usable float, or None when there is no figure.

    `max_loss_usd` and `long_strike` arrive as float, int, str, None or empty
    string depending on the row's age and which writer produced it. Three sites
    in `view_portfolio` each carried the same coercion inline; this is that
    expression, once.

    **Zero counts as absent.** A defined-risk structure recording a max loss of
    0 has a missing figure, not a riskless trade, and the callers fall back to
    cost basis rather than reporting nothing at risk.

    NaN is rejected too: it parses cleanly and then poisons every sum it
    reaches, which would quietly NaN the whole concentration total.
    """
    if value is None or value == "" or value == 0:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN is the only value unequal to itself


def _get_multiplier(ticker: str) -> float:
    """Return the contract multiplier: 1.0 for crypto, 100.0 for stocks."""
    if (ticker or "").upper() in ("BTC", "ETH"):
        return 1.0
    return 100.0


def _lots(quantity: Any) -> float:
    """Contracts held, defaulting to one.

    Every row written before 2026-08-19 carries 1.0 because `log_trade` never
    wrote the column; position sizing (src/book_sizing.py) writes 2 and 3. Each
    dollar figure on this screen is `per-contract x multiplier`, so without this
    a multi-lot position would be displayed at a fraction of what it is doing —
    P&L, cost basis, concentration, portfolio max loss and the Greeks alike.
    A NULL, zero or unparseable quantity is a row that was never sized, not an
    empty position, so it falls back to one rather than to zero.
    """
    try:
        qty = float(quantity)
    except (TypeError, ValueError):
        return 1.0
    return qty if (qty == qty and 0 < qty < float("inf")) else 1.0


def _row_lots(row) -> float:
    """`_lots` for a row mapping, tolerating a missing column."""
    try:
        return _lots(row["quantity"])
    except (KeyError, IndexError, TypeError):
        return 1.0


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


def _price_open_legs(open_trades, pm=None, progress=None) -> dict:
    """Live mark per leg, keyed (ticker, expiration, strike, opt_type).

    ONE option-chain request per (ticker, expiration), not one lookup per leg.

    This priced every leg with its own `yf.Ticker(occ)` call. On the live book
    that is 119 open positions, 87 of them four-legged iron condors — about 381
    round trips, eight at a time. Measured 2026-08-17: over eleven minutes wall
    clock against 10.9 SECONDS of CPU, i.e. essentially all network wait, and
    it had to be killed before it finished.

    Those 381 legs sit on only 41 distinct (ticker, expiration) pairs; QQQ
    2026-09-18 alone carries 15 positions. `PaperManager._fetch_chain_quotes`
    already serves every leg on a pair from one request and memoises it for
    60s, which is the same fix the scan path got when the GEX chain path was
    taking ~90% of every scan.

    A leg the chain cannot answer is left ABSENT rather than guessed: the
    caller has traded-price rungs to fall back on, and a fabricated mark can
    fire an exit.
    """
    from concurrent.futures import ThreadPoolExecutor

    legs: list = []
    for r in open_trades:
        for opt_type, strike, _qty in _legs_for_row(r):
            legs.append((r["ticker"], str(r["expiration"])[:10], strike, opt_type))
    legs = list(dict.fromkeys(legs))
    if not legs:
        return {}

    if pm is None:
        try:
            from .paper_manager import PaperManager
        except ImportError:
            from paper_manager import PaperManager  # type: ignore[no-redef]
        pm = PaperManager(db_path=DB_PATH, config_path="config.json")

    pairs = list(dict.fromkeys((t, e) for t, e, _s, _o in legs))

    def _one_chain(pair):
        try:
            return pair, pm._fetch_chain_quotes(pair[0], pair[1])
        except Exception:
            # One dead pair must not abort the whole book.
            return pair, {}

    chains: dict = {}
    bar = progress(len(pairs), "Pricing open positions") if progress else None
    try:
        with ThreadPoolExecutor(max_workers=min(len(pairs), 8)) as ex:
            for pair, quotes in ex.map(_one_chain, pairs):
                chains[pair] = quotes
                if bar:
                    bar.update(1)
    finally:
        if bar:
            bar.close()

    out: dict = {}
    for ticker, exp, strike, opt_type in legs:
        quote = chains.get((ticker, exp), {}).get((float(strike), opt_type))
        mark = None
        # Defensive unpack: a chain helper that returns something other than
        # {(strike, type): (bid, ask)} — a stub, a mock, a malformed row —
        # must leave the leg unpriced rather than abort marking the book.
        try:
            bid, ask = quote
            if bid is not None and ask is not None and float(ask) > 0:
                mark = (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            mark = None
        out[(ticker, exp, strike, opt_type)] = mark
    return out


def _legs_for_row(r) -> list:
    """Return list of (opt_type, strike, qty_sign) for the given DB row.

    qty_sign: +1 = long leg, -1 = short leg. Used to compute net position
    value as sum(qty * leg_price) so credit spreads / iron condors mark to
    market correctly. Returns [] for unrecognized or malformed rows.
    """
    def _f(v):
        try:
            if v in (None, "", 0):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    structure = _classify_structure(r)
    if structure == "iron_condor":
        sp = _f(r.get("short_put_strike"))
        lp = _f(r.get("long_put_strike"))
        sc = _f(r.get("short_call_strike"))
        lc = _f(r.get("long_call_strike"))
        if None in (sp, lp, sc, lc):
            # Legacy iron-condor row classified by strategy_name but missing
            # the dedicated leg columns — skip valuation rather than crash.
            return []
        return [
            ("put",  sp, -1),
            ("put",  lp, +1),
            ("call", sc, -1),
            ("call", lc, +1),
        ]
    if structure == "spread":
        opt_type = str(r.get("type", "") or "").lower()
        if not opt_type:
            opt_type = "put" if "bull put" in str(r.get("strategy_name", "")).lower() else "call"
        long_strike = _f(r.get("long_strike"))
        if long_strike is None:
            # Legacy SPREAD:long:width:max_loss fallback
            try:
                long_strike = float(str(r.get("strategy_name", "")).split(":")[1])
            except (ValueError, IndexError):
                return []
        short_strike = _f(r.get("strike"))
        if short_strike is None:
            return []
        return [
            (opt_type, short_strike, -1),
            (opt_type, long_strike,  +1),
        ]
    # Single leg
    sign = -1 if _is(_get_strategy_name(r)) else 1
    strike = _f(r.get("strike"))
    if strike is None:
        return []
    return [(str(r.get("type", "")).lower(), strike, sign)]


def _ror_cell(value: Optional[float]) -> str:
    """Format return-on-capital-at-risk for the breakdown table.

    NULL capital_at_risk is normal on rows logged before the column existed, so
    this reads n/a rather than 0% — a strategy with no risk data has no return
    on risk, not a flat one.
    """
    if value is None:
        return f"{'n/a':>7}"
    return f"{value * 100:>+6.1f}%"


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

            # We don't have S_entry/S_exit stored, so use pnl_ratio * entry_price * mult as actual P&L
            # Contracts fold into the multiplier so the realised total and the
            # Greek decomposition it is compared against scale together.
            mult = _get_multiplier(r.get("ticker", "")) * _row_lots(r)
            actual_pnl = pnl_ratio * entry_price * mult

            # Estimate days held
            try:
                trade_date = datetime.strptime(str(r["date"])[:10], "%Y-%m-%d").date()
                exit_dt = datetime.strptime(str(r["exit_date"])[:10], "%Y-%m-%d").date()
                days_held = max((exit_dt - trade_date).days, 1)
            except Exception:
                days_held = 14  # fallback

            # Theta P&L (daily theta * days held * multiplier)
            # For short positions, theta P&L has opposite sign
            sign_mult = -1.0 if is_short else 1.0
            theta_pnl = sign_mult * entry_theta * days_held * mult

            # For delta/gamma/vega, we'd need spot price change.
            # Approximate: actual - theta = delta+gamma+vega+residual
            # Attribute remaining proportionally to Greeks magnitude
            remaining = actual_pnl - theta_pnl

            # Without spot price data, attribute remaining based on Greek magnitudes
            # Scale gamma to comparable units: gamma * S * typical_5pct_move
            # (gamma is per-share per $1 move)
            S_approx = float(stock_prices.get(r.get("ticker", ""), 100.0))
            abs_d = abs(entry_delta)
            abs_g = 0.5 * abs(entry_gamma) * (S_approx * 0.05) ** 2 * mult  # quadratic: 0.5 * gamma * (ΔS)^2 * mult
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
        print(ui.rule(width, title=f"P&L ATTRIBUTION \u2014 {counted} closed trades with entry Greeks"))
    else:
        print(header)
        print(sep)

    total_abs = abs(total_delta_pnl) + abs(total_gamma_pnl) + abs(total_theta_pnl) + abs(total_vega_pnl)
    if total_abs == 0:
        total_abs = 1.0

    def _attr_line(name, val):
        pct = val / total_abs * 100 if total_abs > 0 else 0
        sign = "+" if val >= 0 else "-"
        if HAS_FMT and fmt:
            val_str = fmt.style(f"{sign}${abs(val):>8.0f}  ({pct:>+5.0f}%)",
                                'good' if val >= 0 else 'bad')
            return ui.kv_line(name.rstrip(':'), val_str)
        return f"    {name:<8} {sign}${abs(val):>8.0f}  ({pct:>+5.0f}%)"

    components = [
        ("Delta", total_delta_pnl),
        ("Gamma", total_gamma_pnl),
        ("Theta", total_theta_pnl),
        ("Vega", total_vega_pnl),
    ]

    residual = total_actual_pnl - (total_delta_pnl + total_gamma_pnl + total_theta_pnl + total_vega_pnl)
    if abs(residual) > 1.0:
        components.append(("Other", residual))

    rows = ui.waterfall(components, bar_w=min(30, max(12, width - 40))) \
        if (HAS_FMT and fmt and _HAS_UI_CP) else []
    if rows:
        for ln in rows:
            print(ln)
        # Theta is measured (daily theta x days held); delta/gamma/vega are NOT.
        # Entry/exit spot is not stored, so `actual - theta` is split across them
        # in proportion to Greek magnitude. Say so rather than implying a
        # measured decomposition.
        print(fmt.style(
            "  Theta is measured; Δ/Γ/Vega split the remainder by "
            "Greek magnitude (entry/exit spot not stored).", 'muted'))
    else:
        for _name, _val in components:
            print(_attr_line(_name + ":", _val))

    if HAS_FMT and fmt:
        print(ui.rule(width))
    else:
        print(sep)


def _print_equity_curve(closed_trades: list, width: int, min_trades: int = 10):
    """Inline braille cumulative realized-P&L curve + underwater drawdown strip.

    Display-only. Mirrors the chronological USD equity series used for max-DD,
    so the curve and the analytics line can never disagree.
    """
    chrono = sorted(
        [r for r in closed_trades if r.get("pnl_pct") is not None],
        key=lambda r: (r.get("exit_date") or r.get("date") or "")
    )
    if len(chrono) < min_trades:
        return
    cum = 0.0
    peak = 0.0
    equity = []
    depth = []
    for r in chrono:
        ep = float(r["entry_price"]) if r.get("entry_price") else 0.0
        mult = _get_multiplier(r.get("ticker", "")) * _row_lots(r)
        pnl_u = float(r["pnl_pct"]) * ep * mult if ep > 0 else 0.0
        cum += pnl_u
        peak = max(peak, cum)
        equity.append(cum)
        # Depth below the running peak, as a positive magnitude.
        depth.append(peak - cum)
    max_dd = max(depth) if depth else 0.0

    print()
    title = f"EQUITY CURVE — cumulative realized P&L ({len(chrono)} trades)"
    if HAS_FMT and fmt and _HAS_UI_CP:
        chart_width = min(72, width - 6)
        print(ui.rule(width, title=title))
        for ln in ui.braille_chart(equity, width=chart_width, height=5,
                                   style_name=('good' if cum >= 0 else 'bad')):
            print("  " + ln)
        end_style = 'good' if cum >= 0 else 'bad'
        print(ui.kv_line("Final", fmt.style(f"${cum:+,.0f}", end_style)
                         + fmt.style(f"   peak ${peak:+,.0f}", 'muted')))
        # Drawdown drawn as its own mini equity curve, negated so it droops
        # below a flat "at peak" surface instead of a bar-height-means-badness
        # sparkline — same visual language as the curve above, and correctly
        # width-capped (braille_chart downsamples; a bare sparkline doesn't
        # and would overflow the terminal one character per trade).
        print()
        print(ui.kv_line("Drawdown", fmt.style("below peak equity", 'muted')))
        for ln in ui.braille_chart([-d for d in depth], width=chart_width, height=2,
                                   style_name='bad'):
            print("  " + ln)
        print(ui.kv_line("Max DD", fmt.style(f"-${max_dd:,.0f}", 'bad')))
    else:
        print(f"  {title}")
        print(f"  Final: ${cum:+,.0f}   peak ${peak:+,.0f}   max DD -${max_dd:,.0f}")


def _print_greeks_by_ticker(by_ticker: dict, width: int, top: int = 8):
    """Per-underlying net vega as sorted, sign-colored bars, with net delta.

    Display-only. `by_ticker` maps ticker -> [vega, delta]. Silent when fewer
    than two names carry vega (the aggregate line already says it all).
    """
    items = [(t, v[0], v[1]) for t, v in by_ticker.items()]
    items = [x for x in items if abs(x[1]) > 1e-9]
    if len(items) < 2:
        return
    items.sort(key=lambda x: -abs(x[1]))
    shown = items[:top]
    rest = items[top:]
    max_abs = max(abs(v) for _, v, _ in shown) or 1.0
    bar_w = 14

    print()
    if HAS_FMT and fmt and _HAS_UI_CP:
        print(ui.rule(width, title="VEGA BY UNDERLYING — $/1% IV  (long=+ short=−)"))
    else:
        print("  Vega by underlying ($/1% IV):")
    for tkr, vega, delta in shown:
        fill = int(round(abs(vega) / max_abs * bar_w))
        bar = "█" * fill + " " * (bar_w - fill)
        if HAS_FMT and fmt and _HAS_UI_CP:
            bar = ui.heat_cell(bar, vega, max_abs, glyph=False)
        print(f"  {tkr:<6} {bar} {vega:>+8.0f}   Δ {delta:>+6.2f}")
    if rest:
        other_v = sum(v for _, v, _ in rest)
        other_d = sum(d for _, _, d in rest)
        print(f"  {'others':<6} {' ' * bar_w} {other_v:>+8.0f}   Δ {other_d:>+6.2f}")


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
    by_ticker: dict = {}  # ticker -> [net_vega, net_delta]

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
            # Portfolio Greeks are exposure, so they count contracts: two lots
            # of the same spread carry twice the delta, vega and theta. `leg_qty`
            # here is the leg's SIGN within one structure, not the position size.
            lots = _row_lots(r)
            for leg_type, leg_strike, leg_qty in legs:
                d = float(bs_delta(leg_type, S, leg_strike, T, rfr, sigma))
                g = float(bs_gamma(S, leg_strike, T, rfr, sigma))
                v = float(bs_vega(S, leg_strike, T, rfr, sigma))
                t = float(bs_theta(leg_type, S, leg_strike, T, rfr, sigma))
                row_delta += leg_qty * d * lots
                row_gamma += leg_qty * g * lots
                row_vega  += leg_qty * v * lots
                row_theta += leg_qty * t * lots
            mult = _get_multiplier(ticker)
            net_delta        += row_delta
            net_gamma_dollar += 0.5 * row_gamma * (S * 0.01) ** 2 * mult
            net_vega         += row_vega * mult
            net_theta        += row_theta * mult
            slot = by_ticker.setdefault(ticker, [0.0, 0.0])
            slot[0] += row_vega * mult
            slot[1] += row_delta
            counted += 1
        except Exception as _greeks_exc:
            logging.getLogger(__name__).debug("Greeks computation failed for position: %s", _greeks_exc)

    if counted == 0:
        return

    # ── Display ────────────────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(ui.rule(width, title="PORTFOLIO GREEKS \u2014 entry IV when stored | long=+ short=\u2212"))
    else:
        print("  PORTFOLIO GREEKS  (uses entry IV when stored  |  long=+  short=\u2212)")
        print("  " + "-" * (width - 2))

    # Delta
    delta_style = 'good' if net_delta > 0.10 else ('bad' if net_delta < -0.10 else 'warn')
    if HAS_FMT and fmt:
        print(ui.kv_line("Net \u0394", fmt.style(f"{net_delta:+.2f}", delta_style)))
    else:
        print(f"  Net \u0394: {net_delta:+.2f}")

    # Gamma ($ per 1% stock move)
    gd = net_gamma_dollar
    if HAS_FMT and fmt:
        print(ui.kv_line("Net \u0393", fmt.style(f"{gd:+.2f}", 'good' if gd > 0 else 'bad') + "  ($/1% move)"))
    else:
        print(f"  Net \u0393 ($/1% move): {gd:+.2f}")

    # Vega ($ per 1% IV rise)
    if HAS_FMT and fmt:
        print(ui.kv_line("Net Vega", fmt.style(f"{net_vega:+.2f}", 'good' if net_vega > 0 else 'bad') + "  ($/1% IV)"))
    else:
        print(f"  Net Vega ($/1% IV): {net_vega:+.2f}")

    # Theta ($/day)
    if HAS_FMT and fmt:
        print(ui.kv_line("Net \u0398", fmt.style(f"{net_theta:+.2f}", 'good' if net_theta > 0 else 'bad') + "  ($/day)"))
    else:
        print(f"  Net \u0398 ($/day): {net_theta:+.2f}")

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
            print(fmt.style(warn_line, 'warn', bold=True))
        else:
            print(warn_line)

    _print_greeks_by_ticker(by_ticker, width)

    note = f"  [{counted}/{len(open_trades)} positions, entry IV when available]"
    if HAS_FMT and fmt:
        print(fmt.style(note, 'muted'))
    else:
        print(note)


# ── Book split (current vs closed history) ────────────────────────────────────
# The book was restarted on BOOK_RESTART_DATE. Everything closed before it was
# scored under superseded models and is kept as history only, which is why the
# viewer no longer offers the old pre/post-calibration and pre-data/finalized
# splits — they carved up trades nobody reads any more.
#
# The date is frozen on purpose. A rolling date.today() would empty the current
# book every morning and quietly reclassify yesterday's work as history.
#
# CURRENT also carries every still-OPEN position regardless of entry date: 83 of
# the 93 open positions predate the restart, and routing those into history would
# hide live exposure from the one view where the book actually gets looked at.
# The `era` column is untouched by all this — src/portfolio_eras.py still reports
# the old-vs-new process split, which is separate evidence.
BOOK_RESTART_DATE = "2026-08-05"


def _entry_date(r) -> str:
    """Entry date as YYYY-MM-DD. The column is a fixed-width 10-char date string
    with no nulls, so a plain lexicographic compare is chronological. A blank
    date sorts below any real one, i.e. as history — never silently current."""
    return str(r.get("date") or "")[:10]


def _period_for_row(r) -> str:
    if str(r.get("status") or "").upper() == "OPEN":
        return "current"
    return "current" if _entry_date(r) >= BOOK_RESTART_DATE else "before"


def _filter_by_period(rows: list, period: Optional[str]) -> list:
    """period: 'current' = restart date onward, plus every open position;
    'before' = closed trades from before the restart; None = the whole book."""
    if period not in ("current", "before"):
        return rows
    return [r for r in rows if _period_for_row(r) == period]


def resolve_period(choice: Optional[str]) -> Optional[str]:
    """Which slice the viewer shows, given what the caller asked for.

    **No choice means the CURRENT book**, not the whole ledger. The 866 closed
    trades from before the restart are almost all strategies since switched off
    — Long Call, Iron Condor, Bear Call, Long Put — picked by a ranker measured
    at OOS IC -0.12 and scored on EV estimates later found to be the short
    leg's. Opening on them buries the handful of trades the operator is
    actually running.

    Nothing is deleted, and that is deliberate: those rows are what establish
    Bull Put's profit factor (131 of its 134 closes predate the restart), the
    50.9% required win rate over 415 credit trades, and the pre-registered
    ranker test frozen to 2026-11-19. Evidence, not daily reading.
    ``all`` asks for the whole book.
    """
    if choice == "all":
        return None
    if choice in ("current", "before"):
        return choice
    return "current"


def view_portfolio_menu() -> None:
    """Interactive entry point — prompt for which slice of the book, then render."""
    print()
    print("  PORTFOLIO VIEW — choose what to show:")
    print(f"    [C] Current book (default — logged {BOOK_RESTART_DATE} onward, plus every open position)")
    print(f"    [B] Before {BOOK_RESTART_DATE} (closed history, superseded scoring)")
    print("    [A] All trades")
    try:
        choice = input("  Choice [C/B/A]: ").strip().upper() or "C"
    except (EOFError, KeyboardInterrupt):
        choice = "C"
        print()
    period = {"C": "current", "B": "before", "A": "all"}.get(choice, "current")
    view_portfolio(period=resolve_period(period))

    # Auto-refresh the SVG chart. The calibration-cohort views are gone from this
    # menu, so the chart always renders the whole equity book.
    try:
        import subprocess
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_pnl_chart.py"
        args = [sys.executable, str(script), "--equity-only", "--cohort", "all"]
        print()
        print("  Refreshing equity curve chart...")
        subprocess.run(args, check=False)
    except Exception as e:
        print(f"  (chart refresh skipped: {e})")


def view_portfolio(period: Optional[str] = None):
    """Display paper portfolio from paper_trades.db.

    Args:
        period: None for the whole book, 'current' for trades logged
            BOOK_RESTART_DATE onward plus every still-open position,
            'before' for closed trades from before the restart.
    """
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
            from paper_manager import PaperManager  # type: ignore[no-redef]
        print("  Enforcing exit rules...", end="", flush=True)
        PaperManager(db_path=DB_PATH, config_path="config.json").update_positions()
        print("\r" + " " * 30 + "\r", end="")
    except Exception as _e:
        print(f"\r  (exit enforcement skipped: {_e})")

    # Cache exit-rule thresholds once so the per-row milestone hint matches the
    # rule that update_positions() actually enforces. Previously the hint was
    # hardcoded to 50%, which lied about long options (real TP is 100%).
    _exit_cfg: Dict[str, Any] = {}
    try:
        import json as _json
        with open("config.json") as _f:
            _exit_cfg = (_json.load(_f) or {}).get("exit_rules", {}) or {}
    except Exception:
        pass

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
    all_rows = _filter_by_period(all_rows, period)
    period_label = {"current": f"CURRENT BOOK ({BOOK_RESTART_DATE} ONWARD)",
                    "before": f"BEFORE {BOOK_RESTART_DATE}"}.get(period or "", "")
    header_text = f"PAPER PORTFOLIO  \u2014  {now_str}"
    if period_label:
        header_text += f"  [{period_label}]"

    if HAS_FMT and fmt:
        print(ui.banner(header_text, [], width))
    else:
        print("=" * width)
        print(f"  {header_text}")
        print("=" * width)

    # Staleness guard: warn here too, since the portfolio viewer is where the
    # book actually gets looked at. Silent when maintenance is fresh.
    try:
        from src.maintenance import load_state, DEFAULT_STATE_PATH
        from src.maintenance_health import (compute_health, health_banner,
                                            launchd_silence_days, read_launchd_status)
        _hb = health_banner(compute_health(load_state(DEFAULT_STATE_PATH), datetime.now()),
                            launchd_jobs=read_launchd_status(),
                            silence_days=launchd_silence_days())
        if _hb:
            print(_hb)
    except Exception:
        pass

    if not all_rows:
        msg = ("\n  No trades match this filter.\n" if period
               else "\n  No trades logged yet.\n")
        print(msg)
        return

    open_trades  = [r for r in all_rows if r["status"] == "OPEN"]
    closed_trades = [r for r in all_rows if r["status"] == "CLOSED"]

    # ── Open Positions ─────────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(ui.rule(width, title="OPEN POSITIONS"))
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
            print(fmt.style(hdr, 'label', bold=True))
            print(fmt.style(sep, 'muted'))
        else:
            print(hdr)
            print("  " + "-" * (width - 2))

        total_pnl_usd   = 0.0
        total_cost_usd  = 0.0
        fetched_count   = 0

        # Parallel-fetch all live option prices up front. For multi-leg
        # structures (spreads / iron condors) we need ALL legs' marks so the
        # P&L row shows the true net cost-to-close rather than just one leg.
        # One option-chain request per (ticker, expiration) — see
        # `_price_open_legs`. Was one `yf.Ticker(occ)` per LEG: ~381 round
        # trips on the live book for 41 distinct chains, which ran for over
        # eleven minutes on 10.9s of CPU.
        _live_prices = _price_open_legs(open_trades, progress=_progress)

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
                sp = float(r.get("short_put_strike") or 0)
                sc = float(r.get("short_call_strike") or 0)
                sp_str = f"{sp:.0f}" if sp > 0 else "—"
                sc_str = f"{sc:.0f}" if sc > 0 else "—"
                strike_disp = f"{sp_str}/{sc_str}"
            elif structure == "spread":
                sn_low = (r.get("strategy_name") or "").lower()
                opt_type_disp = "BPS" if "bull put" in sn_low else ("BCS" if "bear call" in sn_low else "SPR")
                long_k = _num_or_none(r.get("long_strike"))
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
            # Require at least one leg AND every leg priced — `all([])` is vacuously
            # True, which would mark legacy multi-leg rows with missing strike columns
            # as fully captured (pnl == credit). Guard against that.
            all_legs_priced = bool(leg_marks) and all(lp is not None and lp > 0 for _, lp in leg_marks)

            mult = _get_multiplier(ticker)
            # Contracts held. Every row written before 2026-08-19 carries 1.0
            # because the column was never written; from position sizing on it
            # can be 2 or 3, and an unscaled mark would show the operator a
            # fraction of what the position is actually doing.
            lots = _row_lots(r)
            if structure == "single":
                live_price = leg_marks[0][1] if leg_marks else None
                live_str = f"${live_price:.2f}" if (live_price is not None and live_price > 0) else None
                if live_price is not None and live_price > 0:
                    pnl_per = (entry_price - live_price) if short else (live_price - entry_price)
                    pnl_usd_row = pnl_per * mult * lots
                    pnl_pct_row = pnl_per / entry_price * 100 if entry_price > 0 else 0.0
                    cost_basis = entry_price * mult * lots
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
                    pnl_usd_row = pnl_per * mult * lots
                    pnl_pct_row = pnl_per / entry_price * 100 if entry_price > 0 else 0.0
                    live_str = f"${current_credit:.2f}"
                    # Cost basis ≈ max_loss (true defined risk) for concentration
                    # math. `capital_at_risk` is stored at the sized quantity and
                    # `max_loss_usd` per contract, so prefer the former and scale
                    # the latter — concentration is about the whole position.
                    _car = _num_or_none(r.get("capital_at_risk"))
                    _ml = _num_or_none(r.get("max_loss_usd"))
                    if _car is not None and _car > 0:
                        cost_basis = abs(_car)
                    elif _ml is not None:
                        cost_basis = abs(_ml) * lots
                    else:
                        cost_basis = entry_price * mult * lots
                else:
                    pnl_usd_row = None
                    pnl_pct_row = None
                    live_str = None
                    cost_basis = 0.0

            if pnl_usd_row is not None and pnl_pct_row is not None:
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

            # Take-profit milestone alert. Threshold mirrors the rule that
            # paper_manager.update_positions() enforces, so the advisory matches
            # what auto-close will actually do on the next pass.
            if live_price is not None and live_price > 0:
                if short:
                    # Short premium: DTE-aware TP ladder
                    sp_cfg = _exit_cfg.get("short_premium") or {}
                    try:
                        _dte_now = _dte(expiry)
                    except Exception:
                        _dte_now = 30
                    if _dte_now >= 21:
                        tp_threshold = float(sp_cfg.get("take_profit_ge_21_dte", 0.50))
                    elif _dte_now >= 7:
                        tp_threshold = float(sp_cfg.get("take_profit_7_to_21_dte", 0.35))
                    else:
                        tp_threshold = float(sp_cfg.get("take_profit_lt_7_dte", 0.25))
                    profit_pct = (entry_price - live_price) / entry_price if entry_price > 0 else 0.0
                else:
                    # Long option: flat take-profit (config default 1.0 = 100%)
                    tp_threshold = float((_exit_cfg.get("long_option") or {}).get("take_profit", 1.0))
                    profit_pct = (live_price - entry_price) / entry_price if entry_price > 0 else 0.0

                if profit_pct >= tp_threshold:
                    milestone_line = (
                        f"    ✓ {profit_pct:.0%} profit — at take-profit "
                        f"({tp_threshold:.0%}); next exit-rule pass will auto-close"
                    )
                    if HAS_FMT and fmt:
                        print(fmt.colorize(milestone_line, fmt.Colors.GREEN, bold=True))
                    else:
                        print(milestone_line)
                elif profit_pct >= 0.5 * tp_threshold and profit_pct >= 0.50:
                    # Halfway hint — useful for long options where TP=100% means
                    # +50% is still a notable milestone worth surfacing, just not
                    # actionable per the auto-close rule.
                    halfway_line = (
                        f"    · {profit_pct:.0%} profit — past halfway to TP "
                        f"({tp_threshold:.0%}); not yet at auto-close threshold"
                    )
                    if HAS_FMT and fmt:
                        print(fmt.colorize(halfway_line, fmt.Colors.DIM))
                    else:
                        print(halfway_line)

        # Open totals
        if HAS_FMT and fmt:
            print(fmt.style(sep, 'muted'))
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
                mult = _get_multiplier(t) * _row_lots(r)
                ticker_exp[t] = ticker_exp.get(t, 0.0) + float(r["entry_price"]) * mult
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
            # max_loss_usd is stored per contract; the portfolio's worst case is
            # what the whole position can lose.
            lots = _row_lots(r)
            if structure in ("spread", "iron_condor"):
                _ml = _num_or_none(r.get("max_loss_usd"))
                ml_val = abs(_ml) * lots if _ml is not None else None
                if ml_val is None and sn.startswith("SPREAD:"):
                    # Legacy fallback parsing
                    try:
                        parts = sn.split(":")
                        ml_val = (abs(float(parts[3])) * 100 * lots
                                  if len(parts) >= 4 else None)
                    except (ValueError, IndexError):
                        ml_val = None
                if ml_val is None:
                    has_undefined_risk = True
                else:
                    total_max_loss += ml_val
            else:
                # Single-leg: max loss = entry_price * mult (for longs) or unlimited (shorts)
                ep = abs(float(r.get("entry_price", 0)))
                mult = _get_multiplier(r.get("ticker", "")) * lots
                if _is_short(sn):
                    has_undefined_risk = True
                else:
                    total_max_loss += ep * mult

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

        # Lottery sleeve scorecard (silent when the sleeve is empty)
        try:
            from src.lottery.sleeve import print_lottery_sleeve
            print_lottery_sleeve(db_path=DB_PATH, width=width)
        except Exception:
            pass

        # Stress test — only meaningful with 3+ positions
        if HAS_STRESS and len(open_trades) >= 3:
            try:
                print_stress_test(open_trades, width=width)
            except Exception:
                pass

    # ── Closed Positions ───────────────────────────────────────────────────────
    print()
    if HAS_FMT and fmt:
        print(ui.rule(width, title="CLOSED POSITIONS"))
    else:
        print("  CLOSED POSITIONS")
    print()

    if not closed_trades:
        print(_c("  No closed trades yet.", fmt.Colors.DIM if HAS_FMT and fmt else ""))
    else:
        hdr = (
            f"  {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Expiry':<12}"
            f" {'Opened':<11} {'Closed':<11} {'Entry $':>8} {'Exit $':>8}"
            f" {'P/L $':>10} {'P/L %':>7} {'Peak':>6} {'Result'}"
        )
        sep = "  " + "\u2500" * (width - 2)

        if HAS_FMT and fmt:
            print(fmt.style(hdr, 'label', bold=True))
            print(fmt.style(sep, 'muted'))
        else:
            print(hdr)
            print("  " + "-" * (width - 2))

        closed_pnl_usd = 0.0
        wins = 0
        missed_2x = 0     # peaked ≥2× entry while tracked but exited below 2×
        peaks_seen = 0

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
            # Contracts fold in here so this reconstruction agrees with the
            # stored pnl_usd, which paper_manager writes at the sized quantity.
            mult = _get_multiplier(ticker) * _row_lots(r)
            if r["pnl_pct"] is not None:
                pnl_pct = pnl_ratio * 100
                pnl_usd = pnl_ratio * entry_price * mult
            elif exit_price > 0 and entry_price > 0:
                is_short_closed = _is_short(str(r.get("strategy_name", "")))
                if is_short_closed:
                    pnl_usd = (entry_price - exit_price) * mult
                else:
                    pnl_usd = (exit_price - entry_price) * mult
                pnl_pct = pnl_usd / (entry_price * mult) * 100
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

            # High-water mark: highest premium sampled while the position was
            # tracked (schema v15; NULL on rows closed before it existed).
            try:
                peak_seen = r["max_price_seen"] if "max_price_seen" in r.keys() else None
            except Exception:
                peak_seen = None
            if peak_seen is not None and entry_price > 0:
                peak_mult = float(peak_seen) / entry_price
                raw_peak = f"{peak_mult:.1f}x"
                peaks_seen += 1
                # a missed multiple is a LONG concept: the premium ran ≥2× while
                # held but the exit banked less (for shorts a spike is a stop)
                if (peak_mult >= 2.0 and exit_price < 2.0 * entry_price
                        and not _is_short(str(r["strategy_name"] or "")
                                          if "strategy_name" in r.keys() else "")):
                    missed_2x += 1
            else:
                raw_peak = "—"

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

            if HAS_FMT and fmt:
                # peak ≥2× is the "look what you let go" signal — warn ink
                peak_str = (fmt.style(f"{raw_peak:>6}", 'warn')
                            if raw_peak not in ("—",) and peak_seen is not None
                            and float(peak_seen) >= 2.0 * entry_price
                            else fmt.style(f"{raw_peak:>6}", 'muted'))
            else:
                peak_str = f"{raw_peak:>6}"

            print(
                f"  {ticker:<7} {type_str} {strike:>8.2f} {expiry:<12}"
                f" {opened:<11} {closed_date:<11} ${entry_price:>6.2f} ${exit_price:>6.2f}"
                f" {usd_str} {pct_str} {peak_str}  {result_str}"
            )

        if HAS_FMT and fmt:
            print(fmt.style(sep, 'muted'))
        else:
            print("  " + "-" * (width - 2))

        n = len(closed_trades)
        win_rate_pct = wins / n * 100 if n > 0 else 0.0
        sign = "+" if closed_pnl_usd >= 0 else "-"
        closed_summary = (
            f"  Realized P/L: {sign}${abs(closed_pnl_usd):.2f}"
            f"   Win Rate: {win_rate_pct:.0f}% ({wins}/{n} trades)"
        )
        if peaks_seen:
            closed_summary += (
                f"   Peaks tracked: {peaks_seen}"
                f" (missed ≥2×: {missed_2x})"
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
                mult = _get_multiplier(r.get("ticker", "")) * _row_lots(r)
                pnl_u = float(r["pnl_pct"]) * ep * mult if ep > 0 else 0.0
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

            _print_equity_curve(closed_trades, width)

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
                    n_trades = len(rets)
                    avg = sum(rets) / n_trades
                    spf_val = sum(x for x in rets if x > 0)
                    spl_val = abs(sum(x for x in rets if x <= 0))
                    spf = f"{spf_val/spl_val:.2f}x" if spl_val > 0 else "∞"
                    line = f"    {strat:<24} {sw}/{n_trades} wins  avg {avg:+.1%}  PF {spf}"
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
                    print(fmt.style(sep, 'muted'))
                else:
                    print(strat_db_hdr)
                    print(sep)
                col_hdr = (f"    {'Strategy':<24} {'Trades':>6}  {'Win%':>5}  "
                           f"{'Avg P&L':>9}  {'Total P&L':>10}  {'On Risk':>7}")
                if HAS_FMT and fmt:
                    print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
                else:
                    print(col_hdr)
                for row in breakdown:
                    wr = row["win_rate"] * 100
                    avg = row["avg_pnl"] * 100
                    tot = row["total_pnl"] * 100
                    strat_name = (row["strategy"] or "Unknown").split(":")[0].strip()[:24]
                    line = (f"    {strat_name:<24} {row['total']:>6}  {wr:>4.0f}%  "
                            f"{avg:>+8.1f}%  {tot:>+9.1f}%  "
                            f"{_ror_cell(row.get('return_on_risk'))}")
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
    import argparse

    _p = argparse.ArgumentParser(description="Paper portfolio viewer")
    _p.add_argument("--period", choices=["current", "before", "all"], default=None,
                    help=f"Filter the book: 'current' (DEFAULT) = logged "
                         f"{BOOK_RESTART_DATE} onward plus every open position, "
                         "'before' = closed history, 'all' = the whole ledger")
    _p.add_argument("--all", dest="show_all", action="store_true",
                    help="Show the whole ledger including pre-restart history")
    _p.add_argument("--current", dest="current", action="store_true",
                    help=f"Show the current book ({BOOK_RESTART_DATE} onward + open positions)")
    _p.add_argument("--before", dest="before", action="store_true",
                    help=f"Show only closed trades from before {BOOK_RESTART_DATE}")
    _p.add_argument("--menu", action="store_true",
                    help="Open the interactive view chooser instead")
    _args = _p.parse_args()

    if _args.menu:
        view_portfolio_menu()
    else:
        _period = _args.period
        if _args.show_all:
            _period = "all"
        elif _args.current:
            _period = "current"
        elif _args.before:
            _period = "before"
        view_portfolio(period=resolve_period(_period))
