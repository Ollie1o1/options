#!/usr/bin/env python3
"""
Portfolio P&L Stress Test using full Black-Scholes repricing.

Applies full BS repricing across stock move and IV shock scenarios.
Uses stored entry IV from paper trades DB when available (falls back to 25%).
Delta-gamma approximation is kept as a fallback if BS repricing fails.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

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
    from .utils import bs_delta, bs_gamma, bs_vega, bs_price
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

try:
    from .data_fetching import get_risk_free_rate as _get_rfr
    HAS_RFR = True
except ImportError:
    HAS_RFR = False


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

    def _fetch_one(ticker: str):
        import time as _time
        for attempt in range(3):
            try:
                tkr = yf.Ticker(ticker)
                p = getattr(tkr.fast_info, "last_price", None)
                if p and float(p) > 0:
                    return ticker, float(p)
            except Exception:
                if attempt < 2:
                    _time.sleep(0.5 * (attempt + 1))
        return ticker, None

    with ThreadPoolExecutor(max_workers=min(len(set(tickers)), 8)) as executor:
        for ticker, price in executor.map(_fetch_one, set(tickers)):
            if price is not None:
                prices[ticker] = price
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

    from datetime import datetime

    if stock_prices is None:
        tickers = [r["ticker"] for r in open_trades]
        stock_prices = _fetch_stock_prices(tickers)

    rfr = _get_rfr() if HAS_RFR else 0.045
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

            T = max(dte_days / 365.0, 1.0 / (365 * 24))  # floor at 1 hour, not 1 day

            # Use stored entry IV if available; fall back to 25%
            sigma = 0.25
            stored_iv = trade.get("entry_iv")
            if stored_iv is not None:
                try:
                    sv = float(stored_iv)
                    if 0.01 < sv < 5.0:
                        sigma = sv
                except (ValueError, TypeError):
                    pass

            div_yield = 0.0
            stored_q = trade.get("dividend_yield")
            if stored_q is not None:
                try:
                    qv = float(stored_q)
                    if 0.0 <= qv < 0.20:
                        div_yield = qv
                except (ValueError, TypeError):
                    pass

            sign = -1.0 if _is_short_position(strategy_name) else 1.0

            # For spreads: compute Greeks for both legs (short + long)
            if strategy_name.startswith("SPREAD:"):
                try:
                    parts = strategy_name.split(":")
                    long_strike = float(parts[1])
                    # Short leg (strike = K, sign = -1)
                    d_short = float(bs_delta(opt_type, S, K, T, rfr, sigma, div_yield))
                    g_short = float(bs_gamma(S, K, T, rfr, sigma, div_yield))
                    v_short = float(bs_vega(S, K, T, rfr, sigma, div_yield))
                    # Long leg (strike = long_strike, sign = +1)
                    d_long = float(bs_delta(opt_type, S, long_strike, T, rfr, sigma, div_yield))
                    g_long = float(bs_gamma(S, long_strike, T, rfr, sigma, div_yield))
                    v_long = float(bs_vega(S, long_strike, T, rfr, sigma, div_yield))
                    # Net Greeks = short leg * -1 + long leg * +1
                    delta = -d_short + d_long
                    gamma = -g_short + g_long
                    vega = -v_short + v_long
                    sign = 1.0  # Net sign already baked into the combined Greeks
                except Exception:
                    # Fallback to single-leg
                    delta = float(bs_delta(opt_type, S, K, T, rfr, sigma, div_yield))
                    gamma = float(bs_gamma(S, K, T, rfr, sigma, div_yield))
                    vega = float(bs_vega(S, K, T, rfr, sigma, div_yield))
            else:
                delta = float(bs_delta(opt_type, S, K, T, rfr, sigma, div_yield))
                gamma = float(bs_gamma(S, K, T, rfr, sigma, div_yield))
                vega = float(bs_vega(S, K, T, rfr, sigma, div_yield))

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
                "sigma": sigma,
                "div_yield": div_yield,
                "T": T,
                "is_spread": strategy_name.startswith("SPREAD:"),
                "strategy_name": strategy_name,
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
                    sign = pos["sign"]
                    opt_type = pos["type"]
                    K = pos["strike"]
                    sigma = pos.get("sigma", 0.25)
                    T = pos.get("T", 30.0 / 365)
                    q = pos.get("div_yield", 0.0)
                    rfr_val = _get_rfr() if HAS_RFR else 0.045

                    S_new = S * (1.0 + dS_pct)
                    IV_new = max(sigma + dIV, 0.01)

                    # Full BS repricing (accurate for large moves)
                    try:
                        is_spread = pos.get("is_spread", False)
                        strategy_name = pos.get("strategy_name", "")

                        if is_spread and strategy_name.startswith("SPREAD:"):
                            # Reprice both legs of the spread
                            parts = strategy_name.split(":")
                            long_strike = float(parts[1])
                            # Short leg: sell at K
                            short_base = float(np.float64(bs_price(opt_type, S, K, T, rfr_val, sigma, q)))
                            short_new = float(np.float64(bs_price(opt_type, S_new, K, T, rfr_val, IV_new, q)))
                            # Long leg: buy at long_strike
                            long_base = float(np.float64(bs_price(opt_type, S, long_strike, T, rfr_val, sigma, q)))
                            long_new = float(np.float64(bs_price(opt_type, S_new, long_strike, T, rfr_val, IV_new, q)))
                            if not all(np.isfinite(v) for v in [short_base, short_new, long_base, long_new]):
                                raise ValueError("NaN/Inf from BS spread")
                            # Spread P&L = (short_new - short_base)*-1 + (long_new - long_base)*+1
                            pnl = (-(short_new - short_base) + (long_new - long_base)) * 100
                        else:
                            price_base = float(np.float64(bs_price(opt_type, S, K, T, rfr_val, sigma, q)))
                            price_new = float(np.float64(bs_price(opt_type, S_new, K, T, rfr_val, IV_new, q)))
                            if not np.isfinite(price_base) or not np.isfinite(price_new):
                                raise ValueError("NaN/Inf from BS")
                            pnl = sign * (price_new - price_base) * 100
                    except Exception:
                        # Delta-gamma fallback
                        dS = S * dS_pct
                        pnl_delta = sign * pos["delta"] * dS * 100
                        pnl_gamma = sign * 0.5 * pos["gamma"] * (dS ** 2) * 100
                        dIV_pp = dIV * 100
                        pnl_vega = sign * pos["vega"] * dIV_pp * 100
                        pnl = pnl_delta + pnl_gamma + pnl_vega

                    total_pnl += pnl
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

    Uses full Black-Scholes repricing with stored entry IV when available.
    """
    n_pos = len(open_trades)
    if n_pos == 0:
        return

    df = run_stress_test(open_trades, stock_prices)
    if df is None or df.empty:
        print("\n  Stress test unavailable (missing dependencies or positions).")
        return

    print()
    header = f"  PORTFOLIO STRESS TEST  \u2014  {n_pos} open position(s)  [full BS repricing]"
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

    # Correlation risk warnings
    print_correlation_warnings(open_trades, width)

    note = "  [Full Black-Scholes repricing per scenario. Uses stored entry IV when available.]"
    if HAS_FMT and fmt:
        print(fmt.colorize(note, fmt.Colors.DIM))
    else:
        print(note)
    print()


def print_correlation_warnings(open_trades: list, width: int = 90) -> None:
    """Flag concentration risk when portfolio tickers are highly correlated."""
    if not HAS_NP or not HAS_PD or not HAS_YF:
        return

    tickers = list(set(t["ticker"] for t in open_trades if t.get("ticker")))
    if len(tickers) < 2:
        return

    try:
        import yfinance as yf
        # Fetch 30 trading days of close prices
        data = yf.download(tickers, period="2mo", progress=False, auto_adjust=True)
        if data.empty:
            return
        closes = data["Close"] if "Close" in data.columns else data
        if isinstance(closes, pd.Series):
            return  # single ticker after dedup
        returns = closes.pct_change().dropna()
        if len(returns) < 15:
            return
        corr = returns.corr()

        # Find high-correlation pairs
        pairs = []
        seen = set()
        for i, t1 in enumerate(corr.columns):
            for j, t2 in enumerate(corr.columns):
                if i >= j:
                    continue
                key = tuple(sorted([t1, t2]))
                if key in seen:
                    continue
                seen.add(key)
                rho = corr.loc[t1, t2]
                if np.isfinite(rho) and abs(rho) > 0.70:
                    pairs.append((t1, t2, rho))

        if not pairs:
            return

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        pair_strs = [f"{t1}/{t2} (\u03C1={rho:.2f})" for t1, t2, rho in pairs[:4]]
        line = f"  Concentration risk: {', '.join(pair_strs)}"
        if HAS_FMT and fmt:
            print(fmt.colorize(line, fmt.Colors.YELLOW, bold=True))
        else:
            print(line)
    except Exception:
        pass


__all__ = [
    "compute_position_greeks",
    "run_stress_test",
    "print_stress_test",
    "print_correlation_warnings",
    "STOCK_MOVES",
    "IV_SHOCKS",
]
