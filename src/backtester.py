#!/usr/bin/env python3
"""
Walk-Forward Options Backtester

Methodology:
- Uses up to 252 days of stock price history.
- For each day d (from day 30 to day 252-30), computes signals from price data only.
- Simulates entering a 0.30-delta OTM call or put using Black-Scholes with realized vol as
  sigma and 30 DTE. Direction is determined by 5-day momentum: positive = call, negative = put.
- Exits at day d+21 (time exit), TP=50% of premium, SL=-25% of premium (loss on option value).
- Computes Information Coefficient (IC) between entry signals and realized P&L.
- Groups results by score quintile to test whether higher score -> better outcome.

Important limitations:
- Uses Black-Scholes theoretical prices, not actual historical bid/ask data.
- Does not model liquidity, slippage, or bid-ask spreads.
- Realized vol (not IV) is used as the pricing sigma — IV premium not modelled.
- Past signal-return relationships do not guarantee future performance.
- Results should be used for qualitative signal validation, not as a trading system.
"""

import math
import pathlib
import sqlite3
import sys
from typing import Optional, List, Dict, Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(_PROJECT_ROOT / "paper_trades.db")

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
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from . import formatting as fmt
    from .formatting import Colors, supports_color
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None

try:
    from .utils import bs_call, bs_put, bs_delta, _d1d2
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


def _classify_vix_regime(vix_level: float) -> str:
    """Map a VIX level to a regime label."""
    if vix_level < 15:
        return "Low"
    if vix_level < 25:
        return "Normal"
    return "High"


def _calculate_mdd(series: "pd.Series") -> float:
    """Maximum drawdown of a cumulative P&L series."""
    cum = series.cumsum()
    return float((cum.cummax() - cum).max())


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _sep(width: int = 90) -> str:
    line = "  " + "\u2500" * (width - 2)
    if HAS_FMT and fmt:
        return fmt.colorize(line, fmt.Colors.DIM)
    return line


# ─── Signal computation helpers ────────────────────────────────────────────────

def compute_rsi(prices: "pd.Series", period: int = 14) -> "pd.Series":
    """
    Standard Wilder RSI computed via EWMA (exponential weighted moving average).
    Returns a Series aligned with the input prices.
    """
    if not HAS_PD or not HAS_NP:
        raise ImportError("pandas and numpy required for compute_rsi")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's EWMA: alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_hv(log_rets: "pd.Series", window: int = 30) -> "pd.Series":
    """Rolling annualised realized vol from log returns."""
    return log_rets.rolling(window).std() * math.sqrt(252)


def _bs_option_price(opt_type: str, S: float, K: float, T: float,
                     r: float, sigma: float) -> float:
    """Black-Scholes option price; returns 0.0 on any error."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        if opt_type == "call":
            return float(bs_call(S, K, T, r, sigma))
        else:
            return float(bs_put(S, K, T, r, sigma))
    except Exception:
        return 0.0


def _find_30d_otm_strike(opt_type: str, S: float, sigma: float, T: float,
                          r: float, target_delta: float = 0.30) -> float:
    """
    Find the strike that gives approximately target_delta for a call or put.
    Uses analytical Black-Scholes inversion approximation.
    For a call: delta ~ N(d1) = target_delta => d1 = N_inv(target_delta)
    For a put:  delta ~ N(d1) - 1 = -target_delta => N(d1) = 1 - target_delta
    Returns strike K.
    """
    try:
        from scipy.stats import norm as sp_norm
        if opt_type == "call":
            d1_target = float(sp_norm.ppf(target_delta))
        else:
            d1_target = float(sp_norm.ppf(1.0 - target_delta))
        # d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        # => log(S/K) = d1*sigma*sqrt(T) - (r + 0.5*sigma^2)*T
        log_SK = d1_target * sigma * math.sqrt(T) - (r + 0.5 * sigma ** 2) * T
        K = S * math.exp(-log_SK)
        return float(K)
    except Exception:
        # Fallback: approximate with OTM multiplier
        if opt_type == "call":
            return S * 1.05
        else:
            return S * 0.95


# ─── Main backtest ──────────────────────────────────────────────────────────────

def run_backtest(
    tickers: List[str],
    lookback_days: int = 252,
    tp: float = 0.50,
    sl: float = -0.25,
    dte: int = 30,
    exit_dte: int = 21,
    risk_free: float = 0.045,
) -> Dict[str, Any]:
    """
    Walk-forward backtest for a list of tickers.

    For each ticker:
    1. Fetch price history (~252 trading days).
    2. At each day d (from day 30 to day 252-exit_dte), compute a directional signal score.
    3. Simulate entering a ~0.30-delta OTM option (call if momentum positive, put if negative)
       using Black-Scholes with 30 DTE and rolling 30d realized vol as sigma.
    4. Exit at day d+exit_dte (time decay exit), with early exit at TP=50% or SL=-25%.
    5. Compute IC between score and realized P&L, grouped by quintile.

    Returns a list of result dicts (one per ticker) plus a combined summary.
    """
    if not HAS_NP or not HAS_PD or not HAS_YF or not HAS_UTILS:
        return {
            "error": "Missing dependencies (numpy, pandas, yfinance, or utils)",
            "results": [],
            "combined": {},
        }

    import json as _json
    import warnings
    # Load config for spread cost
    _config: dict = {}
    try:
        _cfg_path = str(_PROJECT_ROOT / "config.json")
        with open(_cfg_path) as _f:
            _config = _json.load(_f)
    except Exception:
        pass
    spread_cost_per_side = _config.get("backtest", {}).get("spread_cost_per_side", 0.07)

    all_ticker_results = []

    # Fetch VIX history once for regime tagging
    vix_hist: "Optional[pd.Series]" = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _vix_raw = yf.Ticker("^VIX").history(period="2y")["Close"]
        if not _vix_raw.empty:
            vix_hist = _vix_raw
    except Exception:
        pass

    # Normalise VIX once so per-ticker reindex needs no tz/dedup work
    vix_hist_normalized: "Optional[pd.Series]" = None
    if vix_hist is not None and HAS_PD:
        try:
            vix_n = vix_hist.copy()
            if hasattr(vix_n.index, "tz") and vix_n.index.tz is not None:
                vix_n.index = vix_n.index.tz_localize(None)
            vix_n.index = vix_n.index.normalize()
            vix_n = vix_n[~vix_n.index.duplicated(keep="last")]
            vix_hist_normalized = vix_n
        except Exception:
            pass

    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = tkr.history(period="2y")
            if hist.empty or len(hist) < lookback_days // 2:
                all_ticker_results.append({"ticker": ticker, "error": "Insufficient history"})
                continue

            close = hist["Close"].dropna()
            # Use only the last lookback_days
            if len(close) > lookback_days:
                close = close.iloc[-lookback_days:]

            N = len(close)
            if N < 60:
                all_ticker_results.append({"ticker": ticker, "error": "Insufficient history"})
                continue

            log_rets = np.log(close / close.shift(1)).dropna()
            hv_30 = _compute_hv(log_rets, window=30)
            rsi_14 = compute_rsi(close, period=14)

            # Rolling 252d percentile of hv_30 (for hv_rank)
            hv_rank = hv_30.rolling(min(252, len(hv_30))).rank(pct=True)

            # 5d return
            ret_5d = close.pct_change(5)

            # Align all series
            signals_df = pd.DataFrame({
                "close": close,
                "log_ret": log_rets,
                "hv_30": hv_30,
                "hv_rank": hv_rank,
                "rsi_14": rsi_14,
                "ret_5d": ret_5d,
            }).dropna()

            if len(signals_df) < 30:
                all_ticker_results.append({"ticker": ticker, "error": "Too few signal rows"})
                continue

            trades = []
            close_arr = signals_df["close"].values
            hv_arr = signals_df["hv_30"].values
            hv_rank_arr = signals_df["hv_rank"].values
            rsi_arr = signals_df["rsi_14"].values
            ret5_arr = signals_df["ret_5d"].values

            # Align pre-normalised VIX to signals_df dates for regime tagging.
            vix_aligned: "Optional[pd.Series]" = None
            if vix_hist_normalized is not None and HAS_PD:
                try:
                    sig_idx = signals_df.index
                    if hasattr(sig_idx, "tz") and sig_idx.tz is not None:
                        sig_idx = sig_idx.tz_localize(None)
                    sig_idx = sig_idx.normalize()
                    vix_aligned = vix_hist_normalized.reindex(sig_idx, method="ffill", limit=5)
                except Exception:
                    pass

            for i in range(len(signals_df) - exit_dte - 2):
                try:
                    S = float(close_arr[i])
                    hv = float(hv_arr[i])
                    hv_rank_i = float(hv_rank_arr[i]) if not math.isnan(float(hv_rank_arr[i])) else 0.5
                    rsi = float(rsi_arr[i]) if not math.isnan(float(rsi_arr[i])) else 50.0
                    ret5 = float(ret5_arr[i]) if not math.isnan(float(ret5_arr[i])) else 0.0

                    if hv <= 0 or S <= 0:
                        continue

                    # Direction
                    direction = "call" if ret5 >= 0 else "put"

                    # Signal score components
                    hv_daily = hv / math.sqrt(252)
                    momentum_raw = ret5 / (2 * max(hv_daily, 1e-6))
                    momentum_raw = max(-1.0, min(1.0, momentum_raw))
                    abs_momentum = abs(momentum_raw)

                    # hv_rank_score: prefer LOW current HV (cheap options)
                    hv_rank_score = 1.0 - hv_rank_i

                    # RSI momentum: for calls prefer high RSI, for puts prefer low RSI
                    if direction == "call":
                        abs_rsi_momentum = rsi / 100.0
                    else:
                        abs_rsi_momentum = (100.0 - rsi) / 100.0

                    # Momentum-only proxy signal (NOT the full quality_score)
                    momentum_signal = (0.5 * abs_momentum
                             + 0.3 * hv_rank_score
                             + 0.2 * abs_rsi_momentum)
                    momentum_signal = max(0.0, min(1.0, momentum_signal))
                    score = momentum_signal

                    # BS entry pricing (with bid-ask spread cost)
                    T_entry = dte / 365.0
                    sigma = max(hv, 0.05)
                    K = _find_30d_otm_strike(direction, S, sigma, T_entry, risk_free, target_delta=0.30)
                    entry_price_gross = _bs_option_price(direction, S, K, T_entry, risk_free, sigma)
                    if entry_price_gross <= 0.001:
                        continue
                    entry_price = entry_price_gross * (1 + spread_cost_per_side)

                    # Exit at day i+exit_dte — use exit-day realized vol (not entry-day)
                    exit_idx = min(i + exit_dte, len(close_arr) - 1)
                    S_exit = float(close_arr[exit_idx])
                    T_exit = max((dte - exit_dte) / 365.0, 1 / 365.0)
                    sigma_exit = float(hv_arr[exit_idx]) if exit_idx < len(hv_arr) and hv_arr[exit_idx] > 0 else sigma
                    exit_price_time = _bs_option_price(direction, S_exit, K, T_exit, risk_free, sigma_exit)

                    # Apply TP/SL: check if TP or SL was triggered before time exit
                    # Short position: profit when option price falls, loss when it rises
                    actual_exit_price = exit_price_time
                    tp_price = entry_price * (1 - tp)    # price decays to tp% of entry → buy back at profit
                    sl_price = entry_price * (1 - sl)    # price rises by |sl|% of entry → buy back at loss (sl is negative)

                    # Scan intra-period for TP/SL hit (simplified: daily check)
                    for j in range(1, exit_dte + 1):
                        if i + j >= len(close_arr):
                            break
                        S_j = float(close_arr[i + j])
                        T_j = max((dte - j) / 365.0, 1 / 365.0)
                        sigma_j = float(hv_arr[i + j]) if (i + j) < len(hv_arr) and hv_arr[i + j] > 0 else sigma
                        px_j = _bs_option_price(direction, S_j, K, T_j, risk_free, sigma_j)
                        if px_j <= tp_price:   # short: profit when price drops
                            actual_exit_price = px_j
                            break
                        if px_j >= sl_price:   # short: loss when price rises
                            actual_exit_price = px_j
                            break

                    exit_price_after_costs = actual_exit_price * (1 + spread_cost_per_side)
                    pnl_per_share = entry_price - exit_price_after_costs  # short: profit from premium decay
                    pnl_pct = pnl_per_share / entry_price
                    pnl_pct_gross = (entry_price_gross - actual_exit_price) / entry_price_gross

                    # VIX regime for this trade day
                    vix_val = 20.0  # default "Normal"
                    if vix_aligned is not None:
                        try:
                            vix_val = float(vix_aligned.iloc[i])
                            if math.isnan(vix_val):
                                vix_val = 20.0
                        except Exception:
                            pass

                    # EM calibration: track expected vs realized move
                    em_1sigma = S * sigma * math.sqrt(dte / 365.0)
                    realized_move = abs(S_exit - S)
                    within_em = realized_move <= em_1sigma

                    trades.append({
                        "day": i,
                        "direction": direction,
                        "score": score,
                        "entry_price": entry_price,
                        "exit_price": actual_exit_price,
                        "pnl_pct": pnl_pct,
                        "pnl_pct_gross": pnl_pct_gross,
                        "win": pnl_pct > 0,
                        "vix_level": vix_val,
                        "vix_regime": _classify_vix_regime(vix_val),
                        "within_em": within_em,
                    })
                except Exception:
                    continue

            if not trades:
                all_ticker_results.append({"ticker": ticker, "error": "No valid trades simulated"})
                continue

            trades_df = pd.DataFrame(trades)
            n_trades = len(trades_df)
            win_rate = float(trades_df["win"].mean())
            avg_return = float(trades_df["pnl_pct"].mean())
            avg_return_gross = float(trades_df["pnl_pct_gross"].mean()) if "pnl_pct_gross" in trades_df.columns else avg_return

            # EM calibration: what fraction of realized moves fell within 1σ EM (target: ~68%)
            em_within_pct = float(trades_df["within_em"].mean()) if "within_em" in trades_df.columns else None

            # Sharpe (annualised, assuming ~12 trades/year roughly)
            ret_std = float(trades_df["pnl_pct"].std())
            sharpe = (avg_return / ret_std * math.sqrt(n_trades / (lookback_days / 252))) if ret_std > 0 else 0.0

            # IC: Pearson correlation between score and pnl_pct
            ic = float("nan")
            ic_pvalue = float("nan")
            if HAS_SCIPY and n_trades >= 10:
                try:
                    ic_val, pval = scipy_stats.pearsonr(trades_df["score"], trades_df["pnl_pct"])
                    ic = float(ic_val)
                    ic_pvalue = float(pval)
                except Exception:
                    pass

            # Quintile breakdown
            by_quintile = {}
            if n_trades >= 5:
                trades_df["quintile"] = pd.qcut(
                    trades_df["score"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
                )
                for q in [1, 2, 3, 4, 5]:
                    qdf = trades_df[trades_df["quintile"] == q]
                    if len(qdf) > 0:
                        by_quintile[q] = {
                            "n": len(qdf),
                            "win_rate": float(qdf["win"].mean()),
                            "avg_ret": float(qdf["pnl_pct"].mean()),
                        }

            # Optimal threshold: sweep score thresholds
            optimal_threshold = 0.5
            best_sharpe = float("-inf")
            for thr in np.arange(0.3, 0.9, 0.05):
                sub = trades_df[trades_df["score"] >= thr]
                if len(sub) < 5:
                    continue
                s_mean = sub["pnl_pct"].mean()
                s_std = sub["pnl_pct"].std()
                if s_std > 0:
                    s_sharpe = s_mean / s_std * math.sqrt(len(sub))
                    if s_sharpe > best_sharpe:
                        best_sharpe = s_sharpe
                        optimal_threshold = float(thr)

            # Max drawdown (on cumulative pnl_pct)
            cum = trades_df["pnl_pct"].cumsum()
            peak = cum.cummax()
            drawdown = (peak - cum)
            max_drawdown = float(drawdown.max())

            # Profit factor
            wins_sum = float(trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].sum())
            loss_sum = float(abs(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].sum()))
            profit_factor = wins_sum / loss_sum if loss_sum > 0 else float("inf")

            # Regime summary
            regime_summary = {}
            if "vix_regime" in trades_df.columns:
                try:
                    regime_summary = build_regime_summary(trades_df)
                except Exception:
                    pass

            all_ticker_results.append({
                "ticker": ticker,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_return_gross": avg_return_gross,
                "sharpe": sharpe,
                "ic": ic,
                "ic_pvalue": ic_pvalue,
                "by_quintile": by_quintile,
                "optimal_threshold": optimal_threshold,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "regime_summary": regime_summary,
                "em_within_pct": em_within_pct,
            })

        except Exception as e:
            all_ticker_results.append({"ticker": ticker, "error": str(e)})

    return {
        "results": all_ticker_results,
        "lookback_days": lookback_days,
        "dte": dte,
        "exit_dte": exit_dte,
        "tp": tp,
        "sl": sl,
        "spread_cost_per_side": spread_cost_per_side,
    }


# ─── Regime analysis ─────────────────────────────────────────────────────────────

def build_regime_summary(trades_df: "pd.DataFrame") -> Dict[str, Any]:
    """
    Compute per-VIX-regime performance statistics from a trades DataFrame.

    Requires a 'vix_regime' column with values "Low", "Normal", or "High".
    Returns a nested dict keyed by regime name.
    """
    if not HAS_PD or trades_df.empty or "vix_regime" not in trades_df.columns:
        return {}

    summary: Dict[str, Any] = {}
    for regime in ["Low", "Normal", "High"]:
        sub = trades_df[trades_df["vix_regime"] == regime]
        if sub.empty:
            continue
        n = len(sub)
        win_rate = float(sub["win"].mean())
        avg_return = float(sub["pnl_pct"].mean())

        ret_std = float(sub["pnl_pct"].std())
        sharpe = (avg_return / ret_std * math.sqrt(n)) if ret_std > 0 else 0.0

        wins_sum = float(sub.loc[sub["pnl_pct"] > 0, "pnl_pct"].sum())
        loss_sum = float(abs(sub.loc[sub["pnl_pct"] <= 0, "pnl_pct"].sum()))
        profit_factor = wins_sum / loss_sum if loss_sum > 0 else float("inf")

        max_drawdown = _calculate_mdd(sub["pnl_pct"])

        summary[regime] = {
            "n_trades": n,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }
    return summary


def print_regime_report(results: Dict[str, Any], width: int = 90) -> None:
    """
    Print a 4-column regime performance table:
      Regime | Trades | Win% | Sharpe | PF | MDD
    Aggregates across all tickers in results.
    """
    if not HAS_PD:
        return

    ticker_results = [r for r in results.get("results", []) if "error" not in r]
    if not ticker_results:
        return

    # Aggregate across tickers
    combined: Dict[str, Dict] = {}
    for r in ticker_results:
        rs = r.get("regime_summary", {})
        for regime, data in rs.items():
            if regime not in combined:
                combined[regime] = {
                    "n_trades": 0,
                    "wins": 0,
                    "total_ret": 0.0,
                    "sharpe_sum": 0.0,
                    "pf_sum": 0.0,
                    "mdd_sum": 0.0,
                    "count": 0,
                }
            n = data["n_trades"]
            combined[regime]["n_trades"] += n
            combined[regime]["wins"] += round(data["win_rate"] * n)
            combined[regime]["total_ret"] += data["avg_return"] * n
            combined[regime]["sharpe_sum"] += data["sharpe"]
            pf = data["profit_factor"]
            combined[regime]["pf_sum"] += pf if math.isfinite(pf) else 0.0
            combined[regime]["mdd_sum"] += data["max_drawdown"]
            combined[regime]["count"] += 1

    if not combined:
        return

    print()
    hdr = "  VIX REGIME PERFORMANCE BREAKDOWN"
    if HAS_FMT and fmt:
        print(fmt.colorize(hdr, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(hdr)
    print(_sep(width))

    col_hdr = (
        f"  {'Regime':<8}  {'Trades':>6}  {'Win%':>5}  {'Sharpe':>7}"
        f"  {'PF':>5}  {'MDD':>7}"
    )
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    for regime in ["Low", "Normal", "High"]:
        if regime not in combined:
            continue
        d = combined[regime]
        n = d["n_trades"]
        if n == 0:
            continue
        wr = d["wins"] / n
        avg_r = d["total_ret"] / n
        cnt = d["count"]
        avg_sharpe = d["sharpe_sum"] / cnt if cnt > 0 else 0.0
        avg_pf = d["pf_sum"] / cnt if cnt > 0 else 0.0
        avg_mdd = d["mdd_sum"] / cnt if cnt > 0 else 0.0

        pf_str = f"{avg_pf:.2f}" if math.isfinite(avg_pf) else "  inf"
        row = (
            f"  {regime:<8}  {n:>6}  {wr*100:>4.0f}%  {avg_sharpe:>7.2f}"
            f"  {pf_str:>5}  {avg_mdd*100:>6.1f}%"
        )
        if HAS_FMT and fmt:
            color = fmt.Colors.GREEN if avg_r > 0 else fmt.Colors.RED
            print(fmt.colorize(row, color))
        else:
            print(row)

    print(_sep(width))
    print()


# ─── Paper trade IC ─────────────────────────────────────────────────────────────

def run_paper_trade_ic(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Compute IC between quality_score and realized pnl_pct from paper trades DB.
    Reads paper_trades.db via sqlite3.
    Returns structured IC analysis.
    """
    empty_result = {
        "n_trades": 0,
        "ic": None,
        "ic_pvalue": None,
        "significant": False,
        "by_quintile": {},
        "verdict": "Insufficient trades for IC analysis (need >= 10 closed)",
    }
    _COMPONENT_COLS = [
        "pop_score", "ev_score", "rr_score", "liquidity_score",
        "momentum_score", "iv_rank_score", "theta_score",
        "iv_edge_score", "vrp_score", "iv_mispricing_score",
        "skew_align_score", "vega_risk_score", "term_structure_score",
    ]
    empty_result["component_ic"] = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        _col_list = ", ".join(["quality_score", "pnl_pct"] + _COMPONENT_COLS)
        rows = conn.execute(
            f"SELECT {_col_list} "
            "FROM trades WHERE status='CLOSED' "
            "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception as e:
        empty_result["verdict"] = f"Could not read database: {e}"
        return empty_result

    if not rows:
        return empty_result

    scores = []
    returns = []
    for r in rows:
        try:
            s = float(r["quality_score"])
            p = float(r["pnl_pct"])
            if math.isfinite(s) and math.isfinite(p):
                scores.append(s)
                returns.append(p)
        except Exception:
            continue

    n = len(scores)
    if n < 10:
        empty_result["n_trades"] = n
        empty_result["verdict"] = f"Insufficient trades for IC analysis (need >= 10 closed, have {n})"
        return empty_result

    # IC
    ic = float("nan")
    ic_pvalue = float("nan")
    significant = False
    if HAS_SCIPY and HAS_NP:
        try:
            ic_val, pval = scipy_stats.pearsonr(scores, returns)
            ic = float(ic_val)
            ic_pvalue = float(pval)
            significant = (ic_pvalue < 0.05)
        except Exception:
            pass

    spearman_ic = float("nan")
    spearman_ic_pvalue = float("nan")
    if HAS_SCIPY and HAS_NP:
        try:
            sp_val, sp_pval = scipy_stats.spearmanr(scores, returns)
            spearman_ic = float(sp_val)
            spearman_ic_pvalue = float(sp_pval)
        except Exception:
            pass

    # Quintile breakdown
    by_quintile = {}
    if HAS_PD and n >= 5:
        try:
            df = pd.DataFrame({"score": scores, "pnl_pct": returns})
            df["quintile"] = pd.qcut(df["score"], q=min(5, n // 2), labels=False, duplicates="drop") + 1
            for q in sorted(df["quintile"].unique()):
                qdf = df[df["quintile"] == q]
                by_quintile[int(q)] = {
                    "n": len(qdf),
                    "win_rate": float((qdf["pnl_pct"] > 0).mean()),
                    "avg_ret": float(qdf["pnl_pct"].mean()),
                }
        except Exception:
            pass

    if not math.isnan(ic) and significant:
        verdict = f"SIGNIFICANT EDGE: quality_score predicts returns (IC={ic:.2f}, p={ic_pvalue:.3f})"
    elif not math.isnan(ic) and ic_pvalue < 0.10:
        verdict = f"BORDERLINE: weak signal detected (IC={ic:.2f}, p={ic_pvalue:.3f}) — needs more data"
    elif not math.isnan(ic):
        verdict = f"NO SIGNIFICANT EDGE detected (IC={ic:.2f}, p={ic_pvalue:.3f}) — more trades needed"
    else:
        verdict = "IC could not be computed (scipy unavailable)"

    # Per-component IC
    component_ic: dict = {}
    component_pvalues: dict = {}
    if HAS_SCIPY and HAS_NP and HAS_PD:
        try:
            full_df = pd.DataFrame(
                [{k: (float(r[k]) if r[k] is not None else float("nan")) for k in list(r.keys())} for r in rows]
            )
            for col in _COMPONENT_COLS:
                if col not in full_df.columns:
                    continue
                try:
                    sub = full_df[[col, "pnl_pct"]].dropna()
                    if len(sub) >= 10:
                        comp_ic_val, comp_p = scipy_stats.pearsonr(sub[col].values, sub["pnl_pct"].values)
                        component_ic[col] = float(comp_ic_val)
                        component_pvalues[col] = float(comp_p)
                except Exception:
                    pass
        except Exception:
            pass

    return {
        "n_trades": n,
        "ic": ic,
        "ic_pvalue": ic_pvalue,
        "significant": significant,
        "spearman_ic": spearman_ic,
        "spearman_ic_pvalue": spearman_ic_pvalue,
        "by_quintile": by_quintile,
        "verdict": verdict,
        "component_ic": component_ic,
        "component_pvalues": component_pvalues,
    }


# ─── Print functions ─────────────────────────────────────────────────────────────

def _ic_color(ic: float, pvalue: float) -> str:
    """Return color for IC based on significance."""
    if not HAS_FMT or not fmt:
        return ""
    if math.isnan(ic):
        return fmt.Colors.DIM
    if pvalue < 0.05:
        return fmt.Colors.GREEN
    elif pvalue < 0.10:
        return fmt.Colors.YELLOW
    return fmt.Colors.DIM


def _mini_bar(win_rate: float, width: int = 10) -> str:
    """ASCII bar proportional to win_rate (0..1)."""
    filled = int(round(win_rate * width))
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


def print_backtest_report(results: dict, width: int = 90) -> None:
    """
    Print a formatted backtest report from the output of run_backtest().
    Includes per-ticker table and combined quintile analysis.
    Note: Uses BS theoretical prices — see module docstring for limitations.
    """
    ticker_results = [r for r in results.get("results", []) if "error" not in r]
    if not ticker_results:
        print("\n  No backtest results to display.")
        return

    n_tickers = len(ticker_results)
    dte = results.get("dte", 30)
    exit_dte = results.get("exit_dte", 21)
    lookback = results.get("lookback_days", 252)
    spread_cost = results.get("spread_cost_per_side", 0.07)

    print()
    header = (
        f"  BACKTEST RESULTS  \u2014  {n_tickers} ticker(s)"
        f"  |  {lookback}d lookback  |  {dte} DTE  |  {exit_dte}d time exit"
        f"  [THEORETICAL PRICES — see module docstring]"
    )
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    col_hdr = (
        f"  {'Ticker':<8} {'Trades':>6}  {'Win%':>5}  {'Avg Ret':>8}"
        f"  {'Sharpe':>6}  {'IC':>6}  {'p-val':>7}  {'PF':>5}"
    )
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    for r in ticker_results:
        ticker = r["ticker"]
        n_trades = r.get("n_trades", 0)
        win_rate = r.get("win_rate", 0.0)
        avg_ret = r.get("avg_return", 0.0)
        sharpe = r.get("sharpe", 0.0)
        ic = r.get("ic", float("nan"))
        ic_pval = r.get("ic_pvalue", float("nan"))
        pf = r.get("profit_factor", 0.0)

        ic_str = f"{ic:.2f}" if not math.isnan(ic) else " n/a"
        pval_str = f"{ic_pval:.3f}" if not math.isnan(ic_pval) else " n/a"

        # Significance stars
        stars = ""
        if not math.isnan(ic_pval):
            if ic_pval < 0.01:
                stars = "**"
            elif ic_pval < 0.05:
                stars = "*"

        pf_str = f"{pf:.2f}" if math.isfinite(pf) else "  inf"

        row = (
            f"  {ticker:<8} {n_trades:>6}  {win_rate*100:>4.0f}%  {avg_ret*100:>+7.1f}%"
            f"  {sharpe:>6.2f}  {ic_str:>6}  {pval_str}{stars:<2}  {pf_str:>5}"
        )

        if HAS_FMT and fmt:
            ic_col = _ic_color(ic, ic_pval if not math.isnan(ic_pval) else 1.0)
            if ic_col:
                print(fmt.colorize(row, ic_col))
            else:
                print(row)
        else:
            print(row)

    print(_sep(width))

    # Gross vs after-cost summary
    all_gross = [r.get("avg_return_gross", r.get("avg_return", 0.0)) for r in ticker_results]
    all_net = [r.get("avg_return", 0.0) for r in ticker_results]
    if all_gross and all_net:
        avg_gross = sum(all_gross) / len(all_gross)
        avg_net = sum(all_net) / len(all_net)
        cost_line = (
            f"  Spread cost ({spread_cost*100:.0f}% per side):  "
            f"Gross avg return: {avg_gross*100:+.1f}%  |  "
            f"After-cost avg return: {avg_net*100:+.1f}%  |  "
            f"Cost drag: {(avg_gross - avg_net)*100:.1f}%"
        )
        if HAS_FMT and fmt:
            print(fmt.colorize(cost_line, fmt.Colors.YELLOW))
        else:
            print(cost_line)

    # EM calibration: what fraction of realized moves fell within 1σ expected move
    em_pcts = [r["em_within_pct"] for r in ticker_results if r.get("em_within_pct") is not None]
    if em_pcts:
        avg_em = sum(em_pcts) / len(em_pcts) * 100
        em_color = (fmt.Colors.GREEN if abs(avg_em - 68) < 5 else fmt.Colors.YELLOW) if HAS_FMT and fmt else ""
        em_line = f"  EM calibration: {avg_em:.0f}% of realized moves within 1\u03c3 EM (target: ~68%)"
        if HAS_FMT and fmt and em_color:
            print(fmt.colorize(em_line, em_color))
        else:
            print(em_line)

    print(_sep(width))

    # Combined quintile analysis
    # Aggregate all quintile data across tickers
    combined_quintiles: dict = {}
    for r in ticker_results:
        for q, qdata in r.get("by_quintile", {}).items():
            if q not in combined_quintiles:
                combined_quintiles[q] = {"n": 0, "wins": 0, "total_ret": 0.0}
            combined_quintiles[q]["n"] += qdata["n"]
            combined_quintiles[q]["wins"] += int(qdata["win_rate"] * qdata["n"])
            combined_quintiles[q]["total_ret"] += qdata["avg_ret"] * qdata["n"]

    if combined_quintiles:
        q_hdr = "  SCORE QUINTILE ANALYSIS (combined)"
        if HAS_FMT and fmt:
            print(fmt.colorize(q_hdr, fmt.Colors.BRIGHT_CYAN))
        else:
            print(q_hdr)

        q_labels = {1: "Quintile 1 (low score): ", 2: "Quintile 2:             ",
                    3: "Quintile 3:             ", 4: "Quintile 4:             ",
                    5: "Quintile 5 (high score):"}

        for q in sorted(combined_quintiles.keys()):
            qd = combined_quintiles[q]
            n = qd["n"]
            if n == 0:
                continue
            wr = qd["wins"] / n
            avg_r = qd["total_ret"] / n
            bar = _mini_bar(wr, width=10)
            label = q_labels.get(q, f"Quintile {q}:             ")
            row = f"  {label}  {wr*100:.0f}% win   avg {avg_r*100:+.1f}%   {bar}"
            if HAS_FMT and fmt:
                color = fmt.Colors.GREEN if avg_r > 0 else fmt.Colors.RED
                print(fmt.colorize(row, color))
            else:
                print(row)

        print(_sep(width))

        # Monotonicity check
        q_avgs = []
        for q in sorted(combined_quintiles.keys()):
            qd = combined_quintiles[q]
            n = qd["n"]
            if n > 0:
                q_avgs.append(qd["total_ret"] / n)

        monotone = all(q_avgs[i] <= q_avgs[i + 1] for i in range(len(q_avgs) - 1))
        mono_str = "CONFIRMED" if monotone else "NOT confirmed"
        mono_color = fmt.Colors.GREEN if monotone else fmt.Colors.YELLOW if HAS_FMT and fmt else ""
        mono_line = f"  IC monotonicity: {mono_str} \u2014 score rank {'does' if monotone else 'does not'} predict return rank"
        if HAS_FMT and fmt and mono_color:
            print(fmt.colorize(mono_line, mono_color))
        else:
            print(mono_line)

    # Optimal threshold summary
    thresholds = [r.get("optimal_threshold", 0.5) for r in ticker_results]
    if thresholds:
        avg_thr = sum(thresholds) / len(thresholds)
        # Find win rate and PF at this threshold
        thr_wins = 0
        thr_n = 0
        for r in ticker_results:
            thr_wins += int(r.get("win_rate", 0) * r.get("n_trades", 0))
            thr_n += r.get("n_trades", 0)
        opt_wr = thr_wins / thr_n if thr_n > 0 else 0.0
        opt_pf = sum(r.get("profit_factor", 1.0) for r in ticker_results) / len(ticker_results)
        thr_line = f"  Optimal threshold: quality_score >= {avg_thr:.2f}  ->  {opt_wr*100:.0f}% win rate, {opt_pf:.2f}x avg PF"
        if HAS_FMT and fmt:
            print(fmt.colorize(thr_line, fmt.Colors.YELLOW))
        else:
            print(thr_line)
    _caveat = ("  Note: Walk-forward backtest uses a momentum-only proxy signal, "
               "not the full quality_score.\n"
               "  For quality_score validation, use: python -m src.backtester --paper-ic")
    if HAS_FMT and fmt:
        print(fmt.colorize(_caveat, fmt.Colors.DIM))
    else:
        print(_caveat)
    print()

    # Regime breakdown
    any_regime = any(r.get("regime_summary") for r in ticker_results)
    if any_regime:
        print_regime_report(results, width=width)


def print_paper_trade_ic(db_path: str = DEFAULT_DB_PATH, width: int = 90) -> None:
    """
    Print IC analysis of paper trades from paper_trades.db.
    Shows per-quintile breakdown and verdict.
    """
    ic_data = run_paper_trade_ic(db_path)

    print()
    header = "  PAPER TRADE SIGNAL QUALITY (IC Analysis)"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    n = ic_data.get("n_trades", 0)
    ic = ic_data.get("ic")
    ic_pval = ic_data.get("ic_pvalue")
    significant = ic_data.get("significant", False)
    verdict = ic_data.get("verdict", "")

    print(f"  Closed trades analysed: {n}")

    if ic is not None and not (isinstance(ic, float) and math.isnan(ic)):
        ic_pval_f = float(ic_pval) if ic_pval is not None else 1.0
        stars = "**" if ic_pval_f < 0.01 else ("*" if ic_pval_f < 0.05 else "")
        ic_line = f"  IC (quality_score vs return): {ic:.3f}{stars}   p-value: {ic_pval_f:.4f}"
        if HAS_FMT and fmt:
            color = _ic_color(ic, ic_pval_f)
            print(fmt.colorize(ic_line, color))
        else:
            print(ic_line)

    sp_ic = ic_data.get("spearman_ic")
    sp_pval = ic_data.get("spearman_ic_pvalue")
    if sp_ic is not None and not (isinstance(sp_ic, float) and math.isnan(sp_ic)):
        sp_pval_f = float(sp_pval) if sp_pval is not None else 1.0
        stars = "**" if sp_pval_f < 0.01 else ("*" if sp_pval_f < 0.05 else "")
        sp_line = f"  IC (Spearman rank):              {sp_ic:.3f}{stars}   p-value: {sp_pval_f:.4f}"
        if HAS_FMT and fmt:
            color = _ic_color(sp_ic, sp_pval_f)
            print(fmt.colorize(sp_line, color))
        else:
            print(sp_line)

    by_quintile = ic_data.get("by_quintile", {})
    if by_quintile:
        print()
        q_hdr = "  Score quintile breakdown:"
        if HAS_FMT and fmt:
            print(fmt.colorize(q_hdr, fmt.Colors.DIM))
        else:
            print(q_hdr)
        for q in sorted(by_quintile.keys()):
            qd = by_quintile[q]
            bar = _mini_bar(qd["win_rate"], width=8)
            row = f"    Q{q}: {qd['n']:>4} trades  {qd['win_rate']*100:.0f}% win  avg {qd['avg_ret']*100:+.1f}%  {bar}"
            if HAS_FMT and fmt:
                color = fmt.Colors.GREEN if qd["avg_ret"] > 0 else fmt.Colors.RED
                print(fmt.colorize(row, color))
            else:
                print(row)
        print()

    verdict_line = f"  Verdict: {verdict}"
    if HAS_FMT and fmt:
        color = fmt.Colors.GREEN if significant else (fmt.Colors.YELLOW if n >= 5 else fmt.Colors.DIM)
        print(fmt.colorize(verdict_line, color, bold=significant))
    else:
        print(verdict_line)
    print()


# ─── CLI entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "SPY", "NVDA", "QQQ", "TSLA"]
    print(f"Running backtest for: {', '.join(tickers)}")
    results = run_backtest(tickers)
    print_backtest_report(results)
    print_paper_trade_ic()


__all__ = [
    "compute_rsi",
    "run_backtest",
    "run_paper_trade_ic",
    "print_backtest_report",
    "print_paper_trade_ic",
]
