"""CLI display functions extracted from options_screener.py."""

import math
import sys
import shutil
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import numpy as np

from .utils import format_pct, format_money, determine_moneyness
from .data_fetching import get_vix_level
from .oi_snapshot import save_oi_snapshot

try:
    from . import formatting as fmt
    from .trade_analysis import (
        generate_trade_thesis, calculate_entry_exit_levels, calculate_confidence_score,
        categorize_by_strategy, assess_risk_factors, format_trade_plan,
        explain_quality_score, format_risk_alerts, build_scenario_table,
        generate_execution_guidance,
    )
    from tqdm import tqdm
    HAS_ENHANCED_CLI = True
except ImportError:
    HAS_ENHANCED_CLI = False

try:
    from .simulation import monte_carlo_pop
    HAS_SIMULATION = True
except ImportError:
    HAS_SIMULATION = False

try:
    from .visualize_results import create_visualizations
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    from .vol_analytics import print_vol_cone, print_iv_surface, classify_vol_regime, print_regime_summary
    from .backtester import print_paper_trade_ic
    HAS_VOL_ANALYTICS = True
except ImportError:
    HAS_VOL_ANALYTICS = False


def _get_config() -> dict:
    """Lazy import of load_config to avoid circular imports with options_screener."""
    try:
        from .options_screener import load_config
        return load_config()
    except Exception:
        return {}


def get_display_width() -> int:
    """Return terminal width clamped to a readable range (60–120)."""
    try:
        w = shutil.get_terminal_size(fallback=(100, 24)).columns
        return max(60, min(w, 120))
    except Exception:
        return 100


def format_dte_bucket(dte: float) -> str:
    """Return DTE bucket name for a given days-to-expiration value."""
    dte = float(dte)
    if dte <= 14:
        return "Short (7-14 DTE)"
    if dte <= 30:
        return "Standard (15-30 DTE)"
    return "Swing (31-45 DTE)"


def print_score_breakdown(score_drivers_str: str) -> str:
    """Format score drivers string into a readable multi-part display."""
    if not score_drivers_str:
        return ""
    parts = str(score_drivers_str).split()
    pos_parts = [p for p in parts if p.startswith("+") and p != "|"]
    neg_parts = [p for p in parts if p.startswith("-") and p != "|"]
    out = ""
    if pos_parts:
        out += "  Positives: " + "  ".join(pos_parts)
    if neg_parts:
        out += "\n  Negatives: " + "  ".join(neg_parts)
    return out or score_drivers_str


def print_top_n_table(contracts: pd.DataFrame, n: int) -> None:
    """Print a ranked cross-ticker table grouped by DTE bucket."""
    if contracts.empty:
        print("No contracts to display.")
        return

    width = get_display_width()
    sep = "-" * width

    header = (
        f"{'Rank':<5} {'Ticker':<7} {'Type':<5} {'Strike':>7} {'Expiry':<12} "
        f"{'DTE':>4} {'Delta':>6} {'IV%':>6} {'PoP%':>6} {'Prem':>7} "
        f"{'Score':>7}  Score Drivers"
    )

    dte_bucket_order = ["Short (7-14 DTE)", "Standard (15-30 DTE)", "Swing (31-45 DTE)"]
    contracts = contracts.copy()
    contracts["_dte"] = (contracts["T_years"] * 365.0).round(0) if "T_years" in contracts.columns else 0
    contracts["_bucket"] = contracts["_dte"].apply(format_dte_bucket)

    rank = 1
    printed_any = False
    for bucket in dte_bucket_order:
        bucket_df = contracts[contracts["_bucket"] == bucket]
        if bucket_df.empty:
            continue
        if HAS_ENHANCED_CLI:
            try:
                from . import formatting as fmt
                print("\n" + fmt.draw_separator(width))
                print(fmt.colorize(f"  {bucket}", fmt.Colors.BRIGHT_CYAN, bold=True))
            except Exception:
                print(f"\n{sep}")
                print(f"  {bucket}")
        else:
            print(f"\n{sep}")
            print(f"  {bucket}")
        print(header)
        print(sep)
        for _, row in bucket_df.iterrows():
            dte_val = int(row.get("_dte", 0))
            iv_pct = row.get("iv_percentile_30", row.get("iv_percentile", 0)) or 0
            pop = row.get("prob_profit", 0) or 0
            prem = row.get("premium", 0) or 0
            delta = row.get("delta", 0) or 0
            iv = row.get("impliedVolatility", 0) or 0
            score = row.get("quality_score", 0) or 0
            drivers = str(row.get("score_drivers", ""))[:30]
            line = (
                f"{rank:<5} {str(row.get('symbol','')):<7} {str(row.get('type','')):<5} "
                f"{row.get('strike', 0):>7.1f} {str(row.get('expiration', '')):<12} "
                f"{dte_val:>4} {delta:>6.2f} {iv*100:>5.1f}% {pop*100:>5.1f}% "
                f"${prem:>6.2f} {score:>7.3f}  {drivers}"
            )
            print(line)
            rank += 1
            printed_any = True
        print(sep)

    if not printed_any:
        print("No contracts matched any DTE bucket.")
    print(f"\nTop {n} contracts shown. Run with --export csv to save full results.")


def format_analysis_row(row: pd.Series, chain_iv_median: float, mode: str) -> str:
    """Formats the second detail row for the screener report (analysis)."""
    parts = []

    # IV context — visual bar or plain fallback
    iv = row.get("impliedVolatility", pd.NA)
    if pd.notna(iv) and math.isfinite(iv):
        if HAS_ENHANCED_CLI:
            iv_pct = row.get("iv_percentile_30", row.get("iv_percentile", 0.5)) or 0.5
            hv = row.get("hv_30d", 0) or 0
            iv_conf = row.get("iv_confidence", "")
            parts.append(fmt.format_iv_rank_bar(iv_pct, hv, float(iv), iv_confidence=iv_conf))
        else:
            rel = "\u2248" if abs(float(iv) - chain_iv_median) <= 0.02 else ("above" if iv > chain_iv_median else "below")
            parts.append(f"IV: {format_pct(iv)} ({rel} median)")

    # Probability of Profit — dual PoP/PoT when prob_touch is available
    pop = row.get("prob_profit", pd.NA)
    if pd.notna(pop) and math.isfinite(pop):
        if HAS_ENHANCED_CLI:
            pop_str = fmt.format_pop(float(pop))
            prob_touch = row.get("prob_touch", None)
            if prob_touch is not None and pd.notna(prob_touch) and float(prob_touch) > 0:
                pot_str = fmt.format_pop(float(prob_touch))
                parts.append(f"PoP(exp): {pop_str}  PoT(touch): {pot_str}")
            else:
                parts.append(f"PoP: {pop_str}")
        else:
            parts.append(f"PoP: {format_pct(pop)}")

    # Risk/Reward or Return on Risk
    if mode == "Premium Selling":
        ror = row.get("return_on_risk", pd.NA)
        if pd.notna(ror):
            parts.append(f"RoR: {format_pct(ror)}")
    else:
        rr = row.get("rr_ratio", pd.NA)
        if pd.notna(rr):
            rr_str = fmt.format_rr(float(rr)) if HAS_ENHANCED_CLI else f"{float(rr):.1f}x"
            parts.append(f"RR: {rr_str}")

    # EV
    ev = row.get("ev_per_contract", pd.NA)
    if pd.notna(ev) and math.isfinite(float(ev)):
        ev_str = fmt.format_ev(float(ev)) if HAS_ENHANCED_CLI else f"${float(ev):.0f}"
        parts.append(f"EV: {ev_str}")

    # PCR per expiry
    pcr = row.get("pcr", pd.NA)
    if pd.notna(pcr):
        pcr_signal = row.get("pcr_signal", "")
        pcr_str = f"PCR: {float(pcr):.2f}"
        if pcr_signal:
            pcr_str += f" ({pcr_signal})"
        if HAS_ENHANCED_CLI and pcr_signal:
            pcr_color = fmt.Colors.RED if pcr_signal == "HEAVY HEDGING" else (fmt.Colors.GREEN if pcr_signal == "BULLISH FLOW" else fmt.Colors.DIM)
            parts.append(fmt.colorize(pcr_str, pcr_color))
        else:
            parts.append(pcr_str)

    # Momentum
    rsi = row.get("rsi_14", pd.NA)
    ret5 = row.get("ret_5d", pd.NA)
    if pd.notna(rsi) and pd.notna(ret5):
        parts.append(f"Momentum: RSI {float(rsi):.0f}, 5d {format_pct(ret5)}")

    # Sentiment
    sentiment = row.get("sentiment_tag", "Neutral")
    if HAS_ENHANCED_CLI:
        s_color = fmt.Colors.GREEN if sentiment == "Bullish" else (fmt.Colors.RED if sentiment == "Bearish" else fmt.Colors.YELLOW)
        sentiment_str = fmt.colorize(sentiment, s_color)
    else:
        sentiment_str = sentiment
    parts.append(f"Sentiment: {sentiment_str}")

    # Quality Score + leading drivers
    quality = row.get('quality_score', 0.0)
    drivers = row.get("score_drivers", "")
    if HAS_ENHANCED_CLI:
        stars, q_color = fmt.format_quality_score(quality)
        q_str = f"Quality: {fmt.colorize(f'{quality:.2f}', q_color)} {stars}"
        if drivers:
            q_str += f"  {fmt.colorize(f'[{drivers}]', fmt.Colors.DIM)}"
        parts.append(q_str)
    else:
        parts.append(f"Quality: {quality:.2f}" + (f"  [{drivers}]" if drivers else ""))

    # Earnings Play
    if row.get("Earnings Play") == "YES":
        underpriced_status = "Underpriced" if row.get("is_underpriced") else "Overpriced"
        parts.append(f"Earnings: YES ({underpriced_status})")

    # Earnings implied move vs historical
    if row.get("Earnings Play") == "YES" and pd.notna(row.get("implied_earnings_move")):
        try:
            imp = float(row["implied_earnings_move"])
            hist = row.get("hist_earnings_move")
            beat = row.get("earnings_beat_rate")
            cheap = row.get("earnings_iv_cheap")
            move_str = f"Implied Move: \u00b1{imp:.1%}"
            if hist is not None and pd.notna(hist):
                move_str += f" | Hist Avg: \u00b1{float(hist):.1%}"
            if beat is not None and pd.notna(beat):
                move_str += f" | Beat Rate: {float(beat):.0%}"
            if cheap is not None and pd.notna(cheap):
                label = "CHEAP" if cheap else "RICH"
                if HAS_ENHANCED_CLI:
                    color = fmt.Colors.GREEN if cheap else fmt.Colors.RED
                    move_str += f" | IV {fmt.colorize(label, color)}"
                else:
                    move_str += f" | IV {label}"
            parts.append(move_str)
        except Exception:
            pass

    # Term structure
    tss = row.get("term_structure_spread", None)
    if tss is not None and pd.notna(tss):
        ts_label = "CONTANGO" if tss > 0.02 else ("BACKWARDATION" if tss < -0.02 else "FLAT")
        if HAS_ENHANCED_CLI:
            ts_color = fmt.Colors.GREEN if tss > 0.02 else (fmt.Colors.RED if tss < -0.02 else fmt.Colors.DIM)
            parts.append(f"Term: {fmt.colorize(ts_label, ts_color)} ({tss:+.1%})")
        else:
            parts.append(f"Term: {ts_label} ({tss:+.1%})")

    # Stock Price, DTE, and breakeven distance
    stock_price = row.get('underlying', 0.0)
    dte = int(row.get('T_years', 0) * 365)
    be_dist = row.get('be_dist_pct', pd.NA)
    stock_dte_str = f"Stock: {format_money(stock_price)} | DTE: {dte}d"
    if pd.notna(be_dist) and math.isfinite(float(be_dist)):
        stock_dte_str += f" | BE move: {float(be_dist):.1f}%"
    parts.append(stock_dte_str)

    # --- Seasonality ---
    if pd.notna(row.get("seasonal_win_rate")):
        win_rate = row["seasonal_win_rate"]
        current_month_name = datetime.now().strftime("%b")
        parts.append(f"{current_month_name} Hist: {win_rate:.0%}")

    # --- Warnings & Squeeze ---
    if row.get("gamma_ramp"):
        w = fmt.colorize("GAMMA RAMP", fmt.Colors.BRIGHT_RED, bold=True) if HAS_ENHANCED_CLI else "GAMMA RAMP"
        parts.append(w)
    if row.get("decay_warning"):
        w = fmt.format_warning("HIGH DECAY RISK") if HAS_ENHANCED_CLI else "HIGH DECAY RISK"
        parts.append(w)
    if row.get("sr_warning"):
        w = fmt.colorize(row["sr_warning"], fmt.Colors.BRIGHT_RED) if HAS_ENHANCED_CLI else row["sr_warning"]
        parts.append(w)
    if row.get("oi_wall_warning"):
        w = fmt.colorize(row["oi_wall_warning"], fmt.Colors.BRIGHT_RED) if HAS_ENHANCED_CLI else row["oi_wall_warning"]
        parts.append(w)
    if row.get("squeeze_play"):
        parts.append("\U0001f525 SQUEEZE PLAY")

    # Macro / yield warnings
    if row.get("macro_warning"):
        w = fmt.colorize(row["macro_warning"], fmt.Colors.BRIGHT_RED, bold=True) if HAS_ENHANCED_CLI else row["macro_warning"]
        parts.append(w)
    if row.get("max_pain_warning"):
        parts.append(row["max_pain_warning"])
    if row.get("yield_warning"):
        parts.append(row["yield_warning"])
    if row.get("high_premium_turnover"):
        parts.append("\U0001f40b WHALE FLOW")

    return " | ".join(parts)


def format_mechanics_row(row: pd.Series) -> str:
    """Formats the first detail row for the screener report (market mechanics)."""
    parts = []

    # Liquidity
    vol = int(row.get('volume', 0))
    oi = int(row.get('openInterest', 0))
    if HAS_ENHANCED_CLI:
        vol_color = fmt.Colors.GREEN if vol > 200 else (fmt.Colors.YELLOW if vol > 50 else fmt.Colors.RED)
        oi_color = fmt.Colors.GREEN if oi > 500 else (fmt.Colors.YELLOW if oi > 100 else fmt.Colors.RED)
        parts.append(f"Vol: {fmt.colorize(str(vol), vol_color)} OI: {fmt.colorize(str(oi), oi_color)}")
    else:
        parts.append(f"Vol: {vol} OI: {oi}")

    # Spread
    sp = row.get("spread_pct", pd.NA)
    if HAS_ENHANCED_CLI:
        parts.append(f"Spread: {fmt.format_spread(float(sp) if pd.notna(sp) else 0.0)}")
    else:
        parts.append(f"Spread: {format_pct(sp)}")

    # Delta
    d = row.get("delta", pd.NA)
    if pd.notna(d) and math.isfinite(d):
        opt_type = row.get("type", "call")
        if HAS_ENHANCED_CLI:
            parts.append(f"Delta: {fmt.format_delta(d, is_call=(opt_type.lower() == 'call'))}")
        else:
            parts.append(f"Delta: {d:+.2f}")

    # Greeks
    gamma = row.get("gamma", pd.NA)
    vega = row.get("vega", pd.NA)
    theta = row.get("theta", pd.NA)
    vega_dollar = row.get("vega_dollar", pd.NA)
    if pd.notna(gamma) and pd.notna(vega) and pd.notna(theta):
        vd_str = f" V$: ${float(vega_dollar):.0f}" if pd.notna(vega_dollar) else ""
        greeks_str = f"Greeks: \u0393 {gamma:.3f}, V {vega:.2f}, \u0398 {theta:.2f}{vd_str}"
        parts.append(fmt.colorize(greeks_str, fmt.Colors.DIM) if HAS_ENHANCED_CLI else greeks_str)

    # Annualized return — particularly relevant for premium sellers
    ann_ret = row.get("annualized_return", pd.NA)
    if pd.notna(ann_ret) and math.isfinite(float(ann_ret)) and float(ann_ret) > 0:
        ann_str = f"Ann.Yield: {float(ann_ret):.1%}"
        parts.append(fmt.colorize(ann_str, fmt.Colors.DIM) if HAS_ENHANCED_CLI else ann_str)

    # Charm and Vanna
    charm = row.get("charm", None)
    vanna = row.get("vanna", None)
    if charm is not None and pd.notna(charm) and abs(float(charm)) > 0.001:
        parts.append(f"Charm: {float(charm):.4f}/d")
    if vanna is not None and pd.notna(vanna) and abs(float(vanna)) > 0.01:
        parts.append(f"Vanna: {float(vanna):.3f}")

    # OI change
    oi_chg = int(row.get("oi_change", 0) or 0)
    if oi_chg != 0:
        sign = "+" if oi_chg > 0 else ""
        oi_chg_str = f"OI \u0394: {sign}{oi_chg}"
        if HAS_ENHANCED_CLI:
            chg_color = fmt.Colors.GREEN if oi_chg > 0 else fmt.Colors.RED
            parts.append(fmt.colorize(oi_chg_str, chg_color))
        else:
            parts.append(oi_chg_str)

    # Cost
    cost = row.get('premium', 0.0) * 100
    if HAS_ENHANCED_CLI:
        parts.append(f"Cost: {fmt.format_money(cost)}")
    else:
        parts.append(f"Cost: {format_money(cost)}")

    # Theta acceleration warning
    theta_val = row.get("theta", pd.NA)
    premium_val = row.get("premium", 0.0) or 0.0
    dte_val = float(row.get("T_years", 0) or 0) * 365
    if pd.notna(theta_val) and math.isfinite(float(theta_val)) and premium_val > 0:
        daily_bleed = abs(float(theta_val))
        daily_bleed_pct = daily_bleed / premium_val * 100
        bleed_str = f"\u0398 bleed: -${daily_bleed:.2f}/day ({daily_bleed_pct:.1f}%/day)"
        if HAS_ENHANCED_CLI:
            if dte_val < 14 and daily_bleed_pct > 5:
                bleed_str += " " + fmt.colorize("\u26a0 ACCELERATING", fmt.Colors.BRIGHT_RED)
            elif daily_bleed_pct > 2:
                bleed_str += " " + fmt.colorize("HIGH DECAY", fmt.Colors.YELLOW)
        else:
            if dte_val < 14 and daily_bleed_pct > 5:
                bleed_str += " \u26a0 ACCELERATING"
            elif daily_bleed_pct > 2:
                bleed_str += " HIGH DECAY"
        parts.append(bleed_str)

    return " | ".join(parts)


def _format_breakeven_line(row: pd.Series, arrow: str) -> str:
    """Format a breakeven vs expected-move ratio line for a single pick."""
    try:
        breakeven = row.get('breakeven', None)
        spot = float(row.get('underlying', 0) or 0)
        required_move = row.get('required_move', None)
        expected_move = row.get('expected_move', None)

        if breakeven is None or required_move is None or expected_move is None:
            return ""

        breakeven = float(breakeven)
        required_move = float(required_move)
        expected_move = float(expected_move)

        if spot <= 0 or expected_move <= 0:
            return ""

        req_pct = required_move / spot * 100
        em_pct = expected_move / spot * 100
        em_ratio = required_move / expected_move

        opt_type = str(row.get('type', 'call')).lower()
        sign = "+" if opt_type == 'call' else "-"

        if em_ratio < 0.75:
            label, color = "ACHIEVABLE", fmt.Colors.GREEN if HAS_ENHANCED_CLI else None
        elif em_ratio < 1.0:
            label, color = "FAIR", fmt.Colors.YELLOW if HAS_ENHANCED_CLI else None
        elif em_ratio < 1.5:
            label, color = "STRETCHED", fmt.Colors.YELLOW if HAS_ENHANCED_CLI else None
        else:
            label, color = "UNLIKELY", fmt.Colors.RED if HAS_ENHANCED_CLI else None

        body = (
            f"BE: ${breakeven:.2f} | Needs {sign}${required_move:.2f} ({sign}{req_pct:.1f}%)"
            f"  1\u03c3 EM: ${expected_move:.2f} ({em_pct:.1f}%)  \u2192  {em_ratio:.1f}\u00d7 EM"
        )
        prefix_sym = "\u26a0 " if em_ratio >= 1.0 else ""
        if HAS_ENHANCED_CLI and color:
            label_str = fmt.colorize(f"{prefix_sym}{label}", color)
        else:
            label_str = f"{prefix_sym}{label}"

        be_label = fmt.colorize("BE:", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "BE:"
        full_line = f"BE: ${breakeven:.2f} | Needs {sign}${required_move:.2f} ({sign}{req_pct:.1f}%)  1\u03c3 EM: ${expected_move:.2f} ({em_pct:.1f}%)  \u2192  {em_ratio:.1f}\u00d7 EM  {label_str}"
        be_dim = fmt.colorize("Breakevn: ", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "Breakevn: "
        return f"    {arrow} {be_dim}{full_line[4:]}"  # strip leading "BE: " and use dim label
    except Exception:
        return ""


def _print_strategy_panel(df_picks: pd.DataFrame, width: int) -> None:
    """Print strategy classification mix panel before the bucket breakdown."""
    if not HAS_ENHANCED_CLI or df_picks.empty:
        return
    try:
        df_cat = categorize_by_strategy(df_picks)
    except Exception:
        return

    counts = df_cat['strategy_category'].value_counts()
    con_count = counts.get('CONSERVATIVE', 0)
    bal_count = counts.get('BALANCED', 0)
    agg_count = counts.get('AGGRESSIVE', 0)

    con_str = fmt.colorize(f"{con_count} Conservative", fmt.Colors.GREEN) if con_count else fmt.colorize("0 Conservative", fmt.Colors.DIM)
    bal_str = fmt.colorize(f"{bal_count} Balanced", fmt.Colors.YELLOW) if bal_count else fmt.colorize("0 Balanced", fmt.Colors.DIM)
    agg_str = fmt.colorize(f"{agg_count} Aggressive", fmt.Colors.RED) if agg_count else fmt.colorize("0 Aggressive", fmt.Colors.DIM)

    print(fmt.draw_separator(width))
    print(f"  Strategy Mix: {con_str}  {bal_str}  {agg_str}")

    # Best pick per category
    for cat, color in [('CONSERVATIVE', fmt.Colors.GREEN), ('AGGRESSIVE', fmt.Colors.RED)]:
        sub = df_cat[df_cat['strategy_category'] == cat]
        if sub.empty:
            continue
        best = sub.loc[sub['quality_score'].idxmax()]
        sym = best.get('symbol', 'N/A')
        strike = best.get('strike', 0)
        opt_type = str(best.get('type', 'call')).upper()
        quality = best.get('quality_score', 0)
        pop = best.get('prob_profit', 0)
        rr = best.get('rr_ratio', 0)
        stars, _ = fmt.format_quality_score(quality)
        thesis = generate_trade_thesis(best) if HAS_ENHANCED_CLI else ""
        thesis_short = thesis.split('|')[0].strip()[:40] if thesis else ""
        label = fmt.colorize(f"Best {cat.capitalize()}:", color)
        pop_str = fmt.format_pop(float(pop))
        rr_str = fmt.format_rr(float(rr))
        print(f"  {label} {sym} {opt_type} ${strike:.0f}  {stars}  PoP {pop_str}  RR {rr_str}  \"{thesis_short}\"")

    print(fmt.draw_separator(width))
    print()


def print_executive_summary(df_picks: pd.DataFrame, config: Dict, mode: str = "Discovery",
                            market_trend: str = "Unknown", volatility_regime: str = "Unknown",
                            macro_risk: bool = False, num_tickers: int = 0):
    """
    Print an executive summary with top picks and key warnings.

    Args:
        df_picks: DataFrame with all options
        config: Configuration dictionary
        mode: Scan mode
        market_trend: Market trend (Bullish/Bearish/Sideways)
        volatility_regime: VIX regime (Low/Normal/High)
        macro_risk: Whether macro risk is active
        num_tickers: Number of tickers scanned
    """
    if df_picks.empty:
        return

    if not HAS_ENHANCED_CLI:
        # Fallback to simple summary
        print(f"\n{'='*80}")
        print(f"  SUMMARY: Found {len(df_picks)} opportunities")
        print(f"{'='*80}\n")
        return

    # Use new formatting
    width = get_display_width()

    print("\n" + fmt.draw_box("⚡ EXECUTIVE SUMMARY", width, double=True))

    # Market Context
    vix = get_vix_level()
    vix_str = f"{vix:.1f}" if vix else "N/A"

    context_parts = []
    if mode in ["Discovery", "Budget"]:
        context_parts.append(f"{mode} Scan ({num_tickers} tickers)")
    else:
        context_parts.append(mode)

    print(f"\n{fmt.format_header('📊 MARKET CONTEXT', '')}")
    trend_color = fmt.Colors.GREEN if market_trend == "Bullish" else (
        fmt.Colors.RED if market_trend == "Bearish" else fmt.Colors.YELLOW
    )
    vol_color = fmt.Colors.GREEN if volatility_regime == "Low" else (
        fmt.Colors.RED if volatility_regime == "High" else fmt.Colors.YELLOW
    )

    print(f"   Trend: {fmt.colorize(market_trend, trend_color, bold=True)} | "
          f"VIX: {fmt.colorize(vix_str, vol_color)} ({volatility_regime}) | "
          f"Risk: {fmt.colorize('HIGH', fmt.Colors.RED, bold=True) if macro_risk else fmt.colorize('LOW', fmt.Colors.GREEN)}")

    # Portfolio VaR
    try:
        from .portfolio_risk import RiskAggregator
        _agg = RiskAggregator(config=config)
        _var_data = _agg.calculate_portfolio_var(
            confidence=config.get("var_confidence", 0.95),
            n_simulations=config.get("var_n_simulations", 10_000),
        )
        if _var_data["n_positions"] > 0:
            print(f"\n{fmt.format_header('📉 PORTFOLIO RISK (OPEN POSITIONS)', '')}")
            print(f"   {_var_data['n_positions']} open position(s)  |  "
                  f"{int(config.get('var_confidence', 0.95) * 100)}% 1-day VaR: ${_var_data['var_95']:,.0f}  |  "
                  f"CVaR: ${_var_data['cvar_95']:,.0f}  |  "
                  f"Expected P&L: ${_var_data['mean_pnl']:+,.0f}")
    except Exception:
        pass

    # Top 3 Opportunities
    print(f"\n{fmt.format_header('🏆 TOP 3 OPPORTUNITIES', '')}")

    top3 = df_picks.nlargest(3, 'quality_score')

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        symbol = row.get('symbol', 'N/A')
        strike = row.get('strike', 0)
        opt_type = row.get('type', 'call').upper()
        premium = row.get('premium', 0)
        pop = row.get('prob_profit', 0)
        rr = row.get('rr_ratio', 0)
        ev = row.get('ev_per_contract', 0)
        quality = row.get('quality_score', 0)

        stars, _ = fmt.format_quality_score(quality)

        # Box for each pick
        print(fmt.draw_separator(width - 4, fmt.BoxChars.HORIZONTAL))
        print(f"{fmt.BoxChars.VERTICAL} {i}. {symbol} ${strike} {opt_type} @ ${premium:.2f} • "
              f"{fmt.format_pop(pop)} PoP • {fmt.format_rr(rr)} RR • {fmt.format_ev(ev)} EV • {stars}")

        # Thesis
        thesis = generate_trade_thesis(row) if HAS_ENHANCED_CLI else "Standard setup"
        print(f"{fmt.BoxChars.VERTICAL}    💡 {thesis}")

        # Entry/Exit
        if config.get('display', {}).get('show_entry_exit_levels', True):
            levels = calculate_entry_exit_levels(row, config)
            print(f"{fmt.BoxChars.VERTICAL}    📍 Entry: ≤${levels['entry_price']:.2f} | "
                  f"Target: ${levels['profit_target']:.2f} (+50%) | "
                  f"Stop: ${levels['stop_loss']:.2f} (-25%)")

    print(fmt.draw_separator(width - 4, fmt.BoxChars.HORIZONTAL))

    # Warnings
    print(f"\n{fmt.format_header('⚠️  WATCH OUT', '')}")

    high_spread = df_picks[df_picks['spread_pct'] > 0.20]
    if not high_spread.empty:
        print(fmt.format_warning(f"{len(high_spread)} options with spreads >20% - use limit orders!"))

    neg_ev = df_picks[df_picks['ev_per_contract'] < 0]
    if not neg_ev.empty:
        print(fmt.format_warning(f"{len(neg_ev)} trades have negative expected value"))

    earnings = df_picks[df_picks.get('Earnings Play', 'NO') == 'YES']
    if not earnings.empty:
        print(fmt.format_warning(f"{len(earnings)} earnings plays - IV crush risk post-announcement"))

    low_liquid = df_picks[(df_picks['volume'] < 100) | (df_picks['openInterest'] < 100)]
    if not low_liquid.empty:
        print(fmt.format_warning(f"{len(low_liquid)} low-liquidity options - execution risk"))

    print("\n" + fmt.draw_separator(width, fmt.BoxChars.D_HORIZONTAL))
    print()


def print_best_setup_callout(df_picks: pd.DataFrame, width: int) -> None:
    """Print a prominent callout box for the single best pick by quality_score."""
    if df_picks.empty or len(df_picks) < 3:
        return
    if not HAS_ENHANCED_CLI:
        return

    top = df_picks.nlargest(1, "quality_score").iloc[0]
    if top.get("quality_score", 0) < 0.75:
        return

    total = len(df_picks)
    sym = top.get("symbol", "N/A")
    opt_type = str(top.get("type", "CALL")).upper()
    strike = top.get("strike", 0)
    exp = str(pd.to_datetime(top.get("expiration", "")).date())
    pop = float(top.get("prob_profit", 0) or 0)
    rr = float(top.get("rr_ratio", 0) or 0)
    ev_raw = top.get("ev_per_contract", None)
    ev = float(ev_raw) if (ev_raw is not None and pd.notna(ev_raw) and math.isfinite(float(ev_raw))) else 0.0
    quality = float(top.get("quality_score", 0))
    stars, _ = fmt.format_quality_score(quality)
    thesis = generate_trade_thesis(top) if HAS_ENHANCED_CLI else ""
    thesis_short = thesis.split("|")[0].strip()[:55] if thesis else ""
    pop_str = fmt.format_pop(pop)
    rr_str = fmt.format_rr(rr)
    ev_str = fmt.format_ev(ev)

    line1 = f"BEST SETUP  \u2014  {sym} {opt_type} ${strike:.0f}  exp {exp}   #1/{total}"
    line2 = f"PoP {pop_str}  RR {rr_str}  EV {ev_str}  {stars}  Score {quality:.2f}"
    line3 = f'"{thesis_short}"' if thesis_short else ""

    # Build box manually with double-border
    inner_w = width - 4  # 2 chars each side: ╔ + space ... space + ╗
    def pad(s):
        # strip ANSI for length measurement
        import re
        plain = re.sub(r'\033\[[0-9;]*m', '', s)
        pad_len = max(0, inner_w - len(plain))
        return s + " " * pad_len

    border_h = fmt.BoxChars.D_HORIZONTAL * (width - 2)
    top_border = fmt.colorize(fmt.BoxChars.D_TOP_LEFT + border_h + fmt.BoxChars.D_TOP_RIGHT, fmt.Colors.BRIGHT_GREEN)
    bot_border = fmt.colorize(fmt.BoxChars.D_BOTTOM_LEFT + border_h + fmt.BoxChars.D_BOTTOM_RIGHT, fmt.Colors.BRIGHT_GREEN)
    v = fmt.colorize(fmt.BoxChars.D_VERTICAL, fmt.Colors.BRIGHT_GREEN)

    lines_to_print = [line1, line2]
    if line3:
        lines_to_print.append(line3)

    print(top_border)
    for ln in lines_to_print:
        colored_ln = fmt.colorize(ln, fmt.Colors.BRIGHT_GREEN, bold=(ln == line1))
        print(f"{v}  {pad(colored_ln)}{v}")
    print(bot_border)
    print()


def print_comparison_table(df_top: pd.DataFrame, mode: str = "Discovery") -> None:
    """Print a compact side-by-side comparison table of top picks per DTE bucket."""
    if df_top.empty or not HAS_ENHANCED_CLI:
        return

    width = get_display_width()
    is_seller = (mode == "Premium Selling")

    # Select top 5 per DTE bucket
    df_top = df_top.copy()
    df_top["_dte"] = (df_top["T_years"] * 365.0).round(0) if "T_years" in df_top.columns else 0

    rows = df_top.sort_values("quality_score", ascending=False).head(10)
    if rows.empty:
        return

    print()
    hdr = "  QUICK COMPARISON  —  Top Picks"
    print(fmt.colorize(hdr, fmt.Colors.BRIGHT_CYAN, bold=True))
    sep_line = "  " + "\u2500" * (width - 4)
    print(fmt.colorize(sep_line, fmt.Colors.DIM))

    # Table header
    col_hdr = (
        f"  {'#':>3}  {'Tick':<5} {'Strike':<8} {'Exp':>8} {'Score':>6} {'PoP':>5}"
        f" {'R/R':>5} {'IV%':>5} {'EV':>7} {'Sprd':>5}"
    )
    if "iv_surface_residual" in rows.columns:
        col_hdr += f" {'SVI':>6}"
    print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    print(fmt.colorize(sep_line, fmt.Colors.DIM))

    for rank_i, (_, r) in enumerate(rows.iterrows(), 1):
        strike = r.get("strike", 0)
        opt_type = str(r.get("type", "C"))[0].upper()
        exp_raw = str(r.get("expiration", ""))[:10]
        try:
            exp_str = pd.to_datetime(exp_raw).strftime("%m/%d")
        except Exception:
            exp_str = exp_raw[-5:]
        score = r.get("quality_score", 0)
        pop = r.get("prob_profit", 0) or 0
        rr = r.get("rr_ratio", 0) or 0
        iv_pct = (r.get("iv_percentile_30", r.get("iv_percentile", 0)) or 0) * 100
        ev = r.get("ev_per_contract", 0) or 0
        spread = (r.get("spread_pct", 0) or 0) * 100

        # Color score — pad before colorize to keep alignment
        sc_color = fmt.Colors.GREEN if score >= 0.70 else (fmt.Colors.YELLOW if score >= 0.50 else fmt.Colors.RED)
        score_padded = f"{score:>6.2f}"
        score_str = fmt.colorize(score_padded, sc_color)
        pop_str = f"{pop*100:>4.0f}%"
        rr_str = f"{rr:>4.1f}x" if rr > 0 else "  n/a"
        ev_str = f"${ev:>+5.0f}" if abs(ev) < 10000 else f"${ev:>+5.0f}"

        # Symbol prefix for multi-ticker
        sym = str(r.get("symbol", ""))[:5]
        strike_str = f"${strike:.0f}{opt_type}"

        line = (
            f"  {rank_i:>3}  {sym:<5} {strike_str:<8} {exp_str:>8} {score_str} {pop_str:>5}"
            f" {rr_str:>5} {iv_pct:>4.0f}% {ev_str:>7} {spread:>4.1f}%"
        )
        if "iv_surface_residual" in rows.columns:
            resid = r.get("iv_surface_residual", 0) or 0
            if abs(resid) > 0.15:
                tag = "CHEAP" if resid < 0 else "RICH"
                tag_color = fmt.Colors.GREEN if resid < 0 else fmt.Colors.RED
                tag_padded = f"{tag:>6}"
                line += f" {fmt.colorize(tag_padded, tag_color)}"
            else:
                line += f" {'':>6}"

        print(line)

    print(fmt.colorize(sep_line, fmt.Colors.DIM))
    print()


def print_report(df_picks: pd.DataFrame, underlying_price: float, rfr: float, num_expiries: int, min_dte: int, max_dte: int, mode: str = "Single-stock", budget: Optional[float] = None, market_trend: str = "Unknown", volatility_regime: str = "Unknown", config: Optional[Dict] = None):
    """Enhanced report with context, formatting, top pick, and summary."""
    if df_picks.empty:
        print("No picks available after filtering.")
        return

    if config is None:
        config = _get_config()

    WIDTH = get_display_width()
    chain_iv_median = df_picks["impliedVolatility"].median(skipna=True)
    is_multi = mode in ["Budget scan", "Discovery scan", "Premium Selling"]

    # ── Header ──────────────────────────────────────────────────────────────
    if mode == "Budget scan":
        title = f"OPTIONS SCREENER  \u2014  MULTI-TICKER  (Budget: ${budget:.2f})"
    elif mode == "Discovery scan":
        unique_tickers = df_picks["symbol"].nunique() if "symbol" in df_picks.columns else 1
        title = f"OPTIONS SCREENER  \u2014  DISCOVERY  ({unique_tickers} Tickers)"
    elif mode == "Premium Selling":
        unique_tickers = df_picks["symbol"].nunique() if "symbol" in df_picks.columns else 1
        title = f"OPTIONS SCREENER  \u2014  PREMIUM SELLING  ({unique_tickers} Tickers)"
    else:
        symbol_name = df_picks.iloc[0]['symbol'] if "symbol" in df_picks.columns else "UNKNOWN"
        title = f"OPTIONS SCREENER  \u2014  {symbol_name}"

    print()
    if HAS_ENHANCED_CLI:
        print(fmt.draw_box(title, WIDTH, double=True))
    else:
        print("=" * WIDTH)
        print(f"  {title}")
        print("=" * WIDTH)

    # Context lines
    trend_color = fmt.Colors.GREEN if market_trend == "Bullish" else (fmt.Colors.RED if market_trend == "Bearish" else fmt.Colors.YELLOW)
    vol_color = fmt.Colors.GREEN if volatility_regime == "Low" else (fmt.Colors.RED if volatility_regime == "High" else fmt.Colors.YELLOW)

    if mode == "Single-stock":
        sp_str = f"${underlying_price:.2f}"
        print(f"  Stock Price: {fmt.colorize(sp_str, fmt.Colors.BRIGHT_WHITE, bold=True) if HAS_ENHANCED_CLI else sp_str}")
    elif mode == "Budget scan":
        print(f"  Budget: ${budget:.2f}/contract  |  LOW <${budget*0.33:.0f}  |  MED ${budget*0.33:.0f}\u2013${budget*0.66:.0f}  |  HIGH >${budget*0.66:.0f}")
    else:
        print(f"  Top opportunities across all price ranges  |  LOW / MED / HIGH by premium")

    trend_str = fmt.colorize(market_trend, trend_color, bold=True) if HAS_ENHANCED_CLI else market_trend
    vol_str = fmt.colorize(volatility_regime, vol_color) if HAS_ENHANCED_CLI else volatility_regime
    print(f"  Trend: {trend_str}  |  Volatility: {vol_str}  |  RFR: {rfr*100:.2f}%")
    print(f"  Nearest Expiries: {num_expiries}  |  DTE: {min_dte}\u2013{max_dte}d  |  Chain IV: {format_pct(chain_iv_median)}")

    if HAS_ENHANCED_CLI:
        print(fmt.draw_separator(WIDTH))
    else:
        print("=" * WIDTH)

    # ── Strategy Classification Panel ────────────────────────────────────────
    _print_strategy_panel(df_picks, WIDTH)

    # ── Best Setup Callout ────────────────────────────────────────────────────
    print_best_setup_callout(df_picks, WIDTH)

    # ── Rank all picks by quality_score ──────────────────────────────────────
    df_picks = df_picks.copy()
    df_picks["_rank"] = df_picks["quality_score"].rank(ascending=False, method="first").astype(int)

    # ── Buckets ──────────────────────────────────────────────────────────────
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df_picks[df_picks["price_bucket"] == bucket]
        if sub.empty:
            continue

        # Bucket header
        if HAS_ENHANCED_CLI:
            bucket_color = fmt.Colors.DIM if bucket == "LOW" else (fmt.Colors.BRIGHT_YELLOW if bucket == "MEDIUM" else fmt.Colors.BRIGHT_GREEN)
            label = fmt.colorize(f"  {bucket} PREMIUM", bucket_color, bold=(bucket == "HIGH"))
            print(f"\n{label}  \u00b7  Top {len(sub)} Picks")
        else:
            print(f"\n  {bucket} PREMIUM  \u00b7  Top {len(sub)} Picks")

        # Summary stats
        avg_iv = sub["impliedVolatility"].mean()
        avg_spread = sub["spread_pct"].mean(skipna=True)
        median_delta = sub["abs_delta"].median()
        if HAS_ENHANCED_CLI:
            iv_str = fmt.format_percentage(avg_iv, good=40, bad=80, higher_is_better=False)
            sp_str = fmt.format_spread(float(avg_spread) if pd.notna(avg_spread) else 0.0)
            print(f"  Summary: Avg IV {iv_str} | Avg Spread {sp_str} | Median |\u0394| {median_delta:.2f}\n")
        else:
            print(f"  Summary: Avg IV {format_pct(avg_iv)} | Avg Spread {format_pct(avg_spread)} | Median |\u0394| {median_delta:.2f}\n")

        # Column headers
        if is_multi:
            hdr = f"  {'Rk':<4} {'Tkr':<6} {'W':<2} {'Type':<5} {'Strike':>8} {'Expiry':<12} {'Prem':<9} {'IV':<8} {'OI':>7} {'Vol':>7} {'Delta':>7}  {'Tag':<4}  Quality"
        else:
            hdr = f"  {'Rk':<4} {'W':<2} {'Type':<5} {'Strike':>8} {'Expiry':<12} {'Prem':<9} {'IV':<8} {'OI':>7} {'Vol':>7} {'Delta':>7}  {'Tag':<4}  Quality"
        print(fmt.colorize(hdr, fmt.Colors.BOLD + fmt.Colors.UNDERLINE) if HAS_ENHANCED_CLI else hdr)

        if HAS_ENHANCED_CLI:
            print("  " + fmt.draw_separator(WIDTH - 2))
        else:
            print("  " + "-" * (WIDTH - 2))

        for _, r in sub.iterrows():
            exp = pd.to_datetime(r["expiration"]).date()
            moneyness = determine_moneyness(r)
            dte = int(r["T_years"] * 365)
            whale = "\U0001f40b" if r.get("high_premium_turnover", False) else "  "
            opt_type = r["type"].upper()
            strike = r["strike"]
            premium = r.get("premium", 0.0)
            iv = r.get("impliedVolatility", 0.0)
            oi = int(r.get("openInterest", 0))
            vol = int(r.get("volume", 0))
            delta = r.get("delta", 0.0)
            quality = r.get("quality_score", 0.0)
            rank = int(r.get("_rank", 0))

            if HAS_ENHANCED_CLI:
                rank_str = fmt.colorize(f"#{rank:<2}", fmt.Colors.DIM)
                sym_str = fmt.colorize(f"{r['symbol']:<6}", fmt.Colors.BRIGHT_WHITE, bold=True)
                type_color = fmt.Colors.BRIGHT_GREEN if opt_type == "CALL" else fmt.Colors.BRIGHT_RED
                type_str = fmt.colorize(f"{opt_type:<5}", type_color)
                # Pad before colorize to avoid ANSI breaking alignment
                exp_padded = f"{str(exp):<12}"
                exp_str = fmt.colorize(exp_padded, fmt.Colors.YELLOW) if dte < 14 else exp_padded
                prem_padded = f"${premium:.2f}"
                prem_color = fmt.Colors.GREEN if premium > 0 else fmt.Colors.YELLOW
                prem_str = fmt.colorize(f"{prem_padded:<9}", prem_color)
                iv_color = fmt.Colors.GREEN if iv < chain_iv_median * 0.9 else (fmt.Colors.RED if iv > chain_iv_median * 1.2 else fmt.Colors.YELLOW)
                iv_str = fmt.colorize(f"{format_pct(iv):<8}", iv_color)
                oi_color = fmt.Colors.GREEN if oi > 500 else (fmt.Colors.YELLOW if oi > 100 else fmt.Colors.RED)
                oi_str = fmt.colorize(f"{oi:>7}", oi_color)
                vol_color = fmt.Colors.GREEN if vol > 200 else (fmt.Colors.YELLOW if vol > 50 else fmt.Colors.RED)
                vol_str = fmt.colorize(f"{vol:>7}", vol_color)
                # Pad delta before colorize
                delta_raw = f"{delta:>+7.2f}"
                d_aligned = (opt_type == "CALL" and delta > 0) or (opt_type != "CALL" and delta < 0)
                d_color = fmt.Colors.GREEN if d_aligned else fmt.Colors.RED
                delta_str = fmt.colorize(delta_raw, d_color)
                mon_color = fmt.Colors.GREEN if moneyness == "ITM" else (fmt.Colors.YELLOW if moneyness == "ATM" else fmt.Colors.DIM)
                mon_str = fmt.colorize(f"{moneyness:<4}", mon_color)
                stars, _ = fmt.format_quality_score(quality)
                if is_multi:
                    print(f"  {rank_str} {sym_str} {whale} {type_str} {strike:>8.2f} {exp_str} {prem_str} {iv_str} {oi_str} {vol_str}  {delta_str}  {mon_str}  {stars}")
                else:
                    print(f"  {rank_str} {whale} {type_str} {strike:>8.2f} {exp_str} {prem_str} {iv_str} {oi_str} {vol_str}  {delta_str}  {mon_str}  {stars}")
            else:
                rank_plain = f"#{rank:<2}"
                if is_multi:
                    print(f"  {rank_plain} {r['symbol']:<6} {whale} {opt_type:<5} {strike:>8.2f} {exp} {format_money(premium):<9} {format_pct(iv):<8} {oi:>7} {vol:>7} {delta:>+7.2f}  {moneyness:<4}")
                else:
                    print(f"  {rank_plain} {whale} {opt_type:<5} {strike:>8.2f} {exp} {format_money(premium):<9} {format_pct(iv):<8} {oi:>7} {vol:>7} {delta:>+7.2f}  {moneyness:<4}")

            arrow = fmt.colorize("\u21b3", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "\u21b3"

            # Risk alerts — before detail rows
            if HAS_ENHANCED_CLI:
                alerts_line = format_risk_alerts(r)
                if alerts_line:
                    print(alerts_line)

            # Detail rows
            mechanics_line = format_mechanics_row(r)
            analysis_line = format_analysis_row(r, chain_iv_median, mode)
            mech_label = fmt.colorize("Mechanics:", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "Mechanics:"
            anal_label = fmt.colorize("Analysis: ", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "Analysis: "
            print(f"    {arrow} {mech_label} {mechanics_line}")
            print(f"    {arrow} {anal_label} {analysis_line}")

            # Breakeven vs expected move
            if HAS_ENHANCED_CLI:
                be_line = _format_breakeven_line(r, arrow)
                if be_line:
                    print(be_line)

            # Trade plan (thesis + entry/exit)
            if HAS_ENHANCED_CLI:
                thesis = generate_trade_thesis(r)
                # Append IV surface mispricing and crush info to thesis
                _extra_thesis = []
                if "iv_surface_residual" in r.index:
                    resid = r.get("iv_surface_residual", 0) or 0
                    if resid < -0.15:
                        _extra_thesis.append(fmt.colorize("CHEAP vs surface", fmt.Colors.GREEN, bold=True))
                    elif resid > 0.15:
                        _extra_thesis.append(fmt.colorize("RICH vs surface", fmt.Colors.RED, bold=True))
                if "avg_iv_crush" in r.index and pd.notna(r.get("avg_iv_crush")):
                    _extra_thesis.append(f"Avg IV crush: -{r['avg_iv_crush']:.0%} post-earnings")
                if _extra_thesis:
                    thesis += " | " + " | ".join(_extra_thesis)
                thesis_label = fmt.colorize("Thesis:   ", fmt.Colors.BRIGHT_CYAN)
                thesis_text = fmt.colorize(thesis, fmt.Colors.BRIGHT_CYAN) if not _extra_thesis else thesis
                print(f"    {arrow} {thesis_label} {thesis_text}")
                levels = calculate_entry_exit_levels(r, config)
                tp_pct = config.get("exit_rules", {}).get("take_profit", 0.50)
                sl_pct = abs(config.get("exit_rules", {}).get("stop_loss", -0.25))
                entry_str = fmt.colorize(f"\u2264${levels['entry_price']:.2f}", fmt.Colors.BRIGHT_WHITE)
                target_str = fmt.colorize(f"${levels['profit_target']:.2f} (+{tp_pct:.0%})", fmt.Colors.GREEN)
                stop_str = fmt.colorize(f"${levels['stop_loss']:.2f} (-{sl_pct:.0%})", fmt.Colors.RED)
                from .trade_analysis import calculate_confidence_score, assess_risk_factors
                conf_score, conf_label = calculate_confidence_score(r)
                risks = assess_risk_factors(r)
                conf_color = fmt.Colors.GREEN if conf_label == "HIGH" else (fmt.Colors.YELLOW if conf_label == "MEDIUM" else fmt.Colors.RED)
                conf_str = fmt.colorize(f"{conf_label} ({conf_score:.0%})", conf_color, bold=True)
                print(f"    {arrow} {fmt.colorize('Entry:    ', fmt.Colors.DIM)} {entry_str}  |  Target: {target_str}  |  Stop: {stop_str}")
                print(f"         Breakeven: ${levels['breakeven']:.2f}  |  Max Loss: ${levels['max_loss']:.0f}  |  Confidence: {conf_str}")

                # Execution guidance
                exec_guide = generate_execution_guidance(r)
                if exec_guide:
                    exec_label = fmt.colorize("Exec:     ", fmt.Colors.DIM)
                    print(f"         {exec_label}{fmt.colorize(exec_guide, fmt.Colors.YELLOW)}")

                # Quality score breakdown
                qscore_line = explain_quality_score(r)
                if qscore_line:
                    score_label = fmt.colorize("Score:    ", fmt.Colors.DIM)
                    print(f"         {score_label}{qscore_line}")

                if risks and risks != ["Standard risks apply"]:
                    risk_parts = [fmt.colorize(ri, fmt.Colors.BRIGHT_RED) for ri in risks[:2]]
                    print(f"         Risks: {' | '.join(risk_parts)}")

                # Scenario P/L matrix
                scenario = build_scenario_table(r, rfr, WIDTH)
                if scenario:
                    print(scenario)

            # Institutional metrics
            if "short_interest" in r and pd.notna(r["short_interest"]):
                print(f"      \u2022 Short Interest: {r['short_interest']*100:.2f}%")
            if "rvol" in r and pd.notna(r["rvol"]):
                print(f"      \u2022 RVOL: {r['rvol']:.2f}x")
            if "gex_flip_price" in r and pd.notna(r["gex_flip_price"]):
                print(f"      \u2022 GEX Flip: ${r['gex_flip_price']:.2f}")
            if "vwap" in r and pd.notna(r["vwap"]):
                print(f"      \u2022 VWAP: ${r['vwap']:.2f}")

            print("")  # Newline

    # Save OI snapshot for next run
    save_oi_snapshot(df_picks)

    # Compact comparison table for quick-glance ranking
    print_comparison_table(df_picks, mode)


def print_news_panel(news_map: dict, picks_df: pd.DataFrame, width: int = 100) -> None:
    """Print a consolidated news & events panel for all tickers in picks_df."""
    if not news_map:
        return
    # Only show news for tickers that actually have picks
    symbols = picks_df["symbol"].unique().tolist() if "symbol" in picks_df.columns else []
    shown = [sym for sym in symbols if sym in news_map and news_map[sym] is not None]
    if not shown:
        return

    try:
        from .news_fetcher import format_news_panel
    except Exception:
        return

    use_color = HAS_ENHANCED_CLI
    print()
    if HAS_ENHANCED_CLI:
        print(fmt.draw_separator(width))
    else:
        print("=" * width)
    print("  NEWS & EVENTS DIGEST")

    for sym in shown[:6]:  # cap at 6 tickers for output length
        nd = news_map[sym]
        if nd is None:
            continue
        panel = format_news_panel(nd, width=width, use_color=use_color)
        print(panel)


def print_spreads_report(df_spreads: pd.DataFrame):
    """Prints a report of the vertical spreads found."""
    if df_spreads.empty:
        return

    print("\n" + "="*80)
    print("  VERTICAL SPREADS REPORT")
    print("="*80)

    print(f"  {'Symbol':<7} {'Type':<12} {'Long Strike':<12} {'Short Strike':<13} {'Expiration':<12} {'Cost':<8} {'Max Profit':<12} {'Risk':<8}")
    print("  " + "-"*78)

    for _, row in df_spreads.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        print(
            f"  {row['symbol']:<7} "
            f"{row['type']:<12} "
            f"{row['long_strike']:>11.2f} "
            f"{row['short_strike']:>12.2f} "
            f"{exp} "
            f"{format_money(row['spread_cost']):<8} "
            f"{format_money(row['max_profit']):<12} "
            f"{format_money(row['risk']):<8}"
        )


def print_credit_spreads_report(df_spreads: pd.DataFrame):
    """Prints a dedicated report for credit spreads."""
    if df_spreads.empty:
        print("\nNo credit spreads meeting the criteria were found.")
        return

    WIDTH = get_display_width()
    print()
    if HAS_ENHANCED_CLI:
        print(fmt.draw_box("CREDIT SPREADS REPORT  \u2014  INCOME ENGINE", WIDTH, double=True))
    else:
        print("=" * WIDTH)
        print("  CREDIT SPREADS REPORT (INCOME ENGINE)")
        print("=" * WIDTH)

    hdr = f"  {'Symbol':<7} {'Type':<10} {'Short':>8} {'Long':>8} {'Expiration':<12} {'Credit':<10} {'Max Profit':<12} {'Max Loss':<10} Score"
    print(fmt.colorize(hdr, fmt.Colors.BOLD) if HAS_ENHANCED_CLI else hdr)
    print(("  " + fmt.draw_separator(WIDTH - 2)) if HAS_ENHANCED_CLI else ("  " + "-" * (WIDTH - 2)))

    for _, row in df_spreads.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        quality = row["quality_score"]
        credit_str = fmt.format_money(row['net_credit']) if HAS_ENHANCED_CLI else format_money(row['net_credit'])
        profit_str = fmt.format_money(row['max_profit']) if HAS_ENHANCED_CLI else format_money(row['max_profit'])
        loss_str = fmt.colorize(f"${row['max_loss']:.2f}", fmt.Colors.RED) if HAS_ENHANCED_CLI else format_money(row['max_loss'])
        if HAS_ENHANCED_CLI:
            stars, _ = fmt.format_quality_score(quality)
            score_str = f"{quality:.2f} {stars}"
        else:
            score_str = f"{quality:.2f}"
        print(
            f"  {row['symbol']:<7} "
            f"{row['type']:<10} "
            f"{row['short_strike']:>8.2f} "
            f"{row['long_strike']:>8.2f} "
            f"{exp} "
            f"{credit_str:<10} "
            f"{profit_str:<12} "
            f"{loss_str:<10} "
            f"{score_str}"
        )


def print_iron_condor_report(df_condors: pd.DataFrame):
    """Prints a dedicated report for iron condors."""
    if df_condors.empty:
        print("\nNo iron condors meeting the criteria were found.")
        return

    WIDTH = get_display_width()
    print()
    if HAS_ENHANCED_CLI:
        print(fmt.draw_box("IRON CONDOR REPORT  \u2014  RANGE-BOUND STRATEGIES", WIDTH, double=True))
    else:
        print("=" * WIDTH)
        print("  IRON CONDOR REPORT (RANGE-BOUND STRATEGIES)")
        print("=" * WIDTH)

    _delta_hdr = "Net \u0394"
    hdr = f"  {'Symbol':<7} {'Exp':<12} {'Put Wing':<18} {'Call Wing':<18} {'Credit':<10} {'Max Risk':<10} {'RoR':<8} {_delta_hdr:<8} Score"
    print(fmt.colorize(hdr, fmt.Colors.BOLD) if HAS_ENHANCED_CLI else hdr)
    print(("  " + fmt.draw_separator(WIDTH - 2)) if HAS_ENHANCED_CLI else ("  " + "-" * (WIDTH - 2)))

    for _, row in df_condors.iterrows():
        exp = pd.to_datetime(row["expiration"]).date()
        put_wing = f"{row['long_put_strike']:.0f}/{row['short_put_strike']:.0f}"
        call_wing = f"{row['short_call_strike']:.0f}/{row['long_call_strike']:.0f}"
        net_delta = row['net_delta']
        quality = row['quality_score']

        credit_str = fmt.format_money(row['total_credit']) if HAS_ENHANCED_CLI else format_money(row['total_credit'])
        risk_str = fmt.colorize(f"${row['max_risk']:.2f}", fmt.Colors.RED) if HAS_ENHANCED_CLI else format_money(row['max_risk'])
        ror_str = fmt.colorize(f"{row['return_on_risk']:.2f}x", fmt.Colors.GREEN if row['return_on_risk'] > 0.25 else fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"{row['return_on_risk']:.2f}x"
        delta_color = fmt.Colors.GREEN if abs(net_delta) < 0.05 else fmt.Colors.RED
        delta_str = fmt.colorize(f"{net_delta:>+.3f}", delta_color) if HAS_ENHANCED_CLI else f"{net_delta:>+.3f}"
        if HAS_ENHANCED_CLI:
            stars, _ = fmt.format_quality_score(quality)
            score_str = f"{quality:.2f} {stars}"
        else:
            score_str = f"{quality:.2f}"

        print(
            f"  {row['symbol']:<7} {exp} "
            f"{put_wing:<18} {call_wing:<18} "
            f"{credit_str:<10} "
            f"{risk_str:<10} "
            f"{ror_str:<8} "
            f"{delta_str:<8} "
            f"{score_str}"
        )

    if HAS_ENHANCED_CLI:
        print("\n" + fmt.draw_separator(WIDTH))
        print(fmt.colorize("  Iron Condors profit from range-bound movement with defined risk on both sides.", fmt.Colors.DIM))
        print(fmt.colorize("  Net Delta closer to 0 = more neutral position.", fmt.Colors.DIM))
    else:
        print("\n  Iron Condors profit from range-bound movement with defined risk on both sides.")
        print("  Net Delta shows directional bias (closer to 0 = more neutral)")
