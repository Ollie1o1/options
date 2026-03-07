#!/usr/bin/env python3
"""
Trade analysis utilities for generating actionable trading insights.
Provides trade thesis generation, entry/exit level calculations, and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def generate_trade_thesis(row: pd.Series) -> str:
    """
    Generate a concise trade thesis explaining why this option is recommended.

    Args:
        row: DataFrame row containing option data

    Returns:
        Trade thesis string with key reasons for the trade
    """
    reasons = []
    warnings = []

    # Positive factors
    pop = row.get('prob_profit', 0)
    if pop > 0.65:
        reasons.append("High probability (>65%)")
    elif pop > 0.55:
        reasons.append("Good probability (>55%)")

    iv_pct = row.get('iv_percentile_30', row.get('iv_percentile', 0))
    if iv_pct > 0.70:
        reasons.append("IV elevated - mean reversion play")
    elif iv_pct < 0.30:
        reasons.append("IV suppressed - volatility expansion play")

    if row.get('Unusual_Whale', False):
        reasons.append("Unusual flow detected 🐋")

    if row.get('squeeze_play', False):
        reasons.append("Gamma squeeze setup 🔥")

    if row.get('Trend_Aligned', False):
        reasons.append("Trend aligned")

    # Risk/Reward
    rr = row.get('rr_ratio', 0)
    if rr > 1.5:
        reasons.append(f"Strong R/R ({rr:.1f}x)")

    # Momentum
    rsi = row.get('rsi_14', 50)
    if rsi > 70:
        reasons.append("RSI overbought (contrarian)")
    elif rsi < 30:
        reasons.append("RSI oversold (bounce play)")

    # Seasonality
    seasonal_win = row.get('seasonal_win_rate', 0)
    if seasonal_win > 0.75:
        reasons.append(f"Strong seasonality ({seasonal_win:.0%} win rate)")

    # Warnings
    ev = row.get('ev_per_contract', 0)
    if ev < 0:
        warnings.append("Negative EV")

    spread = row.get('spread_pct', 0)
    if spread > 0.20:
        warnings.append("Wide spread - use limits")
    elif spread > 0.15:
        warnings.append("Moderate spread")

    if row.get('Earnings Play', 'NO') == 'YES':
        warnings.append("Earnings play - IV crush risk")

    oi_warning = row.get('oi_wall_warning', '')
    if oi_warning:
        warnings.append(oi_warning)

    # Construct thesis
    thesis_parts = []

    if reasons:
        thesis_parts.append(" • ".join(reasons[:3]))  # Limit to top 3 reasons

    if warnings:
        warning_str = " • ".join([f"⚠️ {w}" for w in warnings[:2]])  # Limit to top 2 warnings
        thesis_parts.append(warning_str)

    if not thesis_parts:
        return "Standard setup"

    return " | ".join(thesis_parts)


def calculate_entry_exit_levels(row: pd.Series, config: Dict) -> Dict[str, float]:
    """
    Calculate recommended entry price, profit target, and stop loss.

    Args:
        row: DataFrame row containing option data
        config: Configuration dictionary with exit rules

    Returns:
        Dictionary with entry, target, stop, and breakeven levels
    """
    premium = row.get('premium', 0)
    bid = row.get('bid', premium * 0.95)
    ask = row.get('ask', premium * 1.05)
    spread_pct = row.get('spread_pct', 0.10)
    strike = row.get('strike', 0)
    opt_type = row.get('type', 'call').lower()

    # Get exit rules from config
    exit_rules = config.get('exit_rules', {})
    entry_improvement = config.get('entry_exit_rules', {}).get('entry_improvement_pct', 0.05)
    profit_target_pct = exit_rules.get('take_profit', 0.50)
    stop_loss_pct = abs(exit_rules.get('stop_loss', -0.25))

    # Calculate entry price (aim for better than mid)
    # For wide spreads, be more aggressive in improving entry
    improvement_factor = min(entry_improvement + (spread_pct * 0.3), 0.15)

    # Buying: try to enter below mid
    # Selling: try to enter above mid
    entry_price = premium * (1 - improvement_factor)

    # Ensure entry is between bid and ask
    entry_price = max(bid, min(ask, entry_price))

    # Calculate profit target and stop loss
    profit_target = entry_price * (1 + profit_target_pct)
    stop_loss = entry_price * (1 - stop_loss_pct)

    # Calculate breakeven
    if opt_type == 'call':
        breakeven = strike + entry_price
    else:  # put
        breakeven = strike - entry_price

    # Calculate maximum loss (cost of premium × 100)
    max_loss = entry_price * 100

    # Calculate potential profit (at profit target)
    potential_profit = (profit_target - entry_price) * 100

    return {
        'entry_price': entry_price,
        'profit_target': profit_target,
        'stop_loss': stop_loss,
        'breakeven': breakeven,
        'max_loss': max_loss,
        'potential_profit': potential_profit,
        'risk_reward_ratio': potential_profit / max_loss if max_loss > 0 else 0
    }


def calculate_confidence_score(row: pd.Series) -> Tuple[float, str]:
    """
    Calculate a confidence score based on data quality and reliability.

    Args:
        row: DataFrame row containing option data

    Returns:
        Tuple of (confidence_score, confidence_label)
    """
    confidence = 1.0
    factors = []

    # Penalize wide spreads (poor execution)
    spread = row.get('spread_pct', 0)
    if spread > 0.30:
        confidence -= 0.3
        factors.append("wide spread")
    elif spread > 0.20:
        confidence -= 0.15
        factors.append("moderate spread")

    # Penalize low volume (poor liquidity)
    volume = row.get('volume', 0)
    if volume < 50:
        confidence -= 0.25
        factors.append("low volume")
    elif volume < 100:
        confidence -= 0.1
        factors.append("moderate volume")

    # Penalize low open interest
    oi = row.get('openInterest', 0)
    if oi < 50:
        confidence -= 0.2
        factors.append("low OI")
    elif oi < 100:
        confidence -= 0.1
        factors.append("moderate OI")

    # Penalize if IV data is missing or suspicious
    iv = row.get('impliedVolatility', 0)
    if iv <= 0 or iv > 5.0:  # IV > 500% is suspicious
        confidence -= 0.2
        factors.append("suspicious IV")

    # Penalize if near expiration (gamma risk)
    dte = row.get('T_years', 0) * 365
    if dte < 3:
        confidence -= 0.2
        factors.append("very short DTE")
    elif dte < 7:
        confidence -= 0.1
        factors.append("short DTE")

    # Boost confidence for institutional activity
    if row.get('Unusual_Whale', False):
        confidence += 0.1
        factors.append("unusual activity")

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # Label
    if confidence >= 0.8:
        label = "HIGH"
    elif confidence >= 0.6:
        label = "MEDIUM"
    elif confidence >= 0.4:
        label = "LOW"
    else:
        label = "VERY LOW"

    return confidence, label


def categorize_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize options by strategy type (Conservative/Balanced/Aggressive).

    Args:
        df: DataFrame with options data

    Returns:
        DataFrame with 'strategy_category' and 'risk_level' columns added
    """
    if df.empty:
        return df

    df = df.copy()

    def assign_category(row):
        pop = row.get('prob_profit', 0)
        iv_pct = row.get('iv_percentile_30', row.get('iv_percentile', 0))
        volume = row.get('volume', 0)
        spread = row.get('spread_pct', 1.0)
        rr = row.get('rr_ratio', 0)

        # Conservative: High PoP, good liquidity, tight spreads
        if (pop >= 0.60 and
            iv_pct >= 0.50 and
            volume >= 200 and
            spread <= 0.15):
            return 'CONSERVATIVE', 'LOW RISK'

        # Aggressive: High R/R, lower PoP, or earnings plays
        if (rr > 1.5 or
            pop < 0.45 or
            row.get('Earnings Play', 'NO') == 'YES' or
            row.get('squeeze_play', False)):
            return 'AGGRESSIVE', 'HIGH RISK'

        # Everything else is Balanced
        return 'BALANCED', 'MED RISK'

    df[['strategy_category', 'risk_level']] = df.apply(
        lambda row: pd.Series(assign_category(row)),
        axis=1
    )

    return df


def get_position_sizing_recommendation(row: pd.Series, account_size: float,
                                       risk_per_trade: float = 0.02) -> Dict:
    """
    Calculate recommended position size based on account size and risk tolerance.

    Args:
        row: DataFrame row containing option data
        account_size: Total account size in dollars
        risk_per_trade: Percentage of account to risk per trade (default 2%)

    Returns:
        Dictionary with position sizing recommendations
    """
    max_loss = row.get('premium', 0) * 100  # Cost per contract
    risk_amount = account_size * risk_per_trade

    if max_loss <= 0:
        return {
            'contracts': 0,
            'total_cost': 0,
            'percent_of_account': 0
        }

    # Calculate number of contracts based on risk
    contracts = int(risk_amount / max_loss)

    # Ensure at least 1 contract if affordable
    if contracts == 0 and max_loss < account_size:
        contracts = 1

    total_cost = max_loss * contracts
    pct_of_account = (total_cost / account_size) * 100

    return {
        'contracts': contracts,
        'total_cost': total_cost,
        'percent_of_account': pct_of_account,
        'risk_amount': risk_amount
    }


def assess_risk_factors(row: pd.Series) -> List[str]:
    """
    Identify all risk factors for an options trade.

    Args:
        row: DataFrame row containing option data

    Returns:
        List of risk factor descriptions
    """
    risks = []

    # Liquidity risk
    volume = row.get('volume', 0)
    oi = row.get('openInterest', 0)
    if volume < 50 or oi < 50:
        risks.append("Low liquidity - may be difficult to exit")

    # Spread risk
    spread = row.get('spread_pct', 0)
    if spread > 0.25:
        risks.append(f"Very wide spread ({spread:.0%}) - poor execution risk")
    elif spread > 0.15:
        risks.append(f"Wide spread ({spread:.0%}) - use limit orders")

    # Time decay risk
    dte = row.get('T_years', 0) * 365
    theta = row.get('theta', 0)
    if dte < 7:
        risks.append("Extreme time decay - gamma/theta risk high")
    elif dte < 14:
        risks.append("High time decay - monitor closely")

    # Volatility risk
    iv = row.get('impliedVolatility', 0) * 100
    if iv > 150:
        risks.append("Very high IV - volatility crush risk")

    # Earnings risk
    if row.get('Earnings Play', 'NO') == 'YES':
        risks.append("Earnings event nearby - IV crush expected post-announcement")

    # Expected value risk
    ev = row.get('ev_per_contract', 0)
    if ev < -50:
        risks.append(f"Negative expected value (${ev:.0f}) - unfavorable odds")
    elif ev < 0:
        risks.append("Slight negative EV - breakeven or small loss expected")

    # Open interest walls
    oi_warning = row.get('oi_wall_warning', '')
    if oi_warning:
        risks.append(f"OI resistance: {oi_warning}")

    # Max pain
    mp_warning = row.get('max_pain_warning', '')
    if mp_warning:
        risks.append("Trading against max pain level")

    # Macro risks
    macro_warning = row.get('macro_warning', '')
    if macro_warning:
        risks.append(f"Macro risk: {macro_warning}")

    # Yield warning
    yield_warning = row.get('yield_warning', '')
    if yield_warning:
        risks.append("Treasury yield spike - rate-sensitive")

    return risks if risks else ["Standard risks apply"]


def format_trade_plan(row: pd.Series, config: Dict, include_sizing: bool = False,
                     account_size: Optional[float] = None) -> str:
    """
    Format a complete trade plan with entry, exit, and risk management.

    Args:
        row: DataFrame row containing option data
        config: Configuration dictionary
        include_sizing: Whether to include position sizing recommendations
        account_size: Account size for position sizing

    Returns:
        Formatted trade plan string
    """
    levels = calculate_entry_exit_levels(row, config)
    confidence, conf_label = calculate_confidence_score(row)
    risks = assess_risk_factors(row)

    lines = []

    # Entry/Exit levels
    lines.append(f"📍 Entry: ≤${levels['entry_price']:.2f} | "
                f"Target: ${levels['profit_target']:.2f} (+{config.get('exit_rules', {}).get('take_profit', 0.50):.0%}) | "
                f"Stop: ${levels['stop_loss']:.2f} (-{abs(config.get('exit_rules', {}).get('stop_loss', -0.25)):.0%})")

    lines.append(f"   Breakeven: ${levels['breakeven']:.2f} | Max Loss: ${levels['max_loss']:.0f}")

    # Position sizing
    if include_sizing and account_size:
        sizing = get_position_sizing_recommendation(row, account_size)
        if sizing['contracts'] > 0:
            lines.append(f"   Recommended size: {sizing['contracts']} contract(s) = "
                        f"${sizing['total_cost']:.0f} ({sizing['percent_of_account']:.1f}% of account)")

    # Confidence
    lines.append(f"   Confidence: {conf_label} ({confidence:.0%})")

    # Top risks
    if len(risks) <= 3:
        lines.append(f"   Risks: {'; '.join(risks)}")
    else:
        lines.append(f"   Key Risks: {'; '.join(risks[:2])}")

    return "\n".join(lines)


__all__ = [
    'generate_trade_thesis',
    'calculate_entry_exit_levels',
    'calculate_confidence_score',
    'categorize_by_strategy',
    'get_position_sizing_recommendation',
    'assess_risk_factors',
    'format_trade_plan',
]
