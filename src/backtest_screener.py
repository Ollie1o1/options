#!/usr/bin/env python3
"""
Backtesting engine for options screener.
Evaluates historical performance of screener picks.
"""

import os
import sys
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import glob

try:
    from .visualize_results import create_backtest_charts
except ImportError:
    create_backtest_charts = None


def load_historical_logs(logs_dir: str = "logs") -> List[Dict]:
    """Load all JSONL log files from logs directory."""
    log_files = glob.glob(os.path.join(logs_dir, "run_*.jsonl"))
    
    if not log_files:
        print(f"No log files found in {logs_dir}/")
        return []
    
    all_entries = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    all_entries.append(entry)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return all_entries


def fetch_price_at_expiration(symbol: str, expiration_date: str) -> Optional[float]:
    """Fetch the closing price on or near expiration date."""
    try:
        exp_dt = pd.to_datetime(expiration_date).date()
        ticker = yf.Ticker(symbol)
        
        # Fetch a window around expiration
        start_date = exp_dt - timedelta(days=5)
        end_date = exp_dt + timedelta(days=5)
        
        hist = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist.empty:
            return None
        
        # Try to get exact expiration date, or closest available
        if exp_dt in hist.index.date:
            return float(hist.loc[hist.index.date == exp_dt, "Close"].iloc[0])
        else:
            # Get closest date
            hist["date_diff"] = abs((hist.index.date - exp_dt).days)
            closest_idx = hist["date_diff"].idxmin()
            return float(hist.loc[closest_idx, "Close"])
    
    except Exception as e:
        return None


def calculate_realized_pnl(
    option_type: str,
    strike: float,
    entry_premium: float,
    final_price: float
) -> Dict:
    """
    Calculate realized P/L for an option held to expiration.
    
    Returns:
        Dict with 'pnl_per_share', 'pnl_per_contract', 'pnl_pct'
    """
    try:
        option_type = option_type.lower()
        
        # Calculate intrinsic value at expiration
        if option_type == "call":
            intrinsic_value = max(0, final_price - strike)
        else:  # put
            intrinsic_value = max(0, strike - final_price)
        
        # P/L per share = intrinsic value - premium paid
        pnl_per_share = intrinsic_value - entry_premium
        
        # P/L per contract (multiplier of 100)
        pnl_per_contract = pnl_per_share * 100
        
        # P/L percentage (relative to initial investment)
        pnl_pct = (pnl_per_share / entry_premium) * 100 if entry_premium > 0 else 0
        
        return {
            "pnl_per_share": pnl_per_share,
            "pnl_per_contract": pnl_per_contract,
            "pnl_pct": pnl_pct,
            "final_intrinsic": intrinsic_value
        }
    except Exception:
        return {
            "pnl_per_share": None,
            "pnl_per_contract": None,
            "pnl_pct": None,
            "final_intrinsic": None
        }


def backtest_entries(log_entries: List[Dict]) -> pd.DataFrame:
    """
    Process log entries and calculate realized P/L for expired contracts.
    
    Returns:
        DataFrame with backtest results
    """
    results = []
    
    print(f"\nProcessing {len(log_entries)} log entries...")
    
    for entry in log_entries:
        picks = entry.get("picks", [])
        context = entry.get("context", {})
        timestamp = entry.get("timestamp", "")
        
        for pick in picks:
            exp_date = pick.get("expiration")
            symbol = pick.get("symbol")
            
            if not exp_date or not symbol:
                continue
            
            # Check if option has expired
            exp_dt = pd.to_datetime(exp_date).date()
            if exp_dt > datetime.now().date():
                continue  # Not yet expired
            
            # Fetch price at expiration
            final_price = fetch_price_at_expiration(symbol, exp_date)
            
            if final_price is None:
                continue
            
            # Calculate P/L
            pnl_data = calculate_realized_pnl(
                pick.get("type", "call"),
                pick.get("strike", 0),
                pick.get("premium", 0),
                final_price
            )
            
            # Compile result
            result = {
                "timestamp": timestamp,
                "symbol": symbol,
                "type": pick.get("type"),
                "strike": pick.get("strike"),
                "expiration": exp_date,
                "entry_premium": pick.get("premium"),
                "entry_underlying": pick.get("underlying"),
                "final_underlying": final_price,
                "delta": pick.get("delta"),
                "iv": pick.get("impliedVolatility"),
                "hv": pick.get("hv_30d"),
                "prob_profit": pick.get("prob_profit"),
                "p_itm": pick.get("p_itm"),
                "rr_ratio": pick.get("rr_ratio"),
                "theo_value": pick.get("theo_value"),
                "ev_per_contract": pick.get("ev_per_contract"),
                "quality_score": pick.get("quality_score"),
                "mode": context.get("mode"),
                **pnl_data
            }
            
            results.append(result)
            print(f"  ✓ {symbol} {pick.get('type')} ${pick.get('strike')} exp {exp_date}: P/L ${pnl_data.get('pnl_per_contract', 0):.2f}")
    
    return pd.DataFrame(results)


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between metrics and realized P/L."""
    if df.empty or "pnl_per_contract" not in df.columns:
        return pd.DataFrame()
    
    metrics = [
        "prob_profit", "p_itm", "rr_ratio", "ev_per_contract",
        "quality_score", "delta", "iv", "hv"
    ]
    
    correlations = {}
    for metric in metrics:
        if metric in df.columns:
            corr = df[[metric, "pnl_per_contract"]].corr().iloc[0, 1]
            correlations[metric] = corr
    
    corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation"])
    corr_df = corr_df.sort_values("correlation", ascending=False)
    
    return corr_df


def generate_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate performance statistics."""
    if df.empty or "pnl_per_contract" not in df.columns:
        return pd.DataFrame()
    
    summary = {
        "Total Trades": len(df),
        "Winners": len(df[df["pnl_per_contract"] > 0]),
        "Losers": len(df[df["pnl_per_contract"] <= 0]),
        "Win Rate %": f"{(len(df[df['pnl_per_contract'] > 0]) / len(df) * 100):.1f}",
        "Avg P/L": f"${df['pnl_per_contract'].mean():.2f}",
        "Median P/L": f"${df['pnl_per_contract'].median():.2f}",
        "Total P/L": f"${df['pnl_per_contract'].sum():.2f}",
        "Best Trade": f"${df['pnl_per_contract'].max():.2f}",
        "Worst Trade": f"${df['pnl_per_contract'].min():.2f}"
    }
    
    return pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])


def main():
    print("=" * 80)
    print("  OPTIONS SCREENER BACKTESTING ENGINE")
    print("=" * 80)
    
    # Load historical logs
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"\nNo logs directory found. Run the screener first to generate logs.")
        sys.exit(1)
    
    log_entries = load_historical_logs(logs_dir)
    
    if not log_entries:
        print("No historical log entries found.")
        sys.exit(0)
    
    print(f"Loaded {len(log_entries)} historical entries from logs.")
    
    # Backtest
    results_df = backtest_entries(log_entries)
    
    if results_df.empty:
        print("\nNo expired contracts found to backtest.")
        sys.exit(0)
    
    print(f"\n{'=' * 80}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 80}\n")
    
    # Performance summary
    performance = generate_performance_summary(results_df)
    print(performance)
    
    # Metric correlations
    print(f"\n{'=' * 80}")
    print("  METRIC CORRELATIONS WITH REALIZED P/L")
    print(f"{'=' * 80}\n")
    correlations = calculate_correlations(results_df)
    print(correlations)
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = f"reports/backtest_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n  📄 Detailed results saved to: {results_file}")
    
    corr_file = f"reports/metric_correlations_{timestamp}.csv"
    correlations.to_csv(corr_file)
    print(f"  📄 Correlations saved to: {corr_file}")
    
    # Generate charts
    if create_backtest_charts:
        create_backtest_charts(correlations, performance, output_dir="reports")
    
    print(f"\n{'=' * 80}")
    print("  Backtest complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
