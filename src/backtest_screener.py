#!/usr/bin/env python3
"""
Backtesting engine for options screener.
Evaluates historical performance of screener picks.
"""

import os
import sys
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import glob

try:
    from .visualize_results import create_backtest_charts
except ImportError:
    create_backtest_charts = None


def load_config() -> Dict:
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: config.json not found at {config_path}, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config.json: {e}, using defaults")
        return {}


def generate_option_symbol(row: pd.Series) -> str:
    """Generates a yfinance-compatible option symbol."""
    exp_date = pd.to_datetime(row['expiration']).strftime('%y%m%d')
    option_type = row['type'][0].upper()
    strike_price = f"{int(row['strike'] * 1000):08d}"
    return f"{row['symbol']}{exp_date}{option_type}{strike_price}"


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


def managed_trade_simulation(log_entries: List[Dict]) -> pd.DataFrame:
    """
    Simulates a managed trade strategy with profit-taking and stop-loss.
    """
    results = []
    today = datetime.now().date()
    config = load_config()
    backtest_config = config.get("backtest", {})

    # Load multipliers from config or use defaults
    profit_mult_long = backtest_config.get("profit_target_multiplier_long", 1.5)
    profit_mult_short = backtest_config.get("profit_target_multiplier_short", 0.5)
    stop_mult_long = backtest_config.get("stop_loss_multiplier_long", 0.5)
    stop_mult_short = backtest_config.get("stop_loss_multiplier_short", 1.5)

    print(f"\nRunning Managed Trade Simulation on {len(log_entries)} log entries...")
    print(f"Using profit/stop multipliers: Long ({profit_mult_long}x/{stop_mult_long}x), Short ({profit_mult_short}x/{stop_mult_short}x)")

    for entry in log_entries:
        picks = entry.get("picks", [])
        for pick in picks:
            try:
                time.sleep(0.5) # Rate limit
                option_symbol = generate_option_symbol(pd.Series(pick))
                entry_date = pd.to_datetime(entry.get("timestamp")).date()
                exp_date = pd.to_datetime(pick.get("expiration")).date()

                if exp_date >= today:
                    print(f"  - Skipping {option_symbol} (not expired)")
                    continue

                hist = yf.download(option_symbol, start=entry_date, end=exp_date, progress=False)
                if hist.empty:
                    print(f"  - No historical data for {option_symbol}")
                    continue

                # --- Realistic Slippage Logic ---
                bid = pick.get("bid")
                ask = pick.get("ask")
                mid = pick.get("premium")  # 'premium' is the mid-price
                entry_price = None

                # Determine trade direction from the context of the log entry
                trade_mode = entry.get("context", {}).get("mode", "Unknown")
                is_short_trade = (trade_mode == "Premium Selling")

                if not is_short_trade: # Long trade (buy to open)
                    # Primary: Use Ask if available
                    if ask and ask > 0:
                        entry_price = ask
                    # Secondary: Use Mid + 10% of spread width
                    elif mid and bid and ask and (ask - bid) > 0:
                        entry_price = mid + (0.10 * (ask - bid))
                    # Tertiary: Use Mid + 1% slippage
                    elif mid:
                        entry_price = mid * 1.01
                else: # Short trade (sell to open)
                    # Primary: Use Bid if available
                    if bid and bid > 0:
                        entry_price = bid
                    # Secondary: Use Mid - 10% of spread width
                    elif mid and bid and ask and (ask - bid) > 0:
                        entry_price = mid - (0.10 * (ask - bid))
                    # Tertiary: Use Mid - 1% slippage
                    elif mid:
                        entry_price = mid * 0.99

                # Quaternary (Final Fallback): Use historical close
                if not entry_price:
                    entry_price = hist.iloc[0]["Close"]

                if not entry_price or entry_price <= 0:
                    print(f"  - Could not determine valid entry price for {option_symbol}")
                    continue

                # Apply profit/stop multipliers from config
                if is_short_trade:
                    profit_target = entry_price * profit_mult_short
                    stop_loss = entry_price * stop_mult_short
                else:
                    profit_target = entry_price * profit_mult_long
                    stop_loss = entry_price * stop_mult_long

                outcome = "EXPIRED"
                for _, day in hist.iterrows():
                    if is_short_trade:
                        if day["Low"] <= profit_target:
                            outcome = "WIN"
                            break
                        if day["High"] >= stop_loss:
                            outcome = "LOSS"
                            break
                    else:
                        if day["High"] >= profit_target:
                            outcome = "WIN"
                            break
                        if day["Low"] <= stop_loss:
                            outcome = "LOSS"
                            break

                pnl_data = {}
                if outcome == "EXPIRED":
                    final_price = fetch_price_at_expiration(pick["symbol"], pick["expiration"])
                    if final_price is not None:
                        pnl_data = calculate_realized_pnl(pick["type"], pick["strike"], entry_price, final_price)
                        if pnl_data.get("pnl_per_contract", 0) > 0:
                            outcome = "WIN"
                        else:
                            outcome = "LOSS"

                elif outcome == "WIN":
                    # Long: bought at entry_price, sold at profit_target (1.5x)
                    # Short: sold at entry_price, bought back at profit_target (0.5x)
                    if is_short_trade:
                        pnl_data = {"pnl_per_contract": (entry_price - profit_target) * 100}
                    else:
                        pnl_data = {"pnl_per_contract": (profit_target - entry_price) * 100}
                elif outcome == "LOSS":
                    # Long: bought at entry_price, forced to sell at stop_loss (0.5x)
                    # Short: sold at entry_price, forced to buy back at stop_loss (1.5x)
                    if is_short_trade:
                        pnl_data = {"pnl_per_contract": (entry_price - stop_loss) * 100}
                    else:
                        pnl_data = {"pnl_per_contract": (stop_loss - entry_price) * 100}

                pick["outcome"] = outcome
                pick.update(pnl_data)
                results.append(pick)
                print(f"  ✓ {option_symbol}: {outcome}")

            except Exception as e:
                print(f"  - Skipping {pick.get('symbol')}: {e}")
                continue

    return pd.DataFrame(results)


def fetch_price_at_expiration(symbol: str, expiration_date: str) -> Optional[float]:
    """Fetch the closing price on or near expiration date."""
    try:
        exp_dt = pd.to_datetime(expiration_date).date()
        ticker = yf.Ticker(symbol)
        
        start_date = exp_dt - timedelta(days=5)
        end_date = exp_dt + timedelta(days=5)
        
        hist = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist.empty:
            return None
        
        if exp_dt in hist.index.date:
            return float(hist.loc[hist.index.date == exp_dt, "Close"].iloc[0])
        else:
            closest_idx = min(hist.index, key=lambda d: abs((d.date() - exp_dt).days))
            return float(hist.loc[closest_idx, "Close"])
    
    except Exception:
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
    if df.empty or "outcome" not in df.columns:
        return pd.DataFrame()
    
    summary = {
        "Total Trades": len(df),
        "Winners": len(df[df["outcome"] == "WIN"]),
        "Losers": len(df[df["outcome"] == "LOSS"]),
        "Expired": len(df[df["outcome"] == "EXPIRED"]),
        "Win Rate %": f"{(len(df[df['outcome'] == 'WIN']) / len(df) * 100):.1f}",
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
    print(f"Loading logs from: {logs_dir}")
    if not os.path.exists(logs_dir):
        print("\nNo logs directory found. Run the screener first to generate logs.")
        sys.exit(1)
    
    log_entries = load_historical_logs(logs_dir)
    
    if not log_entries:
        print("No historical log entries found.")
        sys.exit(0)
    
    print(f"Loaded {len(log_entries)} historical entries from logs.")
    
    # Backtest
    results_df = managed_trade_simulation(log_entries)
    
    if results_df.empty:
        print("\nNo expired contracts found to backtest.")
        sys.exit(0)
    
    print(f"\n{'=' * 80}")
    print("  BACKTEST RESULTS")
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
