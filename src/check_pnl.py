#!/usr/bin/env python3
"""
Check Unrealized P/L for Active Options Trades
Reads from trades_log/entries.csv and fetches current market prices.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional


def generate_occ_symbol(symbol: str, expiration: str, option_type: str, strike: float) -> str:
    """
    Generate OCC-compliant option symbol for yfinance.
    Format: TICKER + YYMMDD + C/P + STRIKE (8 digits, padded)
    
    Example: AAPL251219C00220000
    """
    try:
        # Parse expiration date
        exp_date = pd.to_datetime(expiration)
        date_str = exp_date.strftime('%y%m%d')
        
        # Format option type
        opt_type = 'C' if option_type.lower() == 'call' else 'P'
        
        # Format strike (multiply by 1000 and pad to 8 digits)
        strike_int = int(float(strike) * 1000)
        strike_str = f"{strike_int:08d}"
        
        # Combine
        occ_symbol = f"{symbol.upper()}{date_str}{opt_type}{strike_str}"
        return occ_symbol
    except Exception as e:
        print(f"  ⚠️  Error generating OCC symbol for {symbol}: {e}")
        return None


def get_current_price(occ_symbol: str) -> Optional[float]:
    """
    Fetch current market price for an option using yfinance.
    """
    try:
        ticker = yf.Ticker(occ_symbol)
        
        # Try fast_info first
        try:
            fast_info = getattr(ticker, 'fast_info', None)
            if fast_info:
                price = getattr(fast_info, 'last_price', None)
                if price and price > 0:
                    return float(price)
        except Exception:
            pass
        
        # Fallback to history
        hist = ticker.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        return None
    except Exception as e:
        return None


def view_portfolio():
    log_file = "trades_log/entries.csv"
    
    # Check if file exists
    if not os.path.exists(log_file):
        print("❌ No trade log found at 'trades_log/entries.csv'")
        print("   Run the screener first to log some trades.")
        return
    
    # Load trades
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"❌ Error reading trade log: {e}")
        return
    
    # Filter for open trades
    open_trades = df[df['status'] == 'OPEN'].copy()
    
    if open_trades.empty:
        print("✓ No open trades found.")
        print("  All positions are closed or no trades have been logged yet.")
        # Still show activity summary even if no open trades
    
    # === TRADE ACTIVITY SUMMARY ===
    print("\n" + "=" * 80)
    print("  TRADE ACTIVITY SUMMARY")
    print("=" * 80)
    
    # Extract date from timestamp
    if 'timestamp' in df.columns:
        # Handle potential parsing errors gracefully
        try:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size().sort_index(ascending=False)
            
            print(f"  {'Date':<15} {'Trades Taken':<15}")
            print("  " + "-" * 30)
            
            for date, count in daily_counts.items():
                print(f"  {str(date):<15} {count:<15}")
                
        except Exception as e:
            print(f"  ⚠️  Could not parse dates for summary: {e}")
    else:
        print("  ⚠️  Timestamp column missing in log.")
        
    print("=" * 80 + "\n")

    if open_trades.empty:
        return

    print("=" * 120)
    print("  UNREALIZED P/L TRACKER - Active Options Positions")
    print("=" * 120)
    print(f"\n  Found {len(open_trades)} open position(s)\n")
    
    # Table header
    print(f"  {'Symbol':<10} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Entry $':<10} {'Current $':<10} {'P/L $':<12} {'P/L %':<10}")
    print("  " + "-" * 118)
    
    total_pnl = 0.0
    successful_fetches = 0
    
    for idx, trade in open_trades.iterrows():
        symbol = trade['symbol']
        exp = trade['expiration']
        opt_type = trade['type']
        strike = float(trade['strike'])
        entry_premium = float(trade['entry_premium'])
        mode = trade.get('mode', '')
        
        # Generate OCC symbol
        occ_symbol = generate_occ_symbol(symbol, exp, opt_type, strike)
        if not occ_symbol:
            print(f"  {symbol:<10} {opt_type.upper():<5} {strike:<8.2f} {exp:<12} {'N/A':<10} {'N/A':<10} {'ERROR':<12} {'N/A':<10}")
            continue
        
        # Fetch current price
        current_price = get_current_price(occ_symbol)
        
        if current_price is None:
            print(f"  {symbol:<10} {opt_type.upper():<5} {strike:<8.2f} {exp:<12} ${entry_premium:<9.2f} {'N/A':<10} {'ERROR':<12} {'N/A':<10}")
            continue
        
        # Calculate P/L
        # For Long positions: profit = (current - entry) * 100
        # For Premium Selling (short): profit = (entry - current) * 100
        if 'premium selling' in mode.lower() or 'credit' in mode.lower():
            # Short position
            pnl_per_share = entry_premium - current_price
        else:
            # Long position (default)
            pnl_per_share = current_price - entry_premium
        
        pnl_dollars = pnl_per_share * 100  # Per contract
        pnl_percent = (pnl_per_share / entry_premium) * 100 if entry_premium > 0 else 0.0
        
        total_pnl += pnl_dollars
        successful_fetches += 1
        
        # Format P/L with color indicators
        pnl_sign = "+" if pnl_dollars >= 0 else ""
        pct_sign = "+" if pnl_percent >= 0 else ""
        
        print(f"  {symbol:<10} {opt_type.upper():<5} {strike:<8.2f} {exp:<12} "
              f"${entry_premium:<9.2f} ${current_price:<9.2f} "
              f"{pnl_sign}${pnl_dollars:<11.2f} {pct_sign}{pnl_percent:<9.1f}%")
    
    # Summary
    print("  " + "-" * 118)
    total_sign = "+" if total_pnl >= 0 else ""
    status_emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"\n  {status_emoji} TOTAL UNREALIZED P/L: {total_sign}${total_pnl:.2f}")
    print(f"  📊 Successfully fetched {successful_fetches}/{len(open_trades)} positions")
    print("\n" + "=" * 120)


if __name__ == "__main__":
    view_portfolio()
