#!/usr/bin/env python3
"""
Paper Trading Manager for Options Screener.
Handles logging forward tests and updating open positions.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

class PaperManager:
    """Manages paper trades stored in a CSV file."""
    
    def __init__(self, csv_path: str = "paper_trades.csv", config_path: str = "config.json"):
        self.csv_path = csv_path
        self.config_path = config_path
        self.columns = [
            "date", "ticker", "expiration", "strike", "type", 
            "entry_price", "quality_score", "strategy_name", 
            "status", "exit_price", "exit_date", "pnl_pct"
        ]
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Creates the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_path, index=False)

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration for exit rules."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "exit_rules": {
                    "take_profit": 0.50,
                    "stop_loss": -0.25
                }
            }

    def log_trade(self, trade_dict: Dict[str, Any]):
        """
        Logs a new paper trade.
        Required keys: ticker, expiration, strike, type, entry_price, quality_score, strategy_name
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "date": trade_dict.get("date", now),
            "ticker": trade_dict["ticker"].upper(),
            "expiration": trade_dict["expiration"],
            "strike": float(trade_dict["strike"]),
            "type": trade_dict["type"].lower(),
            "entry_price": float(trade_dict["entry_price"]),
            "quality_score": float(trade_dict["quality_score"]),
            "strategy_name": trade_dict["strategy_name"],
            "status": "OPEN",
            "exit_price": np.nan,
            "exit_date": "",
            "pnl_pct": np.nan
        }
        
        df = pd.read_csv(self.csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        print(f"Logged {new_row['type'].upper()} on {new_row['ticker']} at ${new_row['entry_price']:.2f}")

    def _get_option_symbol(self, ticker: str, expiration: str, strike: float, option_type: str) -> str:
        """Generates a yfinance-compatible option symbol."""
        # Convert 2026-06-19 to 260619
        exp_date = pd.to_datetime(expiration).strftime('%y%m%d')
        # Type: C or P
        otype = 'C' if option_type.lower() == 'call' else 'P'
        # Strike: 8 digits (5 for integer part, 3 for decimals)
        strike_price = f"{int(strike * 1000):08d}"
        return f"{ticker}{exp_date}{otype}{strike_price}"

    def update_positions(self):
        """Updates all OPEN positions and checks exit rules."""
        df = pd.read_csv(self.csv_path)
        if df.empty:
            print("No paper trades found.")
            return

        open_mask = df["status"] == "OPEN"
        if not open_mask.any():
            print("No open positions to update.")
            return

        config = self._load_config()
        exit_rules = config.get("exit_rules", {"take_profit": 0.50, "stop_loss": -0.25})
        tp = exit_rules.get("take_profit", 0.50)
        sl = exit_rules.get("stop_loss", -0.25)

        print(f"Updating {open_mask.sum()} open positions...")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, row in df[open_mask].iterrows():
            symbol = self._get_option_symbol(row["ticker"], row["expiration"], row["strike"], row["type"])
            try:
                tkr = yf.Ticker(symbol)
                current_price = None
                
                # Prioritize fast_info for speed and reliability on option tickers
                try:
                    current_price = getattr(tkr.fast_info, "last_price", None)
                except Exception:
                    pass
                
                if current_price is None or np.isnan(current_price) or current_price <= 0:
                    # Fallback 1: history
                    hist = tkr.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist["Close"].iloc[-1])
                
                if current_price is None or np.isnan(current_price) or current_price <= 0:
                    # Fallback 2: info (slowest)
                    info = tkr.info
                    current_price = info.get("regularMarketPrice") or info.get("lastPrice")
                
                if current_price is not None and not np.isnan(current_price) and current_price > 0:
                    entry_price = float(row["entry_price"])
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    hit_tp = pnl_pct >= tp
                    hit_sl = pnl_pct <= sl
                    
                    if hit_tp or hit_sl:
                        reason = "Take Profit" if hit_tp else "Stop Loss"
                        print(f"  - {reason} hit for {symbol}: {pnl_pct:.1%}")
                        df.at[idx, "status"] = "CLOSED"
                        df.at[idx, "exit_price"] = current_price
                        df.at[idx, "exit_date"] = now
                        df.at[idx, "pnl_pct"] = pnl_pct
                    else:
                        print(f"  - {symbol} remains OPEN: {pnl_pct:.1%}")
                else:
                    print(f"  - ⚠️ Could not fetch price for {symbol}")
            except Exception as e:
                print(f"  - ⚠️ Error updating {symbol}: {e}")

        df.to_csv(self.csv_path, index=False)

    def get_performance_summary(self) -> pd.DataFrame:
        """Returns a summary of trading performance."""
        df = pd.read_csv(self.csv_path)
        if df.empty:
            return pd.DataFrame()

        closed_df = df[df["status"] == "CLOSED"].copy()
        if closed_df.empty:
            return pd.DataFrame({
                "Total Trades": [len(df)],
                "Closed Trades": [0],
                "Win Rate": ["0%"],
                "Total PnL %": ["0%"],
                "Avg Return": ["0%"]
            })

        wins = (closed_df["pnl_pct"] > 0).sum()
        total_closed = len(closed_df)
        win_rate = wins / total_closed if total_closed > 0 else 0
        total_pnl = closed_df["pnl_pct"].sum()
        avg_return = closed_df["pnl_pct"].mean()

        summary = {
            "Total Trades": [len(df)],
            "Closed Trades": [total_closed],
            "Win Rate": [f"{win_rate:.1%}"],
            "Total PnL %": [f"{total_pnl:.1%}"],
            "Avg Return": [f"{avg_return:.1%}"]
        }
        return pd.DataFrame(summary)

if __name__ == "__main__":
    # Test script
    manager = PaperManager(csv_path="test_paper_trades.csv")
    
    # Mock trade (using a likely real symbol for testing price fetch)
    # yfinance often needs specific strike format, e.g. AAPL260619C00150000
    # Let's use a strike that likely exists
    test_trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-19",
        "strike": 150.0,
        "type": "call",
        "entry_price": 50.0, # High entry so we don't accidentally hit TP/SL if live
        "quality_score": 0.85,
        "strategy_name": "Test Strategy"
    }
    
    manager.log_trade(test_trade)
    manager.update_positions()
    print("\nPerformance Summary:")
    print(manager.get_performance_summary())
    
    # Cleanup test file
    if os.path.exists("test_paper_trades.csv"):
        os.remove("test_paper_trades.csv")