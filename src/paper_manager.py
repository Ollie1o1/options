#!/usr/bin/env python3
"""
Paper Trading Manager for Options Screener.
Handles logging forward tests and updating open positions using SQLite.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

class PaperManager:
    """Manages paper trades stored in a SQLite database."""
    
    def __init__(self, db_path: str = "paper_trades.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self._init_db()

    def _get_connection(self):
        """Returns a new sqlite3 connection."""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates the trades table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            type TEXT,
            entry_price REAL,
            quality_score REAL,
            strategy_name TEXT,
            status TEXT,
            exit_price REAL,
            exit_date TEXT,
            pnl_pct REAL
        )
        """
        with self._get_connection() as conn:
            conn.execute(query)

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
        Logs a new paper trade to the SQLite database.
        Required keys: ticker, expiration, strike, type, entry_price, quality_score, strategy_name
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        query = """
        INSERT INTO trades (
            date, ticker, expiration, strike, type, 
            entry_price, quality_score, strategy_name, 
            status, exit_price, exit_date, pnl_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            trade_dict.get("date", now),
            trade_dict["ticker"].upper(),
            trade_dict["expiration"],
            float(trade_dict["strike"]),
            trade_dict["type"].lower(),
            float(trade_dict["entry_price"]),
            float(trade_dict["quality_score"]),
            trade_dict["strategy_name"],
            "OPEN",
            None, # exit_price
            "",   # exit_date
            None  # pnl_pct
        )
        
        with self._get_connection() as conn:
            conn.execute(query, params)
            
        print(f"Logged {trade_dict['type'].upper()} on {trade_dict['ticker']} at ${float(trade_dict['entry_price']):.2f}")

    def _get_option_symbol(self, ticker: str, expiration: str, strike: float, option_type: str) -> str:
        """Generates a yfinance-compatible option symbol."""
        try:
            exp_date = pd.to_datetime(expiration).strftime('%y%m%d')
            otype = 'C' if option_type.lower() == 'call' else 'P'
            strike_price = f"{int(strike * 1000):08d}"
            return f"{ticker}{exp_date}{otype}{strike_price}"
        except Exception:
            return ""

    def update_positions(self):
        """Updates all OPEN positions using SQLite and checks exit rules."""
        config = self._load_config()
        exit_rules = config.get("exit_rules", {"take_profit": 0.50, "stop_loss": -0.25})
        tp = exit_rules.get("take_profit", 0.50)
        sl = exit_rules.get("stop_loss", -0.25)

        # Fetch open trades
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades WHERE status='OPEN'")
            open_trades = cursor.fetchall()

        if not open_trades:
            print("No open positions to update.")
            return

        print(f"Updating {len(open_trades)} open positions...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for row in open_trades:
            entry_id = row["entry_id"]
            ticker = row["ticker"]
            expiration = row["expiration"]
            strike = row["strike"]
            option_type = row["type"]
            entry_price = row["entry_price"]
            
            symbol = self._get_option_symbol(ticker, expiration, strike, option_type)
            if not symbol:
                continue
                
            try:
                tkr = yf.Ticker(symbol)
                current_price = None
                
                try:
                    current_price = getattr(tkr.fast_info, "last_price", None)
                except Exception:
                    pass
                
                if current_price is None or np.isnan(current_price) or current_price <= 0:
                    hist = tkr.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist["Close"].iloc[-1])
                
                if current_price is None or np.isnan(current_price) or current_price <= 0:
                    info = tkr.info
                    current_price = info.get("regularMarketPrice") or info.get("lastPrice")
                
                if current_price is not None and not np.isnan(current_price) and current_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    hit_tp = pnl_pct >= tp
                    hit_sl = pnl_pct <= sl
                    
                    if hit_tp or hit_sl:
                        reason = "Take Profit" if hit_tp else "Stop Loss"
                        print(f"  - {reason} hit for {symbol}: {pnl_pct:.1%}")
                        
                        update_query = """
                        UPDATE trades 
                        SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=? 
                        WHERE entry_id=?
                        """
                        with self._get_connection() as conn:
                            conn.execute(update_query, (current_price, now, pnl_pct, entry_id))
                    else:
                        print(f"  - {symbol} remains OPEN: {pnl_pct:.1%}")
                else:
                    print(f"  - ⚠️ Could not fetch price for {symbol}")
            except Exception as e:
                print(f"  - ⚠️ Error updating {symbol}: {e}")

    def get_all_trades(self) -> pd.DataFrame:
        """Returns all trades as a pandas DataFrame."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM trades", conn)

    def get_performance_summary(self) -> pd.DataFrame:
        """Returns a summary of trading performance using SQL aggregations."""
        queries = {
            "total_count": "SELECT COUNT(*) FROM trades",
            "closed_count": "SELECT COUNT(*) FROM trades WHERE status='CLOSED'",
            "win_count": "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct > 0",
            "avg_pnl": "SELECT AVG(pnl_pct) FROM trades WHERE status='CLOSED'",
            "sum_pnl": "SELECT SUM(pnl_pct) FROM trades WHERE status='CLOSED'"
        }
        
        results = {}
        with self._get_connection() as conn:
            for key, q in queries.items():
                res = conn.execute(q).fetchone()[0]
                results[key] = res if res is not None else 0

        total_closed = results["closed_count"]
        win_rate = (results["win_count"] / total_closed) if total_closed > 0 else 0
        
        summary = {
            "Total Trades": [results["total_count"]],
            "Closed Trades": [total_closed],
            "Win Rate": [f"{win_rate:.1%}"],
            "Total PnL %": [f"{results['sum_pnl']:.1%}"],
            "Avg Return": [f"{results['avg_pnl']:.1%}"]
        }
        return pd.DataFrame(summary)

if __name__ == "__main__":
    # Test script with temporary database
    test_db = "test_paper_trades.db"
    manager = PaperManager(db_path=test_db)
    
    test_trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-19",
        "strike": 150.0,
        "type": "call",
        "entry_price": 50.0,
        "quality_score": 0.85,
        "strategy_name": "Test Strategy"
    }
    
    manager.log_trade(test_trade)
    manager.update_positions()
    print("\nPerformance Summary:")
    print(manager.get_performance_summary())
    
    # Cleanup test database
    if os.path.exists(test_db):
        os.remove(test_db)
