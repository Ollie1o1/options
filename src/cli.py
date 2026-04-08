#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import csv
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
from .scanner import run_scan, load_config

from .types import ScanResult
from .data_fetching import get_market_context, get_risk_free_rate, get_vix_level, determine_vix_regime, get_dynamic_tickers
from .paper_manager import PaperManager
from .watchlist import load_watchlist, add_to_watchlist, remove_from_watchlist
from .cli_display import get_display_width, print_report, print_news_panel, print_executive_summary

try:
    from . import formatting as fmt
    HAS_ENHANCED_CLI = True
except ImportError:
    HAS_ENHANCED_CLI = False

def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("options_screener")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"logs/scan_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger

def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    p_str = f"  {prompt} [{default}]: " if default else f"  {prompt}: "
    val = input(p_str).strip()
    return val if val else (default if default else "")

def prompt_for_tickers() -> List[str]:
    print("\nSelect Ticker Source:")
    print("  1. Curated Liquid (default)")
    print("  2. Top Gainers (Finviz)")
    print("  3. High IV Stocks (Finviz)")
    choice = prompt_input("Enter 1, 2, or 3", "1")
    if choice == "1":
        return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]
    else:
        stype = "gainers" if choice == "2" else "high_iv"
        return get_dynamic_tickers(stype, max_tickers=50)

def export_to_csv(df: pd.DataFrame, mode: str, budget: Optional[float] = None) -> str:
    os.makedirs("exports", exist_ok=True)
    fname = f"exports/scan_{mode.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    return fname

def log_trade_entry(df: pd.DataFrame, mode: str) -> None:
    os.makedirs("trades_log", exist_ok=True)
    fname = "trades_log/trades.csv"
    df.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

def close_trades():
    print("Feature coming soon: Closing trades update.")

def _check_market_hours() -> tuple:
    return True, "Market status check placeholder"

def _run_ai_pipeline(picks, regime, verbose=True, sector_ctx=None):
    print("AI Ranking skipped (modularized).")
    return picks

def run_top_scan(tickers, top_n=10, **kwargs):
    print(f"Running top-{top_n} scan (modularized).")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--ticker", type=str)
    args, _ = parser.parse_known_args()

    if args.help:
        print("Modularized Options Screener CLI")
        sys.exit(0)

    config = load_config()
    logger = setup_logging()
    
    print("\n" + "="*60)
    print("  OPTIONS SCREENER (Modularized)")
    print("="*60)

    ticker = args.ticker.upper() if args.ticker else prompt_input("Enter ticker (or DISCOVER)", "DISCOVER").upper()
    
    if ticker == "DISCOVER":
        tickers = prompt_for_tickers()
        mode = "Discovery scan"
    else:
        tickers = [ticker]
        mode = "Single-stock"

    print(f"Scanning {len(tickers)} tickers...")
    m_trend, v_regime, m_risk, tnx = get_market_context()
    
    results = run_scan(mode, tickers, None, 4, 7, 45, "swing", logger, m_trend, v_regime, macro_risk_active=m_risk, tnx_change_pct=tnx)
    
    if not results.picks.empty:
        print_report(results.picks.head(10), results.underlying_price, results.rfr, 4, 7, 45, mode=mode, config=config)
    else:
        print("No picks found.")

if __name__ == "__main__":
    main()
