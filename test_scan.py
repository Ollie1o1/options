
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.options_screener import run_scan, setup_logging

def test_scan():
    print("Testing Scan with Default Filters...")
    logger = setup_logging()
    
    # Test with a known liquid ticker
    tickers = ["SPY", "AAPL"]
    
    results = run_scan(
        mode="Discovery scan",
        tickers=tickers,
        budget=None,
        max_expiries=3,
        min_dte=7,
        max_dte=45,
        trader_profile="swing",
        logger=logger,
        market_trend="Bullish",
        volatility_regime="Normal",
        verbose=True
    )
    
    picks = results.get('picks', None)
    if picks is not None and not picks.empty:
        print(f"\nSUCCESS: Found {len(picks)} picks.")
        print(picks[['symbol', 'strike', 'type', 'quality_score', 'spread_pct', 'rr_ratio']].head())
    else:
        print("\nFAILURE: No picks found. Filters might be too strict.")

if __name__ == "__main__":
    test_scan()
