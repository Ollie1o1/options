#!/usr/bin/env python3
"""
Streamlit Dashboard for Options Screener
A professional web UI for the options screener backend.
"""

import sys
import os
from pathlib import Path

# Add parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

# Import from the options screener module
from src.options_screener import run_scan, load_config, get_market_context, setup_logging, prompt_for_tickers

# Configure page
st.set_page_config(
    page_title="Options Screener Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for market context and scan results
if 'market_context_loaded' not in st.session_state:
    st.session_state.market_context_loaded = False
    st.session_state.market_trend = "Unknown"
    st.session_state.volatility_regime = "Unknown"
    st.session_state.macro_risk_active = False
    st.session_state.tnx_change_pct = 0.0

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# Load market context immediately on app start
if not st.session_state.market_context_loaded:
    with st.spinner("Loading market context..."):
        try:
            market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
            st.session_state.market_trend = market_trend
            st.session_state.volatility_regime = volatility_regime
            st.session_state.macro_risk_active = macro_risk_active
            st.session_state.tnx_change_pct = tnx_change_pct
            st.session_state.market_context_loaded = True
        except Exception as e:
            st.error(f"Could not load market context: {e}")

# Header
st.markdown('<div class="main-header">:chart_with_upwards_trend: Options Screener Dashboard</div>', unsafe_allow_html=True)

# Market Context Header (Always Visible)
st.markdown("### :earth_americas: Market Context")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("SPY Trend", st.session_state.market_trend)

with col2:
    st.metric("Volatility Regime", st.session_state.volatility_regime)

with col3:
    macro_status = ":warning: Risk Detected" if st.session_state.macro_risk_active else ":white_check_mark: Normal"
    st.metric("Macro Risk", macro_status)

with col4:
    yield_status = f"+{st.session_state.tnx_change_pct:.1%}" if st.session_state.tnx_change_pct > 0.025 else "Normal"
    st.metric("10Y Yield", yield_status)

st.divider()

# Sidebar Configuration
st.sidebar.header(":gear: Configuration")

# 1. Scan Mode Dropdown
scan_mode = st.sidebar.selectbox(
    "Scan Mode",
    ["Single-stock", "Discovery scan", "Budget scan", "Premium Selling", "Credit Spreads", "Iron Condor"],
    index=1  # Default to Discovery scan
)

# 2. Dynamic Inputs based on mode
if scan_mode == "Single-stock":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    tickers = [ticker] if ticker else []
elif scan_mode == "Budget scan":
    budget = st.sidebar.number_input("Budget (USD per contract)", min_value=50.0, max_value=10000.0, value=500.0, step=50.0)
    num_tickers = st.sidebar.number_input("Number of tickers to scan", min_value=1, max_value=100, value=20)
    tickers = []  # Will be populated programmatically
else:
    # Discovery, Premium Selling, Credit Spreads, Iron Condor
    num_tickers = st.sidebar.number_input("Number of tickers to scan", min_value=1, max_value=100, value=20)
    tickers = []  # Will be populated programmatically

# 3. Filter Controls
st.sidebar.subheader(":dart: Filters")
min_dte = st.sidebar.slider("Minimum DTE", min_value=1, max_value=365, value=7)
max_dte = st.sidebar.slider("Maximum DTE", min_value=1, max_value=365, value=60)
max_expiries = st.sidebar.slider("Max Expirations", min_value=1, max_value=10, value=3)

# 4. Advanced Settings (Expandable)
with st.sidebar.expander(":wrench: Advanced Settings"):
    trader_profile = st.selectbox("Trader Profile", ["scalp", "swing", "long-term"], index=1)
    
    st.markdown("**Scoring Weights**")
    st.caption("Adjust weights to customize the quality score calculation")
    
    # Load default weights from config
    config = load_config("config.json")
    default_weights = config.get('composite_weights', {})
    
    # Weight sliders matching config.json keys
    weight_pop = st.slider("PoP (Probability of Profit)", 0.0, 1.0, default_weights.get('pop', 0.18), 0.01)
    weight_em_realism = st.slider("EM Realism", 0.0, 1.0, default_weights.get('em_realism', 0.12), 0.01)
    weight_rr = st.slider("Risk/Reward Ratio", 0.0, 1.0, default_weights.get('rr', 0.15), 0.01)
    weight_momentum = st.slider("Momentum", 0.0, 1.0, default_weights.get('momentum', 0.10), 0.01)
    weight_liquidity = st.slider("Liquidity", 0.0, 1.0, default_weights.get('liquidity', 0.15), 0.01)
    weight_catalyst = st.slider("Catalyst", 0.0, 1.0, default_weights.get('catalyst', 0.05), 0.01)
    weight_theta = st.slider("Theta", 0.0, 1.0, default_weights.get('theta', 0.10), 0.01)
    weight_ev = st.slider("Expected Value", 0.0, 1.0, default_weights.get('ev', 0.05), 0.01)
    weight_trader_pref = st.slider("Trader Preference", 0.0, 1.0, default_weights.get('trader_pref', 0.10), 0.01)
    
    # Build custom_weights dict
    custom_weights = {
        'pop': weight_pop,
        'em_realism': weight_em_realism,
        'rr': weight_rr,
        'momentum': weight_momentum,
        'liquidity': weight_liquidity,
        'catalyst': weight_catalyst,
        'theta': weight_theta,
        'ev': weight_ev,
        'trader_pref': weight_trader_pref
    }

# Main Area - Run Scan Button
st.sidebar.divider()
run_button = st.sidebar.button(":rocket: Run Scan", type="primary", use_container_width=True)

if run_button:
    # Populate tickers if needed
    if scan_mode != "Single-stock":
        with st.spinner("Fetching tickers..."):
            # Use curated liquid tickers
            tickers = [
                # Major Indices & ETFs
                "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "GLD", "SLV", "TLT",
                # Mega Cap Tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
                # Financial
                "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
                # Healthcare & Pharma
                "JNJ", "UNH", "PFE", "ABBV", "MRK",
                # Consumer & Retail
                "WMT", "HD", "DIS", "NKE", "MCD", "COST",
                # Energy
                "XOM", "CVX", "COP",
                # Industrial
                "BA", "CAT", "GE"
            ][:num_tickers]
    
    # Setup logger
    logger = setup_logging()
    
    # Determine budget for budget mode
    budget_value = budget if scan_mode == "Budget scan" else None
    
    # Run scan with verbose=False and custom_weights
    with st.spinner(f"Running {scan_mode} scan on {len(tickers)} ticker(s)..."):
        try:
            results = run_scan(
                mode=scan_mode,
                tickers=tickers,
                budget=budget_value,
                max_expiries=max_expiries,
                min_dte=min_dte,
                max_dte=max_dte,
                trader_profile=trader_profile,
                logger=logger,
                market_trend=st.session_state.market_trend,
                volatility_regime=st.session_state.volatility_regime,
                macro_risk_active=st.session_state.macro_risk_active,
                tnx_change_pct=st.session_state.tnx_change_pct,
                verbose=False,  # Suppress console output
                custom_weights=custom_weights  # Pass custom weights
            )
            st.session_state.scan_results = results
            st.success(f":white_check_mark: Scan complete! Found {len(results.get('picks', pd.DataFrame()))} options.")
        except Exception as e:
            st.error(f"Error during scan: {e}")
            st.session_state.scan_results = None

# Display Results
if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    
    # Results Header
    st.markdown("## :bar_chart: Scan Results")
    
    # Display scan timestamp and underlying price
    col1, col2 = st.columns(2)
    with col1:
        st.info(f":alarm_clock: Scan Time: {results.get('timestamp', 'N/A')}")
    with col2:
        if results.get('underlying_price', 0.0) > 0:
            st.info(f":dollar: Underlying Price: ${results['underlying_price']:.2f}")
    
    # Analysis Tabs
    tab1, tab2, tab3 = st.tabs([":bar_chart: Results", ":mag: Dynamic Filtering", ":briefcase: Portfolio Manager"])
    
    with tab1:
        st.markdown("### Options Picks")
        
        picks_df = results.get('picks', pd.DataFrame())
        if not picks_df.empty:
            # Select relevant columns for display
            display_cols = ['symbol', 'type', 'strike', 'expiration', 'premium', 'underlying', 
                          'impliedVolatility', 'delta', 'prob_profit', 'quality_score', 
                          'volume', 'openInterest', 'rr_ratio']
            # Filter to existing columns
            display_cols = [c for c in display_cols if c in picks_df.columns]
            
            # Format the dataframe
            display_df = picks_df[display_cols].copy()
            
            # Format percentage columns
            if 'impliedVolatility' in display_df.columns:
                display_df['IV'] = display_df['impliedVolatility'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            if 'prob_profit' in display_df.columns:
                display_df['PoP'] = display_df['prob_profit'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            if 'delta' in display_df.columns:
                display_df['Delta'] = display_df['delta'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            if 'premium' in display_df.columns:
                display_df['Premium'] = display_df['premium'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            if 'quality_score' in display_df.columns:
                display_df['Quality'] = display_df['quality_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            
            # Display interactive dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500
            )
            
            # Download button
            csv = picks_df.to_csv(index=False)
            st.download_button(
                label=":floppy_disk: Download Results CSV",
                data=csv,
                file_name=f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No options found matching criteria.")
    
    with tab2:
        st.markdown("### Post-Scan Filtering")
        st.caption("Filter results without re-running the scan")
        
        picks_df = results.get('picks', pd.DataFrame())
        if not picks_df.empty and 'quality_score' in picks_df.columns:
            min_score = st.slider(
                "Minimum Quality Score",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0
            )
            
            filtered_df = picks_df[picks_df['quality_score'] >= min_score].copy()
            st.metric("Filtered Results", len(filtered_df), delta=f"{len(filtered_df) - len(picks_df)}")
            
            if not filtered_df.empty:
                # Same display as Tab 1
                display_cols = ['symbol', 'type', 'strike', 'expiration', 'premium', 'underlying', 
                              'impliedVolatility', 'delta', 'prob_profit', 'quality_score']
                display_cols = [c for c in display_cols if c in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[display_cols],
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No results match the filter criteria.")
        else:
            st.info("No results available for filtering.")
    
    with tab3:
        st.markdown("### Portfolio Manager")
        st.caption("View and edit your trade log")
        
        trades_log_path = "trades_log/entries.csv"
        if os.path.exists(trades_log_path):
            try:
                trades_df = pd.read_csv(trades_log_path)
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
                    st.metric("Open Positions", open_trades)
                with col2:
                    closed_trades = len(trades_df[trades_df['status'] == 'CLOSED'])
                    st.metric("Closed Positions", closed_trades)
                with col3:
                    if 'realized_pnl' in trades_df.columns:
                        total_pnl = trades_df['realized_pnl'].sum()
                        st.metric("Total Realized P/L", f"${total_pnl:.2f}")
                
                st.divider()
                
                # Editable dataframe
                st.markdown("**Edit Trade Log**")
                edited_df = st.data_editor(
                    trades_df,
                    use_container_width=True,
                    height=400,
                    num_rows="dynamic"
                )
                
                # Save changes button
                if st.button(":floppy_disk: Save Changes", type="primary"):
                    try:
                        edited_df.to_csv(trades_log_path, index=False)
                        st.success(":white_check_mark: Changes saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving changes: {e}")
                
            except Exception as e:
                st.error(f"Error loading trade log: {e}")
        else:
            st.info("No trade log found. Run a scan and log some trades first.")
else:
    # Welcome message when no scan has been run
    st.markdown("""
    ## :wave: Welcome to the Options Screener Dashboard
    
    Configure your scan parameters in the sidebar and click **:rocket: Run Scan** to get started.
    
    ### Features:
    - **Market Context**: View real-time SPY trend, VIX regime, and macro risk indicators
    - **Multiple Scan Modes**: Single-stock, Discovery, Budget, Premium Selling, Credit Spreads, Iron Condors
    - **Custom Weights**: Fine-tune the scoring algorithm with advanced weight sliders
    - **Dynamic Filtering**: Filter results by quality score without re-running the scan
    - **Portfolio Manager**: Track and manage your open positions
    
    """)
