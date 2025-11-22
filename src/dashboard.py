#!/usr/bin/env python3
"""
Streamlit Dashboard for Options Screener
Professional "Dark Financial" Trading Terminal UI
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the options screener module
from src.options_screener import run_scan, load_config, get_market_context, setup_logging, categorize_by_premium, pick_top_per_bucket

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Options Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK FINANCIAL THEME) ---
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ticker Tape Styling */
    .ticker-tape {
        display: flex;
        justify-content: space-between;
        background-color: #161b22;
        padding: 10px 20px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .ticker-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .ticker-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
    }
    .ticker-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #e6edf3;
    }
    .ticker-value.positive { color: #3fb950; }
    .ticker-value.negative { color: #f85149; }
    .ticker-value.neutral { color: #e6edf3; }
    .ticker-value.warning { color: #d29922; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0d1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'market_context_loaded' not in st.session_state:
    st.session_state.market_context_loaded = False
    st.session_state.market_trend = "Unknown"
    st.session_state.volatility_regime = "Unknown"
    st.session_state.macro_risk_active = False
    st.session_state.tnx_change_pct = 0.0

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None

# --- HELPER FUNCTIONS ---
def load_market_data():
    """Loads market context if not already loaded."""
    if not st.session_state.market_context_loaded:
        try:
            market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
            st.session_state.market_trend = market_trend
            st.session_state.volatility_regime = volatility_regime
            st.session_state.macro_risk_active = macro_risk_active
            st.session_state.tnx_change_pct = tnx_change_pct
            st.session_state.market_context_loaded = True
        except Exception as e:
            st.error(f"Market Data Error: {e}")

def render_ticker_tape():
    """Renders the top ticker tape with market context."""
    load_market_data()
    
    trend = st.session_state.market_trend
    vix = st.session_state.volatility_regime
    macro = st.session_state.macro_risk_active
    tnx = st.session_state.tnx_change_pct
    
    # Determine colors
    trend_class = "positive" if "Bullish" in trend else ("negative" if "Bearish" in trend else "neutral")
    macro_class = "negative" if macro else "positive"
    macro_text = "RISK ON" if not macro else "RISK OFF"
    tnx_class = "negative" if tnx > 0.025 else "neutral"
    tnx_text = f"{tnx:+.2%}"
    
    st.markdown(f"""
    <div class="ticker-tape">
        <div class="ticker-item">
            <span class="ticker-label">SPY Trend</span>
            <span class="ticker-value {trend_class}">{trend}</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-label">VIX Regime</span>
            <span class="ticker-value neutral">{vix}</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-label">Macro Cond.</span>
            <span class="ticker-value {macro_class}">{macro_text}</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-label">10Y Yield Δ</span>
            <span class="ticker-value {tnx_class}">{tnx_text}</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-label">Terminal Time</span>
            <span class="ticker-value neutral">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def calculate_payoff(option_row, spot_price, range_pct=0.2):
    """Calculates P/L curve for the selected option."""
    strike = float(option_row['strike'])
    premium = float(option_row['premium'])
    op_type = option_row['type'].lower()
    
    # Generate price range
    lower_bound = spot_price * (1 - range_pct)
    upper_bound = spot_price * (1 + range_pct)
    prices = np.linspace(lower_bound, upper_bound, 100)
    
    # Calculate P/L at expiration
    pnl = []
    for p in prices:
        if op_type == 'call':
            intrinsic = max(0, p - strike)
        else:
            intrinsic = max(0, strike - p)
            
        # Assuming long position for screener results (buying options)
        # If we want to support short strategies, we need to know the strategy type
        # For now, let's assume long single leg as per "Picks"
        profit = intrinsic - premium
        pnl.append(profit * 100) # Per contract
        
    return prices, pnl

# --- SIDEBAR ---
def render_sidebar():
    st.sidebar.markdown("## ⚙️ CONTROL PANEL")
    
    # 1. Scan Settings
    with st.sidebar.expander("SCAN SETTINGS", expanded=True):
        scan_mode = st.selectbox(
            "Strategy",
            ["Discovery scan", "Single-stock", "Budget scan", "Premium Selling"],
            index=0
        )
        
        tickers = []
        budget = None
        
        if scan_mode == "Single-stock":
            ticker_input = st.text_input("Ticker", value="SPY").upper()
            tickers = [ticker_input] if ticker_input else []
        elif scan_mode == "Budget scan":
            budget = st.number_input("Budget ($)", min_value=50.0, value=500.0, step=50.0)
            num_tickers = st.number_input("Universe Size", 10, 100, 20)
        else:
            num_tickers = st.number_input("Universe Size", 10, 100, 20)
            
    # 2. Filters
    with st.sidebar.expander("FILTERS", expanded=False):
        min_dte = st.slider("Min DTE", 0, 90, 7)
        max_dte = st.slider("Max DTE", 0, 180, 45)
        max_expiries = st.slider("Max Expiries", 1, 5, 3)
        
    # 3. Weights
    with st.sidebar.expander("ALGO WEIGHTS", expanded=False):
        config = load_config("config.json")
        default_weights = config.get('composite_weights', {})
        
        w_pop = st.slider("PoP", 0.0, 1.0, default_weights.get('pop', 0.18))
        w_liq = st.slider("Liquidity", 0.0, 1.0, default_weights.get('liquidity', 0.15))
        w_rr = st.slider("Risk/Reward", 0.0, 1.0, default_weights.get('rr', 0.15))
        
        custom_weights = default_weights.copy()
        custom_weights.update({'pop': w_pop, 'liquidity': w_liq, 'rr': w_rr})

    # Run Button
    st.sidebar.divider()
    if st.sidebar.button("🚀 RUN SCANNER", type="primary", use_container_width=True):
        with st.spinner("Scanning Market..."):
            # Ticker logic
            if scan_mode != "Single-stock":
                # Simplified liquid list for demo
                tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL"][:num_tickers]
            
            logger = setup_logging()
            try:
                results = run_scan(
                    mode=scan_mode,
                    tickers=tickers,
                    budget=budget,
                    max_expiries=max_expiries,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    trader_profile="swing",
                    logger=logger,
                    market_trend=st.session_state.market_trend,
                    volatility_regime=st.session_state.volatility_regime,
                    macro_risk_active=st.session_state.macro_risk_active,
                    tnx_change_pct=st.session_state.tnx_change_pct,
                    verbose=False,
                    custom_weights=custom_weights
                )
                st.session_state.scan_results = results
                st.success("Scan Complete")
            except Exception as e:
                st.error(f"Scan Failed: {e}")

# --- MAIN CONTENT ---
def render_main():
    render_ticker_tape()
    
    tab1, tab2, tab3 = st.tabs(["🔍 SCANNER", "📊 VISUALIZER", "💼 PORTFOLIO"])
    
    # --- TAB 1: SCANNER ---
    with tab1:
        if st.session_state.scan_results:
            results = st.session_state.scan_results
            picks_df = results.get('picks', pd.DataFrame())
            budget_val = results.get('budget', None) # We might need to pass budget through results or session state
            # In run_scan wrapper, we didn't pass budget back in results dict explicitly, but we can infer or pass it.
            # Actually, let's just use the local variable 'budget' from the sidebar if we can, or better, 
            # let's assume categorize_by_premium was NOT called in run_scan return value (it wasn't).
            
            # We need to know if it was budget mode to pass budget to categorize_by_premium
            # We can infer mode from results or session state? 
            # Let's just re-run categorization here.
            
            if not picks_df.empty:
                # 1. Categorize
                # We need to know the budget used. 
                # Since we don't have it easily in results, let's look at the sidebar input 'budget' 
                # But that might have changed. 
                # Ideally run_scan returns it. 
                # For now, let's try to use the 'contract_cost' if it exists or recalculate.
                
                # Recalculate categorization
                # We need to know if it is budget mode.
                # Let's use a heuristic or just pass None if not sure, it defaults to quantiles which is fine for most cases.
                # If we want strict budget buckets, we need the budget value.
                # Let's assume quantile based for now unless we can get budget.
                
                picks_df = categorize_by_premium(picks_df, budget=budget) 
                
                # 2. Pick Top Candidates
                top_picks_df = pick_top_per_bucket(picks_df, per_bucket=3, diversify_tickers=True)
                
                # 3. Display "Top Overall Pick" Card
                if not top_picks_df.empty:
                    best = top_picks_df.iloc[0]
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: #1f2937; border-radius: 0.5rem; border: 1px solid #374151; margin-bottom: 1rem;">
                        <h3 style="margin: 0; color: #fbbf24;">🏆 Top Pick: {best['symbol']} {best['type'].upper()} ${best['strike']}</h3>
                        <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.9rem; color: #d1d5db;">
                            <span><b>Exp:</b> {best['expiration']}</span>
                            <span><b>Prem:</b> ${best['premium']:.2f}</span>
                            <span><b>PoP:</b> {best['prob_profit']:.1%}</span>
                            <span><b>Score:</b> {best['quality_score']:.1f}</span>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #9ca3af;">
                            {best.get('context_log', ['Strong metrics across the board.'])[0] if isinstance(best.get('context_log'), list) and best.get('context_log') else "High quality score with favorable risk/reward profile."}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Tabs for Buckets
                subtab_names = ["🏆 Top Picks", "🟢 Low Premium", "🟡 Medium Premium", "🔴 High Premium", "📋 All Results"]
                subtabs = st.tabs(subtab_names)
                
                common_cols = [
                    'symbol', 'type', 'strike', 'expiration', 'premium', 
                    'prob_profit', 'quality_score', 'impliedVolatility', 
                    'delta', 'volume', 'openInterest', 'price_bucket'
                ]
                
                column_config = {
                    "Score": st.column_config.ProgressColumn(
                        "Quality Score",
                        help="Proprietary Quality Score (0-100)",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    "PoP": st.column_config.ProgressColumn(
                        "Prob. Profit",
                        help="Probability of Profit",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Price",
                        format="$%.2f"
                    ),
                    "IV": st.column_config.NumberColumn(
                        "IV",
                        format="%.2f"
                    ),
                    "Delta": st.column_config.NumberColumn(
                        "Delta",
                        format="%.2f"
                    )
                }

                def render_df(df, key_suffix):
                    # Filter cols
                    cols = [c for c in common_cols if c in df.columns]
                    d = df[cols].copy()
                    d.rename(columns={
                        'symbol': 'Ticker', 'type': 'Type', 'strike': 'Strike', 
                        'expiration': 'Exp', 'premium': 'Price', 
                        'prob_profit': 'PoP', 'quality_score': 'Score',
                        'impliedVolatility': 'IV', 'delta': 'Delta',
                        'volume': 'Vol', 'openInterest': 'OI',
                        'price_bucket': 'Bucket'
                    }, inplace=True)
                    
                    # Sort by Score descending
                    if 'Score' in d.columns:
                        d = d.sort_values('Score', ascending=False)

                    event = st.dataframe(
                        d,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config,
                        selection_mode="single-row",
                        on_select="rerun",
                        height=500,
                        key=f"df_{key_suffix}"
                    )
                    if len(event.selection['rows']) > 0:
                        # Map back to original df index is tricky if sorted/filtered.
                        # Streamlit returns row index of the *displayed* dataframe.
                        # We need to get the corresponding row from 'd', then find it in 'picks_df' or just use it directly.
                        # Using it directly is safer.
                        selected_row_idx = event.selection['rows'][0]
                        # Get the row from the sorted/filtered dataframe 'd'
                        selected_row_data = d.iloc[selected_row_idx]
                        
                        # Now find this row in the original picks_df to get all columns (like 'underlying', 'theta' etc needed for visualizer)
                        # We match on unique keys: symbol, type, strike, expiration
                        mask = (
                            (picks_df['symbol'] == selected_row_data['Ticker']) &
                            (picks_df['type'] == selected_row_data['Type']) &
                            (picks_df['strike'] == selected_row_data['Strike']) &
                            (picks_df['expiration'] == selected_row_data['Exp'])
                        )
                        match = picks_df[mask]
                        if not match.empty:
                            st.session_state.selected_option = match.iloc[0]

                with subtabs[0]: # Top Picks
                    render_df(top_picks_df, "top")
                
                with subtabs[1]: # Low
                    render_df(picks_df[picks_df['price_bucket'] == 'LOW'], "low")
                    
                with subtabs[2]: # Medium
                    render_df(picks_df[picks_df['price_bucket'] == 'MEDIUM'], "med")
                    
                with subtabs[3]: # High
                    render_df(picks_df[picks_df['price_bucket'] == 'HIGH'], "high")
                    
                with subtabs[4]: # All
                    render_df(picks_df, "all")
                
            else:
                st.warning("No results found. Try adjusting filters.")
        else:
            st.info("👈 Configure and Run Scan to see results.")
            
        # Error/Warning Display
        if st.session_state.scan_results:
            results = st.session_state.scan_results
            # Check for errors in individual tickers if available (we might need to update run_scan to return them more explicitly)
            # For now, let's just check if picks are empty but we had tickers
            pass

    # --- TAB 2: VISUALIZER ---
    with tab2:
        if st.session_state.selected_option is not None:
            opt = st.session_state.selected_option
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📉 Payoff Diagram (at Expiration)")
                spot = opt['underlying']
                prices, pnl = calculate_payoff(opt, spot)
                
                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name='P/L', fill='tozeroy'))
                fig_payoff.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Current Price")
                fig_payoff.add_vline(x=opt['strike'], line_dash="dot", line_color="yellow", annotation_text="Strike")
                
                fig_payoff.update_layout(
                    template="plotly_dark",
                    xaxis_title="Stock Price",
                    yaxis_title="Profit/Loss ($)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400
                )
                st.plotly_chart(fig_payoff, use_container_width=True)
                
            with col2:
                st.markdown("### 🕸️ Quality Radar")
                # Normalize metrics for radar
                categories = ['Liquidity', 'PoP', 'Theta', 'IV Rank', 'Risk/Reward']
                
                # Mock values for radar if not explicitly in df, or normalize real ones
                # Using simple heuristics for demo if exact columns missing
                val_liq = min(opt.get('volume', 0) / 1000, 1.0)
                val_pop = opt.get('prob_profit', 0.5)
                val_theta = min(abs(opt.get('theta', 0)) * 10, 1.0)
                val_iv = min(opt.get('impliedVolatility', 0.2) * 2, 1.0) # Rough normalization
                val_rr = min(opt.get('rr_ratio', 1.0) / 3, 1.0)
                
                values = [val_liq, val_pop, val_theta, val_iv, val_rr]
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color='#58a6ff'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    template="plotly_dark",
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            # Option Details Card
            st.markdown("### 📝 Contract Details")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.metric("Ticker", opt['symbol'])
            d_col2.metric("Strike", f"{opt['strike']} {opt['type'].upper()}")
            d_col3.metric("Expiration", opt['expiration'])
            d_col4.metric("Premium", f"${opt['premium']:.2f}")
            
        else:
            st.info("Select a row in the Scanner tab to view analysis.")

    # --- TAB 3: PORTFOLIO ---
    with tab3:
        st.markdown("### 💼 Portfolio Manager")
        trades_log_path = "trades_log/entries.csv"
        if os.path.exists(trades_log_path):
            trades_df = pd.read_csv(trades_log_path)
            
            # Summary Metrics
            p_col1, p_col2 = st.columns(2)
            open_pos = len(trades_df[trades_df['status'] == 'OPEN'])
            realized = trades_df['realized_pnl'].sum() if 'realized_pnl' in trades_df.columns else 0.0
            
            p_col1.metric("Open Positions", open_pos)
            p_col2.metric("Realized P/L", f"${realized:.2f}", delta_color="normal")
            
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.warning("No trade log found.")

# --- APP ENTRY POINT ---
if __name__ == "__main__":
    render_sidebar()
    render_main()
