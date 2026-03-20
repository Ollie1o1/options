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
from src.options_screener import run_scan, load_config, get_market_context, setup_logging
from src.filters import categorize_by_premium, pick_top_per_bucket
from src.paper_manager import PaperManager

# AI scoring imports (graceful degradation if unavailable)
_AI_AVAILABLE = False
try:
    from src.ai_scorer import AIScorer
    from src.ranking import combine_scores
    from src.config_ai import AI_CONFIG
    _AI_AVAILABLE = True
except Exception:
    pass

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

if 'ai_results' not in st.session_state:
    st.session_state.ai_results = None

if 'ai_status' not in st.session_state:
    st.session_state.ai_status = None  # None | "ok" | "error" | "disabled"

if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None

if 'scan_params_hash' not in st.session_state:
    st.session_state.scan_params_hash = None

# --- GLOBAL CONFIG & MANAGERS ---
config = load_config("config.json")
pm = PaperManager(config_path="config.json")

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
    trend_class = "positive" if "Bull" in trend else ("negative" if "Bear" in trend else "neutral")
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
        num_tickers = 20
        
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
        dte_bucket_filter = st.selectbox(
            "DTE Bucket",
            ["All", "Short (7-14 DTE)", "Standard (15-30 DTE)", "Swing (31-45 DTE)"],
            index=0,
        )
        
    # 3. Weights
    with st.sidebar.expander("ALGO WEIGHTS", expanded=False):
        default_weights = config.get('composite_weights', {})

        w_pop = st.slider("PoP", 0.0, 1.0, default_weights.get('pop', 0.18))
        w_liq = st.slider("Liquidity", 0.0, 1.0, default_weights.get('liquidity', 0.15))
        w_rr = st.slider("Risk/Reward", 0.0, 1.0, default_weights.get('rr', 0.15))

        custom_weights = default_weights.copy()
        custom_weights.update({'pop': w_pop, 'liquidity': w_liq, 'rr': w_rr})

    # 4. AI Settings
    ai_enabled = False
    ai_weight_override = None
    with st.sidebar.expander("AI SCORING", expanded=True):
        if _AI_AVAILABLE:
            ai_enabled = st.toggle("Enable AI Analysis", value=True)
            if ai_enabled:
                default_ai_weight = AI_CONFIG.get("ai_weight", 0.30)
                ai_weight_override = st.slider(
                    "AI Weight", 0.05, 0.80, float(default_ai_weight), step=0.05,
                    help="How much the AI score influences the final rank"
                )
                model_name = AI_CONFIG.get("model", "").split("/")[-1]
                st.caption(f"Model: {model_name}")
            # Show last AI run status
            if st.session_state.ai_status == "ok":
                st.success("AI: Active", icon="✅")
            elif st.session_state.ai_status == "error":
                st.warning("AI: Unavailable (tech-only)", icon="⚠️")
            elif st.session_state.ai_status == "disabled":
                st.info("AI: Disabled", icon="ℹ️")
        else:
            st.caption("AI module not installed")

    # Run Button
    import hashlib as _hashlib
    _params_str = f"{scan_mode}|{sorted(tickers)}|{budget}|{max_expiries}|{min_dte}|{max_dte}|{sorted(custom_weights.items())}"
    _new_hash = _hashlib.md5(_params_str.encode()).hexdigest()

    st.sidebar.divider()
    if st.sidebar.button("🚀 RUN SCANNER", type="primary", use_container_width=True):
        if _new_hash != st.session_state.scan_params_hash:
            with st.spinner("Scanning Market..."):
                if scan_mode != "Single-stock":
                    tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL",
                               "META", "NFLX", "JPM", "GS", "V", "MA", "COIN", "INTC", "PYPL", "SQ"][:num_tickers]

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
                    st.session_state.ai_results = None
                except Exception as e:
                    st.error(f"Scan Failed: {e}")
                    return budget, dte_bucket_filter
            st.session_state.scan_params_hash = _new_hash
        else:
            st.info("Params unchanged — showing cached results.")

        # AI Scoring — runs automatically after technical scan
        if ai_enabled and _AI_AVAILABLE and st.session_state.scan_results:
            picks = st.session_state.scan_results.picks
            if not picks.empty:
                with st.spinner("Running AI analysis..."):
                    try:
                        vix_regime_map = {"Low": "low", "Normal": "normal", "High": "high"}
                        vix_regime = vix_regime_map.get(str(st.session_state.volatility_regime), "normal")
                        ai_cfg = {"ai_weight": ai_weight_override} if ai_weight_override is not None else None
                        scorer = AIScorer(config=ai_cfg)
                        ticker_contexts = st.session_state.scan_results.ticker_contexts
                        ai_df = scorer.score_candidates(picks, ticker_contexts=ticker_contexts)
                        kwargs = {}
                        if ai_weight_override is not None:
                            kwargs["ai_weight"] = ai_weight_override
                            kwargs["technical_weight"] = 1.0 - ai_weight_override
                        ranked = combine_scores(picks, ai_df, vix_regime=vix_regime, **kwargs)
                        st.session_state.ai_results = ranked
                        st.session_state.ai_status = "ok"
                    except Exception as e:
                        st.session_state.ai_status = "error"
                        logger.warning("AI scoring failed: %s", e)
            else:
                st.session_state.ai_status = "disabled"
        elif not ai_enabled:
            st.session_state.ai_status = "disabled"

        st.success("Scan Complete")

    if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.scan_params_hash = None
        st.session_state.scan_results = None
        st.session_state.ai_results = None
        st.session_state.ai_status = None
        st.rerun()

    return budget, dte_bucket_filter

# --- SCANNER TAB CONTENT ---
def render_scanner_tab(budget, dte_bucket_filter="All"):
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        picks_df = results.picks

        # Merge AI scores onto picks_df if available
        ai_ranked = st.session_state.ai_results
        has_ai = ai_ranked is not None and not ai_ranked.empty

        # Apply DTE bucket filter
        if not picks_df.empty and dte_bucket_filter != "All" and "T_years" in picks_df.columns:
            _dte_vals = picks_df["T_years"] * 365.0
            if dte_bucket_filter == "Short (7-14 DTE)":
                picks_df = picks_df[(_dte_vals >= 7) & (_dte_vals <= 14)].copy()
            elif dte_bucket_filter == "Standard (15-30 DTE)":
                picks_df = picks_df[(_dte_vals > 14) & (_dte_vals <= 30)].copy()
            elif dte_bucket_filter == "Swing (31-45 DTE)":
                picks_df = picks_df[(_dte_vals > 30) & (_dte_vals <= 45)].copy()

        if not picks_df.empty:
            picks_df = categorize_by_premium(picks_df, budget=budget)
            top_picks_df = pick_top_per_bucket(picks_df, per_bucket=3, diversify_tickers=True)

            # Merge AI columns into display frames when available
            ai_merge_cols = ['symbol', 'type', 'strike', 'expiration', 'ai_score', 'ai_confidence',
                             'final_score', 'catalyst_risk', 'ai_reasoning', 'ai_flags', 'rank']
            if has_ai:
                ai_sub = ai_ranked[[c for c in ai_merge_cols if c in ai_ranked.columns]].copy()
                picks_df = picks_df.merge(ai_sub, on=['symbol', 'type', 'strike', 'expiration'], how='left')
                top_picks_df = top_picks_df.merge(ai_sub, on=['symbol', 'type', 'strike', 'expiration'], how='left')

            if not top_picks_df.empty:
                best = top_picks_df.iloc[0]
                ai_badge = ""
                if has_ai and pd.notna(best.get('ai_score')):
                    ai_badge = f"<span style='color:#58a6ff'> | AI: {best['ai_score']:.0f}</span>"
                st.markdown(f"""
                <div style="padding: 1rem; background-color: #1f2937; border-radius: 0.5rem; border: 1px solid #374151; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #fbbf24;">🏆 Top Pick: {best['symbol']} {best['type'].upper()} ${best['strike']}</h3>
                    <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.9rem; color: #d1d5db;">
                        <span><b>Exp:</b> {best['expiration']}</span>
                        <span><b>Prem:</b> ${best['premium']:.2f}</span>
                        <span><b>PoP:</b> {best['prob_profit']:.1%}</span>
                        <span><b>Score:</b> {best['quality_score'] * 100:.1f}{ai_badge}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            subtab_names = ["🏆 Top Picks", "🟢 Low Premium", "🟡 Medium Premium", "🔴 High Premium", "📋 All Results"]
            subtabs = st.tabs(subtab_names)

            base_cols = ['symbol', 'type', 'strike', 'expiration', 'premium', 'prob_profit',
                         'quality_score', 'impliedVolatility', 'delta', 'volume', 'openInterest', 'price_bucket']
            ai_display_cols = ['ai_score', 'ai_confidence', 'final_score', 'catalyst_risk'] if has_ai else []
            common_cols = base_cols + [c for c in ai_display_cols if c in picks_df.columns]

            col_rename = {
                'symbol': 'Ticker', 'type': 'Type', 'strike': 'Strike', 'expiration': 'Exp',
                'premium': 'Price', 'prob_profit': 'PoP', 'quality_score': 'Tech Score',
                'impliedVolatility': 'IV', 'delta': 'Delta', 'volume': 'Vol',
                'openInterest': 'OI', 'price_bucket': 'Bucket',
                'ai_score': 'AI Score', 'ai_confidence': 'AI Conf', 'final_score': 'Final',
                'catalyst_risk': 'Catalyst'
            }
            sort_col = 'final_score' if (has_ai and 'final_score' in picks_df.columns) else 'quality_score'
            column_config = {
                "Tech Score": st.column_config.NumberColumn("Tech Score (0-100)", format="%.1f"),
                "AI Score": st.column_config.NumberColumn("AI Score", format="%.0f"),
                "Final": st.column_config.ProgressColumn("Final Score", format="%.2f", min_value=0, max_value=1),
                "PoP": st.column_config.ProgressColumn("Prob. Profit", format="%.2f", min_value=0, max_value=1),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            }

            def render_df(df, key_suffix):
                cols = [c for c in common_cols if c in df.columns]
                d = df[cols].copy()
                # Normalize quality_score 0-1 → 0-100 for display parity with AI Score
                if "quality_score" in d.columns:
                    d["quality_score"] = (d["quality_score"] * 100).round(1)
                d.rename(columns=col_rename, inplace=True)
                sc = col_rename.get(sort_col, sort_col)
                if sc in d.columns:
                    d = d.sort_values(sc, ascending=False)
                event = st.dataframe(d, use_container_width=True, hide_index=True,
                                     column_config=column_config, selection_mode="single-row",
                                     on_select="rerun", height=400, key=f"df_{key_suffix}")
                if len(event.selection['rows']) > 0:
                    selected_row_idx = event.selection['rows'][0]
                    selected_row_data = d.iloc[selected_row_idx]
                    ticker_col = col_rename.get('symbol', 'symbol')
                    type_col = col_rename.get('type', 'type')
                    strike_col = col_rename.get('strike', 'strike')
                    exp_col = col_rename.get('expiration', 'expiration')
                    mask = (
                        (picks_df['symbol'] == selected_row_data[ticker_col]) &
                        (picks_df['type'] == selected_row_data[type_col]) &
                        (picks_df['strike'] == selected_row_data[strike_col]) &
                        (picks_df['expiration'] == selected_row_data[exp_col])
                    )
                    match = picks_df[mask]
                    if not match.empty:
                        st.session_state.selected_option = match.iloc[0]

            with subtabs[0]: render_df(top_picks_df, "top")
            with subtabs[1]: render_df(picks_df[picks_df['price_bucket'] == 'LOW'], "low")
            with subtabs[2]: render_df(picks_df[picks_df['price_bucket'] == 'MEDIUM'], "med")
            with subtabs[3]: render_df(picks_df[picks_df['price_bucket'] == 'HIGH'], "high")
            with subtabs[4]: render_df(picks_df, "all")

            # AI Ticker Regime Summary (from Pass 1 ticker analysis)
            ticker_contexts = results.ticker_contexts
            regime_rows = []
            for ticker, ctx in ticker_contexts.items():
                regime = ctx.get("regime")
                cat = ctx.get("catalyst_risk")
                bias = ctx.get("directional_bias")
                summary = ctx.get("summary")
                if all(v is not None for v in [regime, cat, bias, summary]):
                    regime_rows.append({
                        "Ticker": ticker,
                        "Regime": regime,
                        "Catalyst Risk": cat,
                        "Directional Bias": bias,
                        "Summary": summary,
                    })
            with st.expander("AI Ticker Regime Summary"):
                if regime_rows:
                    st.dataframe(
                        pd.DataFrame(regime_rows).set_index("Ticker"),
                        use_container_width=True,
                    )
                else:
                    st.caption("Run with AI enabled to see ticker regime analysis.")

            # Action Buttons
            if st.session_state.selected_option is not None:
                opt = st.session_state.selected_option
                btn_col, csv_col = st.columns([1, 1])
                with btn_col:
                    if st.button(f"📝 Paper Trade {opt['symbol']} ${opt['strike']} {opt['type'].upper()}"):
                        ask_val = opt.get("ask")
                        entry_px = (ask_val if (ask_val is not None and float(ask_val) > 0)
                                    else opt.get("lastPrice") or opt.get("premium", 0.0))
                        trade_dict = {
                            "ticker": opt["symbol"],
                            "expiration": opt["expiration"],
                            "strike": opt["strike"],
                            "type": opt["type"],
                            "entry_price": float(entry_px) if entry_px else 0.0,
                            "quality_score": opt["quality_score"],
                            "strategy_name": f"Long {opt['type'].capitalize()}"
                        }
                        pm.log_trade(trade_dict)
                        st.success(f"Logged paper trade for {opt['symbol']}")
                with csv_col:
                    export_df = (st.session_state.ai_results
                                 if st.session_state.ai_results is not None and not st.session_state.ai_results.empty
                                 else picks_df)
                    csv_bytes = export_df.to_csv(index=False).encode()
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button("⬇️ Export CSV", data=csv_bytes,
                                       file_name=f"options_picks_{ts}.csv",
                                       mime="text/csv")
        else:
            st.warning("No results found. Try adjusting filters.")
    else:
        st.info("👈 Configure and Run Scan to see results.")

def render_visualizer_tab():
    if st.session_state.selected_option is not None:
        opt = st.session_state.selected_option
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📉 Payoff Diagram")
            spot = opt['underlying']
            prices, pnl = calculate_payoff(opt, spot)
            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name='P/L', fill='tozeroy'))
            fig_payoff.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Current")
            fig_payoff.add_vline(x=opt['strike'], line_dash="dot", line_color="yellow", annotation_text="Strike")
            fig_payoff.update_layout(template="plotly_dark", xaxis_title="Price", yaxis_title="P/L ($)", height=400)
            st.plotly_chart(fig_payoff, use_container_width=True)
        with col2:
            st.markdown("### 🕸️ Quality Radar")
            categories = ['Liquidity', 'PoP', 'Theta', 'IV Rank', 'Risk/Reward']
            values = [min(opt.get('volume', 0) / 1000, 1.0), opt.get('prob_profit', 0.5), min(abs(opt.get('theta', 0)) * 10, 1.0), min(opt.get('impliedVolatility', 0.2) * 2, 1.0), min(opt.get('rr_ratio', 1.0) / 3, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#58a6ff'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)

        # Score Breakdown Panel
        _sd = opt.get("score_drivers", "")
        if _sd:
            st.markdown("### Score Breakdown")
            _sd_str = str(_sd)
            _pos = [p for p in _sd_str.split() if p.startswith("+") and p != "|"]
            _neg = [p for p in _sd_str.split() if p.startswith("-") and p != "|"]
            _bd_col1, _bd_col2 = st.columns(2)
            with _bd_col1:
                st.markdown("**Positives**")
                for _p in _pos:
                    st.markdown(f"- `{_p}`")
            with _bd_col2:
                st.markdown("**Negatives**")
                for _n in _neg:
                    st.markdown(f"- `{_n}`")

        # AI Reasoning Panel
        ai_ranked = st.session_state.ai_results
        if ai_ranked is not None and not ai_ranked.empty:
            st.markdown("### 🤖 AI Analysis")
            mask = (
                (ai_ranked['symbol'] == opt['symbol']) &
                (ai_ranked['type'] == opt['type']) &
                (ai_ranked['strike'] == opt['strike']) &
                (ai_ranked['expiration'] == opt['expiration'])
            )
            ai_row = ai_ranked[mask]
            if not ai_row.empty:
                row = ai_row.iloc[0]
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("AI Score", f"{row.get('ai_score', 0):.0f}/100")
                a2.metric("Confidence", f"{row.get('ai_confidence', 0):.1f}/10")
                a3.metric("Catalyst Risk", str(row.get('catalyst_risk', 'medium')).upper())
                a4.metric("Final Score", f"{row.get('final_score', 0):.3f}")

                reasoning = row.get('ai_reasoning', '')
                if reasoning:
                    st.markdown(f"> {reasoning}")
                flags = row.get('ai_flags', '')
                if flags:
                    flag_list = flags if isinstance(flags, list) else str(flags).split(',')
                    st.markdown(" ".join([f"`{f.strip()}`" for f in flag_list if f.strip()]))
            else:
                st.caption("No AI data for this contract.")
    else:
        st.info("Select a row in the Scanner tab to view analysis.")

# --- PAPER PORTFOLIO TAB CONTENT ---
def render_paper_portfolio_tab():
    st.markdown("### 📈 Forward Testing Portfolio")
    
    # Action Header
    col_a, col_b = st.columns([1, 5])
    if col_a.button("🔄 Refresh Prices", use_container_width=True):
        with st.spinner("Updating positions..."):
            pm.update_positions()
        st.success("Positions updated.")
        st.rerun()
    
    # Performance Metrics
    summary = pm.get_performance_summary()
    if not summary.empty:
        df_paper = pm.get_all_trades()
        open_count = len(df_paper[df_paper["status"] == "OPEN"])
        closed_df = df_paper[df_paper["status"] == "CLOSED"]
        # pnl_pct is already sign-corrected for short/credit positions by update_positions()
        total_pnl_usd = (closed_df["pnl_pct"] * closed_df["entry_price"] * 100).sum() if not closed_df.empty else 0.0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total PnL ($)", f"${total_pnl_usd:,.2f}")
        m2.metric("Win Rate", summary["Win Rate"].iloc[0])
        m3.metric("Open Positions", open_count)
        m4.metric("Avg Return", summary["Avg Return"].iloc[0])
        
        st.divider()
        
        # Positions Display
        st.markdown("#### 📂 Open Positions")
        open_df = df_paper[df_paper["status"] == "OPEN"].copy()
        if not open_df.empty:
            st.dataframe(open_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")
            
        st.markdown("#### 📜 Trade History")
        closed_df_display = df_paper[df_paper["status"] == "CLOSED"].copy()
        if not closed_df_display.empty:
            st.dataframe(closed_df_display.sort_values("exit_date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet.")
    else:
        st.info("Start paper trading from the Scanner tab to see results here.")

# --- MAIN RENDERER ---
def render_main():
    render_ticker_tape()
    
    # Top-level Navigation
    tab_scanner, tab_portfolio = st.tabs(["🔍 Options Scanner", "📈 Paper Portfolio"])
    
    with tab_scanner:
        budget, dte_bucket_filter = render_sidebar()
        
        # Sub-navigation for Scanner
        s_tab1, s_tab2 = st.tabs(["📋 Results", "📊 Deep Analysis"])
        with s_tab1:
            render_scanner_tab(budget, dte_bucket_filter)
        with s_tab2:
            render_visualizer_tab()
            
    with tab_portfolio:
        render_paper_portfolio_tab()

# --- APP ENTRY POINT ---
if __name__ == "__main__":
    render_main()