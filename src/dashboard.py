#!/usr/bin/env python3
"""Visual Web Cockpit for Options Screener (`streamlit run src/dashboard.py`).

Shares the exact same truth layer (`DeskDataProvider`, `configs/playbooks.json`,
`src/structure/express.py`, `src/equity_playbooks/rebalance.py`, and
`src/pick_ranking.gate_board`) as the `opt` CLI and Textual TUI, adding rich
interactive Plotly visualizations:
  1. Today & TFSA Core    — Market Mood + Kalshi odds + Interactive TFSA Rebalance Calculator
  1b. Discover            — ① Market → ② Sectors (PROVEN rank) → ③ Ideas (labelled lenses) → ④ Evidence card
  2. Opportunity Radar    — Filter presets, sortable table, live Dossier + 60d Price/Levels Chart
  3. Shares vs Options    — Interactive sliders, Real-Chain Table, Market-vs-View RND & Payoff Overlay
  4. Book & Track Record  — Stock/ETF Picks (with exit alerts), Paper Options, Holdings & Fill Concession k
  5. Legacy Option Scan   — Multi-ticker live chain scan properly routed through `gate_board()`
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure repo root is on sys.path when invoked via `streamlit run src/dashboard.py`
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cli.guide import (
    TIER_PLAIN,
    plain_english_action_plan,
    plain_english_traffic_light,
    plain_english_vehicle_verdict,
)
from src.equity_playbooks.rebalance import compute_tfsa_rebalance, parse_holdings_spec
from src.probability_lab.view import apply_view
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.chain_data import load_chain_for_horizon
from src.structure.express import _build_lognormal_density, compare_vehicles
from src.structure.types import DetailedView, VehicleComparison
from src.tui.app import filter_opportunities
from src.tui.state import DeskDataProvider


# ── Pure Plotly Chart Builders (100% unit-testable without Streamlit runtime) ──


def build_dossier_price_figure(dos: TickerDossier) -> go.Figure:
    """Interactive 60-session price tape with support/resistance horizontal levels."""
    fig = go.Figure()
    closes = dos.closes_60d or []
    if closes:
        xs = list(range(-len(closes) + 1, 1))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=closes,
                mode="lines",
                name=f"{dos.ticker} Close",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
            )
        )
        for sup in dos.supports[:2]:
            fig.add_hline(
                y=float(sup["level"]),
                line_dash="dot",
                line_color="#3fb950",
                annotation_text=f"Sup ${float(sup['level']):.2f}",
            )
        for res in dos.resistances[:2]:
            fig.add_hline(
                y=float(res["level"]),
                line_dash="dot",
                line_color="#f85149",
                annotation_text=f"Res ${float(res['level']):.2f}",
            )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Sessions Ago",
        yaxis_title="Price ($)",
        title=f"{dos.ticker} 60-Session Structure & Levels",
    )
    return fig


def build_rnd_comparison_figure(
    spot: float,
    iv: float,
    view: DetailedView,
    *,
    chain_db: str = "data/chain_archive.db",
) -> Tuple[go.Figure, str]:
    """Compare Market-Implied RND (SVI when available) vs User's Tilted View Density."""
    t_h = max(int(view.horizon_days), 5) / 365.0
    sl = load_chain_for_horizon(
        view.symbol,
        target_dte=view.horizon_days,
        min_dte=max(7, view.horizon_days - 20),
        max_dte=view.horizon_days + 30,
        spot=spot,
        chain_db=chain_db,
        allow_live=False,
        build_rnd=True,
    )
    if sl is not None and sl.rnd_density is not None:
        q_mkt = sl.rnd_density
        src_lbl = f"{sl.source}/{sl.rnd_source} ({sl.dte}d)"
    else:
        q_mkt = _build_lognormal_density(spot, iv, t_h, 0.045)
        src_lbl = f"lognormal proxy ({view.horizon_days}d)"

    q_view = apply_view(q_mkt, drift_pct=view.target_move_pct, vol_mult=view.vol_mult)

    lo = spot * 0.72
    hi = spot * 1.28
    mask = (q_mkt.K >= lo) & (q_mkt.K <= hi)
    ks = q_mkt.K[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=q_mkt.pdf[mask],
            mode="lines",
            name="Market Implied RND",
            line=dict(color="#8b949e", width=2, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(139,148,158,0.10)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=q_view.pdf[mask],
            mode="lines",
            name=f"Your View ({view.target_move_pct*100:+.1f}%)",
            line=dict(color="#3fb950" if view.target_move_pct >= 0 else "#f85149", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.12)" if view.target_move_pct >= 0 else "rgba(248,81,73,0.12)",
        )
    )
    fig.add_vline(x=spot, line_dash="dot", line_color="#e6edf3", annotation_text=f"Spot ${spot:.2f}")
    target_px = spot * (1.0 + view.target_move_pct)
    if abs(view.target_move_pct) > 0.005:
        fig.add_vline(
            x=target_px,
            line_dash="dot",
            line_color="#58a6ff",
            annotation_text=f"Target ${target_px:.2f}",
        )
    fig.update_layout(
        template="plotly_dark",
        height=310,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title=f"{view.symbol} Price at Horizon ($)",
        yaxis_title="Probability Density",
        title=f"Market Implied RND vs Your View [{src_lbl}]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig, src_lbl


def build_vehicle_edge_figure(vehicles: List[VehicleComparison], risk_budget: float) -> go.Figure:
    """Horizontal bar comparison of Net EV(View) vs Friction across candidate vehicles."""
    fig = go.Figure()
    if vehicles:
        labels = [
            ("★ " if v.recommended else "") + v.vehicle + (" [OVER]" if v.capital_at_risk_usd > risk_budget * 1.15 else "")
            for v in vehicles
        ]
        evs = [v.ev_view_net_usd for v in vehicles]
        colors = ["#3fb950" if e > 0 else "#f85149" for e in evs]
        fig.add_trace(
            go.Bar(
                y=labels,
                x=evs,
                orientation="h",
                marker_color=colors,
                text=[f"${e:+,.0f} (cost {v.friction_pct_of_reward:.1f}%)" for e, v in zip(evs, vehicles)],
                textposition="auto",
                name="Net EV (Your View)",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=310,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title="Expected P&L Net of Friction ($)",
        title=f"Vehicle Edge After Crossing Costs (Risk Budget ${risk_budget:,.0f})",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def calculate_payoff(option_row: Dict[str, Any], spot_price: float, range_pct: float = 0.20):
    """Calculates expiration P/L curve for a single long/short option contract."""
    strike = float(option_row["strike"])
    premium = float(option_row["premium"])
    op_type = str(option_row["type"]).lower()
    is_short = "short" in str(option_row.get("strategy_name", "")).lower()

    prices = np.linspace(spot_price * (1.0 - range_pct), spot_price * (1.0 + range_pct), 100)
    pnl = []
    for p in prices:
        intrinsic = max(0.0, p - strike) if op_type == "call" else max(0.0, strike - p)
        profit = (premium - intrinsic) if is_short else (intrinsic - premium)
        pnl.append(profit * 100.0)
    return prices, pnl


# ── Streamlit Web UI ─────────────────────────────────────────────────────────


def render_main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Options Desk — Unified Visual Cockpit",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .stApp { background-color: #0e1117; font-family: 'Inter', -apple-system, sans-serif; }
            .ribbon-bar {
                display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: space-between;
                background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                padding: 0.7rem 1.1rem; margin-bottom: 1rem;
            }
            .ribbon-item { display: flex; flex-direction: column; }
            .ribbon-lbl { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.04em; }
            .ribbon-val { font-size: 1.05rem; font-weight: 700; color: #e6edf3; font-family: monospace; }
            .good { color: #3fb950 !important; }
            .bad { color: #f85149 !important; }
            .warn { color: #d29922 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    provider = DeskDataProvider()
    if "radar_snap" not in st.session_state:
        st.session_state.radar_snap = provider.get_radar(refresh=False)
    if "focused_ticker" not in st.session_state:
        opps0 = st.session_state.radar_snap.opportunities
        st.session_state.focused_ticker = opps0[0].ticker if opps0 else "SPY"

    # Sidebar quick actions
    with st.sidebar:
        st.markdown("## 🧭 Daily Desk Controls")
        if st.button("🔄 Refresh Local Radar", type="primary", width="stretch"):
            with st.spinner("Running local scanners…"):
                st.session_state.radar_snap = provider.get_radar(refresh=True)
                st.session_state.pop("topdown_hub", None)
                st.session_state.pop("topdown_cards", None)
            st.success("Radar refreshed.")
        if st.button("⚡ Scan Live 60–120d ETF Spreads", width="stretch"):
            with st.spinner("Scanning live ETF option chains (Test O1)…"):
                live_opps, warns = provider.get_live_etf_options()
                if live_opps:
                    from src.radar.engine import merge_confluence

                    live_tickers = {o.ticker for o in live_opps}
                    base = [
                        o
                        for o in st.session_state.radar_snap.opportunities
                        if "LIVE_CHAIN" not in o.badges
                        and not (o.playbook_id == "etf_wide_spread" and o.ticker in live_tickers)
                    ]
                    st.session_state.radar_snap.opportunities = merge_confluence(base + live_opps)
            if live_opps:
                st.success(f"Updated {len(live_opps)} live 60–120d ETF spread(s).")
            else:
                st.warning("0 live ETF spreads passed the Test O1 rule; kept cached quotes.")
            skipped = dict(getattr(provider, "last_live_skipped", {}) or {})
            if skipped:
                with st.expander(f"Not shown ({len(skipped)}), and why"):
                    st.markdown("\n".join(f"- **{tk}** — {why}" for tk, why in sorted(skipped.items())))
        if st.button("📥 Update Daily OHLCV Prices", width="stretch"):
            with st.spinner("Refreshing Yahoo daily closes…"):
                res = provider.update_prices(expand_universe=False)
                st.session_state.radar_snap = provider.get_radar(refresh=True)
                st.session_state.pop("topdown_hub", None)
                st.session_state.pop("topdown_cards", None)
            st.info(f"Checked {res['symbols']} symbols ({res['updated']} updated, {res['failed']} failed).")

        st.divider()
        health = provider.get_health()
        st.caption(f"OHLCV: {health.get('ohlcv_symbols', 0)} syms ({health.get('ohlcv_last', '—')})")
        st.caption(f"Disk Free: {health.get('disk_free_gb', '?')} GB [{health.get('worst', 'OK')}]")
        fills = health.get("fills") or {}
        st.caption(f"Real Fills Logged: {fills.get('total_filled', 0)}/{fills.get('total_attempts', 0)}")
        o1 = health.get("o1_fills") or {}
        if o1:
            from src.fill_experiment import o1_progress_line

            st.caption(o1_progress_line(o1) + " · rule: docs/PREREG_O1_FILL_K_20261125.md")
        st.caption("Live Execution: OFF (research/manual only)")

    snap: RadarSnapshot = st.session_state.radar_snap
    posture_cls = "good" if snap.regime_posture == "RISK_ON" else ("bad" if snap.regime_posture == "RISK_OFF" else "warn")
    kalshi_txt = (
        f"{snap.kalshi_macro[0]['label']} ({snap.kalshi_macro[0]['prob_pct']:.0f}%)"
        if snap.kalshi_macro
        else "—"
    )
    spy_txt = f"${snap.spy_last:.2f} ({snap.spy_chg_1d_pct:+.1f}%)" if snap.spy_last is not None else "—"
    qqq_txt = f"{snap.qqq_chg_1d_pct:+.1f}%" if snap.qqq_chg_1d_pct is not None else "—"
    vix_txt = f"VIX {snap.vix:.1f}" if snap.vix is not None else "VIX —"
    if getattr(snap, "spy_rv20", None) is not None:
        vix_txt += f" · RV20 {snap.spy_rv20:.0f}%"

    st.markdown(
        f"""
        <div class="ribbon-bar">
            <div class="ribbon-item"><span class="ribbon-lbl">SPY</span><span class="ribbon-val">{spy_txt}</span></div>
            <div class="ribbon-item"><span class="ribbon-lbl">QQQ 1d</span><span class="ribbon-val">{qqq_txt}</span></div>
            <div class="ribbon-item"><span class="ribbon-lbl">Market Regime</span><span class="ribbon-val {posture_cls}">{snap.regime_posture} ({vix_txt})</span></div>
            <div class="ribbon-item"><span class="ribbon-lbl">Next Macro</span><span class="ribbon-val">{snap.next_macro_event}</span></div>
            <div class="ribbon-item"><span class="ribbon-lbl">Kalshi Odds</span><span class="ribbon-val">{kalshi_txt}</span></div>
            <div class="ribbon-item"><span class="ribbon-lbl">Updated</span><span class="ribbon-val">{datetime.now().strftime('%H:%M')}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_today, tab_disc, tab_radar, tab_wb, tab_book, tab_legacy = st.tabs(
        [
            "🧭 Today & TFSA Core",
            "🗺️ Discover",
            "📡 Opportunity Radar",
            "⚖️ Shares vs Options",
            "📘 Book & Fills",
            "🔬 Legacy Gated Scan",
        ]
    )

    # ── TAB 1: TODAY & TFSA CORE REBALANCE ───────────────────────────────────
    with tab_today:
        st.markdown("### Your Plain-English Daily Action Plan")
        for step in plain_english_action_plan(snap):
            st.markdown(f"- {step}")

        col_alloc, col_reb = st.columns([1, 1.3])
        with col_alloc:
            st.markdown("#### Core TFSA Weights (`VALIDATED: VM_OUTLOOK_TOP3`)")
            if snap.core_allocation:
                df_alloc = pd.DataFrame(snap.core_allocation)
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=df_alloc["ticker"],
                            values=[round(float(w) * 100.0, 1) for w in df_alloc["target_weight"]],
                            hole=0.45,
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_pie.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("No Core Allocation available yet — click Refresh Local Radar.")

        with col_reb:
            st.markdown("#### Interactive TFSA Rebalance Calculator")
            r_c1, r_c2, r_c3 = st.columns([1, 1.4, 0.8])
            acct_usd = r_c1.number_input("TFSA Core USD ($)", min_value=100.0, value=5000.0, step=250.0)
            cur_spec = r_c2.text_input("Current Shares (optional)", placeholder="e.g. XLE:10,XLV:5")
            frac = r_c3.toggle("Fractional", value=True)
            if snap.core_allocation:
                plan = compute_tfsa_rebalance(
                    snap.core_allocation,
                    account_usd=float(acct_usd),
                    current_shares=parse_holdings_spec(cur_spec),
                    fractional=bool(frac),
                )
                if plan.rebalance_recommended:
                    st.success(f"Rebalance recommended — max drift {plan.max_drift_pct:.1f}% · Cash/SGOV after: ${plan.estimated_cash_after_usd:,.2f}")
                else:
                    st.info(f"On target — max drift {plan.max_drift_pct:.1f}% (< {plan.drift_threshold_pct:.0f}% threshold).")
                ord_df = pd.DataFrame(
                    [
                        {
                            "Action": o.action,
                            "Ticker": o.ticker,
                            "Target %": f"{o.target_weight_pct:.1f}%",
                            "Spot": f"${o.spot:.2f}" if o.ticker != "CASH/SGOV" else "—",
                            "Order Sh": f"{o.delta_shares:+.4g}" if o.action in ("BUY", "SELL") else "—",
                            "Delta $": f"${o.delta_usd:+,.2f}",
                        }
                        for o in plan.orders
                    ]
                )
                st.dataframe(ord_df, width="stretch", hide_index=True)

    # ── TAB 1b: TOP-DOWN DISCOVER (① market → ② sectors → ③ ideas → ④ evidence card) ──
    with tab_disc:
        from src.topdown.web import render_discover

        if st.button("Rebuild Discover from local archives", key="disc-rebuild"):
            st.session_state.pop("topdown_hub", None)
            st.session_state.pop("topdown_cards", None)
        if "topdown_hub" not in st.session_state:
            with st.spinner("Building the top-down view (local data only)…"):
                st.session_state.topdown_hub = provider.get_hub(refresh=True)
        cards = st.session_state.setdefault("topdown_cards", {})
        hub = st.session_state.topdown_hub

        def _card(tk: str):
            if tk not in cards:
                cards[tk] = hub.card(tk, dossier=provider.get_dossier(tk, fetch_chain=False))
            return cards[tk]

        render_discover(st, hub, get_card=_card, default_ticker=st.session_state.get("focused_ticker"))

    # ── TAB 2: UNIFIED OPPORTUNITY RADAR ─────────────────────────────────────
    with tab_radar:
        fc1, fc2, fc3 = st.columns([1.4, 1.8, 1])
        preset = fc1.selectbox(
            "Filter Preset",
            ["All", "Proven + Promising", "Stocks Only", "Options Only", "Clean (0 Flags)", "Confluence >= 2"],
        )
        preset_map = {
            "All": "",
            "Proven + Promising": "actionable",
            "Stocks Only": "stock",
            "Options Only": "option",
            "Clean (0 Flags)": "clean",
            "Confluence >= 2": "conf>=2",
        }
        custom_q = fc2.text_input("Search / Micro-query", placeholder="e.g. SPY, pullback, fric<10%")
        sort_by = fc3.selectbox("Sort By", ["Evidence Tier", "Confluence", "Lowest Friction", "Ticker"])

        full_q = f"{preset_map[preset]} {custom_q}".strip()
        filtered = filter_opportunities(snap.opportunities, full_q)
        if sort_by == "Confluence":
            filtered.sort(key=lambda o: (-o.confluence_count, o.friction_value, o.ticker))
        elif sort_by == "Lowest Friction":
            filtered.sort(key=lambda o: (o.friction_value, o.ticker))
        elif sort_by == "Ticker":
            filtered.sort(key=lambda o: (o.ticker, o.playbook_id))

        left_r, right_r = st.columns([1.45, 1])
        with left_r:
            st.markdown(f"#### Radar Setups ({len(filtered)} / {len(snap.opportunities)})")
            if filtered:
                rdf = pd.DataFrame(
                    [
                        {
                            "Evidence": TIER_PLAIN.get(o.evidence_tier, o.evidence_tier),
                            "Ticker": o.ticker,
                            "Strategy": o.playbook_label,
                            "Dir": o.direction,
                            "Match": o.confluence_count,
                            "Cost": o.friction_label or f"{o.friction_value*100:.2f}%",
                            "Flags": ", ".join(o.risk_flags) or "—",
                            "Setup": o.setup_summary,
                        }
                        for o in filtered
                    ]
                )
                sel = st.dataframe(
                    rdf,
                    width="stretch",
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    height=380,
                )
                if sel.selection.get("rows"):
                    st.session_state.focused_ticker = filtered[sel.selection["rows"][0]].ticker
            else:
                st.info("No setups match your current filter.")

        with right_r:
            tk_focus = st.session_state.focused_ticker
            dos = provider.get_dossier(tk_focus, fetch_chain=False)
            light, reason = plain_english_traffic_light(dos)
            st.markdown(f"#### Dossier: `{dos.ticker}` — {light}")
            st.caption(reason)
            m1, m2, m3 = st.columns(3)
            m1.metric("Spot", f"${dos.spot:.2f}" if dos.spot else "N/A", f"{dos.chg_1d_pct:+.2f}%" if dos.chg_1d_pct is not None else None)
            m2.metric("ATM IV / RV20", f"{dos.atm_iv:.1f}%" if dos.atm_iv else "—", f"RV {dos.rv_20_ann:.1f}%" if dos.rv_20_ann else None)
            m3.metric("RSI(2) / 200MA", f"{dos.rsi_2:.1f}" if dos.rsi_2 is not None else "—", f"{dos.dist_sma200_pct:+.1f}%" if dos.dist_sma200_pct is not None else None)
            st.plotly_chart(build_dossier_price_figure(dos), width="stretch")
            if dos.headlines:
                st.markdown("**Recent Headlines:**")
                for h in dos.headlines[:2]:
                    st.markdown(f"- {h['title']} _({h.get('date', '')})_")

    # ── TAB 3: SHARES VS OPTIONS EXPRESS WORKBENCH ───────────────────────────
    with tab_wb:
        wc1, wc2, wc3, wc4, wc5, wc6 = st.columns([1, 1, 1, 0.9, 1, 0.9])
        wb_tk = wc1.text_input("Ticker", value=st.session_state.focused_ticker).strip().upper() or "SPY"
        wb_dir = wc2.selectbox("Direction", ["BULLISH", "BEARISH", "NEUTRAL"])
        wb_tgt = wc3.slider("Expected Move %", -25, 25, 6 if wb_dir == "BULLISH" else (-6 if wb_dir == "BEARISH" else 0))
        wb_days = wc4.number_input("Horizon (d)", min_value=5, max_value=180, value=30, step=5)
        wb_risk = wc5.number_input("Risk Budget ($)", min_value=50.0, value=500.0, step=50.0)
        wb_owns = wc6.toggle("Own 100 sh", value=False)

        dos_wb = provider.get_dossier(wb_tk, fetch_chain=False)
        spot_wb = dos_wb.spot if dos_wb.spot is not None else 100.0
        iv_wb = (dos_wb.atm_iv / 100.0) if dos_wb.atm_iv is not None else ((dos_wb.rv_20_ann / 100.0) if dos_wb.rv_20_ann else 0.25)
        t_move = 0.0 if wb_dir == "NEUTRAL" else (float(wb_tgt) / 100.0)
        if wb_dir == "BEARISH" and t_move > 0:
            t_move = -t_move

        view_wb = DetailedView(
            symbol=wb_tk,
            direction=wb_dir,
            target_move_pct=t_move,
            horizon_days=int(wb_days),
            risk_budget_usd=float(wb_risk),
            account_type="tfsa",
            owns_100_shares=bool(wb_owns),
        )
        vehicles = provider.get_express(view_wb, spot=spot_wb, iv=iv_wb)
        st.success(plain_english_vehicle_verdict(vehicles, view_wb))

        vdf = pd.DataFrame(
            [
                {
                    "Vehicle": ("★ " if v.recommended else "") + v.vehicle,
                    "Structure": v.description,
                    "TFSA": "Yes" if v.tfsa_legal else "No",
                    "Max Risk $": f"${v.capital_at_risk_usd:,.0f}",
                    "Spread %": f"{v.friction_pct_of_reward:.1f}%",
                    "Market EV": f"${v.ev_mkt_net_usd:+,.0f}",
                    "Your View EV": f"${v.ev_view_net_usd:+,.0f}",
                    "Win Odds": f"{v.pop_view*100:.0f}%",
                    "Walk-Away Limit": v.walk_away_label,
                }
                for v in vehicles
            ]
        )
        st.dataframe(vdf, width="stretch", hide_index=True)

        p_left, p_right = st.columns(2)
        with p_left:
            rnd_fig, _src = build_rnd_comparison_figure(spot_wb, iv_wb, view_wb)
            st.plotly_chart(rnd_fig, width="stretch")
        with p_right:
            st.plotly_chart(build_vehicle_edge_figure(vehicles, float(wb_risk)), width="stretch")

    # ── TAB 4: BOOK, EXIT ALERTS & REAL-FILL EXPERIMENT ──────────────────────
    with tab_book:
        book = provider.get_book()
        if book.get("exit_alerts"):
            for al in book["exit_alerts"]:
                st.warning(f"EXIT ALERT — {al}")
        b1, b2 = st.columns([1.4, 1])
        with b1:
            st.markdown("#### Stock & ETF Picks (`data/own_picks.db`)")
            picks = book.get("equity_picks") or []
            if picks:
                st.dataframe(pd.DataFrame(picks), width="stretch", hide_index=True)
            else:
                st.info("No stock/ETF picks logged yet. Use `opt log` or `[l]` in `opt` TUI.")
            st.markdown("#### Open Paper Option Positions (`paper_trades.db`)")
            p_opts = book.get("paper_options") or []
            if p_opts:
                st.dataframe(pd.DataFrame(p_opts), width="stretch", hide_index=True)
            else:
                st.caption("0 open paper option positions.")
        with b2:
            st.markdown("#### Long-Term Accumulation Plan (`longterm_plan.json`)")
            lt = book.get("longterm_holdings") or []
            if lt:
                st.dataframe(pd.DataFrame(lt), width="stretch", hide_index=True)
            else:
                st.caption("No long-term tranches configured.")
            st.markdown("#### Log Real Limit-Order Fill (`data/fills.db`)")
            with st.form("fill_form", clear_on_submit=True):
                ff_broker = st.selectbox("Broker", ["ibkr", "wealthsimple"])
                ff_tk = st.text_input("Ticker", value=wb_tk).strip().upper()
                ff_struct = st.text_input("Structure", value="Bull Put")
                ff_side = st.selectbox("Side", ["SELL", "BUY"])
                fc_a, fc_b, fc_c = st.columns(3)
                ff_bid = fc_a.number_input("Bid", min_value=0.01, value=4.00, step=0.05)
                ff_ask = fc_b.number_input("Ask", min_value=0.01, value=4.20, step=0.05)
                ff_lim = fc_c.number_input("Limit", min_value=0.01, value=4.12, step=0.05)
                ff_filled = st.checkbox("Filled?", value=True)
                if st.form_submit_button("Save Fill Attempt"):
                    provider.log_fill_attempt(
                        broker=ff_broker,
                        ticker=ff_tk,
                        structure=ff_struct,
                        side=ff_side,
                        bid=float(ff_bid),
                        ask=float(ff_ask),
                        limit_price=float(ff_lim),
                        filled=bool(ff_filled),
                        fill_price=float(ff_lim) if ff_filled else None,
                    )
                    st.success(f"Recorded fill attempt for {ff_tk}.")

    # ── TAB 5: LEGACY OPTIONS SCANNER (PROPERLY GATED) ───────────────────────
    with tab_legacy:
        st.markdown("#### Live Multi-Ticker Options Scan (Routed Through `gate_board()`)")
        st.caption(
            "Every contract passes the repo's friction (<=25% reward), negative-EV consistency, and top-quintile gates. "
            "No quality_score '#1' ranking is claimed."
        )
        lc1, lc2, lc3, lc4 = st.columns([1.2, 1.5, 0.8, 0.8])
        scan_mode = lc1.selectbox("Scan Mode", ["Single-stock", "Discovery scan", "Premium Selling"])
        tk_in = lc2.text_input("Ticker(s) (comma-separated)", value="SPY,QQQ,AAPL")
        min_dte = lc3.number_input("Min DTE", min_value=7, max_value=120, value=30)
        max_dte = lc4.number_input("Max DTE", min_value=14, max_value=180, value=60)

        if st.button("🚀 Run Gated Live Option Scan"):
            from src.options_screener import get_market_context, run_scan, setup_logging
            from src.pick_ranking import gate_board

            tickers = [t.strip().upper() for t in tk_in.split(",") if t.strip()]
            if scan_mode == "Single-stock" and tickers:
                tickers = [tickers[0]]
            with st.spinner(f"Scanning {', '.join(tickers)}…"):
                trend, vreg, macro_r, tnx = get_market_context()
                res = run_scan(
                    mode=scan_mode,
                    tickers=tickers,
                    budget=None,
                    max_expiries=4,
                    min_dte=int(min_dte),
                    max_dte=int(max_dte),
                    trader_profile="swing",
                    logger=setup_logging(),
                    market_trend=trend,
                    volatility_regime=vreg,
                    macro_risk_active=macro_r,
                    tnx_change_pct=tnx,
                    verbose=False,
                )
                board = gate_board(res.picks)
                st.session_state.legacy_board = board
                st.session_state.legacy_mode = scan_mode

        board = st.session_state.get("legacy_board")
        if board is not None:
            st.caption(f"Scanned {board.scanned} contracts · {len(board.kept)} cleared gates · {len(board.refused)} refused ({dict(board.reasons)})")
            if not board.kept.empty:
                show_cols = [
                    c for c in ["symbol", "type", "strike", "expiration", "bid", "ask", "premium", "impliedVolatility", "delta", "ev_per_contract", "volume", "openInterest"]
                    if c in board.kept.columns
                ]
                st.dataframe(board.kept[show_cols], width="stretch", hide_index=True)
            else:
                st.warning("Nothing cleared the friction/EV gates — try Wide 60–120d ETF Spreads or Shares vs Options above.")


if __name__ == "__main__":
    render_main()
