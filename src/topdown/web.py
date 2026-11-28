"""🗺️ Discover tab for the Streamlit web cockpit — same four numbered sections as the CLI/TUI.

The DataFrame builders are pure (no Streamlit) so they can be unit-tested; only
`render_discover` touches `st`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .card import EvidenceCard
from .hub import DiscoveryHub
from .lenses import LENSES, LENS_TAG_MEANING, NULL_LEDGER, LensResult, tags_by_ticker
from .render import LABEL_MEANING

CHIP_BG = {
    "PROVEN": "#5ec98d", "PROMISING": "#82aad2", "TESTING": "#c678dd", "WATCHLIST": "#d6a452",
    "CONTEXT": "#8b949e", "AVOID": "#e06c75", "RETIRED": "#8b949e",
}
VERDICT_BG = {"green": "#5ec98d", "amber": "#d6a452", "red": "#e06c75", "grey": "#8b949e"}


def chip_html(label: str) -> str:
    bg = CHIP_BG.get(label, "#8b949e")
    return (f'<span style="background:{bg};color:#0e1117;padding:1px 7px;border-radius:4px;'
            f'font-weight:700;font-size:0.78rem;letter-spacing:0.03em">{label}</span>')


def board_frame(hub: DiscoveryHub) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": r.rank, "ETF": r.etf, "Sector / asset": r.name, "Call": r.posture,
        "Sleeve %": round(r.weight * 100.0, 1), "RS vs SPY 1m %": r.rs_21d_pct, "RS vs SPY 3m %": r.rs_63d_pct,
        "vs 200d %": r.vs_200d_pct, "Stocks >50d %": r.pct_above_50d, "Stocks >200d %": r.pct_above_200d,
        "Stocks (n)": r.n_members or None,
    } for r in hub.board.rows])


def ratios_frame(hub: DiscoveryHub) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ratio": r.name, "What it compares": r.label, "A rise means": r.rise_means,
        "1m %": r.chg_21d_pct, "3m %": r.chg_63d_pct, "vs 200d %": r.vs_200d_pct, "1y percentile": r.pctile_1y,
        "As of": r.as_of or (r.missing or ""),
    } for r in hub.macro.ratios])


def lens_frame(res: LensResult, all_results: List[LensResult]) -> pd.DataFrame:
    tags = tags_by_ticker(all_results)
    return pd.DataFrame([{
        "Ticker": r.ticker, "Sector": r.sector_etf or "", "Listed in": " ".join(t for t, _ in tags.get(r.ticker, [])),
        "Evidence": r.evidence_label or res.lens.label,
        "Warnings": ", ".join(r.flags), "Why it is listed": r.why, "As of": r.as_of or "",
    } for r in res.rows])


def members_frame(hub: DiscoveryHub, etf: str) -> pd.DataFrame:
    from .peers import _metrics
    from .sectors import sector_members

    tags = tags_by_ticker(hub.lenses)
    rows = []
    for sym in sector_members(etf, hub.ohlcv):
        m = _metrics(hub.ohlcv, sym)
        rows.append({
            "Ticker": sym, "12-1m %": m["ret_12_1_pct"], "1m %": m["ret_21d_pct"], "3m %": m["ret_63d_pct"],
            "vs 200d %": m["vs_200d_pct"], "Vol 20d %": m["rv20_pct"], "From 52w high %": m["from_52w_high_pct"],
            "Listed in": " ".join(t for t, _ in tags.get(sym, [])),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("12-1m %", ascending=False, na_position="last") if not df.empty else df


def peers_frame(card: EvidenceCard) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ticker": ("▶ " if r.is_subject else "") + r.ticker + ("" if r.in_sector else " ·"),
        f"Corr w/ {card.ticker}": r.corr_126d, "Beta vs SPY": r.beta_126d, "1m %": r.ret_21d_pct,
        "3m %": r.ret_63d_pct, "12-1m %": r.ret_12_1_pct, "Vol 20d %": r.rv20_pct, "vs 200d %": r.vs_200d_pct,
        "IV/RV": r.iv_rv, "IV as of": r.iv_as_of or "",
    } for r in card.peers.rows])


def provenance_frame(card: EvidenceCard) -> pd.DataFrame:
    return pd.DataFrame([{
        "Section": p.section, "Source": p.source, "As of": p.as_of or "—",
        "Age (days)": p.age_days, "Note": p.note,
    } for p in card.provenance])


def null_ledger_frame() -> pd.DataFrame:
    return pd.DataFrame([{"Idea (tested here)": a, "Result": b, "Date": c, "Source": d} for a, b, c, d in NULL_LEDGER])


def dte_strip_frame(bundle: Any) -> pd.DataFrame:
    rows = []
    for r in bundle.dte_rows:
        p_s = (
            f"{r.p_win*100:.0f}% [{r.p_win_ci95[0]*100:.0f}–{r.p_win_ci95[1]*100:.0f}%] (n={r.n_backtest})"
            if (r.p_win is not None and r.n_backtest > 0 and r.p_win_ci95)
            else "—"
        )
        rows.append(
            {
                "DTE Bucket": r.bucket_label,
                "Verdict": r.evidence_label,
                "Best Setup": f"{r.ticker} {r.strikes} ({r.dte}d)" if r.ticker else "—",
                "Mid $": r.credit_mid,
                "Limit $ (k≤0.35)": getattr(r, "limit_k035", None),
                "Max Risk $": getattr(r, "max_risk_usd", None),
                "Friction %": round(r.friction_pct * 100.0, 1) if r.friction_pct is not None else None,
                "Calibrated P(Win) [95% CI, n]": p_s,
                "PF": r.pf,
                "Avg Loss %": round(r.avg_loss * 100.0, 1) if r.avg_loss is not None else None,
                "Δ-Adj Excess %": round(r.excess_limit * 100.0, 2) if r.excess_limit is not None else None,
                "Why": r.why,
            }
        )
    return pd.DataFrame(rows)


def _box(st: Any, label: str, title: str, body: str) -> None:
    border = CHIP_BG.get(label, "#8b949e")
    st.markdown(
        f'<div style="border-left:4px solid {border};background:#161b22;padding:0.55rem 0.9rem;'
        f'border-radius:4px;margin:0.3rem 0 0.6rem 0">{chip_html(label)} <b>{title}</b><br>'
        f'<span style="color:#c9d1d9">{body}</span></div>',
        unsafe_allow_html=True,
    )


def render_discover(st: Any, hub: DiscoveryHub, *, get_card: Callable[[str], EvidenceCard],
                    default_ticker: Optional[str] = None) -> None:
    f = hub.freshness()
    st.caption(
        f"① Market › ② Sectors › ③ Ideas › ④ Evidence card — top-down, each step labelled by its evidence. "
        f"Data: {f['symbols']} symbols, SPY through {f['spy_last']}"
        + (f" · {f['behind']} symbols behind — run `opt refresh`" if f["behind"] else " · all current")
    )
    st.markdown(" ".join(f"{chip_html(k)} <span style='color:#8b949e;font-size:0.8rem'>{v}</span>"
                         for k, v in LABEL_MEANING.items()), unsafe_allow_html=True)

    # 0 · BEST ACTION TODAY + DTE-SORTED OPTIONS STRIP
    from .best_action import build_best_action

    st.markdown("### 0 · Best Action Today — face-value picks & backtested odds")
    d_c1, d_c2 = st.columns([2, 1])
    sort_choice = d_c1.radio(
        "Options DTE sort",
        ["dte", "evidence"],
        format_func=lambda v: "Shortest DTE first (0DTE → 6M+)" if v == "dte" else "Best evidence first",
        horizontal=True,
        label_visibility="collapsed",
        key="disc-dte-sort",
    )
    live_dte_flag = d_c2.checkbox("Fetch live Alpaca/Yahoo options", value=False, key="disc-dte-live")
    ba = build_best_action(hub, live_options=bool(live_dte_flag), sort_dte_by=str(sort_choice))
    if ba.profile_summary:
        st.caption(f"Profile: {ba.profile_summary} · {ba.o1_fill_meter}")
    for act in ba.actions:
        sz_html = f"<span style='color:#5ec98d;font-weight:600'>{act.sizing_text}</span><br>" if act.sizing_text else ""
        _box(
            st,
            "AVOID" if act.evidence_label == "CLOSED" else act.evidence_label,
            f"[{act.account}] {act.headline}",
            f"{sz_html}<b>{act.prob_text}</b><br><span style='color:#8b949e'>{act.why}</span>",
        )
    with st.expander(f"Options by Expiry (0DTE → 240D) [TAXABLE_ONLY] · Clock: {ba.execution_window}", expanded=True):
        if ba.o1_fill_meter:
            st.caption(f"O1 Fill Quality Meter: **{ba.o1_fill_meter}**")
        st.dataframe(dte_strip_frame(ba), width="stretch", hide_index=True)
        st.caption(
            f"{ba.execution_note}"
            + (f" · {ba.implied_calibration_note}" if ba.implied_calibration_note else "")
        )

    # ① MARKET
    from .render import regime_read

    st.markdown("### ① Market — what kind of market is this?")
    read, _style = regime_read(hub)
    st.markdown(f"**Read:** {read} <span style='color:#8b949e'>(a description of today, not a forecast)</span>",
                unsafe_allow_html=True)
    s, v = hub.macro.spy, hub.macro.vol
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SPY vs 200-day", f"{s.get('vs_200d_pct', 0):+.1f}%" if s.get("as_of") else "—")
    c2.metric("SPY last month", f"{s.get('ret_21d_pct', 0):+.1f}%" if s.get("as_of") else "—")
    c3.metric("VIX", f"{v['vix']:.1f}" if v.get("vix") is not None else "—",
              help=f"{v.get('vix_pctile_1y', '—')}th percentile of the last year")
    c4.metric("VIX curve (VIX/VIX3M)", str(v.get("term", "—")).title(),
              f"{v.get('term_ratio'):.2f}" if v.get("term_ratio") else None, delta_color="off")
    c5.metric("SPY realized vol (20d)", f"{s.get('rv20_pct') or 0:.0f}%" if s.get("as_of") else "—")
    up, n = hub.macro.ratio_tally()
    with st.expander(f"Rotation ratios — {up}/{n} risk-appetite ratios rising over 3m (descriptive)", expanded=False):
        st.dataframe(ratios_frame(hub), width="stretch", hide_index=True)
        fred = [x for x in hub.macro.fred if x.get("value") is not None]
        if fred:
            st.dataframe(pd.DataFrame(fred)[["label", "value", "unit", "chg_1m", "pctile_1y", "as_of"]],
                         width="stretch", hide_index=True)
        else:
            st.caption("FRED credit-spread / yield-curve series not cached — run `opt discover --refresh-macro`.")
    st.caption(f"Source · SPY {s.get('as_of')} (data/equity_ohlcv.db) · Cboe VIX {v.get('as_of')} · "
               "context only: VIX→direction tested NOT_ESTABLISHED; ratios untested.")

    # ② SECTORS
    st.markdown("### ② Sectors — where is the strength?")
    b = hub.board
    held = ", ".join(f"{r.etf} {r.weight*100:.0f}%" for r in b.rows if r.weight > 0) or "nothing"
    _box(st, "PROVEN", "Ranked by the validated rotation model",
         f"This month's TFSA core sleeve: <b>{held}</b> · cash/T-bills {(b.sleeve_cash or 0)*100:.0f}% "
         f"(scaled {b.sleeve_scale or 0:.2f}× — basket 20d vol {(b.basket_rv20 or 0)*100:.1f}% vs 14% target). "
         "FAVOR = held · AVOID = bearish call (only fires when SPY is below trend). "
         "Relative-strength and breadth columns are description only.")
    st.dataframe(
        board_frame(hub), width="stretch", hide_index=True, height=35 * (len(b.rows) + 1) + 3,
        column_config={
            "Sleeve %": st.column_config.NumberColumn(format="%.0f%%"),
            "RS vs SPY 1m %": st.column_config.NumberColumn(format="%+.1f%%"),
            "RS vs SPY 3m %": st.column_config.NumberColumn(format="%+.1f%%"),
            "vs 200d %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Stocks >50d %": st.column_config.NumberColumn(format="%.0f%%"),
            "Stocks >200d %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    sectors = [r.etf for r in b.rows if r.n_members]
    pick = st.selectbox("Drill into a sector's stocks (② › constituents)", sectors, index=0 if sectors else None,
                        format_func=lambda e: f"{e} — {next((r.name for r in b.rows if r.etf == e), e)}")
    if pick:
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(members_frame(hub, pick), width="stretch", hide_index=True, column_config={
            "12-1m %": pct, "1m %": pct, "3m %": pct, "vs 200d %": pct, "From 52w high %": pct,
            "Vol 20d %": st.column_config.NumberColumn(format="%.0f%%"),
        })
        st.caption("Sorted by 12-1 month return — description, not a forecast. 'Listed in' tags: "
                   + " · ".join(f"{k} = {v}" for k, v in LENS_TAG_MEANING.items()))
    st.caption(f"Source · prices as of {b.as_of} · docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md")

    # ③ IDEAS
    st.markdown("### ③ Ideas — why look here?")
    lens_ids = [lz.id for lz in LENSES]
    titles = {lz.id: f"[{lz.label}] {lz.title}" for lz in LENSES}
    lid = st.radio("Lens", lens_ids, format_func=lambda i: titles[i], horizontal=True, label_visibility="collapsed")
    res = hub.lens(lid)
    if res is not None:
        _box(st, res.lens.label, f"{res.lens.title} — {res.lens.question}",
             f"<b>Evidence:</b> {res.lens.evidence}<br><span style='color:#8b949e'>Sorted by: {res.lens.sort_note} · "
             f"{len(res.rows)} row(s) · source {res.lens.source}</span>"
             + (f"<br><span style='color:#d6a452'>{res.note}</span>" if res.note else ""))
        if res.rows:
            st.dataframe(lens_frame(res, hub.lenses), width="stretch", hide_index=True)
        else:
            st.caption("Nothing listed right now.")
    with st.expander("Already tested here — does NOT work (don't re-chase)"):
        st.dataframe(null_ledger_frame(), width="stretch", hide_index=True)

    # ④ EVIDENCE CARD
    st.markdown("### ④ Evidence card — one ticker, every number with its source")
    choices: List[str] = list(dict.fromkeys(
        ([default_ticker] if default_ticker else []) + [x.ticker for r in hub.lenses for x in r.rows]))
    cc1, cc2 = st.columns([1, 1])
    pick_t = cc1.selectbox("Ticker from the lenses above", choices, index=0 if choices else None)
    typed = cc2.text_input("…or type any ticker", value="").strip().upper()
    ticker = typed or pick_t
    if not ticker:
        return
    card = get_card(ticker)
    _render_card(st, card)


def _render_card(st: Any, card: EvidenceCard) -> None:
    d = card.dossier
    bg = VERDICT_BG.get(card.verdict_style, "#8b949e")
    spot = f"${d.spot:.2f}" if d.spot is not None else "price n/a"
    reasons = "".join(f"<li>{r}</li>" for r in card.reasons)
    warns = "".join(f"<li style='color:#d6a452'>⚠ {w}</li>" for w in card.warnings)
    hits = " ".join(f"{chip_html(lab)} {title}" for title, lab in card.lens_hits)
    st.markdown(
        f'<div style="border:2px solid {bg};border-radius:6px;padding:0.6rem 1rem;background:#161b22">'
        f'<span style="background:{bg};color:#0e1117;padding:2px 9px;border-radius:4px;font-weight:800">'
        f'{card.verdict}</span> <b style="font-size:1.15rem">{card.ticker}</b> {spot}'
        f'<ul style="margin:0.4rem 0 0.2rem 1rem">{reasons}{warns}</ul>'
        + (f'<div style="color:#8b949e">Listed in: {hits}</div>' if hits else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**2 · Price structure**")
        try:
            from src.dashboard import build_dossier_price_figure

            st.plotly_chart(build_dossier_price_figure(d), width="stretch", key=f"disc-price-{card.ticker}")
        except Exception:
            pass
        st.markdown(
            f"Trend vs 20/50/200d: {d.dist_sma20_pct or 0:+.1f}% / {d.dist_sma50_pct or 0:+.1f}% / "
            f"{d.dist_sma200_pct or 0:+.1f}% · 12-1m {d.mom_12_1_pct or 0:+.1f}% · RSI(2) {d.rsi_2 or 0:.0f}"
        )
    with right:
        st.markdown("**4 · Options & market-implied odds**")
        o1, o2, o3 = st.columns(3)
        o1.metric("ATM IV (30d)", f"{d.atm_iv:.1f}%" if d.atm_iv is not None else "—",
                  f"IV/RV {d.iv_vs_rv_ratio:.2f}" if d.iv_vs_rv_ratio is not None else None, delta_color="off")
        o2.metric("P(>+5%) 30d", f"{d.rnd_prob_up_5pct*100:.0f}%" if d.rnd_prob_up_5pct is not None else "—")
        o3.metric("P(<−5%) 30d", f"{d.rnd_prob_dn_5pct*100:.0f}%" if d.rnd_prob_dn_5pct is not None else "—")
        st.caption(card.prob_label)
        st.markdown("**5 · Events, revisions & news**")
        earn = (f"in {d.days_to_earnings}d ({d.earnings_date})" if d.days_to_earnings is not None
                else "no upcoming date on file")
        lines = [f"- Earnings: {earn}", f"- Next macro: {d.next_macro_event}"]
        if d.rev_breadth_30d is not None:
            lines.append(f"- EPS revision breadth: {d.rev_breadth_30d:+.2f} (TESTING — readout ≥ 2027-01-15)")
        for h in d.headlines[:3]:
            lines.append(f"- {h.get('date', '')} {h['title']}")
        if d.cdr_ticker:
            lines.append(f"- TFSA: Canadian CDR {d.cdr_ticker}")
        st.markdown("\n".join(lines))
    st.markdown("**3 · Peers** — " + card.peers.method)
    pct = st.column_config.NumberColumn(format="%+.1f%%")
    st.dataframe(peers_frame(card), width="stretch", hide_index=True, column_config={
        "1m %": pct, "3m %": pct, "12-1m %": pct, "vs 200d %": pct,
        "Vol 20d %": st.column_config.NumberColumn(format="%.0f%%"),
        "Beta vs SPY": st.column_config.NumberColumn(format="%.2f"),
        f"Corr w/ {card.ticker}": st.column_config.NumberColumn(format="%.2f"),
        "IV/RV": st.column_config.NumberColumn(format="%.2f"),
    })
    st.markdown("**6 · Portfolio fit** — what this adds to what you already hold (descriptive, never a gate)")
    fit = card.fit
    if fit is None:
        st.caption("Holdings not checked.")
    else:
        by_src: Dict[str, List[str]] = {}
        for h in fit.holdings:
            by_src.setdefault(h.source, []).append(h.ticker + (f" {h.weight * 100:.0f}%" if h.weight else ""))
        lines = ["- You hold: " + (" · ".join(f"{k}: {', '.join(v)}" for k, v in sorted(by_src.items()))
                                  or "nothing recorded")]
        lines += [f"- <span style='color:#d6a452'>⚠ {w}</span>" for w in fit.warnings]
        lines += [f"- {n}" for n in fit.notes]
        if fit.correlations:
            lines.append("- Co-movement (126d): " + ", ".join(f"{tk} {r:+.2f}" for tk, r in fit.correlations[:5]))
        if fit.diversifiers:
            lines.append("- Least correlated sleeve-universe ETFs: "
                         + ", ".join(f"{tk} {r:+.2f}" for tk, r in fit.diversifiers) + " (information, not a recommendation)")
        st.markdown("\n".join(lines), unsafe_allow_html=True)
        st.caption(fit.method)
    st.markdown("**7 · Where every number came from**")
    st.dataframe(provenance_frame(card), width="stretch", hide_index=True)


__all__ = [
    "chip_html", "board_frame", "ratios_frame", "lens_frame", "members_frame", "peers_frame",
    "provenance_frame", "null_ledger_frame", "dte_strip_frame", "render_discover",
]
