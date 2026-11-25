"""Shared Rich renderers for the `opt` CLI (clean, compact, `--no-color` safe)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.types import VehicleComparison

TIER_STYLE = {
    "VALIDATED": "bold #5ec98d",
    "LEAD": "bold #82aad2",
    "CONTEXT": "dim #d6a452",
    "RESEARCH": "dim",
    "CLOSED": "dim #e06c75",
}

TIER_SHORT = {
    "VALIDATED": "VAL",
    "LEAD": "LEAD",
    "CONTEXT": "CTX",
    "RESEARCH": "RES",
    "CLOSED": "CLSD",
}


def make_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color, highlight=False, soft_wrap=False)


def render_ribbon(snap: RadarSnapshot) -> Text:
    t = Text()
    spy_s = f"{snap.spy_last:.2f}" if snap.spy_last is not None else "—"
    spy_c = f"{snap.spy_chg_1d_pct:+.1f}%" if snap.spy_chg_1d_pct is not None else ""
    qqq_c = f"{snap.qqq_chg_1d_pct:+.1f}%" if snap.qqq_chg_1d_pct is not None else ""
    vix_s = f"{snap.vix:.1f}" if snap.vix is not None else "—"
    posture_style = (
        "bold #5ec98d" if snap.regime_posture == "RISK_ON"
        else ("bold #e06c75" if snap.regime_posture == "RISK_OFF" else "bold #d6a452")
    )
    t.append(f"SPY {spy_s} {spy_c}", style="bold")
    t.append("  |  ", style="dim")
    t.append(f"QQQ {qqq_c or '—'}", style="bold")
    t.append("  |  ", style="dim")
    t.append(f"RV/VIX {vix_s} ", style="bold")
    t.append(f"[{snap.regime_posture}]", style=posture_style)
    t.append("  |  ", style="dim")
    t.append(f"Macro: {snap.next_macro_event}", style="dim")
    if snap.from_cache:
        t.append("  |  cached", style="dim")
    return t


def render_core_allocation(snap: RadarSnapshot) -> Table:
    tbl = Table(title="Core TFSA Allocation (VALIDATED: VM_OUTLOOK_TOP3)", box=None, pad_edge=False)
    tbl.add_column("ETF", style="bold #82aad2")
    tbl.add_column("Weight", justify="right", style="bold #5ec98d")
    tbl.add_column("Vol Scale", justify="right")
    tbl.add_column("Spot", justify="right")
    tbl.add_column("12-1m", justify="right")
    for row in snap.core_allocation:
        w = float(row.get("target_weight", 0.0)) * 100.0
        sc = float(row.get("vol_scale", 1.0)) * 100.0
        spot = f"${float(row['spot']):.2f}" if row.get("ticker") != "CASH" else "—"
        m12 = f"{float(row.get('mom_12_1_pct', 0.0)):+.1f}%" if row.get("ticker") != "CASH" else "T-Bills"
        tbl.add_row(str(row["ticker"]), f"{w:.0f}%", f"{sc:.0f}%", spot, m12)
    return tbl


def render_radar_table(opps: List[Opportunity], *, title: str = "Unified Opportunity Radar") -> Table:
    tbl = Table(title=title, show_lines=False, pad_edge=False)
    tbl.add_column("Tier", width=5)
    tbl.add_column("Ticker", style="bold", width=7)
    tbl.add_column("Playbook", width=24)
    tbl.add_column("Dir", width=6)
    tbl.add_column("Conf", justify="right", width=4)
    tbl.add_column("Friction", justify="right", width=10)
    tbl.add_column("Flags", width=18)
    tbl.add_column("Setup")

    for o in opps:
        tier_txt = Text(TIER_SHORT.get(o.evidence_tier, o.evidence_tier[:4]), style=TIER_STYLE.get(o.evidence_tier, ""))
        dir_txt = Text(
            "▲ LNG" if o.direction == "LONG" else ("▼ SRT" if o.direction == "SHORT" else "• NEU"),
            style="#5ec98d" if o.direction == "LONG" else ("#e06c75" if o.direction == "SHORT" else "dim"),
        )
        flags = ", ".join(o.risk_flags[:2]) if o.risk_flags else "—"
        flag_txt = Text(flags, style="#d6a452" if o.risk_flags else "dim")
        tbl.add_row(
            tier_txt,
            o.ticker,
            o.playbook_label[:24],
            dir_txt,
            str(o.confluence_count),
            o.friction_label or f"{o.friction_value*100:.2f}%",
            flag_txt,
            o.setup_summary,
        )
    return tbl


def render_dossier(dos: TickerDossier) -> Panel:
    lines: List[str] = []
    spot_s = f"${dos.spot:.2f}" if dos.spot is not None else "N/A"
    c1 = f"{dos.chg_1d_pct:+.2f}%" if dos.chg_1d_pct is not None else "—"
    c5 = f"{dos.chg_5d_pct:+.2f}%" if dos.chg_5d_pct is not None else "—"
    c21 = f"{dos.chg_21d_pct:+.2f}%" if dos.chg_21d_pct is not None else "—"
    m12 = f"{dos.mom_12_1_pct:+.1f}%" if dos.mom_12_1_pct is not None else "—"
    lines.append(f"[bold]Price[/bold]      {spot_s}  (1d {c1} · 5d {c5} · 21d {c21} · 12-1m {m12})")

    d20 = f"{dos.dist_sma20_pct:+.1f}%" if dos.dist_sma20_pct is not None else "—"
    d50 = f"{dos.dist_sma50_pct:+.1f}%" if dos.dist_sma50_pct is not None else "—"
    d200 = f"{dos.dist_sma200_pct:+.1f}%" if dos.dist_sma200_pct is not None else "—"
    r2 = f"{dos.rsi_2:.1f}" if dos.rsi_2 is not None else "—"
    r14 = f"{dos.rsi_14:.1f}" if dos.rsi_14 is not None else "—"
    lines.append(f"[bold]Structure[/bold]  vs MA20 {d20} · MA50 {d50} · MA200 {d200} · RSI2 {r2} · RSI14 {r14}")

    sup = ", ".join(f"${s['level']:.2f} ({s['pct']*100:+.1f}%)" for s in dos.supports) or "—"
    res = ", ".join(f"${r['level']:.2f} ({r['pct']*100:+.1f}%)" for r in dos.resistances) or "—"
    lines.append(f"[bold]Levels[/bold]     Sup: {sup}  |  Res: {res}")

    sec = f"{dos.sector_etf or '—'} [{dos.sector_posture or 'NEUTRAL'}]"
    cdr = f" · CDR: {dos.cdr_ticker}" if dos.cdr_ticker else ""
    rev = f" · Rev30d: {dos.rev_breadth_30d:+.2f}" if dos.rev_breadth_30d is not None else ""
    lines.append(f"[bold]Context[/bold]    Sector {sec}{cdr}{rev}")

    earn = (
        f"in {dos.days_to_earnings}d ({dos.earnings_date}, {dos.earnings_source})"
        if dos.days_to_earnings is not None
        else "none scheduled in cache"
    )
    lines.append(f"[bold]Landmines[/bold]  Earnings: {earn}  |  Macro: {dos.next_macro_event}")

    rv_s = f"{dos.rv_20_ann:.1f}%" if dos.rv_20_ann is not None else "—"
    iv_s = f"{dos.atm_iv:.1f}%" if dos.atm_iv is not None else "—"
    im_s = f"±{dos.implied_move_1s_pct:.1f}%" if dos.implied_move_1s_pct is not None else "—"
    p5 = (
        f"P(>+5%)={dos.rnd_prob_up_5pct*100:.0f}% / P(<-5%)={dos.rnd_prob_dn_5pct*100:.0f}%"
        if dos.rnd_prob_up_5pct is not None and dos.rnd_prob_dn_5pct is not None
        else "—"
    )
    lines.append(f"[bold]Vol / RND[/bold]  RV20 {rv_s} · ATM IV {iv_s} · 30d 1σ {im_s} · {p5}")

    if dos.risk_flags:
        lines.append(f"[bold #e06c75]Risk Flags[/bold #e06c75] {', '.join(dos.risk_flags)}")
    if dos.active_signals:
        sigs = " · ".join(f"[{s.evidence_tier}] {s.playbook_label}" for s in dos.active_signals)
        lines.append(f"[bold #5ec98d]Radar[/bold #5ec98d]      {sigs}")
    if dos.headlines:
        lines.append("[bold]Headlines[/bold]  " + " | ".join(h["title"][:65] for h in dos.headlines[:2]))

    return Panel("\n".join(lines), title=f"TICKER DOSSIER: {dos.ticker}", border_style="#82aad2")


def render_express_table(rows: List[VehicleComparison], *, symbol: str, spot: float, view_desc: str) -> Table:
    tbl = Table(title=f"Shares vs. Options Workbench — {symbol} @ ${spot:.2f} ({view_desc})", pad_edge=False)
    tbl.add_column("Vehicle", style="bold")
    tbl.add_column("Structure")
    tbl.add_column("TFSA", width=4)
    tbl.add_column("CAR $", justify="right")
    tbl.add_column("Friction", justify="right")
    tbl.add_column("EV(mkt)", justify="right")
    tbl.add_column("EV(view)", justify="right")
    tbl.add_column("PoP", justify="right")
    tbl.add_column("Walk-Away / Notes")

    for r in rows:
        ev_v_style = "bold #5ec98d" if r.ev_view_net_usd > 0 else "#e06c75"
        ev_m_style = "#5ec98d" if r.ev_mkt_net_usd > 0 else "dim"
        prefix = "★ " if r.recommended else "  "
        note = r.walk_away_label + (f" [{r.caution}]" if r.caution else "")
        tbl.add_row(
            prefix + r.vehicle,
            r.description,
            "Yes" if r.tfsa_legal else "No",
            f"${r.capital_at_risk_usd:,.0f}",
            f"{r.friction_pct_of_reward:.1f}%",
            Text(f"${r.ev_mkt_net_usd:+,.0f}", style=ev_m_style),
            Text(f"${r.ev_view_net_usd:+,.0f}", style=ev_v_style),
            f"{r.pop_view*100:.0f}%",
            note,
        )
    return tbl
