"""Formatting helpers & table fillers for `[7] OPTIONS` (Scanner · Chain Matrix · Flow & Vol Intel)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rich.text import Text
from textual.widgets import DataTable

from .charts import (
    braille_iv_smile_and_term,
    braille_multileg_payoff,
    flow_split_bar,
    gex_profile_bars,
    risk_heatmap_grid,
)
from .options_engine import (
    ChainMatrix,
    MultiLegAnalysis,
    OptionLeg,
    OptionScanRow,
    OptionsIntelBundle,
    analyze_custom_legs,
    describe_cal_pop_for_row,
)

OPT_SCAN_COLUMNS = (
    "Badge",
    "Ticker",
    "Strategy",
    "Exp (DTE)",
    "Strikes",
    "Mid",
    "Ref Price",
    "Fric%",
    "Risk $",
    "IV/RV",
    "TFSA",
)

OPT_CHAIN_COLUMNS = (
    "C.OI",
    "C.IV",
    "C.Δ",
    "C.Bid/Ask",
    "STRIKE",
    "P.Bid/Ask",
    "P.Δ",
    "P.IV",
    "P.OI",
)

OPT_FLOW_COLUMNS = (
    "Ticker",
    "Feed",
    "Flow / Vol Metric",
    "Skew / Ratio",
    "Signal / Read",
)

OPT_FILTER_PRESETS = (
    ("opt-pre-all", "All", ""),
    ("opt-pre-bp", "BullPut", "bull_put"),
    ("opt-pre-tfsa", "TFSA", "tfsa"),
    ("opt-pre-cln", "Clean", "clean"),
    ("opt-pre-lowf", "Fric<12%", "fric<12%"),
    ("opt-pre-ivc", "IV Cheap", "iv_cheap"),
)


def _badge_text(badge: str) -> Text:
    b = badge.upper()
    if b == "PROMISING":
        return Text("PROMIS", style="bold #5ec98d")
    if b == "CAUTION":
        return Text("CAUTN", style="bold #e06c75")
    return Text("WATCH", style="#d6a452")


def fill_options_scan_table(tbl: DataTable, rows: Sequence[OptionScanRow]) -> None:
    """Populate the `[7] OPTIONS` Scanner DataTable."""
    tbl.clear()
    for idx, r in enumerate(rows):
        fric_col = "#5ec98d" if r.friction_pct <= 0.10 else ("#d6a452" if r.friction_pct <= 0.20 else "#e06c75")
        iv_rv_s = f"{r.iv_vs_rv:.2f}x" if r.iv_vs_rv is not None else "—"
        if r.iv_vs_rv is not None:
            iv_col = "#5ec98d" if (r.iv_vs_rv <= 0.92 and not r.is_credit) or (r.iv_vs_rv >= 1.15 and r.is_credit) else ""
        else:
            iv_col = "dim"
        mid_lbl = f"+${r.mid_price:.2f}" if r.is_credit else f"-${r.mid_price:.2f}"
        tbl.add_row(
            _badge_text(r.evidence_badge),
            Text(r.ticker, style="bold"),
            r.strategy,
            f"{r.expiration[5:]} ({r.dte}d)" if len(r.expiration) >= 10 else f"{r.expiration} ({r.dte}d)",
            r.strikes_label[:18],
            Text(mid_lbl, style="#5ec98d" if r.is_credit else ""),
            f"${r.walk_away_limit:.2f}",
            Text(f"{r.friction_pct * 100:.1f}%", style=fric_col),
            f"${r.max_risk_usd:,.0f}",
            Text(iv_rv_s, style=iv_col),
            Text("OK", style="#5ec98d") if r.tfsa_legal else Text("TAX", style="#d6a452"),
            key=str(idx),
        )


def format_scan_inspector(
    row: Optional[OptionScanRow],
    *,
    days_elapsed: int = 0,
    iv_shock_pct: float = 0.0,
) -> str:
    """Right-hand inspector pane for a highlighted Options Scanner row."""
    if row is None:
        return (
            "[bold]OPTIONS INSPECTOR[/]\n"
            "[dim]Select any candidate row to inspect its 2D Risk Surface, quoted reference price, "
            "and Greeks.\nPress [b]c[/b] for Chain Matrix · [b]w[/b] for Workbench · [b]p[/b] to Paper-Log.[/]"
        )

    tfsa_s = "[#5ec98d]TFSA OK[/]" if row.tfsa_legal else "[#d6a452]TAXABLE / IBKR ONLY (CRA TFSA ban)[/]"
    b_col = "#5ec98d" if row.evidence_badge == "PROMISING" else ("#e06c75" if row.evidence_badge == "CAUTION" else "#d6a452")
    iv_s = f"{row.iv_pct:.1f}%" if row.iv_pct is not None else "—"
    ivrv_s = f" · IV/RV {row.iv_vs_rv:.2f}x" if row.iv_vs_rv is not None else ""
    data_s = "Yahoo request · quote time unknown" if row.source == "live" else row.source
    sk_s = f" · 25Δ Skew {row.skew_vp:+.1f}vp" if row.skew_vp is not None else ""
    ror_s = f" · Max payoff/risk {row.ror_pct:.1f}%" if row.ror_pct is not None else ""
    risk_label = "Debit at Mid" if row.strategy_key == "calendar" else "Max Risk at Mid"
    if row.max_profit_usd is None:
        profit_s = "Unbounded" if row.strategy_key == "long_call" else "Depends on spot and IV at front expiry"
    else:
        profit_s = f"${row.max_profit_usd:,.0f}"
    flags_s = ", ".join(row.risk_flags) if row.risk_flags else "Clean (no earnings/edge flags)"

    lines: List[str] = [
        f"[bold]{row.ticker} {row.strategy}[/]  [{b_col}][{row.evidence_badge}][/]  ·  {tfsa_s}",
        f"Spot ${row.spot:.2f} · Exp {row.expiration} ({row.dte}d) · {row.strikes_label}  [dim][{data_s} {row.as_of}][/]",
        "─" * 56,
        f"[bold]Pricing[/]   {'Credit' if row.is_credit else 'Debit'} Mid ${row.mid_price:.2f} · "
        f"Ref Price ${row.walk_away_limit:.2f} · Friction {row.friction_pct * 100:.1f}%",
        f"[bold]Risk/Rew[/]  {risk_label} ${row.max_risk_usd:,.0f} · Max Profit {profit_s}{ror_s}",
        f"[bold]Vol/Liq[/]   IV {iv_s}{ivrv_s}{sk_s} · Min OI {row.min_oi:,} · Vol {row.total_vol:,}",
        f"[bold]Flags[/]     [#d6a452]{flags_s}[/]" if row.risk_flags else f"[bold]Flags[/]     [#5ec98d]{flags_s}[/]",
    ]

    if row.legs and row.spot > 0:
        try:
            an = analyze_custom_legs(
                row.ticker,
                row.spot,
                row.legs,
                days_elapsed=days_elapsed,
                iv_shock_pct=iv_shock_pct,
            )
            be_s = ", ".join(f"${b:.2f}" for b in an.breakevens) or "—"
            lines.append(
                f"[bold]Greeks[/]    Δ$ {an.delta_dollars:+,.0f} · Θ/d {an.theta_per_day_usd:+.2f} · "
                f"Vega {an.vega_per_1pct_usd:+.2f} · Mkt PoP {an.pop_mkt * 100:.0f}% · BE {be_s}"
            )
            cal_line = describe_cal_pop_for_row(row)
            if cal_line:
                lines.append(f"[bold #5ec98d]Calibrated[/] {cal_line}")
            lines.append("")
            lines.append(
                risk_heatmap_grid(
                    an.grid_price_shocks,
                    an.grid_iv_shocks,
                    an.grid_pnl,
                    title=f"2D P&L SURFACE (T+{an.days_elapsed}d · IV{an.iv_shock_pct:+.0f}%)",
                    rich=True,
                )
            )
        except Exception:
            pass

    lines.append("")
    lines.append("[dim][Enter/c] Chain  [w] Workbench  [p] Paper Log  [f] Fill  [,/.] Time ±5d  [{/}] IV ±5%[/]")
    return "\n".join(lines)


def fill_chain_matrix_table(tbl: DataTable, matrix: ChainMatrix) -> None:
    """Populate the T-shaped Call | Strike | Put Chain Matrix DataTable."""
    tbl.clear()
    for idx, row in enumerate(matrix.rows):
        c = row.call
        p = row.put

        if c is not None:
            c_oi = f"{c.open_interest:,}" if c.open_interest else "—"
            c_iv = f"{c.iv * 100:.0f}%" if c.iv is not None else "—"
            c_dl = f"{c.delta:+.2f}" if c.delta is not None else "—"
            c_ba = f"{c.bid:.2f}/{c.ask:.2f}"
            c_style = "bold" if row.strike < matrix.spot else ""
            if c.half_spread_pct > 0.12:
                c_ba_t = Text(c_ba, style="#d6a452")
            else:
                c_ba_t = Text(c_ba, style=c_style)
        else:
            c_oi = c_iv = c_dl = "—"
            c_ba_t = Text("—", style="dim")

        if p is not None:
            p_oi = f"{p.open_interest:,}" if p.open_interest else "—"
            p_iv = f"{p.iv * 100:.0f}%" if p.iv is not None else "—"
            p_dl = f"{p.delta:+.2f}" if p.delta is not None else "—"
            p_ba = f"{p.bid:.2f}/{p.ask:.2f}"
            p_style = "bold" if row.strike > matrix.spot else ""
            if p.half_spread_pct > 0.12:
                p_ba_t = Text(p_ba, style="#d6a452")
            else:
                p_ba_t = Text(p_ba, style=p_style)
        else:
            p_oi = p_iv = p_dl = "—"
            p_ba_t = Text("—", style="dim")

        tag_s = (" " + "·".join(row.tags)) if row.tags else ""
        if "SPOT" in row.tags:
            k_txt = Text(f"▶ {row.strike:g}{tag_s}", style="bold #ffb03b")
        elif "C-WALL" in row.tags:
            k_txt = Text(f"{row.strike:g}{tag_s}", style="bold #5ec98d")
        elif "P-WALL" in row.tags:
            k_txt = Text(f"{row.strike:g}{tag_s}", style="bold #e06c75")
        elif "FLIP" in row.tags:
            k_txt = Text(f"{row.strike:g}{tag_s}", style="bold #82aad2")
        else:
            k_txt = Text(f"{row.strike:g}{tag_s}")

        tbl.add_row(
            c_oi,
            c_iv,
            c_dl,
            c_ba_t,
            k_txt,
            p_ba_t,
            p_dl,
            p_iv,
            p_oi,
            key=str(idx),
        )


def format_chain_inspector(
    matrix: Optional[ChainMatrix],
    staged_legs: Sequence[OptionLeg],
    *,
    days_elapsed: int = 0,
    iv_shock_pct: float = 0.0,
) -> str:
    """Right-hand visual telemetry pane for the Chain Matrix: IV Smile, Term, GEX & Staged Leg Composer."""
    if matrix is None:
        return (
            "[bold]CHAIN TELEMETRY[/]\n"
            "[dim]Enter a ticker and press Load (or Live Chain) to view T-shaped Calls/Puts, "
            "IV Smile, Term Structure, and Dealer GEX.[/]"
        )

    em_s = f"±${matrix.expected_move_usd:.2f} (±{matrix.expected_move_pct:.1f}%)" if matrix.expected_move_usd else "—"
    iv_s = f"{matrix.atm_iv_pct:.1f}%" if matrix.atm_iv_pct is not None else "—"
    rv_s = f"{matrix.rv20_pct:.1f}%" if matrix.rv20_pct is not None else "—"
    ratio_s = f" ({matrix.iv_vs_rv:.2f}x)" if matrix.iv_vs_rv is not None else ""
    sk_s = f"{matrix.skew_25d_vp:+.1f}vp" if matrix.skew_25d_vp is not None else "—"
    spr_s = f"{matrix.median_spread_pct:.1f}%" if matrix.median_spread_pct is not None else "—"
    gex_col = "#5ec98d" if matrix.gex_regime == "POSITIVE_GEX" else ("#e06c75" if matrix.gex_regime == "NEGATIVE_GEX" else "dim")
    gex_s = f"[{gex_col}]${matrix.net_gex_m:+.1f}M ({matrix.gex_regime})[/]" if matrix.net_gex_m is not None else "[dim]—[/]"
    earn_s = f" · [#e06c75]EARNINGS IN {matrix.earnings_days}d[/]" if (matrix.earnings_days is not None and matrix.earnings_days <= matrix.dte) else ""

    lines: List[str] = [
        f"[bold]{matrix.ticker} ${matrix.spot:.2f}[/] · Exp {matrix.expiration} ({matrix.dte}d) "
        f"[dim][{matrix.source} {matrix.snap_date}][/]{earn_s}",
        f"ATM IV {iv_s} vs RV20 {rv_s}{ratio_s} · 25Δ Skew {sk_s} · Med Spread {spr_s}",
        f"Expected Move (1σ): [bold]{em_s}[/] · Net GEX: {gex_s}",
        "─" * 56,
    ]

    # If user has staged custom legs, show the LIVE Multi-Leg Strategy Composer!
    if staged_legs:
        an = analyze_custom_legs(
            matrix.ticker,
            matrix.spot,
            staged_legs,
            days_elapsed=days_elapsed,
            iv_shock_pct=iv_shock_pct,
        )
        tfsa_tag = "[#5ec98d]TFSA OK[/]" if an.tfsa_legal else "[#d6a452]TAXABLE ONLY[/]"
        lines.append(f"[bold]LEG COMPOSER: {an.structure_name}[/] ({len(staged_legs)} leg(s)) · {tfsa_tag}")
        for i, lg in enumerate(staged_legs, 1):
            scol = "#5ec98d" if lg.side == "BUY" else "#e06c75"
            lines.append(
                f"  {i}. [{scol}]{lg.side:<4}[/] {lg.opt_type.upper():<4} ${lg.strike:g} "
                f"{lg.expiration[5:]} ({lg.dte}d) @ ${lg.mid:.2f} [dim](bid {lg.bid:.2f}/ask {lg.ask:.2f})[/]"
            )
        be_s = ", ".join(f"${b:.2f}" for b in an.breakevens) or "—"
        mp_s = "Unbounded" if an.max_profit_usd is None else f"${an.max_profit_usd:,.0f}"
        lines.append(
            f"Net {'Credit' if an.is_credit else 'Debit'} [bold]${an.net_mid:.2f}[/] · "
            f"Fric {an.friction_pct * 100:.1f}% · Risk ${an.max_risk_usd:,.0f} · MaxP {mp_s} · BE {be_s}"
        )
        lines.append(
            f"Δ$ {an.delta_dollars:+,.0f} · Θ/d {an.theta_per_day_usd:+.2f} · "
            f"Vega {an.vega_per_1pct_usd:+.2f} · Mkt PoP {an.pop_mkt * 100:.0f}%"
        )
        lines.append(
            braille_multileg_payoff(
                an.prices,
                an.expiry_pnl,
                an.shocked_pnl,
                spot=matrix.spot,
                days_elapsed=an.days_elapsed,
                iv_shock_pct=an.iv_shock_pct,
                width=42,
                height=4,
                rich=True,
            )
        )
        lines.append(
            risk_heatmap_grid(
                an.grid_price_shocks,
                an.grid_iv_shocks,
                an.grid_pnl,
                title=f"2D P&L SURFACE (T+{an.days_elapsed}d · IV{an.iv_shock_pct:+.0f}%)",
                rich=True,
            )
        )
        lines.append(f"[dim]{an.ladder.abort_label}[/]")
        lines.append("[dim]Clear = IV/GEX · [p] Paper Log · [,/.] Time ±5d · [{/}] IV ±5%[/]")
        return "\n".join(lines)

    # Otherwise show IV Smile + Term Structure + Dealer GEX Profile
    lines.append(
        braille_iv_smile_and_term(
            matrix.smile_points,
            matrix.term_points,
            spot=matrix.spot,
            width=38,
            height=4,
            rich=True,
        )
    )
    lines.append("")
    lines.append(
        gex_profile_bars(
            matrix.gex_by_strike,
            spot=matrix.spot,
            gamma_flip=matrix.gamma_flip,
            call_wall=matrix.call_wall,
            put_wall=matrix.put_wall,
            max_rows=7,
            bar_width=10,
            rich=True,
        )
    )
    lines.append("")
    lines.append("[dim]Use buttons above to stage Buy/Sell Call/Put legs on any strike → live Payoff & 2D Heatmap[/]")
    return "\n".join(lines)


def fill_options_intel_table(tbl: DataTable, intel: OptionsIntelBundle, *, mode: str = "all") -> List[Dict[str, Any]]:
    """Populate the Flow & Vol Intel DataTable and return backing row metadata for inspector."""
    tbl.clear()
    items: List[Dict[str, Any]] = []

    if mode in ("all", "flow"):
        for u in intel.uoa_rows[:15]:
            sym = str(u.get("symbol") or "").upper()
            n_u = int(u.get("unusual_count") or len(u.get("unusual") or []))
            raw_cs = u.get("net_call_share")
            c_sh = float(raw_cs) if raw_cs is not None else 0.5
            bar = flow_split_bar(c_sh, width=10, rich=False)
            c_prem = float(u.get("call_premium_added") or 0.0) / 1e6
            p_prem = float(u.get("put_premium_added") or 0.0) / 1e6
            metric = f"{bar} (+${c_prem:.1f}M C / +${p_prem:.1f}M P)"
            read = f"{n_u} unusual contract(s) [UOA]"
            items.append({"ticker": sym, "feed": "UOA", "raw": u})
            tbl.add_row(
                Text(sym, style="bold"),
                Text("UOA", style="#ffb03b"),
                metric,
                f"{c_sh * 100:.0f}% C",
                read,
                key=f"{len(items) - 1}|{sym}",
            )
        for o in intel.occ_rows[:12]:
            sym = str(o.get("symbol") or "").upper()
            vol = int(o.get("total_vol") or 0)
            vo = o.get("vol_to_oi")
            od = o.get("oi_delta_pct")
            metric = f"Vol {vol:,} · V/OI {vo:.2f}x" if vo is not None else f"Vol {vol:,}"
            ratio_s = f"{od:+.1f}% OI" if od is not None else "—"
            read = str(o.get("signal_label") or "OCC_SURGE")
            items.append({"ticker": sym, "feed": "OCC", "raw": o})
            tbl.add_row(
                Text(sym, style="bold"),
                Text("OCC", style="#82aad2"),
                metric,
                ratio_s,
                read,
                key=f"{len(items) - 1}|{sym}",
            )

    if mode in ("all", "vol"):
        for v in intel.vol_rows[:15]:
            sym = str(v.get("symbol") or "").upper()
            iv_p = v.get("iv_pct")
            rv_p = v.get("rv_pct")
            vrp = v.get("vrp_vp")
            rat = v.get("iv_rv_ratio")
            metric = f"IV {iv_p:.1f}% vs RV {rv_p:.1f}%" if (iv_p is not None and rv_p is not None) else "—"
            rat_s = f"{rat:.2f}x" if rat is not None else "—"
            if vrp is not None and vrp >= 4.0:
                read_t = Text(f"RICH VOL (+{vrp:.1f}vp VRP)", style="#5ec98d")
            elif vrp is not None and vrp <= -3.0:
                read_t = Text(f"CHEAP VOL ({vrp:+.1f}vp VRP)", style="#82aad2")
            else:
                read_t = Text(f"VRP {vrp:+.1f}vp" if vrp is not None else "NEUTRAL", style="dim")
            items.append({"ticker": sym, "feed": "VOL", "raw": v})
            tbl.add_row(
                Text(sym, style="bold"),
                Text("VRP", style="#5ec98d"),
                metric,
                rat_s,
                read_t,
                key=f"{len(items) - 1}|{sym}",
            )

    if mode in ("all", "skew"):
        for s in intel.skew_rows[:15]:
            sym = str(s.get("ticker") or "").upper()
            cw = s.get("cw_iv_spread_vp")
            sm = s.get("put_smirk_vp")
            gx = s.get("net_gex_m")
            metric = (
                f"CW {cw:+.1f}vp · Smirk {sm:+.1f}vp"
                if (cw is not None and sm is not None)
                else f"GEX ${gx:+.1f}M" if gx is not None else "—"
            )
            gx_s = f"${gx:+.1f}M" if gx is not None else "—"
            lbl = str(s.get("signal_label") or "NEUTRAL")
            col = "#5ec98d" if lbl == "BULL_FLOW_SHARES" else ("#e06c75" if lbl == "CRASH_SMIRK_AVOID" else "dim")
            items.append({"ticker": sym, "feed": "SKEW", "raw": s})
            tbl.add_row(
                Text(sym, style="bold"),
                Text("SKEW", style="#d6a452"),
                metric,
                gx_s,
                Text(lbl, style=col),
                key=f"{len(items) - 1}|{sym}",
            )

    return items


def format_intel_inspector(item: Optional[Dict[str, Any]]) -> str:
    """Right-hand detail pane for a highlighted Flow / Vol / Skew Intel row."""
    if not item:
        return (
            "[bold]OPTIONS INTEL HUB (ZERO-NETWORK LOCAL FEEDS)[/]\n"
            "[dim]Combines UOA day-over-day contract OI jumps, OCC surges, Vol-Intel VRP (IV vs RV20), "
            "and Cremers-Weinbaum Skew + Dealer GEX.\n\n"
            "Press [b]Enter[/b] on any row to open its full interactive Chain Matrix & GEX Profile.[/]"
        )
    sym = str(item.get("ticker") or "")
    feed = str(item.get("feed") or "")
    raw = item.get("raw") or {}
    lines = [
        f"[bold]{sym}  //  {feed} TELEMETRY[/]  [dim][WATCHLIST / DESCRIPTIVE][/]",
        "─" * 54,
    ]
    if feed == "UOA":
        raw_cs_i = raw.get("net_call_share")
        c_sh = float(raw_cs_i) if raw_cs_i is not None else 0.5
        lines.append(f"Flow Split: {flow_split_bar(c_sh, width=18, rich=True)}")
        lines.append(
            f"Call OI Added: {float(raw.get('call_oi_added') or 0):,.0f}  ·  "
            f"Put OI Added: {float(raw.get('put_oi_added') or 0):,.0f}"
        )
        unusual = raw.get("unusual") or []
        if unusual:
            lines.append("")
            lines.append("[bold]Top Unusual Contracts:[/]")
            for c in unusual[:6]:
                lines.append(
                    f"  • {str(c.get('type', '')).upper():<4} ${float(c.get('strike') or 0):g} "
                    f"exp {str(c.get('expiration') or '')[:10]}  ΔOI {float(c.get('d_oi') or 0):+,.0f} "
                    f"Vol {float(c.get('volume') or 0):,.0f} ({c.get('why', '')})"
                )
    elif feed == "OCC":
        lines.append(f"Trade Date:  {raw.get('trade_date', '—')}")
        lines.append(f"Volume:      {float(raw.get('total_vol') or 0):,.0f}")
        lines.append(f"Open Int:    {float(raw.get('total_oi') or 0):,.0f}")
        lines.append(f"Vol / OI:    {raw.get('vol_to_oi', '—')}x")
        lines.append(f"1d OI Δ%:    {raw.get('oi_delta_pct', '—')}%")
        lines.append(f"Signal:      {raw.get('signal_label', '—')}")
    elif feed == "VOL":
        lines.append(f"ATM IV:      {raw.get('iv_pct', '—')}%")
        lines.append(f"Realized 20d:{raw.get('rv_pct', '—')}%")
        lines.append(f"VRP (IV-RV): {raw.get('vrp_vp', '—')} vp")
        lines.append(f"IV / RV:     {raw.get('iv_rv_ratio', '—')}x")
        lines.append(f"1d IV Move:  {raw.get('iv_chg_vp', '—')} vp")
        lines.append(f"RV %ile:     {raw.get('rv_pctile', '—')}%")
        lines.append("")
        lines.append("[dim]High positive VRP => rich option premium (favors defined-risk credit spreads).\n"
                     "Negative VRP / low IV/RV => cheap gamma (favors long calls/puts/calendars).[/]")
    else:
        lines.append(f"Spot:        ${float(raw.get('spot') or 0):.2f}  [dim]({raw.get('snap_date', '')})[/]")
        lines.append(f"CW Spread:   {raw.get('cw_iv_spread_vp', '—')} vp  (ATM Call IV − Put IV)")
        lines.append(f"Put Smirk:   {raw.get('put_smirk_vp', '—')} vp  (25Δ Put IV − ATM Call IV)")
        lines.append(f"Net GEX:     ${raw.get('net_gex_m', '—')}M  [{raw.get('gex_regime', 'UNKNOWN')}]")
        lines.append(f"Gamma Flip:  ${raw.get('gamma_flip', '—')}")
        lines.append(f"OI Walls:    Call ${raw.get('call_wall', '—')}  /  Put ${raw.get('put_wall', '—')}")
        lines.append(f"Read:        [bold]{raw.get('signal_label', 'NEUTRAL')}[/]")

    lines.append("")
    lines.append(f"[dim][Enter] Open {sym} Chain Matrix  ·  [w] Open {sym} Workbench  ·  [:] Evidence Card[/]")
    return "\n".join(lines)
