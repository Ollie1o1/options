"""Shared Rich/Textual formatters for Phase 3 Alpha & Alt-Data screens (`src/alpha/formatters.py`)."""
from __future__ import annotations

from typing import Tuple

from src.alpha.edgar_events import load_cached_edgar_events
from src.alpha.macro_plumbing import load_macro_plumbing
from src.alpha.options_for_shares import scan_option_signals_for_shares
from src.alpha.premortem import build_premortem
from src.radar.dossier import build_dossier
from src.tui.charts import vix_curve_strip


def format_macro_text(snap=None, *, rich: bool = False) -> str:
    m = snap or load_macro_plumbing()
    vc = m.vix_curve or {}
    lines = [
        vix_curve_strip(
            vix9d=vc.get("vix9d"),
            vix=vc.get("vix"),
            vix3m=vc.get("vix3m"),
            vix6m=vc.get("vix6m"),
            rich=rich,
        ),
    ]
    if m.fed_net_liquidity_b is not None:
        c4 = f"{m.fed_liq_chg_4w_b:+.1f}B (4w)" if m.fed_liq_chg_4w_b is not None else ""
        c13 = f"{m.fed_liq_chg_13w_b:+.1f}B (13w)" if m.fed_liq_chg_13w_b is not None else ""
        lines.append(f"Fed Net Liquidity: ${m.fed_net_liquidity_b:,.1f}B  {c4}  {c13}  [as of {m.fed_liq_as_of}]")
    else:
        lines.append("Fed Net Liquidity: not cached — run `opt alpha macro --refresh`")

    if m.cot_signals:
        lines.append("")
        lines.append("CFTC COT Speculative Net Positioning (% of OI):")
        for c in m.cot_signals:
            lines.append(f"  {c['asset']:<8} {c['spec_net_pct_oi']:+6.1f}%  [{c['bias']}]  ({c['report_date']})")
    if m.polymarket_macro:
        lines.append("")
        lines.append("Polymarket Macro Odds (tight books <=5pt spread only):")
        for p in m.polymarket_macro[:5]:
            lines.append(f"  {p['prob_pct']:5.1f}%  {p['question']} (spread {p['spread_pts']:.1f}pt)")
    if m.updated_at:
        lines.append(f"\nSource: CBOE · FRED · CFTC · Polymarket (cached {m.updated_at[:16]}) [CONTEXT]")
    return "\n".join(lines)


def format_alpha_screen(cmd: str, *, ticker: str = "SPY") -> Tuple[str, str]:
    """Return `(title, rich_markup_body)` for the TUI `AlphaViewerModal`."""
    c = cmd.upper()
    if c == "SKEW":
        rows = scan_option_signals_for_shares()
        lines = [
            "[bold]OPTION-IMPLIED SIGNALS → TRADE IN SHARES[/] [dim][WATCHLIST · data/chain_archive.db][/]",
            "[dim]Cremers-Weinbaum ATM Call-Put IV Spread (+ = bullish flow) · Xing-Zhang-Zhao Put Smirk (high = crash fear)[/]",
            "─" * 78,
            f"{'TICKER':<7} {'SPOT':>8} {'CW SPREAD':>10} {'PUT SMIRK':>10} {'NET GEX':>10} {'SIGNAL':<20}",
        ]
        for r in rows[:25]:
            cw = f"{r.cw_iv_spread_vp:+.2f}vp" if r.cw_iv_spread_vp is not None else "—"
            sm = f"{r.put_smirk_vp:+.2f}vp" if r.put_smirk_vp is not None else "—"
            gx = f"${r.net_gex_m:+.1f}M" if r.net_gex_m is not None else "—"
            col = "#5ec98d" if r.signal_label == "BULL_FLOW_SHARES" else ("#e06c75" if r.signal_label == "CRASH_SMIRK_AVOID" else "#8b949e")
            lines.append(f"{r.ticker:<7} ${r.spot:7.2f} {cw:>10} {sm:>10} {gx:>10} [{col}]{r.signal_label}[/]")
        if not rows:
            lines.append("[dim]No archived chains found in last 30 days.[/]")
        return "ALPHA LAB // OPTION-IMPLIED SIGNALS FOR SHARES", "\n".join(lines)

    if c == "GEX":
        tk = ticker.upper()
        rows = scan_option_signals_for_shares(symbols=(tk,))
        if not rows:
            return f"DEALER GEX // {tk}", f"[#d6a452]No recent chain snapshot for {tk} in data/chain_archive.db.[/]"
        r = rows[0]
        gcol = "#5ec98d" if r.gex_regime == "POSITIVE_GEX" else "#e06c75"
        body = "\n".join(
            [
                f"[bold]{r.ticker}[/] @ ${r.spot:.2f}  [dim](chain snap {r.snap_date})[/]",
                f"Dealer Net GEX: [{gcol}]{r.net_gex_m:+.2f} $M / 1% move [{r.gex_regime}][/]" if r.net_gex_m is not None else "Dealer Net GEX: —",
                f"Gamma Flip:     ${r.gamma_flip:.2f}" if r.gamma_flip is not None else "Gamma Flip:     —",
                f"Call OI Wall:   ${r.call_wall:g}" if r.call_wall is not None else "Call OI Wall:   —",
                f"Put OI Wall:    ${r.put_wall:g}" if r.put_wall is not None else "Put OI Wall:    —",
                f"CW IV Spread:   {r.cw_iv_spread_vp:+.2f}vp" if r.cw_iv_spread_vp is not None else "CW IV Spread:   —",
                f"Put Smirk:      {r.put_smirk_vp:+.2f}vp" if r.put_smirk_vp is not None else "Put Smirk:      —",
                "",
                "[dim]Positive GEX -> dealers dampen swings (favors pullbacks/mean reversion).\n"
                "Negative GEX -> dealers amplify moves (trending/volatile; tighten risk).[/]",
            ]
        )
        return f"DEALER GAMMA EXPOSURE (GEX) // {tk}", body

    if c in ("EDGAR", "13D", "SPIN", "BUYBACK"):
        emap = {"13D": "ACTIVIST_13D", "SPIN": "SPINOFF_FORM10", "BUYBACK": "NET_BUYBACK_QUALITY"}
        ev_type = emap.get(c)
        evs = load_cached_edgar_events(event_type=ev_type, limit=30)
        lines = [
            f"[bold]SEC EDGAR FORCED-FLOW EVENTS ({ev_type or 'ALL'})[/] [dim][WATCHLIST · data/edgar_events.db][/]",
            "─" * 78,
        ]
        for e in evs:
            lines.append(f"[#d6a452]{e.event_type:<19}[/] [bold]{e.ticker:<6}[/] {e.filed_date}  {e.summary}")
        if not evs:
            lines.append("[dim]No cached EDGAR events yet — run `opt alpha edgar --refresh` (100% free SEC API).[/]")
        return "ALPHA LAB // SEC EDGAR FORCED-FLOW", "\n".join(lines)

    if c == "MACRO":
        return "MACRO PLUMBING // VIX CURVE · FED LIQUIDITY · COT · POLYMARKET", format_macro_text(rich=True)

    # Default: Pre-Mortem Devil's Advocate (`PM <TICKER>`)
    tk = ticker.upper()
    dos = build_dossier(tk, fetch_chain=False)
    sk_rows = scan_option_signals_for_shares(symbols=(tk,))
    sk = sk_rows[0] if sk_rows else None
    mac = load_macro_plumbing()
    pm = build_premortem(tk, dos, skew_signal=sk, macro=mac, account_type="tfsa")
    gcol = "#5ec98d" if pm.risk_grade == "CLEAN" else ("#e06c75" if pm.risk_grade == "HIGH_RISK" else "#d6a452")
    lines = [
        f"[bold {gcol}][{pm.risk_grade}][/] {pm.summary}",
        "",
        "[bold]HOW THIS TRADE LOSES (Bear Case & Red Flags):[/]",
        *[f"  • {b}" for b in pm.bear_thesis_points],
        "",
        "[bold]OBJECTIVE KILL-SWITCHES (Set Before Entry):[/]",
        *[f"  ▸ {k}" for k in pm.kill_switches],
        "",
        "[bold]WEALTHSIMPLE TFSA NOTES:[/]",
        *[f"  · {n}" for n in pm.tfsa_notes],
    ]
    return f"PRE-MORTEM DEVIL'S ADVOCATE // {tk}", "\n".join(lines)
