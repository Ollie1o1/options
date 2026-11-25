"""`opt alpha` — Phase 3 Unexhausted TFSA-Legal Alpha & External Alt-Data CLI."""
from __future__ import annotations

import json
from typing import Optional

import click
from rich.panel import Panel
from rich.table import Table

from src.alpha.edgar_events import load_cached_edgar_events, refresh_edgar_events
from src.alpha.formatters import format_macro_text
from src.alpha.macro_plumbing import load_macro_plumbing, refresh_macro_plumbing
from src.alpha.options_for_shares import EXPANDED_CHAIN_UNIVERSE, scan_option_signals_for_shares
from src.alpha.premortem import build_premortem, run_llm_premortem_audit
from src.cli.render import accent_color, make_console
from src.radar.dossier import build_dossier


@click.group("alpha")
def alpha_group() -> None:
    """Phase 3 TFSA-Legal Alpha & Alt-Data Labs (Option-Skew->Shares, GEX, SEC EDGAR, Macro, Pre-Mortem)."""


@alpha_group.command("skew")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_skew(ctx: click.Context, as_json: bool) -> None:
    """Option-Implied Signals (Cremers-Weinbaum IV Spread + Xing-Zhang-Zhao Put Smirk) -> Execute in SHARES."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    rows = scan_option_signals_for_shares()
    if use_json:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="Option-Implied Signals -> Trade in Shares [WATCHLIST]", pad_edge=False)
    tbl.add_column("Ticker", style=f"bold {accent_color()}")
    tbl.add_column("Snap")
    tbl.add_column("Spot", justify="right")
    tbl.add_column("CW IV Spread", justify="right")
    tbl.add_column("Put Smirk", justify="right")
    tbl.add_column("Net GEX ($M)", justify="right")
    tbl.add_column("Read")
    for r in rows[:25]:
        cw = f"{r.cw_iv_spread_vp:+.2f}vp" if r.cw_iv_spread_vp is not None else "—"
        sm = f"{r.put_smirk_vp:+.2f}vp" if r.put_smirk_vp is not None else "—"
        gx = f"{r.net_gex_m:+.1f}" if r.net_gex_m is not None else "—"
        col = "#5ec98d" if r.signal_label == "BULL_FLOW_SHARES" else ("#e06c75" if r.signal_label == "CRASH_SMIRK_AVOID" else "dim")
        tbl.add_row(r.ticker, r.snap_date, f"${r.spot:.2f}", cw, sm, gx, f"[{col}]{r.signal_label}[/{col}]")
    console.print(tbl)
    console.print("[dim]Source: data/chain_archive.db · Label: WATCHLIST (literature signal; harvest in $0-comm TFSA shares).[/dim]")


@alpha_group.command("gex")
@click.argument("ticker", default="SPY")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_gex(ctx: click.Context, ticker: str, as_json: bool) -> None:
    """Dealer Gamma Exposure (GEX), Gamma Flip strike, and Call/Put OI Walls from archived chains."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    rows = scan_option_signals_for_shares(symbols=(ticker.upper(),))
    if not rows:
        click.echo(f"No recent chain snapshot for {ticker.upper()} in data/chain_archive.db.")
        return
    r = rows[0]
    if use_json:
        click.echo(json.dumps(r.to_dict(), indent=2))
        return
    console = make_console(no_color=no_color)
    col = "#5ec98d" if r.gex_regime == "POSITIVE_GEX" else "#e06c75"
    console.print(
        Panel(
            f"[bold]Spot:[/bold] ${r.spot:.2f} (snap {r.snap_date})\n"
            f"[bold]Dealer Net GEX:[/bold] [{col}]{r.net_gex_m:+.2f} $M / 1% move ({r.gex_regime})[/{col}]\n"
            f"[bold]Gamma Flip:[/bold]     {f'${r.gamma_flip:.2f}' if r.gamma_flip else '—'}\n"
            f"[bold]Call OI Wall:[/bold]   {f'${r.call_wall:g}' if r.call_wall else '—'}  ·  "
            f"[bold]Put OI Wall:[/bold] {f'${r.put_wall:g}' if r.put_wall else '—'}\n"
            f"[bold]CW IV Spread:[/bold]   {f'{r.cw_iv_spread_vp:+.2f}vp' if r.cw_iv_spread_vp is not None else '—'}  ·  "
            f"[bold]Put Smirk:[/bold] {f'{r.put_smirk_vp:+.2f}vp' if r.put_smirk_vp is not None else '—'}",
            title=f"DEALER GEX & VOL STRUCTURE // {r.ticker}",
            border_style=accent_color(),
        )
    )


@alpha_group.command("edgar")
@click.option("--refresh", is_flag=True, help="Fetch live SEC EDGAR Form 10 spinoffs, 13D activists, 8-K guidance, 10-K supply chain & XBRL buybacks.")
@click.option("--type", "ev_type", type=click.Choice(["all", "SPINOFF_FORM10", "ACTIVIST_13D", "GUIDANCE_8K", "SUPPLY_CHAIN_10K", "NET_BUYBACK_QUALITY"]), default="all")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_edgar(ctx: click.Context, refresh: bool, ev_type: str, as_json: bool) -> None:
    """SEC EDGAR Forced-Flow Events: Form 10 Spinoffs, 5-day 13D Activists & XBRL Net Buybacks."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    if refresh:
        res = refresh_edgar_events()
        if not use_json:
            click.echo(f"SEC EDGAR refresh: saved {res['saved']} event(s)" + (f" · errors: {res['errors']}" if res["errors"] else ""))
    rows = load_cached_edgar_events(event_type=None if ev_type == "all" else ev_type, limit=40)
    if use_json:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="SEC EDGAR Forced-Flow Corporate Events [WATCHLIST]", pad_edge=False)
    tbl.add_column("Type", style="#d6a452")
    tbl.add_column("Ticker", style=f"bold {accent_color()}")
    tbl.add_column("Filed")
    tbl.add_column("Summary")
    tbl.add_column("Source", style="dim")
    for r in rows:
        tbl.add_row(r.event_type, r.ticker, r.filed_date, r.summary, r.source)
    console.print(tbl)
    if not rows:
        console.print("[dim]Empty cache — run `opt alpha edgar --refresh` (set SEC_EDGAR_CONTACT in .env for best rate limits).[/dim]")


@alpha_group.command("macro")
@click.option("--refresh", is_flag=True, help="Fetch live CBOE 4-pt VIX curve, FRED Net Liquidity, CFTC COT & Polymarket.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_macro(ctx: click.Context, refresh: bool, as_json: bool) -> None:
    """Macro Plumbing: 4-Point VIX Curve, FRED Fed Net Liquidity, CFTC COT & Polymarket Odds."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    snap = refresh_macro_plumbing() if refresh else load_macro_plumbing()
    if use_json:
        click.echo(json.dumps(snap.to_dict(), indent=2))
        return
    console = make_console(no_color=no_color)
    console.print(
        Panel(
            format_macro_text(snap, rich=True),
            title="MACRO PLUMBING & ALT-DATA [CONTEXT]",
            border_style=accent_color(),
        )
    )


@alpha_group.command("premortem")
@click.argument("ticker")
@click.option("--account", type=click.Choice(["tfsa", "taxable"], case_sensitive=False), default="tfsa")
@click.option("--ai", is_flag=True, help="Also run opt-in OpenRouter Devil's Advocate risk audit.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_premortem(ctx: click.Context, ticker: str, account: str, ai: bool, as_json: bool) -> None:
    """Pre-Mortem Devil's Advocate: list exact failure modes & kill-switches BEFORE buying."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    tk = ticker.upper().strip()
    dos = build_dossier(tk, fetch_chain=False)
    sk_rows = scan_option_signals_for_shares(symbols=(tk,))
    mac = load_macro_plumbing()
    rep = build_premortem(tk, dos, skew_signal=sk_rows[0] if sk_rows else None, macro=mac, account_type=account)
    if ai:
        rep.ai_audit = run_llm_premortem_audit(rep)
    if use_json:
        click.echo(json.dumps(rep.to_dict(), indent=2))
        return
    console = make_console(no_color=no_color)
    col = "#5ec98d" if rep.risk_grade == "CLEAN" else ("#e06c75" if rep.risk_grade == "HIGH_RISK" else "#d6a452")
    body = [
        f"[bold {col}][{rep.risk_grade}][/bold {col}] {rep.summary}",
        "",
        "[bold]Bear Case & Local Landmines:[/bold]",
        *[f"  • {b}" for b in rep.bear_thesis_points],
        "",
        "[bold]Objective Kill-Switches:[/bold]",
        *[f"  ▸ {k}" for k in rep.kill_switches],
        "",
        "[bold]Account Notes:[/bold]",
        *[f"  · {n}" for n in rep.tfsa_notes],
    ]
    if rep.ai_audit:
        body += ["", "[bold]OpenRouter Devil's Advocate Audit:[/bold]", rep.ai_audit]
    console.print(Panel("\n".join(body), title=f"PRE-MORTEM DEVIL'S ADVOCATE // {tk}", border_style=col))


@alpha_group.command("archive-chains")
@click.option("--limit", type=int, default=60, help="Number of liquid tickers to snapshot into data/chain_archive.db.")
def alpha_archive_chains(limit: int) -> None:
    """Snapshot expanded liquid ETF + large-cap chains into data/chain_archive.db (CBOE + Yahoo fallback)."""
    from src.chain_archive import archive_symbols

    syms = list(EXPANDED_CHAIN_UNIVERSE[: max(1, int(limit))])
    stats: dict = {}
    written = archive_symbols(syms, stats=stats)
    click.echo(
        f"Chain archive: {written:,} rows written · archived {stats.get('archived', 0)} · "
        f"already {stats.get('already', 0)} · failed {len(stats.get('failed') or [])}"
    )


@alpha_group.command("archive-occ")
@click.option("--date", "trade_date", default=None, help="Trade date YYYY-MM-DD (default: latest trading day).")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_archive_occ(ctx: click.Context, trade_date: Optional[str], as_json: bool) -> None:
    """Archive free whole-market OCC daily option volume & open interest into data/occ_oi.db."""
    from src.occ_archive import archive_occ_daily

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    res = archive_occ_daily(trade_date=trade_date)
    if use_json:
        click.echo(json.dumps(res, indent=2))
        return
    click.echo(
        f"OCC daily OI ({res['trade_date']}): {res['saved']:,} symbol(s) saved via {res['source']}"
        + (f" · notes: {', '.join(res['errors'])}" if res["errors"] else "")
    )


@alpha_group.command("occ-uoa")
@click.option("--limit", type=int, default=20)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_occ_uoa(ctx: click.Context, limit: int, as_json: bool) -> None:
    """Rank unusual option volume/OI and 1-day OI surges from data/occ_oi.db [WATCHLIST]."""
    from src.occ_archive import scan_occ_uoa

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    rows = scan_occ_uoa(limit=limit)
    if use_json:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="OCC Market-Wide Option Volume & OI Surges [WATCHLIST]", pad_edge=False)
    tbl.add_column("Ticker", style=f"bold {accent_color()}")
    tbl.add_column("Date")
    tbl.add_column("Vol", justify="right")
    tbl.add_column("OI", justify="right")
    tbl.add_column("Vol/OI", justify="right")
    tbl.add_column("1d OI Δ%", justify="right")
    tbl.add_column("Call%", justify="right")
    tbl.add_column("Read")
    for r in rows:
        vo = f"{r.vol_to_oi:.2f}x" if r.vol_to_oi is not None else "—"
        od = f"{r.oi_delta_pct:+.1f}%" if r.oi_delta_pct is not None else "—"
        cs = f"{r.call_share_pct:.0f}%" if r.call_share_pct is not None else "—"
        tbl.add_row(r.symbol, r.trade_date, f"{r.total_vol:,.0f}", f"{r.total_oi:,.0f}", vo, od, cs, r.signal_label)
    console.print(tbl)
    if not rows:
        console.print("[dim]No rows yet — run `opt alpha archive-occ` to populate data/occ_oi.db.[/dim]")
