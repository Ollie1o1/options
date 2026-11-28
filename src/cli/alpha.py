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


@alpha_group.command("fundamentals")
@click.option("--refresh", is_flag=True, help="Refresh SEC EDGAR XBRL Net Buyback Yield, GP/A & FCF/A.")
@click.option("--zip-path", default=None, help="Optional path to local SEC bulk companyfacts.zip.")
@click.option("--limit", type=int, default=40, help="Max tickers to refresh or display.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_fundamentals(ctx: click.Context, refresh: bool, zip_path: Optional[str], limit: int, as_json: bool) -> None:
    """SEC EDGAR Point-in-Time Net Share Buybacks (net of SBC), Novy-Marx GP/A & FCF/Assets [WATCHLIST]."""
    from src.alpha.edgar_events import load_quality_buyback_leaders, refresh_edgar_fundamentals

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    if refresh or zip_path:
        res = refresh_edgar_fundamentals(zip_path=zip_path, limit=limit)
        if not use_json:
            click.echo(f"SEC fundamentals: saved {res['saved']} issuer(s)" + (f" · errors: {len(res['errors'])}" if res["errors"] else ""))
    rows = load_quality_buyback_leaders(limit=limit)
    if use_json:
        click.echo(json.dumps(rows, indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="SEC XBRL Net Buybacks + Profitability (GP/A) [WATCHLIST]", pad_edge=False)
    tbl.add_column("Ticker", style=f"bold {accent_color()}")
    tbl.add_column("Filed")
    tbl.add_column("Net Buyback%", justify="right")
    tbl.add_column("GP/Assets%", justify="right")
    tbl.add_column("FCF/Assets%", justify="right")
    tbl.add_column("Company", style="dim")
    for r in rows:
        bb = f"{float(r['net_buyback_pct']):+.1f}%" if r.get("net_buyback_pct") is not None else "—"
        gp = f"{float(r['gross_profit_assets_pct']):.1f}%" if r.get("gross_profit_assets_pct") is not None else "—"
        fc = f"{float(r['fcf_assets_pct']):.1f}%" if r.get("fcf_assets_pct") is not None else "—"
        tbl.add_row(str(r["ticker"]), str(r["filed_date"]), bb, gp, fc, str(r.get("entity_name") or "")[:36])
    console.print(tbl)
    if not rows:
        console.print("[dim]Empty cache — run `opt alpha fundamentals --refresh` (uses SEC_EDGAR_CONTACT).[/dim]")


@alpha_group.command("finra-flow")
@click.option("--refresh", is_flag=True, help="Fetch latest FINRA daily Reg SHO short volume & weekly ATS dark pools.")
@click.option("--days", type=int, default=30, help="Lookback calendar days when refreshing.")
@click.option("--symbols", default=None, help="Comma-separated tickers (defaults to expanded liquid universe).")
@click.option("--limit", type=int, default=25)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_finra_flow(ctx: click.Context, refresh: bool, days: int, symbols: Optional[str], limit: int, as_json: bool) -> None:
    """FINRA Daily Reg SHO Short-Sale Volume Ratio & Weekly ATS Dark-Pool Blocks [WATCHLIST]."""
    import datetime as _dt
    from src.finra_flow import fetch_ats_for_symbols, fetch_regsho_for_symbols, scan_finra_flow_signals

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else list(EXPANDED_CHAIN_UNIVERSE)
    if refresh:
        start_d = (_dt.date.today() - _dt.timedelta(days=max(7, int(days)))).isoformat()
        r1 = fetch_regsho_for_symbols(sym_list, start_date=start_d)
        r2 = fetch_ats_for_symbols(sym_list, start_date=(_dt.date.today() - _dt.timedelta(days=max(45, int(days)))).isoformat())
        if not use_json:
            click.echo(f"FINRA flow refresh: RegSHO {r1['saved']} rows · ATS {r2['saved']} rows")
    rows = scan_finra_flow_signals(symbols=sym_list if symbols else None, limit=limit)
    if use_json:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="FINRA Daily Short-Sale Ratio & ATS Dark-Pool Blocks [WATCHLIST]", pad_edge=False)
    tbl.add_column("Ticker", style=f"bold {accent_color()}")
    tbl.add_column("Date")
    tbl.add_column("Short%", justify="right")
    tbl.add_column("20d Avg", justify="right")
    tbl.add_column("Z-Score", justify="right")
    tbl.add_column("ATS Block", justify="right")
    tbl.add_column("Read")
    for r in rows:
        a20 = f"{r.avg_20d_ratio*100:.1f}%" if r.avg_20d_ratio is not None else "—"
        zs = f"{r.ratio_z:+.2f}" if r.ratio_z is not None else "—"
        ab = f"{r.ats_avg_block:,.0f} sh" if r.ats_avg_block is not None else "—"
        tbl.add_row(r.symbol, r.trade_date, f"{r.short_ratio*100:.1f}%", a20, zs, ab, r.signal_label)
    console.print(tbl)
    if not rows:
        console.print("[dim]No rows yet — run `opt alpha finra-flow --refresh` (free, keyless FINRA API).[/dim]")


@alpha_group.command("opra-cost")
@click.option("--symbols", default="SPY,QQQ,IWM", help="Comma-separated underlying tickers.")
@click.option("--dates", default="2025-11-03", help="Comma-separated trade dates YYYY-MM-DD.")
@click.option("--time", "hhmm", default="15:45", help="ET snapshot HH:MM (default 15:45).")
@click.option("--schema", default="cbbo-1m")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_opra_cost(ctx: click.Context, symbols: str, dates: str, hhmm: str, schema: str, as_json: bool) -> None:
    """Dry-run ($0) Databento OPRA historical snapshot cost estimator."""
    from src.databento_opra import fetch_opra_snapshots

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    dts = [d.strip() for d in dates.split(",") if d.strip()]
    res = fetch_opra_snapshots(syms, dts, hhmm_et=hhmm, schema=schema, dry_run=True)
    if use_json:
        click.echo(json.dumps(res, indent=2))
        return
    click.echo(
        f"Databento OPRA estimate ({res['dataset']} {res['schema']} @ {hhmm} ET): "
        f"${res['total_est_usd']:.4f} across {len(syms)} symbol(s) x {len(dts)} date(s) "
        f"(prior spend: ${res['prior_spend_usd']:.4f} / $125 credit)"
    )


@alpha_group.command("surface")
@click.argument("ticker")
@click.option("--side", type=click.Choice(["bull_put", "bear_call"], case_sensitive=False), default="bull_put")
@click.option("--min-dte", type=int, default=21, show_default=True)
@click.option("--max-dte", type=int, default=125, show_default=True)
@click.option("--max-expirations", type=int, default=4, show_default=True)
@click.option("--max-friction", type=float, default=0.25, show_default=True)
@click.option("--limit", type=int, default=12, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def alpha_surface(
    ctx: click.Context,
    ticker: str,
    side: str,
    min_dte: int,
    max_dte: int,
    max_expirations: int,
    max_friction: float,
    limit: int,
    as_json: bool,
) -> None:
    """Search the live (expiry x delta x width) credit-spread surface via CBOE/Alpaca/Yahoo [TAXABLE_ONLY]."""
    from src.radar.live_options import options_execution_window_guidance
    from src.structure.chain_data import load_live_chain_slices
    from src.tui.options_engine import scan_surface_verticals

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    sym = ticker.upper().strip()
    diag: dict = {}
    slices = load_live_chain_slices(
        sym,
        min_dte=min_dte,
        max_dte=max_dte,
        max_expirations=max_expirations,
        diagnostics=diag,
    )
    spot = float(slices[0].spot) if slices and slices[0].spot else 0.0
    ex_div: Optional[str] = None
    try:
        from src.alpaca_data import has_credentials, upcoming_ex_dividend_dates

        if has_credentials():
            ex_map = upcoming_ex_dividend_dates([sym])
            if sym in ex_map:
                ex_div = ex_map[sym][0]
    except Exception:
        pass

    cands = scan_surface_verticals(
        slices,
        spot=spot,
        side=side.lower(),
        max_friction_pct=max_friction,
        ex_div_date=ex_div,
    )[: max(1, int(limit))]
    exec_win = options_execution_window_guidance()

    if use_json:
        click.echo(
            json.dumps(
                {
                    "ticker": sym,
                    "spot": spot,
                    "side": side.lower(),
                    "provider": diag.get("provider"),
                    "expirations": [s.expiration for s in slices],
                    "ex_div_date": ex_div,
                    "execution_window": exec_win,
                    "candidates": cands,
                },
                indent=2,
            )
        )
        return

    console = make_console(no_color=no_color)
    tbl = Table(
        title=(
            f"{sym} {side.upper()} Surface Search · Spot ${spot:.2f} · "
            f"Feed {diag.get('provider') or '?'} · {exec_win['window']} [TAXABLE_ONLY]"
        ),
        pad_edge=False,
    )
    tbl.add_column("Exp (DTE)", style=f"bold {accent_color()}", no_wrap=True)
    tbl.add_column("Strikes (W%)", no_wrap=True)
    tbl.add_column("Δ", justify="right", no_wrap=True)
    tbl.add_column("Mid/Cross", justify="right", no_wrap=True)
    tbl.add_column("Fric", justify="right", no_wrap=True)
    tbl.add_column("Dep", justify="right", no_wrap=True)
    tbl.add_column("Skew", justify="right", no_wrap=True)
    tbl.add_column("Cr/Δ", justify="right", no_wrap=True)
    tbl.add_column("RoR", justify="right", no_wrap=True)
    tbl.add_column("Flags", style="dim")
    for c in cands:
        dp = f"{c['min_depth']:.0f}" if c.get("min_depth") is not None else "—"
        sk = f"{c['skew_capture_vp']:+.1f}" if c.get("skew_capture_vp") is not None else "—"
        cd = f"{c['credit_per_delta']:.1f}" if c.get("credit_per_delta") is not None else "—"
        exp_short = str(c["expiration"])[2:]
        tbl.add_row(
            f"{exp_short} ({c['dte']}d)",
            f"{c['short_strike']:g}/{c['long_strike']:g} ({c['width_pct']*100:.0f}%)",
            f"{c['short_delta']:.2f}",
            f"{c['credit_mid']:.2f}/{c['credit_cross']:.2f}",
            f"{c['friction_pct']*100:.1f}%",
            dp,
            sk,
            cd,
            f"{c['ror_pct']:.1f}%",
            ",".join(f for f in c["flags"] if f != "TAXABLE_ONLY") or "CLEAN",
        )
    console.print(tbl)
    console.print(f"[dim]{exec_win['note']}[/dim]")
    if not cands:
        console.print(f"[dim]No two-sided spreads <= {max_friction*100:.0f}% friction across {len(slices)} expiration(s).[/dim]")
