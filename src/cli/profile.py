"""`opt profile` — Manage Wealthsimple TFSA profile, holdings, cash ETF, and CSV import."""
from __future__ import annotations

import json
from typing import Optional

import click
from rich.panel import Panel
from rich.table import Table

from src.equity_playbooks.rebalance import parse_holdings_spec
from src.profile import (
    DEFAULT_PROFILE_PATH,
    estimate_cash_yield_annual,
    find_latest_wealthsimple_csv,
    import_wealthsimple_csv,
    load_profile,
    save_profile,
)
from src.tfsa_tax import compute_tfsa_safety_report
from .render import make_console


@click.command("profile")
@click.option("--tfsa", type=float, default=None, help="Set TFSA Core sleeve balance in USD.")
@click.option("--taxable", type=float, default=None, help="Set Taxable / IBKR balance in USD.")
@click.option("--risk", type=float, default=None, help="Default per-trade risk budget in USD.")
@click.option("--broker", type=click.Choice(["wealthsimple_usd", "ibkr_canada"], case_sensitive=False), default=None)
@click.option("--account-type", type=click.Choice(["tfsa", "taxable"], case_sensitive=False), default=None)
@click.option("--cash-etf", default=None, help="Defensive cash T-bill ETF (e.g. SGOV, BIL, CBIL.TO).")
@click.option("--room-cad", type=float, default=None, help="Set annual CRA TFSA contribution room in CAD.")
@click.option("--contributed-cad", type=float, default=None, help="Set YTD TFSA contributions made in CAD.")
@click.option("--holdings", default=None, help="Set current Core ETF shares (e.g. XLE:10,XLV:4,XLP:5).")
@click.option("--import-ws", "ws_csv", default=None, help="Import Wealthsimple Activity/Statement CSV.")
@click.option("--watch-downloads", "watch_dl", is_flag=True, help="Auto-find & import newest Wealthsimple CSV in ~/Downloads.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def profile_cmd(
    ctx: click.Context,
    tfsa: Optional[float],
    taxable: Optional[float],
    risk: Optional[float],
    broker: Optional[str],
    account_type: Optional[str],
    cash_etf: Optional[str],
    room_cad: Optional[float],
    contributed_cad: Optional[float],
    holdings: Optional[str],
    ws_csv: Optional[str],
    watch_dl: bool,
    as_json: bool,
) -> None:
    """View or configure your Wealthsimple TFSA profile, Core holdings, SGOV cash ETF & CSV sync."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    profile_path = (ctx.obj or {}).get("profile_path", DEFAULT_PROFILE_PATH)
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")

    if watch_dl and not ws_csv:
        found = find_latest_wealthsimple_csv()
        if not found:
            click.echo("No Wealthsimple activity/statement CSV found in ~/Downloads.")
            return
        ws_csv = found

    if ws_csv:
        summary = import_wealthsimple_csv(ws_csv, profile_path=profile_path, own_picks_db=own_picks_db)
        if use_json:
            click.echo(json.dumps(summary, indent=2))
        else:
            click.echo(
                f"✓ Imported {summary['file']}: {summary['core_updates']} Core ETF update(s), "
                f"{summary['picks_logged']} pick(s) logged ({summary['skipped_duplicates']} duplicate(s) skipped)."
            )
        return

    prof = load_profile(profile_path)
    changed = False
    if tfsa is not None:
        prof.tfsa_usd = max(100.0, float(tfsa))
        changed = True
    if taxable is not None:
        prof.taxable_usd = max(0.0, float(taxable))
        changed = True
    if risk is not None:
        prof.risk_budget_usd = max(25.0, float(risk))
        changed = True
    if broker is not None:
        prof.broker = broker.lower()
        changed = True
    if account_type is not None:
        prof.account_type = account_type.lower()
        changed = True
    if cash_etf is not None and cash_etf.strip():
        prof.cash_etf = cash_etf.strip().upper()
        changed = True
    if room_cad is not None:
        prof.tfsa_room_cad = max(0.0, float(room_cad))
        changed = True
    if contributed_cad is not None:
        prof.tfsa_contributed_ytd_cad = max(0.0, float(contributed_cad))
        changed = True
    if holdings is not None:
        prof.core_holdings = parse_holdings_spec(holdings)
        changed = True
    if changed:
        save_profile(prof, path=profile_path)

    y_ann, y_lbl = estimate_cash_yield_annual()
    from src.own_picks import list_equity_picks
    from src.radar import load_cached_radar

    snap = load_cached_radar()
    safety = compute_tfsa_safety_report(
        profile=prof,
        core_allocation=snap.core_allocation if snap else (),
        equity_picks=list_equity_picks(db_path=own_picks_db),
    )
    if use_json:
        click.echo(json.dumps({**prof.to_dict(), "cash_yield_ann": y_ann, "cash_yield_label": y_lbl, "tfsa_safety": safety.to_dict()}, indent=2))
        return

    console = make_console(no_color=no_color)
    tfsa_note = (
        "CRA TFSA rules enforced: shares, ETFs, covered calls & long options only (spreads locked)."
        if prof.account_type == "tfsa"
        else "Taxable/margin account: spreads unlocked."
    )
    room_s = (
        f"${safety.room_remaining_cad:,.0f} CAD left (of ${safety.room_cad:,.0f})"
        if safety.room_remaining_cad is not None and safety.room_cad is not None
        else "not set (--room-cad)"
    )
    console.print(
        Panel(
            f"[bold]Broker:[/bold] {prof.broker} · [bold]Default Account:[/bold] {prof.account_type.upper()}\n"
            f"[bold]TFSA Core USD:[/bold] ${prof.tfsa_usd:,.2f} · [bold]Taxable USD:[/bold] ${prof.taxable_usd:,.2f} · "
            f"[bold]Per-Trade Risk:[/bold] ${prof.risk_budget_usd:,.2f}\n"
            f"[bold]Cash Buffer ETF:[/bold] {prof.cash_etf} (est. yield {y_lbl}) · "
            f"[bold]Last Rebalance:[/bold] {prof.last_rebalance_date or 'not recorded'}\n"
            f"[bold]CRA Safety:[/bold] {safety.cra_status} ({safety.trades_ytd} YTD trades) · "
            f"[bold]15% US Div Drag:[/bold] ~${safety.total_tax_drag_usd_yr:,.2f}/yr ({safety.weighted_tax_drag_bps:.0f} bps) · "
            f"[bold]TFSA Room:[/bold] {room_s}\n"
            f"[dim]{tfsa_note}[/dim]",
            title="TRADER PROFILE (data/tfsa_profile.json)",
            border_style="#5ec98d" if changed else "#82aad2",
        )
    )
    if prof.core_holdings:
        tbl = Table(title="Saved Core TFSA Holdings", pad_edge=False)
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Shares", justify="right")
        for tk, sh in sorted(prof.core_holdings.items()):
            tbl.add_row(tk, f"{sh:g}")
        console.print(tbl)
    else:
        console.print("[dim]No Core holdings saved yet — run `opt rebalance --apply` or `opt profile --import-ws <file.csv>`.[/dim]")
