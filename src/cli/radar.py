"""`opt radar` — Unified Stock + Option Opportunity Table."""
from __future__ import annotations

import json

import click

from src.radar import load_cached_radar, run_radar
from .render import make_console, render_radar_table, render_ribbon


@click.command("radar")
@click.option("--playbook", default="all", help="Filter by playbook id or substring (e.g. pullback, outlook, bear_call, holdings).")
@click.option("--asset", type=click.Choice(["all", "stock", "etf", "option"], case_sensitive=False), default="all")
@click.option("--tier", type=click.Choice(["all", "validated", "lead", "context"], case_sensitive=False), default="all")
@click.option("--min-confluence", type=int, default=1, help="Minimum confluence count.")
@click.option("--limit", type=int, default=20, help="Max rows to display.")
@click.option("--refresh", is_flag=True, help="Re-run scanners instead of reading cache.")
@click.option("--live-options", is_flag=True, help="Also scan live 60-120d wide ETF Bull Puts (Test O1).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def radar_cmd(
    ctx: click.Context,
    playbook: str,
    asset: str,
    tier: str,
    min_confluence: int,
    limit: int,
    refresh: bool,
    live_options: bool,
    as_json: bool,
) -> None:
    """Unified Stock + Option Opportunity Radar sorted by Tier -> Confluence -> Friction."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")

    snap = None if refresh else load_cached_radar(cache_path)
    if snap is None:
        snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)

    rows = list(snap.opportunities)
    if live_options:
        from src.radar.live_options import scan_live_etf_wide_spreads

        live_rows, _warns = scan_live_etf_wide_spreads()
        rows = rows + live_rows
    if playbook.lower() != "all":
        q = playbook.lower()
        rows = [o for o in rows if q in o.playbook_id.lower() or q in o.playbook_label.lower()]
    if asset.lower() != "all":
        rows = [o for o in rows if o.asset_class.lower() == asset.lower()]
    if tier.lower() != "all":
        rows = [o for o in rows if o.evidence_tier.lower() == tier.lower()]
    if min_confluence > 1:
        rows = [o for o in rows if o.confluence_count >= min_confluence]
    rows = rows[: max(1, limit)]

    if use_json:
        click.echo(json.dumps({"generated_at": snap.generated_at, "count": len(rows), "opportunities": [r.to_dict() for r in rows]}, indent=2))
        return

    console = make_console(no_color=no_color)
    console.print(render_ribbon(snap))
    console.print()
    if not rows:
        console.print("[dim]No Radar opportunities match the selected filters.[/dim]")
        return
    console.print(render_radar_table(rows, title=f"Unified Radar ({len(rows)} shown)"))
